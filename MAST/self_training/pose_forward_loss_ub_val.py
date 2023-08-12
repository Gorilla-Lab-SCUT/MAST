import numpy as np
import torch
import torch.nn.functional as F
from MAST.models.loss_ops import (
    TCO_init_from_boxes, TCO_init_from_boxes_zup_autodepth, cal_gt_dR_vxyz,
    loss_CO_symmetric, loss_refiner_CO_disentangled,
    loss_refiner_CO_disentangled_bindelta,
    loss_refiner_CO_disentangled_quaternions, xentropy)
from MAST.models.loss_ops import compute_ADD_L1_loss
from MAST.lib3d.transform_ops import add_noise
from MAST.models.bin_delta_Rt import (Rts_to_bin_delta_batch,
                                          bin_delta_to_Rts_batch_by_idx,
                                          grid_xyz, sample_rotations_60)

R_bin_ctrs = torch.tensor(sample_rotations_60("matrix")).float().cuda()
xy_bin_ctrs, z_bin_ctrs = grid_xyz()
xy_bin_ctrs = torch.tensor(xy_bin_ctrs).float().cuda()
z_bin_ctrs = torch.tensor(z_bin_ctrs).float().cuda()

def cast(obj):
    return obj.cuda(non_blocking=True)


def h_pose_ub_val(model, mesh_db, data, meters,
           cfg, n_iterations=1, input_generator='fixed'):

    batch_size, _, h, w = data.images.shape

    images = cast(data.images).float() / 255.
    K = cast(data.K).float()
    TCO_gt = cast(data.TCO).float()
    labels = np.array([obj['name'] for obj in data.objects])
    bboxes = cast(data.bboxes).float()

    meshes = mesh_db.select(labels)
    points = meshes.sample_points(cfg.n_points_loss, deterministic=False)
    TCO_possible_gt = TCO_gt.unsqueeze(1) @ meshes.symmetries

    if input_generator == 'fixed':
        TCO_init = TCO_init_from_boxes(z_range=(1.0, 1.0), boxes=bboxes, K=K)
    elif input_generator == 'gt+noise':
        TCO_init = add_noise(TCO_possible_gt[:, 0], euler_deg_std=[15, 15, 15], trans_std=[0.01, 0.01, 0.05])
    elif input_generator == 'fixed+trans_noise':
        assert cfg.init_method == 'z-up+auto-depth'
        TCO_init = TCO_init_from_boxes_zup_autodepth(bboxes, points, K)
        TCO_init = add_noise(TCO_init,
                             euler_deg_std=[0, 0, 0],
                             trans_std=[0.01, 0.01, 0.05])
    else:
        raise ValueError('Unknown input generator', input_generator)

    # model.module.enable_debug()
    outputs = model(images=images, K=K, labels=labels,
                    TCO=TCO_init, n_iterations=n_iterations)
    # raise ValueError

    for n in range(n_iterations):
        iter_outputs = outputs[f'iteration={n+1}']
        K_crop = iter_outputs['K_crop']
        TCO_input = iter_outputs['TCO_input']
        TCO_pred = iter_outputs['TCO_output']
        model_outputs = iter_outputs['model_outputs']
        dR_bin, dx_bin, dy_bin, dz_bin = iter_outputs['bins']
        
        gt_dR, gt_vxvyvz = cal_gt_dR_vxyz(TCO_input, TCO_possible_gt[:, 0], K_crop)
        Rb_gt, _, xb_gt, yb_gt, zb_gt, _,_,_, = Rts_to_bin_delta_batch(gt_dR, gt_vxvyvz, R_bin_ctrs, xy_bin_ctrs, z_bin_ctrs)
        classification_loss = xentropy(dR_bin, Rb_gt) +\
                                xentropy(dx_bin, xb_gt) +\
                                xentropy(dy_bin, yb_gt) +\
                                xentropy(dz_bin, zb_gt)
    R_l = torch.argmax(Rb_gt, dim=-1); R_p = torch.argmax(dR_bin, dim=-1)
    x_l = torch.argmax(xb_gt, dim=-1); x_p = torch.argmax(dx_bin, dim=-1)
    y_l = torch.argmax(yb_gt, dim=-1); y_p = torch.argmax(dy_bin, dim=-1)
    z_l = torch.argmax(zb_gt, dim=-1); z_p = torch.argmax(dz_bin, dim=-1)
    R_acc = (R_l == R_p).sum()
    x_acc = (x_l == x_p).sum()
    y_acc = (y_l == y_p).sum()
    z_acc = (z_l == z_p).sum()
    accs = dict(
        R_acc=R_acc,
        x_acc=x_acc,
        y_acc=y_acc,
        z_acc=z_acc,
        batch_size=len(R_l)
    )
    loss = classification_loss
    meters['loss_total'].add(loss.item())
    return loss, accs
