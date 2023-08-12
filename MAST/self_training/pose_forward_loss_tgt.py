import numpy as np
import torch
import torch.nn.functional as F
from MAST.models.loss_ops import (
    ExplicitInterClassGraphLoss, TCO_init_from_boxes,
    TCO_init_from_boxes_zup_autodepth, cal_gt_dR_vxyz, loss_CO_symmetric,
    loss_refiner_CO_disentangled, loss_refiner_CO_disentangled_bindelta,
    loss_refiner_CO_disentangled_quaternions, xentropy)
from MAST.models.loss_ops import compute_ADD_L1_loss
from MAST.lib3d.transform_ops import add_noise
from MAST.models.bin_delta_Rt import (Rts_to_bin_delta_batch,
                                          bin_delta_to_Rs_batch_by_idx,
                                          bin_delta_to_ts_batch_by_idx,
                                          grid_xyz, sample_rotations_60)

R_bin_ctrs = torch.tensor(sample_rotations_60("matrix")).float().cuda()
xy_bin_ctrs, z_bin_ctrs = grid_xyz()
xy_bin_ctrs = torch.tensor(xy_bin_ctrs).float().cuda()
z_bin_ctrs = torch.tensor(z_bin_ctrs).float().cuda()
EICGLoss = ExplicitInterClassGraphLoss()

def cast(obj):
    return obj.cuda(non_blocking=True)


def h_pose_tgt(model, mesh_db, data, meters,
           cfg, n_iterations=1, input_generator='fixed'):

    images = data['images'].cuda()
    K = data['K'].cuda()
    TCO_gt = data['TCO_gt'].cuda()
    TCO_pred_from_selection = data['TCO_pred'].cuda()

    TCO_gt = TCO_pred_from_selection  #! ATTENTION: Using pseudo label

    labels = np.array(data['labels'])
    bboxes = data['bboxes'].cuda()
    zb_pseudo = data['zb_pseudo'].cuda()

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

    losses_TCO_iter = []
    losses_cls_iter = []
    for n in range(n_iterations):
        iter_outputs = outputs[f'iteration={n+1}']
        K_crop = iter_outputs['K_crop']
        TCO_input = iter_outputs['TCO_input']
        TCO_pred = iter_outputs['TCO_output']
        model_outputs = iter_outputs['model_outputs']
        dR_bin, dx_bin, dy_bin, dz_bin = iter_outputs['bins']
        dR_delta, dx_delta, dy_delta, dz_delta = iter_outputs['deltas']
        
        _, TCO_assign = loss_CO_symmetric(TCO_possible_gt, TCO_pred, points)
        gt_dR, gt_vxvyvz = cal_gt_dR_vxyz(TCO_input, TCO_assign, K_crop)
        Rb_gt, _, xb_gt, yb_gt, zb_gt, _, _, _ = Rts_to_bin_delta_batch(gt_dR, gt_vxvyvz, R_bin_ctrs, xy_bin_ctrs, z_bin_ctrs)

        knnR = 4
        knnt = 7
        R_soft_idx = torch.topk(Rb_gt, k=knnR, dim=-1)[1]
        x_soft_idx = torch.topk(xb_gt, k=knnt, dim=-1)[1]
        y_soft_idx = torch.topk(yb_gt, k=knnt, dim=-1)[1]
        z_soft_idx = torch.topk(zb_gt, k=knnt, dim=-1)[1]
        pose_outputs_soft_dR = []
        pose_outputs_soft_vxvyvz = []
        for i in range(knnR):
            pose_outputs_soft_dR.append(bin_delta_to_Rs_batch_by_idx(dR_delta, R_soft_idx[:,i], R_bin_ctrs))
        for i in range(knnt):
            pose_outputs_soft_vxvyvz.append(bin_delta_to_ts_batch_by_idx(
                                            dx_delta,x_soft_idx[:,i],
                                            dy_delta, y_soft_idx[:,i],
                                            dz_delta, z_soft_idx[:,i],
                                            xy_bin_ctrs, z_bin_ctrs))
        pose_outputs_soft = (pose_outputs_soft_dR, pose_outputs_soft_vxvyvz)
        
        if cfg.loss_disentangled:
            if cfg.n_pose_dims == 9:
                loss_fn = loss_refiner_CO_disentangled_bindelta
            elif cfg.n_pose_dims == 7:
                loss_fn = loss_refiner_CO_disentangled_quaternions
            else:
                raise ValueError
            pose_outputs = model_outputs  # ['pose']
            loss_TCO_iter = loss_fn(
                TCO_possible_gt=TCO_possible_gt,
                TCO_input=TCO_input,
                refiner_outputs=pose_outputs_soft,
                K_crop=K_crop, points=points,knnR=knnR, knnt=knnt
            )
        else:
            loss_TCO_iter = compute_ADD_L1_loss(
                TCO_possible_gt[:, 0], TCO_pred, points
            )

        #meters[f'loss_TCO-iter={n+1}'].add(loss_TCO_iter.mean().item())
        losses_TCO_iter.append(loss_TCO_iter)
        loss_cls_iter = xentropy(dR_bin, Rb_gt) +\
                        xentropy(dx_bin, xb_gt) +\
                        xentropy(dy_bin, yb_gt) +\
                        xentropy(dz_bin, zb_gt)
        losses_cls_iter.append(loss_cls_iter)

    loss_TCO = losses_TCO_iter[0]  # torch.cat(losses_TCO_iter).mean()
    loss_cls = losses_cls_iter[0]
    loss = loss_TCO * 0.0 + loss_cls
    meters['loss_TCO'].add(loss_TCO.item())
    meters['loss_cls'].add(loss_cls.item())
    meters['loss_total'].add(loss.item())
    if cfg.icg_loss:
        loss_icg = EICGLoss(torch.max(zb_gt, dim=-1)[1], iter_outputs['z_feat'])
        meters['loss_icg'].add(loss_icg.item())
        loss += loss_icg
    return loss, iter_outputs['R_feat'], iter_outputs['z_feat'], torch.max(Rb_gt, dim=-1)[1], torch.max(zb_gt, dim=-1)[1]
