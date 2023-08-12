import numpy as np
import torch
import torch.nn.functional as F
from MAST.models.loss_ops import (TCO_init_from_boxes,
                                         TCO_init_from_boxes_zup_autodepth,
                                         cal_gt_dR_vxyz, loss_CO_symmetric)
from MAST.lib3d.transform_ops import add_noise
from MAST.models.bin_delta_Rt import (Rts_to_bin_delta_batch,
                                          bin_delta_to_Rs_batch_by_idx,
                                          bin_delta_to_ts_batch_by_idx,
                                          grid_xyz, sample_rotations_60)
from .st_utils import \
    select_batch_idx_by_z_for_self_regression

R_bin_ctrs = torch.tensor(sample_rotations_60("matrix")).float().cuda()
xy_bin_ctrs, z_bin_ctrs = grid_xyz()
xy_bin_ctrs = torch.tensor(xy_bin_ctrs).float().cuda()
z_bin_ctrs = torch.tensor(z_bin_ctrs).float().cuda()

def cast(obj):
    return obj.cuda(non_blocking=True)


def h_pose_select_tgt(model_coarse, model_refiner, mesh_db, data, meters,
           cfg, n_iterations=1, input_generator='fixed', self_train_cfg=None):

    images_origin = cast(data.images_origin).float() / 255.
    mask = cast(data.mask).float()

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
    outputs = model_coarse(images=images, K=K, labels=labels,
                    TCO=TCO_init, n_iterations=n_iterations)

    iter_outputs = outputs[f'iteration={1}']
    K_crop = iter_outputs['K_crop']
    TCO_input = iter_outputs['TCO_input']
    TCO_pred = iter_outputs['TCO_output']
    model_outputs = iter_outputs['model_outputs']
    dR_bin, dx_bin, dy_bin, dz_bin = iter_outputs['bins']
    dR_delta, dx_delta, dy_delta, dz_delta = iter_outputs['deltas']

    selection = select_batch_idx_by_z_for_self_regression(dR_bin,
                                                          dx_bin,
                                                          dy_bin,
                                                          dz_bin,
                                                          cfg,
                                                          labels,
                                                          self_train_cfg.obj_threshold)

    ## refiner
    if model_refiner is not None:
        outputs = model_refiner(images=images, K=K, labels=labels,
                        TCO=TCO_pred, n_iterations=4)
        iter_outputs = outputs[f'iteration={1}']
        K_crop = iter_outputs['K_crop']
        TCO_input = iter_outputs['TCO_input']
        TCO_pred = iter_outputs['TCO_output']
        model_outputs = iter_outputs['model_outputs']
        dR_bin, dx_bin, dy_bin, dz_bin = iter_outputs['bins']
        dR_delta, dx_delta, dy_delta, dz_delta = iter_outputs['deltas']


    if selection is not None:
        (all_meet_batch_idx, 
        R_full_batch_label, 
        x_full_batch_label, 
        y_full_batch_label, 
        z_full_batch_label) = selection

        pseudo_R = R_bin_ctrs[R_full_batch_label[all_meet_batch_idx].detach()]
        pseudo_x = xy_bin_ctrs[x_full_batch_label[all_meet_batch_idx].detach()].unsqueeze(-1)
        pseudo_y =xy_bin_ctrs[y_full_batch_label[all_meet_batch_idx].detach()].unsqueeze(-1)
        pseudo_z = z_bin_ctrs[z_full_batch_label[all_meet_batch_idx].detach()].unsqueeze(-1)
        pseudo_vxvyvz = torch.cat([pseudo_x, pseudo_y, pseudo_z], dim=-1)

        Rb_pseudo, _, xb_pseudo, yb_pseudo, zb_pseudo, _,_,_, = Rts_to_bin_delta_batch(
            pseudo_R,
            pseudo_vxvyvz,
            R_bin_ctrs, xy_bin_ctrs, z_bin_ctrs)

        ### this block change pseudo label to gt label in order to explore upper bound of the model
        _, TCO_assign = loss_CO_symmetric(TCO_possible_gt, TCO_pred, points)
        gt_dR, gt_vxvyvz = cal_gt_dR_vxyz(TCO_input, TCO_assign, K_crop)
        Rb_gt, _, xb_gt, yb_gt, zb_gt, _,_,_, = Rts_to_bin_delta_batch(gt_dR, gt_vxvyvz, R_bin_ctrs, xy_bin_ctrs, z_bin_ctrs)
        Rb_gt = Rb_gt[all_meet_batch_idx]
        xb_gt = xb_gt[all_meet_batch_idx]
        yb_gt = yb_gt[all_meet_batch_idx]
        zb_gt = zb_gt[all_meet_batch_idx]
        # R_l = torch.argmax(Rb_gt, dim=-1); R_p = torch.argmax(Rb_pseudo, dim=-1)
        # x_l = torch.argmax(xb_gt, dim=-1); x_p = torch.argmax(xb_pseudo, dim=-1)
        # y_l = torch.argmax(yb_gt, dim=-1); y_p = torch.argmax(yb_pseudo, dim=-1)
        z_l = torch.argmax(zb_gt, dim=-1); z_p = torch.argmax(zb_pseudo, dim=-1)
        # R_acc = (R_l == R_p).sum()
        # x_acc = (x_l == x_p).sum()
        # y_acc = (y_l == y_p).sum()
        z_acc = (z_l == z_p).sum()
        print(f"met num:{len(all_meet_batch_idx)}, sudo z acc: {z_acc/len(z_l):.2f},{labels[0]}")
        # classification_loss = \
        #     xentropy(dR_bin[all_meet_batch_idx, :], Rb_gt) +\
        #     xentropy(dx_bin[all_meet_batch_idx, :], xb_gt) * torch.tensor([0.0]).cuda() +\
        #     xentropy(dy_bin[all_meet_batch_idx, :], yb_gt) * torch.tensor([0.0]).cuda() +\
        #     xentropy(dz_bin[all_meet_batch_idx, :], zb_gt) * torch.tensor([0.0]).cuda()
        ### this block change pseudo label to gt label in order to explore upper bound of the model


        images = images_origin[[all_meet_batch_idx]].cpu()
        K = K[[all_meet_batch_idx]].cpu()
        TCO_gt = TCO_gt[[all_meet_batch_idx]].cpu()
        TCO_pred = TCO_pred[[all_meet_batch_idx]].cpu()
        labels = labels[[all_meet_batch_idx.cpu().numpy()]]
        bboxes = bboxes[[all_meet_batch_idx]].cpu()
        zb_pseudo = zb_pseudo.cpu()
        masks = mask[[all_meet_batch_idx]].cpu()

        selected_data = dict(
            images=images,
            K=K,
            TCO_gt=TCO_gt,
            TCO_pred=TCO_pred,
            labels=labels,
            bboxes=bboxes,
            zb_pseudo=zb_pseudo,
            masks=masks
        )

        return selected_data
    else:
        return None
