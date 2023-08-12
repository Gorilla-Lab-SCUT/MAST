import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from MAST.lib3d.rotations import (compute_rotation_matrix_from_ortho6d,
                                      compute_rotation_matrix_from_quaternions)
from MAST.lib3d.transform_ops import transform_pts
from torch.nn.modules.activation import LogSoftmax

from .bin_delta_Rt import R_bin_ctrs

l1 = lambda diff: diff.abs()
l2 = lambda diff: diff ** 2
smooth_l1 = lambda pred, target: F.smooth_l1_loss(pred, target, reduce=False)


def apply_imagespace_predictions(TCO, K, vxvyvz, dRCO):
    assert TCO.shape[-2:] == (4, 4)
    assert K.shape[-2:] == (3, 3)
    assert dRCO.shape[-2:] == (3, 3)
    assert vxvyvz.shape[-1] == 3
    TCO_out = TCO.clone()

    # Translation in image space
    zsrc = TCO[:, 2, [3]]
    vz = vxvyvz[:, [2]]
    ztgt = vz * zsrc

    vxvy = vxvyvz[:, :2]
    fxfy = K[:, [0, 1], [0, 1]]
    xsrcysrc = TCO[:, :2, 3]
    TCO_out[:, 2, 3] = ztgt.flatten()
    TCO_out[:, :2, 3] = ((vxvy / fxfy) + (xsrcysrc / zsrc.repeat(1, 2))) * ztgt.repeat(1, 2)

    # Rotation in camera frame
    # TC1' = TC2' @  T2'1' where TC2' = T22' = dCRO is predicted and T2'1'=T21=TC1
    TCO_out[:, :3, :3] = dRCO @ TCO[:, :3, :3]
    return TCO_out


def loss_CO_symmetric(TCO_possible_gt, TCO_pred, points, l1_or_l2=l1):
    bsz = TCO_possible_gt.shape[0]
    assert TCO_possible_gt.shape[0] == bsz
    assert len(TCO_possible_gt.shape) == 4 and TCO_possible_gt.shape[-2:] == (4, 4)
    assert TCO_pred.shape == (bsz, 4, 4)
    assert points.dim() == 3 and points.shape[0] == bsz and points.shape[-1] == 3

    TCO_points_possible_gt = transform_pts(TCO_possible_gt, points)
    TCO_pred_points = transform_pts(TCO_pred, points)
    losses_possible = l1_or_l2((TCO_pred_points.unsqueeze(1) - TCO_points_possible_gt).flatten(-2, -1)).mean(-1)
    loss, min_id = losses_possible.min(dim=1)
    TCO_assign = TCO_possible_gt[torch.arange(bsz), min_id]
    return loss, TCO_assign


def loss_refiner_CO_disentangled(TCO_possible_gt,
                                 TCO_input, refiner_outputs,
                                 K_crop, points):
    bsz = TCO_possible_gt.shape[0]
    assert TCO_possible_gt.shape[0] == bsz  # [bs, 64, 4, 4]
    assert TCO_input.shape[0] == bsz
    # assert refiner_outputs.shape == (bsz, 9)
    assert K_crop.shape == (bsz, 3, 3)
    assert points.dim() == 3 and points.shape[0] == bsz and points.shape[-1] == 3
    assert TCO_possible_gt.dim() == 4 and TCO_possible_gt.shape[-2:] == (4, 4)

    # dR = compute_rotation_matrix_from_ortho6d(refiner_outputs[:, 0:6])
    # vxvyvz = refiner_outputs[:, 6:9]

    dR, vxvyvz = refiner_outputs
    TCO_gt = TCO_possible_gt[:, 0]

    TCO_pred_orn = TCO_gt.clone()
    TCO_pred_orn[:, :3, :3] = dR @ TCO_input[:, :3, :3]

    TCO_pred_xy = TCO_gt.clone()
    z_gt = TCO_gt[:, 2, [3]]
    z_input = TCO_input[:, 2, [3]]
    vxvy = vxvyvz[:, :2]
    fxfy = K_crop[:, [0, 1], [0, 1]]
    xsrcysrc = TCO_input[:, :2, 3]
    TCO_pred_xy[:, :2, 3] = ((vxvy / fxfy) + (xsrcysrc / z_input.repeat(1, 2))) * z_gt.repeat(1, 2)

    TCO_pred_z = TCO_gt.clone()
    vz = vxvyvz[:, [2]]
    TCO_pred_z[:, [2], [3]] = vz * z_input

    loss_orn, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_orn, points, l1_or_l2=l1)
    loss_xy, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_xy, points, l1_or_l2=l1)
    loss_z, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_z, points, l1_or_l2=l1)
    return loss_orn + loss_xy + loss_z


def loss_refiner_CO_disentangled_bindelta(TCO_possible_gt,
                                 TCO_input, refiner_outputs,
                                 K_crop, points, knnR, knnt):
    bsz = TCO_possible_gt.shape[0]
    assert TCO_possible_gt.shape[0] == bsz  # [bs, 64, 4, 4]
    assert TCO_input.shape[0] == bsz
    # assert refiner_outputs.shape == (bsz, 9)
    assert K_crop.shape == (bsz, 3, 3)
    assert points.dim() == 3 and points.shape[0] == bsz and points.shape[-1] == 3
    assert TCO_possible_gt.dim() == 4 and TCO_possible_gt.shape[-2:] == (4, 4)


    TCO_gt = TCO_possible_gt[:, 0]
    z_input = TCO_input[:, 2, [3]]
    dR, vxvyvz = refiner_outputs

    def orn_loss(dR):
        TCO_pred_orn = TCO_gt.clone()
        TCO_pred_orn[:, :3, :3] = dR @ TCO_input[:, :3, :3]
        loss_orn, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_orn, points, l1_or_l2=l1)
        return loss_orn

    def xy_loss(vxvy):
        TCO_pred_xy = TCO_gt.clone()
        z_gt = TCO_gt[:, 2, [3]]
        fxfy = K_crop[:, [0, 1], [0, 1]]
        xsrcysrc = TCO_input[:, :2, 3]
        TCO_pred_xy[:, :2, 3] = ((vxvy / fxfy) + (xsrcysrc / z_input.repeat(1, 2))) * z_gt.repeat(1, 2)
        loss_xy, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_xy, points, l1_or_l2=l1)
        return loss_xy

    def z_loss(vz):
        TCO_pred_z = TCO_gt.clone()
        TCO_pred_z[:, [2], [3]] = vz * z_input
        loss_z, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_z, points, l1_or_l2=l1)
        return loss_z

    losses_orn = []
    losses_xy = []
    losses_z = []
    for i in range(knnR):
        losses_orn.append(orn_loss(dR[i]))

    for i in range(knnt):
        vxvy = vxvyvz[i][:, :2]
        vz = vxvyvz[i][:, [2]]
        losses_xy.append(xy_loss(vxvy))
        losses_z.append(z_loss(vz))

    loss_orn = torch.cat(losses_orn).mean()
    loss_xy = torch.cat(losses_xy).mean()
    loss_z = torch.cat(losses_z).mean()
    return loss_orn + loss_xy + loss_z


def loss_refiner_CO_disentangled_quaternions(TCO_possible_gt,
                                             TCO_input, refiner_outputs,
                                             K_crop, points):
    bsz = TCO_possible_gt.shape[0]
    assert TCO_possible_gt.shape[0] == bsz
    assert TCO_input.shape[0] == bsz
    assert refiner_outputs.shape == (bsz, 7)
    assert K_crop.shape == (bsz, 3, 3)
    assert points.dim() == 3 and points.shape[0] == bsz and points.shape[-1] == 3
    assert TCO_possible_gt.dim() == 4 and TCO_possible_gt.shape[-2:] == (4, 4)

    dR = compute_rotation_matrix_from_quaternions(refiner_outputs[:, 0:4])
    vxvyvz = refiner_outputs[:, 4:7]
    TCO_gt = TCO_possible_gt[:, 0]

    TCO_pred_orn = TCO_gt.clone()
    TCO_pred_orn[:, :3, :3] = dR @ TCO_input[:, :3, :3]

    TCO_pred_xy = TCO_gt.clone()
    z_gt = TCO_gt[:, 2, [3]]
    z_input = TCO_input[:, 2, [3]]
    vxvy = vxvyvz[:, :2]
    fxfy = K_crop[:, [0, 1], [0, 1]]
    xsrcysrc = TCO_input[:, :2, 3]
    TCO_pred_xy[:, :2, 3] = ((vxvy / fxfy) + (xsrcysrc / z_input.repeat(1, 2))) * z_gt.repeat(1, 2)

    TCO_pred_z = TCO_gt.clone()
    vz = vxvyvz[:, [2]]
    TCO_pred_z[:, [2], [3]] = vz * z_input

    loss_orn, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_orn, points, l1_or_l2=l1)
    loss_xy, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_xy, points, l1_or_l2=l1)
    loss_z, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_z, points, l1_or_l2=l1)
    return loss_orn + loss_xy + loss_z


def TCO_init_from_boxes(z_range, boxes, K):
    # Used in the paper
    assert len(z_range) == 2
    assert boxes.shape[-1] == 4
    assert boxes.dim() == 2
    bsz = boxes.shape[0]
    uv_centers = (boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2
    z = torch.as_tensor(z_range).mean().unsqueeze(0).unsqueeze(0).repeat(bsz, 1).to(boxes.device).to(boxes.dtype)
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    xy_init = ((uv_centers - cxcy) * z) / fxfy
    TCO = torch.eye(4).unsqueeze(0).to(torch.float).to(boxes.device).repeat(bsz, 1, 1)
    TCO[:, :2, 3] = xy_init
    TCO[:, 2, 3] = z.flatten()
    return TCO


def TCO_init_from_boxes_zup_autodepth(boxes_2d, model_points_3d, K):
    # User in BOP20 challenge
    assert boxes_2d.shape[-1] == 4
    assert boxes_2d.dim() == 2
    bsz = boxes_2d.shape[0]
    z_guess = 1.0
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    TCO = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [-1, 0, 0, z_guess],
        [0, 0, 0, 1]
    ]).to(torch.float).to(boxes_2d.device).repeat(bsz, 1, 1)
    bb_xy_centers = (boxes_2d[:, [0, 1]] + boxes_2d[:, [2, 3]]) / 2
    xy_init = ((bb_xy_centers - cxcy) * z_guess) / fxfy
    TCO[:, :2, 3] = xy_init

    C_pts_3d = transform_pts(TCO, model_points_3d)
    deltax_3d = C_pts_3d[:, :, 0].max(dim=1).values - C_pts_3d[:, :, 0].min(dim=1).values
    deltay_3d = C_pts_3d[:, :, 1].max(dim=1).values - C_pts_3d[:, :, 1].min(dim=1).values

    bb_deltax = (boxes_2d[:, 2] - boxes_2d[:, 0]) + 1
    bb_deltay = (boxes_2d[:, 3] - boxes_2d[:, 1]) + 1

    z_from_dx = fxfy[:, 0] * deltax_3d / bb_deltax
    z_from_dy = fxfy[:, 1] * deltay_3d / bb_deltay

    # z = z_from_dx.unsqueeze(1)
    # z = z_from_dy.unsqueeze(1)
    z = (z_from_dy.unsqueeze(1) + z_from_dx.unsqueeze(1)) / 2

    xy_init = ((bb_xy_centers - cxcy) * z) / fxfy
    TCO[:, :2, 3] = xy_init
    TCO[:, 2, 3] = z.flatten()
    return TCO


class ExplicitInterClassGraphLoss(nn.Module):
    'explicit inter-class relation graph loss'
    def __init__(self) -> None:
        super().__init__()
        self.Rg, self.zg = self.build_R_z_graph()


    def unit_affine(self, x, sign=None):
        '''affine any positive values to [0, 1]'''
        if sign == 'positive':
            y = (x - x.min()) / (x.max() - x.min())
        elif sign == 'negative':
            y = - (x - x.min()) / (x.max() - x.min()) + 1
        else:
            raise ValueError('Function sign not specified!')
        return y


    def AdjMat2Laplacian(self, adj_mat: torch.tensor):
        '''L = D - A'''
        diag_element = torch.sum(adj_mat, dim=-1)
        degree_mat = torch.diag_embed(diag_element)
        Laplacian = degree_mat - adj_mat
        return Laplacian


    def build_R_z_graph(self):
        '''building bin^R, bin^z graph by explicit relations'''
        z_bin_ctrs = torch.tensor([i * np.pi/2/39 for i in range(40)]).cuda()
        def geodesic_dists(R_bin_ctrs, R):
            internal = 0.5 * (torch.diagonal(torch.matmul(R_bin_ctrs, torch.transpose(R,-1, -2)), dim1=-1, dim2=-2).sum(-1) - 1.0)
            internal = torch.clamp(internal, -1.0, 1.0)  # set lower and upper bound as [-1,1]
            return torch.acos(internal)

        Rg = torch.zeros(60, 60).cuda()
        for i in range(len(R_bin_ctrs)):
            Rg[i] = geodesic_dists(R_bin_ctrs, R_bin_ctrs[i])
        Rg = self.unit_affine(Rg, sign='negative')

        zg = torch.zeros(40, 40).cuda()
        for i in range(len(z_bin_ctrs)):
            zg[i] = torch.cos(torch.abs(z_bin_ctrs - z_bin_ctrs[i])).clamp(min=0.0, max=1.0)

        return Rg, zg


    def build_adj_mat_by_label(self, labels: torch.tensor, zg):
        '''given a batch of labels and bin graph,
        build a (bs, bs) adjacency matrix'''
        bs = len(labels)
        batch_lbl_adj_mat = torch.zeros(bs, bs).cuda()
        for i, lb in enumerate(labels):
            batch_lbl_adj_mat[i] = zg[lb, labels]
        return batch_lbl_adj_mat


    def build_adj_mat_by_feat(self, feat: torch.tensor):
        sim_vect = torch.zeros(feat.size(0), feat.size(0)).cuda()
        for i in range(feat.size(0)):
            sim_vect[i] = F.cosine_similarity(feat[[i]], feat)
        return sim_vect


    def forward(self, batch_label, batch_feat):
        batch_label_adj_mat = self.build_adj_mat_by_label(batch_label, self.zg)
        batch_feat_adj_mat = self.build_adj_mat_by_feat(batch_feat)
        loss = F.mse_loss(batch_feat_adj_mat, batch_label_adj_mat)
        return loss



def xentropy(pred, label):
    Lsoftmax = F.log_softmax(pred, dim=-1)
    loss = - (label * Lsoftmax).sum(dim=1).mean()
    return loss


def cal_gt_dR_vxyz(T_in, T_gt, K_crop):
    """calculate gt dR and vxvyvz from input pose and gt pose with K_crop
        for gt bin and delta generation
    """
    gt_dR = torch.matmul(T_gt[..., :3, :3], torch.inverse(T_in)[..., :3, :3])
    fxfy = K_crop[:, [0, 1], [0, 1]]
    
    vz = (T_gt[...,2,3]/T_in[...,2,3]).unsqueeze(-1)
    z_gt = T_gt[:, 2, [3]].repeat(1,2)
    z_in = T_in[:, 2, [3]].repeat(1,2)
    vxvy = (T_gt[:, :2, 3]/z_gt - T_in[:, :2, 3]/z_in) * fxfy
    return gt_dR, torch.cat([vxvy, vz], dim=-1)



def compute_ADDS_loss(TCO_gt, TCO_pred, points):
    assert TCO_gt.dim() == 3 and TCO_gt.shape[-2:] == (4, 4)
    assert TCO_pred.shape[-2:] == (4, 4)
    assert points.dim() == 3 and points.shape[-1] == 3
    TXO_gt_points = transform_pts(TCO_gt, points)
    TXO_pred_points = transform_pts(TCO_pred, points)
    dists_squared = (TXO_gt_points.unsqueeze(1) - TXO_pred_points.unsqueeze(2)) ** 2
    dists = dists_squared
    dists_norm_squared = dists_squared.sum(dim=-1)
    assign = dists_norm_squared.argmin(dim=1)
    ids_row = torch.arange(dists.shape[0]).unsqueeze(1).repeat(1, dists.shape[1])
    ids_col = torch.arange(dists.shape[1]).unsqueeze(0).repeat(dists.shape[0], 1)
    losses = dists_squared[ids_row, assign, ids_col].mean(dim=(-1, -2))
    return losses


def compute_ADD_L1_loss(TCO_gt, TCO_pred, points):
    bsz = len(TCO_gt)
    assert TCO_pred.shape == (bsz, 4, 4) and TCO_gt.shape == (bsz, 4, 4)
    assert points.dim() == 3 and points.shape[-1] == 3
    dists = (transform_pts(TCO_gt, points) - transform_pts(TCO_pred, points)).abs().mean(dim=(-1, -2))
    return dists
