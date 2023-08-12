import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def grid_xyz(z_bin_num=6, z_bin_range=(-0.1, 1.1)):
    bin_size_z = (z_bin_range[1]-z_bin_range[0])/ (2 * z_bin_num)
    z_bin_ctrs = np.linspace(z_bin_range[0], z_bin_range[1],z_bin_num, endpoint=False) + bin_size_z
    return torch.tensor(z_bin_ctrs).float().cuda()


def t_to_bin_delta(z=None, z_bin_ctrs=None, knn=1, theta1=1.0, theta2=0.0):

    dists_z = torch.abs(z_bin_ctrs - z)
    bin_z_delta_z = torch.zeros(2, len(z_bin_ctrs)).cuda()

    _, knnz = torch.topk(dists_z, k=knn, largest=False)
    minz = knnz[0]
    bin_z_delta_z[0, knnz] = theta2
    bin_z_delta_z[0, minz] = theta1

    bin_z_delta_z[1, knnz] = z-z_bin_ctrs[knnz]

    return bin_z_delta_z[0], bin_z_delta_z[1]


def t_to_bin_delta_batch(labels, ctr):
    bins = []
    deltas = []
    for i in range(len(labels)):
        b, d = t_to_bin_delta(labels[i],ctr)
        bins.append(b)
        deltas.append(d)
    return torch.stack(bins).cuda(), torch.stack(deltas).cuda()


def bins_deltas_to_ts_batch(bin_zs, delta_zs, z_bin_ctrs):
    idzs = torch.argmax(bin_zs, dim=-1)
    batch_list = torch.arange(bin_zs.size(0))
    z = z_bin_ctrs[idzs] + delta_zs[batch_list, idzs]
    return z.unsqueeze(-1)


def xentropy(pred, label):
    Lsoftmax = F.log_softmax(pred, dim=-1)
    loss = - (label * Lsoftmax).sum(dim=1).mean()
    return loss


class ExplicitInterClassGraphLoss(nn.Module):
    'explicit inter-class relation graph loss'
    def __init__(self) -> None:
        super().__init__()
        self.zg = self.build_R_z_graph()


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
        z_bin_ctrs = torch.tensor([i * np.pi/2/5 for i in range(6)]).cuda()

        zg = torch.zeros(6, 6).cuda()
        for i in range(len(z_bin_ctrs)):
            zg[i] = torch.cos(torch.abs(z_bin_ctrs - z_bin_ctrs[i])).clamp(min=0.0, max=1.0)

        return zg


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


if __name__ == "__main__":
    ctr = grid_xyz()
    labels = torch.tensor([0, 0.2 , 0.4, 0.6, 0.8, 1.0]).unsqueeze(-1).cuda()
    b, d = t_to_bin_delta_batch(labels, ctr)
    pred = bins_deltas_to_ts_batch(b, d, ctr)
    print(b, d)
