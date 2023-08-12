import math

import numpy as np
import torch
from transforms3d.quaternions import mat2quat


def sample_rotations_12():
    """ tetrahedral_group: 12 rotations
        from https://github.com/mentian/object-posenet
    """
    group = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                      [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                      [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],

                      [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                      [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
                      [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
                      [[0, -1, 0], [0, 0, -1], [1, 0, 0]],

                      [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                      [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                      [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
                      [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]])
    # return group.astype(float)
    quaternion_group = np.zeros((12, 4))
    for i in range(12):
        quaternion_group[i] = mat2quat(group[i])
    return quaternion_group.astype(float)


def sample_rotations_24():
    """ octahedral_group: 24 rotations
        from https://github.com/mentian/object-posenet
    """
    group = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                      [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                      [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],

                      [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                      [[1, 0, 0], [0, 0, -1], [0, -1, 0]],
                      [[-1, 0, 0], [0, 0, 1], [0, -1, 0]],
                      [[-1, 0, 0], [0, 0, -1], [0, 1, 0]],

                      [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                      [[0, 1, 0], [-1, 0, 0], [0, 0, -1]],
                      [[0, -1, 0], [1, 0, 0], [0, 0, -1]],
                      [[0, -1, 0], [-1, 0, 0], [0, 0, 1]],

                      [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                      [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
                      [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
                      [[0, -1, 0], [0, 0, -1], [1, 0, 0]],

                      [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                      [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                      [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
                      [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],

                      [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                      [[0, 0, 1], [0, -1, 0], [-1, 0, 0]],
                      [[0, 0, -1], [0, 1, 0], [-1, 0, 0]],
                      [[0, 0, -1], [0, -1, 0], [1, 0, 0]]])
    # return group.astype(float)
    quaternion_group = np.zeros((24, 4))
    for i in range(24):
        quaternion_group[i] = mat2quat(group[i])
    return quaternion_group.astype(float)


def sample_rotations_60(return_type="quaternion"):
    """ icosahedral_group: 60 rotations
        from https://github.com/mentian/object-posenet
        args:
            return_type: str "matrix" or int 0
                         str "quaternion" or int 1
    """
    phi = (1 + math.sqrt(5)) / 2
    R1 = np.array([[-phi/2, 1/(2*phi), -0.5], [-1/(2*phi), 0.5, phi/2], [0.5, phi/2, -1/(2*phi)]])
    R2 = np.array([[phi/2, 1/(2*phi), -0.5], [1/(2*phi), 0.5, phi/2], [0.5, -phi/2, 1/(2*phi)]])
    group = [np.eye(3, dtype=float)]
    n = 0
    while len(group) > n:
        n = len(group)
        set_so_far = group
        for rot in set_so_far:
            for R in [R1, R2]:
                new_R = np.matmul(rot, R)
                new = True
                for item in set_so_far:
                    if np.sum(np.absolute(item - new_R)) < 1e-6:
                        new = False
                        break
                if new:
                    group.append(new_R)
                    break
            if new:
                break

    if return_type == "matrix" or return_type == 0:
        return np.array(group)

    elif return_type == "quaternion" or return_type == 1:
        group = np.array(group)
        quaternion_group = np.zeros((60, 4))
        for i in range(60):
            quaternion_group[i] = mat2quat(group[i])
        return quaternion_group.astype(float)

    else:
        raise ValueError('Unknown return rotation type')


def grid_xyz(xy_bin_num=20, z_bin_num=40, xy_bin_range=(-200, 200), z_bin_range=(0.0, 2.0)):
    bin_size_xy = (xy_bin_range[1]-xy_bin_range[0])/ (2 * xy_bin_num)
    xy_bin_ctrs = np.linspace(xy_bin_range[0], xy_bin_range[1],xy_bin_num, endpoint=False) + bin_size_xy
    
    bin_size_z = (z_bin_range[1]-z_bin_range[0])/ (2 * z_bin_num)
    z_bin_ctrs = np.linspace(z_bin_range[0], z_bin_range[1],z_bin_num, endpoint=False) + bin_size_z

    # z_bin_ctrs = np.array([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.71, 0.72, 0.73,
    #                        0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 
    #                        0.84, 0.85, 0.86, 0.87, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94,
    #                        0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 
    #                        1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12, 1.13, 1.14,
    #                        1.15, 1.16, 1.17, 1.18, 1.19, 1.20, 1.21, 1.22, 1.23, 1.24, 
    #                        1.25, 1.26, 1.27, 1.28, 1.29, 1.30, 1.32, 1.34, 1.36, 1.38, 
    #                        1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 
    #                        1.90, 1.95, 2.00, 2.05, 2.10, 2.15, 2.20, 1.25, 2.30, 2.35, 
    #                        2.40])

    return xy_bin_ctrs, z_bin_ctrs



R_bin_ctrs = torch.tensor(sample_rotations_60("matrix")).float().cuda()
xy_bin_ctrs, z_bin_ctrs = grid_xyz()
xy_bin_ctrs = torch.tensor(xy_bin_ctrs).float().cuda()
z_bin_ctrs = torch.tensor(z_bin_ctrs).float().cuda()


def R_to_bin_delta(R=None, R_bin_ctrs=None, theta1=0.7, theta2=0.1, knn=4):
    """
    given the rotation bin centers and a rotation matrix, 
    return the rotation bin vector label and delta label
    """
    def geodesic_dists(R_bin_ctrs, R):
        internal = 0.5 * (torch.diagonal(torch.matmul(R_bin_ctrs, torch.transpose(R,-1, -2)), dim1=-1, dim2=-2).sum(-1) - 1.0)
        internal = torch.clamp(internal, -1.0, 1.0)  # set lower and upper bound as [-1,1]
        return torch.acos(internal)


    bin_R = torch.zeros(R_bin_ctrs.size(0)).cuda()
    delta_R = torch.zeros(R_bin_ctrs.size()).cuda()

    dists = geodesic_dists(R_bin_ctrs, R)
    _, nn4 = torch.topk(dists, k=knn, largest=False)
    nn1 = nn4[0]

    bin_R[nn4] = theta2
    bin_R[nn1] = theta1

    delta_R[nn4] = R[..., :3, : 3].matmul(torch.transpose(R_bin_ctrs[nn4], dim0=2, dim1=1))

    return bin_R, delta_R


def t_to_bin_delta(t=None, xy_bin_ctrs=None, z_bin_ctrs=None, knn=7, theta1=0.55, theta2=0.075):

    x = t[0]; y = t[1]; z = t[-1]
    dists_x = torch.abs(xy_bin_ctrs - x)
    dists_y = torch.abs(xy_bin_ctrs - y)
    dists_z = torch.abs(z_bin_ctrs - z)


    bin_xy_delta_xy = torch.zeros(4, len(xy_bin_ctrs)).cuda()
    bin_z_delta_z = torch.zeros(2, len(z_bin_ctrs)).cuda()


    _, knnx = torch.topk(dists_x, k=knn, largest=False)
    minx = knnx[0]
    bin_xy_delta_xy[0, knnx] = theta2
    bin_xy_delta_xy[0, minx] = theta1
    _, knny = torch.topk(dists_y, k=knn, largest=False)
    miny = knny[0]
    bin_xy_delta_xy[1, knny] = theta2
    bin_xy_delta_xy[1, miny] = theta1
    _, knnz = torch.topk(dists_z, k=knn, largest=False)
    minz = knnz[0]
    bin_z_delta_z[0, knnz] = theta2
    bin_z_delta_z[0, minz] = theta1

    bin_xy_delta_xy[2, knnx] = t[0]-xy_bin_ctrs[knnx]
    bin_xy_delta_xy[3, knny] = t[1]-xy_bin_ctrs[knny]
    bin_z_delta_z[1, knnz] = t[2]-z_bin_ctrs[knnz]

    return bin_xy_delta_xy[0], bin_xy_delta_xy[1], bin_z_delta_z[0], bin_xy_delta_xy[2], bin_xy_delta_xy[3], bin_z_delta_z[1]


def Rts_to_bin_delta_batch(Rs, ts, R_bin_ctrs, xy_bin_ctrs, z_bin_ctrs):
    bin_Rs, delta_Rs, bin_xs, bin_ys, bin_zs, delta_xs, delta_ys, delta_zs = [], [], [], [], [], [], [], []
    assert Rs.size(0) == ts.size(0)
    for i in range(len(Rs)):
        bin_R, delta_R = R_to_bin_delta(Rs[i], R_bin_ctrs)
        bin_x, bin_y, bin_z, delta_x, delta_y, delta_z = t_to_bin_delta(ts[i], xy_bin_ctrs, z_bin_ctrs)
        bin_Rs.append(bin_R)
        delta_Rs.append(delta_R)
        bin_xs.append(bin_x)
        bin_ys.append(bin_y)
        bin_zs.append(bin_z)
        delta_xs.append(delta_x)
        delta_ys.append(delta_y)
        delta_zs.append(delta_z)

    return torch.stack(bin_Rs), \
           torch.stack(delta_Rs), \
           torch.stack(bin_xs), \
           torch.stack(bin_ys), \
           torch.stack(bin_zs), \
           torch.stack(delta_xs),\
           torch.stack(delta_ys),\
           torch.stack(delta_zs),\


def bins_deltas_to_Rs_batch(bin_Rs, delta_Rs, R_bin_ctrs):
    delta_R_chosen = delta_Rs[torch.arange(delta_Rs.size(0)), torch.argmax(bin_Rs, dim=-1)]
    R_bin_ctrs_chosen = R_bin_ctrs[torch.argmax(bin_Rs, dim=-1)]
    return torch.matmul(delta_R_chosen[..., :3, :3], R_bin_ctrs_chosen[..., :3, :3])


def bins_deltas_to_ts_batch(bin_xs, bin_ys, bin_zs, delta_xs, delta_ys, delta_zs, xy_bin_ctrs, z_bin_ctrs):
    idxs = torch.argmax(bin_xs, dim=-1)
    idys = torch.argmax(bin_ys, dim=-1)
    idzs = torch.argmax(bin_zs, dim=-1)
    batch_list = torch.arange(delta_xs.size(0))
    x = xy_bin_ctrs[idxs] + delta_xs[batch_list, idxs]
    y = xy_bin_ctrs[idys] + delta_ys[batch_list, idys]
    z = z_bin_ctrs[idzs] + delta_zs[batch_list, idzs]
    return torch.stack((x, y, z), dim=-1)


def bin_delta_to_Rts_batch(bin_Rs, delta_Rs, R_bin_ctrs, bin_xs, bin_ys,
                            bin_zs, delta_xs, delta_ys, delta_zs, xy_bin_ctrs, z_bin_ctrs):
    """
    args: 
    bin_Rs: (bs, bins)
    delta_Rs: (bs, bins, 3, 3)
    R_bin_ctrs: (bins, 3, 3)
    bin/delta_xs, bin/delta_ys, bin/delta_zs: (bs, bins)
    xy_bin_ctrs: (20)
    z_bin_ctrs: (40)

    returns:
    rotation matrix: (bs, 3, 3)
    xyz: (bs, 3)
    """

    assert (bin_Rs.size(0) == delta_Rs.size(0) == bin_xs.size(0)
            == bin_ys.size(0) == bin_zs.size(0) ==delta_zs.size(0))
    Rs = bins_deltas_to_Rs_batch(bin_Rs, delta_Rs, R_bin_ctrs)
    ts = bins_deltas_to_ts_batch(bin_xs, bin_ys, bin_zs, delta_xs, delta_ys, delta_zs, xy_bin_ctrs, z_bin_ctrs)
    return Rs, ts


def bin_delta_to_Rts_batch_by_idx(delta_Rs, idR, R_bin_ctrs, delta_xs, idxs, delta_ys, idys,
                                    delta_zs, idzs, xy_bin_ctrs, z_bin_ctrs):
    """
    args: 
    delta_Rs: (bs, bins, 3, 3)
    idRxyz: (bs, 1)
    R_bin_ctrs: (bins, 3, 3)
    delta_xs, delta_ys, delta_zs: (bs, bins)
    xy_bin_ctrs: (20)
    z_bin_ctrs: (40)

    returns:
    rotation matrix: (bs, 3, 3)
    xyz: (bs, 3)
    """

    def bins_deltas_to_Rs_batch(delta_Rs, R_bin_ctrs):
        delta_R_chosen = delta_Rs[torch.arange(delta_Rs.size(0)), idR]
        R_bin_ctrs_chosen = R_bin_ctrs[idR]
        return torch.matmul(delta_R_chosen[..., :3, :3], R_bin_ctrs_chosen[..., :3, :3])

    def bins_deltas_to_ts_batch(delta_xs, delta_ys, delta_zs, xy_bin_ctrs, z_bin_ctrs):
        batch_list = torch.arange(delta_xs.size(0))
        x = xy_bin_ctrs[idxs] + delta_xs[batch_list, idxs]
        y = xy_bin_ctrs[idys] + delta_ys[batch_list, idys]
        z = z_bin_ctrs[idzs] + delta_zs[batch_list, idzs]
        return torch.stack((x, y, z), dim=-1)

    assert (delta_Rs.size(0) == delta_zs.size(0))
    Rs = bins_deltas_to_Rs_batch(delta_Rs, R_bin_ctrs)
    ts = bins_deltas_to_ts_batch(delta_xs, delta_ys, delta_zs, xy_bin_ctrs, z_bin_ctrs)
    return Rs, ts

def bin_delta_to_Rs_batch_by_idx(delta_Rs, idR, R_bin_ctrs):
    """
    args: 
    delta_Rs: (bs, bins, 3, 3)
    idR: (bs, 1)
    R_bin_ctrs: (bins, 3, 3)

    returns:
    rotation matrix: (bs, 3, 3)
    """

    def bins_deltas_to_Rs_batch(delta_Rs, R_bin_ctrs):
        delta_R_chosen = delta_Rs[torch.arange(delta_Rs.size(0)), idR]
        R_bin_ctrs_chosen = R_bin_ctrs[idR]
        return torch.matmul(delta_R_chosen[..., :3, :3], R_bin_ctrs_chosen[..., :3, :3])

    Rs = bins_deltas_to_Rs_batch(delta_Rs, R_bin_ctrs)
    return Rs


def bin_delta_to_ts_batch_by_idx(delta_xs, idxs, delta_ys, idys,
                                    delta_zs, idzs, xy_bin_ctrs, z_bin_ctrs):
    """
    args: 
    idxyz: (bs, 1)
    delta_xs, delta_ys, delta_zs: (bs, bins)
    xy_bin_ctrs: (20)
    z_bin_ctrs: (40)

    returns:
    xyz: (bs, 3)
    """

    def bins_deltas_to_ts_batch(delta_xs, delta_ys, delta_zs, xy_bin_ctrs, z_bin_ctrs):
        batch_list = torch.arange(delta_xs.size(0))
        x = xy_bin_ctrs[idxs] + delta_xs[batch_list, idxs]
        y = xy_bin_ctrs[idys] + delta_ys[batch_list, idys]
        z = z_bin_ctrs[idzs] + delta_zs[batch_list, idzs]
        return torch.stack((x, y, z), dim=-1)

    ts = bins_deltas_to_ts_batch(delta_xs, delta_ys, delta_zs, xy_bin_ctrs, z_bin_ctrs)
    return ts


if __name__ == "__main__":
    from time import time

    from scipy.spatial.transform import Rotation as R
    Rs = torch.tensor(R.random(58).as_matrix()).float().cuda()
    ts = torch.rand(58, 3).cuda()
    t1 = time()
    for i in range(10):
        a= Rts_to_bin_delta_batch(Rs, ts,R_bin_ctrs, xy_bin_ctrs,z_bin_ctrs)
        b = bin_delta_to_Rts_batch(a[0], a[1],R_bin_ctrs, a[2], a[3], a[4], a[5], a[6], a[7], xy_bin_ctrs, z_bin_ctrs)  # 0.00027 s
    t2 = time() - t1
    print(t2)

    print(((b[0]-Rs)<1e-3).all(), '\n', ((b[1]-ts)<1e-5).all())
    # for i in range(60):
    #     print(R_bin_ctrs[i]@torch.inverse(R_bin_ctrs[i]))
