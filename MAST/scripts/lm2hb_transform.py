import numpy as np
import open3d as o3d
import torch

'''This script calculate the pose from LM model to HomebrewedDB model'''

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''
    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    return R, t


# read hb orientation and lm orientation in lm id model
hb = o3d.io.read_point_cloud('local_data/bop_datasets/lm/models_eval_hb/obj_000015.ply').points
lm = o3d.io.read_point_cloud('local_data/bop_datasets/lm/models_eval/obj_000015.ply').points

# divide by 1000 to fit testing situation
hb = np.array(hb)/1000
lm = np.array(lm)/1000
assert hb.shape == lm.shape

# fitting rigid transformation
R, t = best_fit_transform(lm, hb)

# apply transformation
lm = lm @ R.T + t

# save 2 point cloud together to compare
gt_o3d = o3d.geometry.PointCloud()
gt_o3d.points = o3d.utility.Vector3dVector(np.concatenate((lm, hb), axis=0))
o3d.io.write_point_cloud('temp/com.ply', gt_o3d)

print(torch.tensor(R),torch.tensor(t))
