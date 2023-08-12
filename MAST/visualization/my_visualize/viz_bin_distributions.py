from transforms3d.quaternions import quat2mat
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def sample_rotations_60(return_type="quaternion"):
    """ icosahedral_group: 60 rotations
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
            quaternion_group[i] = quat2mat(group[i])
        return quaternion_group.astype(float)

    else:
        raise ValueError('Unknown return rotation type')




R_bin_ctrs = torch.tensor(sample_rotations_60(0))
R = {0: 8049, 1: 22086, 2: 0, 3: 10988, 4: 0, 5: 7977, 6: 0, 7: 10997, 8: 14, 9: 22488,
 10: 0, 11: 0, 12: 22341, 13: 0, 14: 0, 15: 0, 16: 10914, 17: 0, 18: 0, 19: 0, 20: 0, 
 21: 2063, 22: 0, 23: 0, 24: 0, 25: 1990, 26: 0, 27: 10748, 28: 0, 29: 0, 30: 9, 31: 19,
 32: 14761, 33: 0, 34: 0, 35: 3360, 36: 0, 37: 3106, 38: 0, 39: 0, 40: 22731, 41: 0, 42: 0,
 43: 0, 44: 11, 45: 0, 46: 0, 47: 2016, 48: 0, 49: 3156, 50: 3371, 51: 0, 52: 0, 53: 0, 
 54: 2087, 55: 0, 56: 0, 57: 0, 58: 0, 59: 14718}  # 4 epochs
 
x = {0: 1387, 1: 187, 2: 222, 3: 320, 4: 379, 5: 575, 6: 1183, 7: 3222, 8: 10449,
9: 30135, 10: 32279, 11: 11391, 12: 3654, 13: 1381, 14: 702, 15: 411, 16: 314, 17: 238, 18: 183, 19: 1388}  #2 epochs

z = {0: 215, 1: 306, 2: 318, 3: 340, 4: 340, 5: 375, 6: 406, 7: 440, 8: 538, 9: 549, 
10: 664, 11: 789, 12: 902, 13: 1170, 14: 1643, 15: 2029, 16: 2551, 17: 2923, 18: 2988, 
19: 2948, 20: 2890, 21: 3190, 22: 3203, 23: 3010, 24: 2817, 25: 2449, 26: 2096, 27: 1770, 
28: 1397, 29: 1111, 30: 931, 31: 706, 32: 549, 33: 429, 34: 291, 35: 212, 36: 175, 37: 115, 38: 79, 39: 146}

R_real=[]
for i in R.keys():
    #if R[i]!=0:
        R_real.append(i)

R=R_real
pt = torch.tensor([0,0,1]).double()
ps=[]
for i in R:
    p = torch.matmul(R_bin_ctrs[i], pt.t())
    ps.append(p)
ps = torch.stack(ps, dim=0)

ps = np.array(ps)
ax = plt.subplot(111, projection='3d')
ax.scatter(ps[:,0],ps[:,1],ps[:,2])
ax.set_zlabel('z')
ax.set_ylabel('y')
ax.set_xlabel('x')
print(ps.shape)
plt.show()
