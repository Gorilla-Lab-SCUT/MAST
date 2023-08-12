import os
import os.path as osp
import MAST.utils.tensor_collection as tc
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LMO_Dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.train_on_bop = True  # whether train on bop format dataset
        self.load_depth = None
        # paths
        self.LMO_PATH = "local_data/OCCLUSION_LINEMOD"
        self.rgb_dir = "RGB-D/rgb_noseg"
        self.Rt_dir = "poses"

        self.rgbs_paths = os.listdir(osp.join(self.LMO_PATH, self.rgb_dir))  # RGB images paths
        self.rgbs_paths.sort(key=lambda x:int(x[6:-4]))                      # sort to be ordered

        # lmo to bop_lmo adaptation parameters
        self.cls_id_BOP = {
            'Ape': '000001',
            'Can': '000005',
            'Cat': '000006',
            'Driller': '000008',
            'Duck': '000009',
            'Eggbox': '000010',
            'Glue': '000011',
            'Holepuncher': '000012'
        }

        # proper labels is the hadamard product of original lmo label and the mat below
        # (note: used for model trained on bop dataset)
        self.lmo2bop_label = { 
            '000001': np.array([[[-1.,-1.,1.,1.],
                                 [-1.,-1.,1.,1.],
                                 [-1.,-1.,1.,1.],
                                 [1.,1.,1.,1.]]]),
            '000005': np.array([[[-1.,-1.,1.,1.],
                                 [-1.,-1.,1.,1.],
                                 [-1.,-1.,1.,1.],
                                 [1.,1.,1.,1.]]]),
            '000006': np.array([[[1.,1.,1.,1.],
                                 [1.,1.,1.,1.],
                                 [1.,1.,1.,1.],
                                 [1.,1.,1.,1.]]]),
            '000008': np.array([[[-1.,-1.,1.,1.],
                                 [-1.,-1.,1.,1.],
                                 [-1.,-1.,1.,1.],
                                 [1.,1.,1.,1.]]]),
            '000009': np.array([[[1.,1.,1.,1.],
                                 [1.,1.,1.,1.],
                                 [1.,1.,1.,1.],
                                 [1.,1.,1.,1.]]]),
            '000010': np.array([[[-1.,-1.,1.,1.],
                                 [-1.,-1.,1.,1.],
                                 [-1.,-1.,1.,1.],
                                 [1.,1.,1.,1.]]]),
            '000011': np.array([[[-1.,-1.,1.,1.],
                                 [-1.,-1.,1.,1.],
                                 [-1.,-1.,1.,1.],
                                 [1.,1.,1.,1.]]]),
            '000012': np.array([[[1.,1.,1.,1.],
                                 [1.,1.,1.,1.],
                                 [1.,1.,1.,1.],
                                 [1.,1.,1.,1.]]]),}
        
        # camera intrinsics
        self.K = np.array([[572.4114,    0.,      325.2611 ],
                           [  0.,      573.57043, 242.04899],
                           [  0.,        0.,        1.     ]])


        # preprocess operation on images
        self.img_transform = transforms.Compose([
	    transforms.ToTensor(), 
	    ])


    @staticmethod
    def to_homo_matrix(Rt):
        """formulate homogeneous transformation matrix from 3x4 Rt pose"""
        return np.concatenate((Rt,np.array([[0,0,0,1]])),axis=0)


    @staticmethod
    def read_pose(pose_path):
        """read linemod pose"""
        with open(pose_path) as pose_info:
            lines = [line[:-1] for line in pose_info.readlines()]
            if 'rotation:' not in lines:
                return np.array([])
            row = lines.index('rotation:') + 1
            rotation = np.loadtxt(lines[row:row + 3])
            translation = np.loadtxt(lines[row + 4:row + 5])
        return np.concatenate([rotation, np.reshape(translation, newshape=[3, 1])], axis=-1)

    @staticmethod
    def occlusion_pose_to_blender_pose(pose):
        """original occlusion linemod transformation matrix needs ops"""
        rot, tra = pose[:, :3], pose[:, 3]
        rotation = np.array([[0., 1., 0.],
                             [0., 0., 1.],
                             [1., 0., 0.]])
        rot = np.dot(rot, rotation)
        tra[1:] *= -1
        rot[1:] *= -1
        pose = np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)

        return pose



    def __len__(self):
        return len(self.rgbs_paths)


    def __getitem__(self, idx):

        # read an image, preprocess, and convert to tensor
        img_path = osp.join(self.LMO_PATH, self.rgb_dir, self.rgbs_paths[idx])
        img = Image.open(img_path)
        img = self.img_transform(img)
        # note that at bop the image shape is (480, 640, 3), here is (3, 480, 640)


        # TODO: ADD MASK and depth


        # read corresponding R and t poses
        Trans = {}
        for cls, id in self.cls_id_BOP.items():
            Rt_path = osp.join(self.LMO_PATH, self.Rt_dir, cls, "info_"+self.rgbs_paths[idx][6:-4]+".txt")
            Rt = self.read_pose(Rt_path)
            if len(Rt) > 0:
                Rt = self.occlusion_pose_to_blender_pose(Rt)
                homo_trans_mat = self.to_homo_matrix(Rt)
                Trans[id] = homo_trans_mat * self.lmo2bop_label[id]  # 4x4 homo matrix


        return img, Trans, self.rgbs_paths[idx]


    def collate_fn(self, batch):
        # here batch is a list contains tuples wraps returns of "__getitem__"
        cam_infos, K = [], []
        im_infos = []
        depth = []
        images = []
        transes = []
        for batch_im_id, data in enumerate(batch):
            image, trans, img_id = data
            im_info = {'scene_id': 2, 'view_id': int(img_id[6:-4]),
                       'group_id':0, 'batch_im_id': batch_im_id}
            im_infos.append(im_info)
            cam_info = im_info.copy()
            K.append(self.K * 1.)
            cam_infos.append(cam_info)
            images.append(image)
            transes.append(trans)
            if self.load_depth:
                pass
        
        cameras = tc.PandasTensorCollection(infos=pd.DataFrame(cam_infos),
                                            K=torch.as_tensor(np.stack(K)))
        data = dict(
                    cameras=cameras,
                    images=torch.stack(images),
                    im_infos=im_infos,
                    transes=transes
                    )
        if self.load_depth:
            pass

        return data
                

    def dataloader(self, batch_size=1, n_workers=0, sampler=None, use_collate_fn=False):
        if use_collate_fn:
            collate_function = self.collate_fn
        else:
            collate_function = None

        return DataLoader(self, batch_size=batch_size, shuffle=False,
                          num_workers=n_workers, sampler=sampler,
                          collate_fn=collate_function)



if __name__ == "__main__":
    a=LMO_Dataset()
    a= a.dataloader()
    for i, batch in enumerate(a): 
        print(batch)
        exit()
