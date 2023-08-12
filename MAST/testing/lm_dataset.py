import os
import os.path as osp
import MAST.utils.tensor_collection as tc
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LM_Dataset(Dataset):
    def __init__(self, split="test",category="01") -> None:
        super().__init__()

        self.train_on_bop = True  # whether train on bop format dataset
        self.load_depth = None
        # paths
        self.split = split
        self.category = category
        self.LM_PATH = "local_data/Linemod_preprocessed/data"
        self.rgbs_paths = osp.join(self.LM_PATH, self.category, "rgb")
        self.gt_paths = osp.join(self.LM_PATH, self.category, "gt.yml")  # gt Rt pose paths
        with open(osp.join(self.LM_PATH, self.category, self.split+'.txt'), 'r') as f:
            self.rgbs_paths = [osp.join(self.rgbs_paths, line.strip("\n") + ".png") 
                                for line in f.readlines()]    # RGB images paths
        
        with open(self.gt_paths) as f:
            self.gts=yaml.load(f, Loader=yaml.FullLoader)  # read gt poses and bbox

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


    def __len__(self):
        return len(self.rgbs_paths)


    def __getitem__(self, idx):

        # read an image, preprocess, and convert to tensor
        img = Image.open(self.rgbs_paths[idx])
        img = self.img_transform(img)
        # note that at bop the image shape is (480, 640, 3), here is (3, 480, 640)

        # TODO: ADD MASK and depth

        gt = self.gts[int(self.rgbs_paths[idx][-8:-4])]
        if self.category =="02":
            for j in range(len(gt)):
                if gt[j]['obj_id'] == 2:
                    R = torch.as_tensor(gt[j]['cam_R_m2c']).reshape(3,3)
                    t = torch.as_tensor(gt[j]['cam_t_m2c']) / 1e3
                    bbox = torch.as_tensor(gt[j]['obj_bb'])
                    obj_id = gt[j]['obj_id']
        else:
            R = torch.as_tensor(gt[0]['cam_R_m2c']).reshape(3,3)
            t = torch.as_tensor(gt[0]['cam_t_m2c']) / 1e3
            bbox = torch.as_tensor(gt[0]['obj_bb'])
            obj_id = gt[0]['obj_id']

        return img, R, t, bbox, obj_id, idx


    def collate_fn(self, batch):
        # here batch is a list contains tuples wraps returns of "__getitem__"
        cam_infos, K = [], []
        im_infos = []
        depth = []
        images = []
        transes = []
        for batch_im_id, data in enumerate(batch):
            image, R, t, bbox, obj_id, img_id = data
            im_info = {'scene_id': 2, 'view_id': img_id,
                       'group_id':0, 'batch_im_id': batch_im_id}
            im_infos.append(im_info)
            cam_info = im_info.copy()
            K.append(self.K * 1.)
            cam_infos.append(cam_info)
            images.append(image)
            transes.append((R,t))
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
    dataset = LM_Dataset(category="02")

    a= dataset.dataloader(use_collate_fn=True)
    for i, batch in enumerate(a): 
        print(batch)
