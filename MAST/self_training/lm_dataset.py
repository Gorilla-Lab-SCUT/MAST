import os
import os.path as osp

import MAST.utils.tensor_collection as tc
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LM_Dataset(Dataset):
    '''split: str, train or test
       category: str 'all' or list of each object id
    '''
    def __init__(self,
        split="train",
        category=["01", "02", "04", "05", "06", "08",
        "09", "10", "11", "12", "13", "14", "15"]) -> None:
        super().__init__()

        self.train_on_bop = True  # whether train on bop format dataset
        self.load_depth = None
        # paths
        self.split = split
        if category == 'all': category = ["01", "02", "04", "05", "06", "08",
                                          "09", "10", "11", "12", "13", "14", "15"]
        self.category = category
        self.LM_PATH = "local_data/Linemod_preprocessed/data"

        # camera intrinsics
        self.K = np.array([[572.4114,    0.,      325.2611 ],
                           [  0.,      573.57043, 242.04899],
                           [  0.,        0.,        1.     ]])


        self.rgbs, self.masks, self.Rs, self.ts, self.bboxs, self.obj_ids = [], [], [], [], [], []
        for id in tqdm(self.category):
            rgbs_path = osp.join(self.LM_PATH, id, "rgb")
            masks_path = osp.join(self.LM_PATH, id, "mask")
            gt_path = osp.join(self.LM_PATH, id, "gt.yml")

            with open(osp.join(self.LM_PATH, id, self.split+'.txt'), 'r') as f:
                data_path = [line.strip("\n") for line in f.readlines()]
            with open(gt_path) as gtp:
                gt_single_class=yaml.load(gtp, Loader=yaml.FullLoader)  # read gt poses and bbox

            for idx in range(len(data_path)):
                # read an image, preprocess, and convert to tensor
                with Image.open(osp.join(rgbs_path, f'{data_path[idx]}.png')) as img: 
                    img = np.array(img)
                    img = torch.as_tensor(img)
                with Image.open(osp.join(masks_path, f'{data_path[idx]}.png')) as mask:
                    mask = np.array(mask)
                    mask = (mask > 0).astype("uint8")
                    mask = torch.as_tensor(mask)

                # read gts
                gt = gt_single_class[int(data_path[idx])]
                if id =="02":
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
                
                self.rgbs.append(img)
                self.masks.append(mask)
                self.Rs.append(R)
                self.ts.append(t)
                self.bboxs.append(bbox)
                self.obj_ids.append(obj_id)


    @staticmethod
    def to_homo_matrix(R,t):
        """formulate homogeneous transformation matrix from 3x4 Rt pose"""
        Rt = np.concatenate([R, np.expand_dims(t, axis=1)], axis=-1)
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
        return len(self.rgbs)


    def __getitem__(self, idx):
        # TODO: ADD depth
        img = self.rgbs[idx]
        mask = self.masks[idx][...,0]
        R = self.Rs[idx]
        t = self.ts[idx]
        x, y, w, h = self.bboxs[idx]
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        obj_id = self.obj_ids[idx]

        name = f'obj_{int(obj_id):06d}'

        T0C = self.to_homo_matrix(np.eye(3), np.zeros(3))
        T0O = self.to_homo_matrix(R, t)
        objects = [dict(label=name, name=name, TWO=T0O, T0O=T0O,
            visib_fract=1.,
            id_in_segm=1, bbox=[x1, y1, x2, y2])]

        camera = dict(T0C=T0C, K=self.K, TWC=T0C, resolution=img.shape[:2])

        row = {'scene_id': 1, 'cam_id': 'cam', 'view_id': idx, 'cam_name': 'cam'}
        obs = dict(
            objects=objects,
            camera=camera,
            frame_info=row,
        )

        return img, mask, obs


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


    def dataloader(self, batch_size=1, n_workers=0, sampler=None, use_collate_fn=False, shuffle=None):
        if use_collate_fn:
            collate_function = self.collate_fn
        else:
            collate_function = None

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          num_workers=n_workers, sampler=sampler,
                          collate_fn=collate_function)



if __name__ == "__main__":
    dataset = LM_Dataset()
    print(len(dataset))
    a = dataset.dataloader(use_collate_fn=False, shuffle=False)
    for i, batch in enumerate(a): 
        print(batch);exit()
