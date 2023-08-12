import json
import pickle
from pathlib import Path

import MAST.utils.tensor_collection as tc
import numpy as np
import pandas as pd
import torch
from MAST.datasets.bop import build_index
from MAST.lib3d import Transform
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class YCBV_bop_Dataset(Dataset):
    def __init__(self, ds_dir='local_data/bop_datasets/ycbv', split='test_bop', load_depth=False) -> None:
        super().__init__()

        ds_dir = Path(ds_dir)
        self.ds_dir = ds_dir
        assert ds_dir.exists(), 'Dataset does not exists.'
        self.split = split
        self.base_dir = ds_dir / split
        print(f'Building index and loading annotations...')
        save_file_index = self.ds_dir / f'index_{split}.feather'
        save_file_annotations = self.ds_dir / f'annotations_{split}.pkl'
        build_index(
            ds_dir=ds_dir, save_file=save_file_index,
            save_file_annotations=save_file_annotations,
            split=split)
        self.frame_index = pd.read_feather(save_file_index).reset_index(drop=True)
        self.annotations = pickle.loads(save_file_annotations.read_bytes())

        models_infos = json.loads((ds_dir / 'models' / 'models_info.json').read_text())
        self.all_labels = [f'obj_{int(obj_id):06d}' for obj_id in models_infos.keys()]
        self.load_depth = load_depth

        # preprocess operation on images
        self.img_transform = transforms.Compose([
	    transforms.ToTensor(), 
	    ])


    def __len__(self):
        return len(self.frame_index)


    def __getitem__(self, frame_id):
        row = self.frame_index.iloc[frame_id]
        scene_id, view_id = row.scene_id, row.view_id
        view_id = int(view_id)
        view_id_str = f'{view_id:06d}'
        scene_id_str = f'{int(scene_id):06d}'
        scene_dir = self.base_dir / scene_id_str

        rgb_dir = scene_dir / 'rgb'
        if not rgb_dir.exists(): rgb_dir = scene_dir / 'gray'
        rgb_path = rgb_dir / f'{view_id_str}.png'
        if not rgb_path.exists(): rgb_path = rgb_path.with_suffix('.jpg')
        if not rgb_path.exists(): rgb_path = rgb_path.with_suffix('.tif')
        rgb = np.array(Image.open(rgb_path))
        if rgb.ndim == 2: rgb = np.repeat(rgb[..., None], 3, axis=-1)
        rgb = rgb[..., :3]
        h, w = rgb.shape[:2]
        rgb = torch.as_tensor(rgb).permute(2, 0, 1) / 255

        cam_annotation = self.annotations[scene_id_str]['scene_camera'][str(view_id)]
        if 'cam_R_w2c' in cam_annotation and False:
            RC0 = np.array(cam_annotation['cam_R_w2c']).reshape(3, 3)
            tC0 = np.array(cam_annotation['cam_t_w2c']) * 0.001
            TC0 = Transform(RC0, tC0)
        else:
            TC0 = Transform(np.eye(3), np.zeros(3))
        K = np.array(cam_annotation['cam_K']).reshape(3, 3)
        T0C = TC0.inverse()
        T0C = T0C.toHomogeneousMatrix()
        camera = dict(T0C=T0C, K=K, TWC=T0C, resolution=rgb.shape[:2])
        T0C = TC0.inverse()

        objects = []
        mask = np.zeros((h, w), dtype=np.uint8)
        if 'scene_gt_info' in self.annotations[scene_id_str]:
            annotation = self.annotations[scene_id_str]['scene_gt'][str(view_id)]
            n_objects = len(annotation)
            visib = self.annotations[scene_id_str]['scene_gt_info'][str(view_id)]
            for n in range(n_objects):
                RCO = np.array(annotation[n]['cam_R_m2c']).reshape(3, 3)
                tCO = np.array(annotation[n]['cam_t_m2c']) * 0.001
                TCO = Transform(RCO, tCO)
                T0O = T0C * TCO
                T0O = T0O.toHomogeneousMatrix()

                ## T0C: world frame to camera frame (camera pose); 
                ## TCO: camera frame to obj model frame (obj pose relative to camera frame)
                ## T0O: world frame to obj model frame

                obj_id = annotation[n]['obj_id']
                name = f'obj_{int(obj_id):06d}'
                bbox_visib = np.array(visib[n]['bbox_visib'])
                x, y, w, h = bbox_visib
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                obj = dict(label=name, name=name, TWO=T0O, T0O=T0O,
                           visib_fract=visib[n]['visib_fract'],
                           id_in_segm=n+1, bbox=[x1, y1, x2, y2])
                objects.append(obj)

            mask_path = scene_dir / 'mask_visib' / f'{view_id_str}_all.png'
            if mask_path.exists():
                mask = np.array(Image.open(mask_path))
            else:
                for n in range(n_objects):
                    mask_n = np.array(Image.open(scene_dir / 'mask_visib' / f'{view_id_str}_{n:06d}.png'))
                    mask[mask_n == 255] = n + 1

        mask = torch.as_tensor(mask)
        obs = dict(
            objects=objects,
            camera=camera,
            frame_info=row.to_dict(),
        )
        Trans = {}
        for obj in objects:
            Trans[obj['label'].strip('obj_')] = obj['TWO']


        return rgb, Trans, frame_id, camera


    def collate_fn(self, batch):
        # here batch is a list contains tuples wraps returns of "__getitem__"
        cam_infos, K = [], []
        im_infos = []
        depth = []
        images = []
        transes = []
        for batch_im_id, data in enumerate(batch):
            image, trans, img_id, camera = data
            im_info = {'scene_id': 2, 'view_id': int(img_id),
                       'group_id':0, 'batch_im_id': batch_im_id}
            im_infos.append(im_info)
            cam_info = im_info.copy()
            K.append(camera['K'])
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
    a = YCBV_bop_Dataset('local_data/bop_datasets/ycbv','test_bop')
    dl = a.dataloader()
    for i, d, in enumerate(dl):
        print(i, d)
        exit()
