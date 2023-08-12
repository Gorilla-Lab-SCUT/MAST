import numpy as np
import numpy.ma as ma
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, Compose, ToPILImage
from .st_augmentations import (PillowBlur, PillowBrightness, PillowColor,
                            PillowContrast, PillowSharpness)


class make_selftrain_dataset(Dataset):
    def __init__(self, data, occ_aug=False, rgb_aug=False, occ_rgbs=None, occ_masks=None, occ_id=None) -> None:
        super().__init__()
        self.occ_rgbs = occ_rgbs
        self.occ_masks = occ_masks
        self.occ_id = occ_id
        self.data = data
        self.occ_aug = occ_aug
        self.rgb_aug = rgb_aug
        print(f'Post augmentation on occ: {self.occ_aug}, on RGB: {self.rgb_aug}')
        self.rgb_augmentations = [
                PillowBlur(factor_interval=(1, 2), p=0.1),
                PillowSharpness(factor_interval=(0., 2.)),
                PillowContrast(factor_interval=(0.5, 1.5)),
                PillowBrightness(factor_interval=(0.7, 1.3)),
                PillowColor(factor_interval=(0., 1.)),
            ]
        self.transcolor =  Compose([ColorJitter(0.05, 0.05, 0.05, 0.01)])

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        image = self.data['images'][idx]
        K = self.data['K'][idx]
        TCO_gt = self.data['TCO_gt'][idx]
        TCO_pred = self.data['TCO_pred'][idx]
        labels = self.data['labels'][idx]
        bbox = self.data['bboxes'][idx]
        zb_pseudo = self.data['zb_pseudo'][idx]
        mask = self.data['masks'][idx]

        if self.occ_aug and (np.random.random(1) < 0.7):
            flag = 0
            attempt=0
            image = np.array(image.permute(1,2,0))  # (h,w,c)
            mask = np.array(mask)                   # (h,w)
            full_mask_count = np.sum(mask)
            while flag<3:
                seed = np.random.choice(range(len(self.occ_rgbs)))
                if labels != f'obj_{int(self.occ_id[seed]):06d}':
                    front_rgb = np.array(self.occ_rgbs[seed]) / 255.
                    front_label = np.array(self.occ_masks[seed][...,0])
                    mask_front = ma.getmaskarray(ma.masked_equal(front_label, np.array([1.])))
                    mask_temp = ~mask_front * mask
                    percentage = np.sum(mask_temp) / full_mask_count
                    attempt += 1
                    if percentage < 0.1 or percentage > 0.95:
                        if attempt > 20: pass
                        else: continue
                    mask = mask_temp
                    mask_front_rgb = np.repeat(np.expand_dims(mask_front, axis=-1),3,-1)
                    image = (~mask_front_rgb) * image + mask_front_rgb * front_rgb
                    flag += 1
                    attempt=0
            image = torch.as_tensor(image).float().permute(2, 0, 1)  # (c,h,w)
            mask = torch.as_tensor(mask)
            
        if self.rgb_aug and (np.random.random(1) < 0.5):
            image = self.transcolor(image)
            for aug in self.rgb_augmentations:
                image = aug(image)

        return image, K, TCO_gt, TCO_pred, labels, bbox, zb_pseudo


    def collate_fn(self, batch):
        # here batch is a list contains tuples wraps returns of "__getitem__"
        images, Ks, TCO_gts, TCO_preds, labelss, bboxes, zb_pseudos = [], [], [], [], [], [], []
        for batch_im_id, data in enumerate(batch):
            image, K, TCO_gt, TCO_pred, labels, bbox, zb_pseudo = data
            images.append(image)
            Ks.append(K)
            TCO_gts.append(TCO_gt)
            TCO_preds.append(TCO_pred)
            labelss.append(labels)
            bboxes.append(bbox)
            zb_pseudos.append(zb_pseudo)

        data = dict(
                images=torch.stack(images),
                K=torch.stack(Ks),
                TCO_gt=torch.stack(TCO_gts),
                TCO_pred=torch.stack(TCO_preds),
                labels=labelss,
                bboxes=torch.stack(bboxes),
                zb_pseudo=torch.stack(zb_pseudos)
            )
        return data
