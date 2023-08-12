from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from MAST.datasets.augmentations import CropResizeToAspectAugmentation
from MAST.datasets.wrappers.augmentation_wrapper import AugmentationWrapper
from torchvision import transforms

from .plotter import Plotter


def filter_predictions(preds, scene_id, view_id=None, th=None):
    mask = preds.infos['scene_id'] == scene_id
    if view_id is not None:
        mask = np.logical_and(mask, preds.infos['view_id'] == view_id)
    if th is not None:
        mask = np.logical_and(mask, preds.infos['score'] >= th)
    keep_ids = np.where(mask)[0]
    preds = preds[keep_ids]
    return preds


def render_prediction_wrt_camera(renderer, pred, camera=None, resolution=(640, 480)):
    pred = pred.cpu()
    camera.update(TWC=np.eye(4))

    list_objects = []
    for n in range(len(pred)):
        row = pred.infos.iloc[n]
        obj = dict(
            name=row.label,
            color=(1, 1, 1, 1),
            TWO=pred.poses[n].numpy(),
        )
        list_objects.append(obj)
    rgb_rendered = renderer.render_scene(list_objects, [camera])[0]['rgb']
    return rgb_rendered


def make_singleview_prediction_plots(scene_ds, renderer, predictions, detections=None, resolution=(640, 480)):
    plotter = Plotter()

    scene_id, view_id = np.unique(predictions.infos['scene_id']).item(), np.unique(predictions.infos['view_id']).item()

    scene_ds_index = scene_ds.frame_index
    scene_ds_index['ds_idx'] = np.arange(len(scene_ds_index))
    scene_ds_index = scene_ds_index.set_index(['scene_id', 'view_id'])
    idx = scene_ds_index.loc[(scene_id, view_id), 'ds_idx']

    augmentation = CropResizeToAspectAugmentation(resize=resolution)
    scene_ds = AugmentationWrapper(scene_ds, augmentation)
    rgb_input, mask, state = scene_ds[idx]

    figures = dict()

    figures['input_im'] = plotter.plot_image(rgb_input)

    if detections is not None:
        fig_dets = plotter.plot_image(rgb_input)
        fig_dets = plotter.plot_maskrcnn_bboxes(fig_dets, detections)
        figures['detections'] = fig_dets

    pred_rendered = render_prediction_wrt_camera(renderer, predictions, camera=state['camera'])
    figures['pred_rendered'] = plotter.plot_image(pred_rendered)
    figures['pred_overlay'] = plotter.plot_overlay(rgb_input, pred_rendered)
    return figures


def render_result(renderer, TCO_output, TCO_gt, K_crop, labels, render_size, images_crop, model_outputs,TCO_input):

    gt_rend = renderer.render(obj_infos=[dict(name=l) for l in labels],
                    TCO=TCO_gt,
                    K=K_crop, resolution=render_size).contiguous()
    pred_rend = renderer.render(obj_infos=[dict(name=l) for l in labels],
                    TCO=TCO_output,
                    K=K_crop, resolution=render_size).contiguous()
    for i in range(len(labels)):
        if torch.abs(TCO_gt[i, 2,3]-TCO_output[i,2,3])>0.02 or\
           torch.abs(TCO_gt[i, 2,3]-TCO_output[i,2,3])<0.002:
            msk = gt_rend[i] == 0
            gt_rend[i][msk] = images_crop[i][msk]
            msk = pred_rend[i] == 0
            pred_rend[i][msk] = images_crop[i][msk]
            transforms.ToPILImage()(torch.cat((gt_rend[i], pred_rend[i], images_crop[i]),dim=-2)).save('xxx.png')
            z_prob = torch.max(F.softmax(model_outputs['z_bin'][i],dim=-1)).item()
            R_prob = torch.max(F.softmax(model_outputs['R_bin'][i],dim=-1)).item()
            
            print(f'\nz_input: {TCO_input[i, 2,3]}',
                f'\nz_gt: {TCO_gt[i, 2,3]}',
                f'\nz_pred: {TCO_output[i,2,3]}',
                f'\nR_prob: {R_prob}',
                f'\nz_prob: {z_prob}')
            input()


def add_occlusion(renderer, K_crop, labels, render_size, images_crop, TCO_input):
    '''add occlusion on image when rendering'''
    TCO = deepcopy(TCO_input)
    TCO[..., 2, 3] += 0.2
    TCO[..., :2, 3] += float(torch.rand(1) * 0.1)
    label = np.random.permutation(labels)
    occ_rend = renderer.render(obj_infos=[dict(name=l) for l in label],
                    TCO=TCO,
                    K=K_crop, resolution=render_size).contiguous()
    msk = occ_rend != 0
    images_crop[msk] = occ_rend[msk]
    transforms.ToPILImage()(images_crop[0]).save('xxx.png')
    return images_crop
