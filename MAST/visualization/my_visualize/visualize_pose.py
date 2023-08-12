from MAST.datasets.datasets_cfg import make_scene_dataset, make_object_dataset

# Pose estimator
from MAST.lib3d.rigid_mesh_database import MeshDataBase
from MAST.training.pose_models_cfg import create_model_refiner, create_model_coarse
from MAST.training.pose_models_cfg import check_update_config as check_update_config_pose
from MAST.rendering.bullet_batch_renderer import BulletBatchRenderer
from MAST.integrated.pose_predictor import CoarseRefinePosePredictor

from MAST.utils.logging import get_logger
import MAST.utils.tensor_collection as tc

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = get_logger(__name__)

import ipdb
import pandas as pd
import os
from PIL import Image
from typing import Dict, Optional, Sequence, Union
from collections import defaultdict
import logging
import numpy as np
from copy import deepcopy
import yaml
import torch
import argparse
import time
import glob
from pathlib import Path
import open3d as o3d
import random
import cv2

class SimplePredictionRunner:
    def __init__(self,
                 root: str,
                 pose_predictor: torch.nn.Module,
                 n_coarse_iterations=1,
                 n_refiner_iterations=4,
                 ):
        # paramters initialization
        self.root = root# BOP2020datasets-core/doorholder_v2/test_real_v2
        self.pose_predictor = pose_predictor
        self.n_coarse_iterations = n_coarse_iterations
        self.n_refiner_iterations = n_refiner_iterations

    def get_predictions(self):
        K = torch.tensor([
            [2077.04, 0., 979.327],
            [0., 2078.04, 582.283],
            [0., 0., 1.],
        ])[None, :, :].cuda()# (1, 3, 3)

        predictions = defaultdict(list)
        for im_id in range(1, 21):
            start = time.time()
            image = np.array(Image.open(os.path.join(self.root, 'rgb', f'{im_id:06d}.png')).rotate(180))[:, :, :3]# 
            image = torch.from_numpy(image[None, ...]).cuda().float().permute(0, 3, 1, 2) / 255 # (1, 3, H, W)

            mask_paths = glob.glob(os.path.join(self.root, 'pred_mask', f'{im_id:06d}_*.png'))
            for ins_id, mask_path in enumerate(mask_paths):
                info = dict(
                    batch_im_id=0,
                    label="obj_000001",
                    score=1.0,
                )

                obs_mask = np.array(Image.open(mask_path).rotate(180)).astype(np.float32)# 
                obs_mask[obs_mask<128] = 0.0                
                obs_mask[obs_mask>128] = 1.0

                xs, ys = obs_mask.nonzero()
                # bbox = [xs.min(), ys.min(), xs.max(), ys.max()]
                bbox = [ys.min(), xs.min(), ys.max(), xs.max()]

                detection = tc.PandasTensorCollection(
                    infos=pd.DataFrame([info]),
                    bboxes=torch.tensor([bbox]).cuda().float(),# (1, 4)
                )    
                obs_mask = torch.from_numpy(obs_mask[None, None, ...]).cuda()# (1, 1200, 1920)

                im_infos = [{
                    "im_id": im_id,
                    "ins_id": ins_id,
                    } for _ in range(len(mask_paths))
                ]
                for key in ('im_id', 'ins_id'):
                    detection.infos[key] = detection.infos['batch_im_id'].apply(lambda idx: im_infos[idx][key])

                # pose estimation
                final_preds, all_preds = self.pose_predictor.get_predictions(
                    images=image, obs_masks=obs_mask, K=K, detections=detection,
                    n_coarse_iterations=self.n_coarse_iterations,
                    n_refiner_iterations=self.n_refiner_iterations,
                )
                duration = time.time() - start

                # NOTE: time isn't correct for n iterations < max number of iterations
                for k, v in all_preds.items():
                    v.infos = v.infos.loc[:, ['im_id', 'ins_id', 'label', 'score']]
                    v.infos['time'] = duration
                    predictions[k].append(v.cpu())
                predictions['detections'].append(detection.cpu())

        predictions = dict(predictions)
        for k, v in predictions.items():
            predictions[k] = tc.concatenate(v)
        return predictions

def load_pose_models(coarse_dir, refiner_dir=None, n_workers=8):
    cfg = yaml.load((Path(coarse_dir) / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_pose(cfg)

    object_ds = make_object_dataset(cfg.object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_set=cfg.urdf_ds_name, n_workers=n_workers)
    mesh_db_batched = mesh_db.batched().cuda()

    def load_model(exp_dir):
        cfg = yaml.load((Path(exp_dir) / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config_pose(cfg)
        if cfg.train_refiner:
            model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        else:
            model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        ckpt = torch.load(os.path.join(exp_dir, 'checkpoint.pth.tar'))
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        model.config = cfg
        return model

    coarse_model = load_model(coarse_dir)
    refiner_model = load_model(refiner_dir)
    model = CoarseRefinePosePredictor(coarse_model=coarse_model,
                                      refiner_model=refiner_model)
    return model, mesh_db

def run_inference(args):
    logger.info(f"{'-'*80}")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    logger.info(f"{'-'*80}")

    pred_kwargs = dict()
    pose_predictor, mesh_db = load_pose_models(
        args.coarse_dir, args.refiner_dir, n_workers=8,
    )
    pred_runner = SimplePredictionRunner(
        root='BOP2020datasets-core/doorholder_v2/test_real_v2', 
        pose_predictor=pose_predictor
    )
    all_predictions = pred_runner.get_predictions()
    save_and_vis(all_predictions['refiner/iteration=4'])      
    return

def save_and_vis(results):
    poses = results.poses# (66, 4, 4) tensor
    im_id = results.infos['im_id'].values# (66) array
    ins_id = results.infos['ins_id'].values# (66) array
    result_dict = {
        'poses': poses,
        'im_id': im_id,
        'ins_id': ins_id,
    }
    torch.save(result_dict, './results.pth')

    def compute_proj(pts_3d, R, t, K):
        pts_2d = np.zeros((2, pts_3d.shape[0]), dtype=np.float32)#2*n
        pts_3d = np.dot(K, np.add(np.dot(R, pts_3d.T), t.T))#3*n
        pts_2d[0, :] = pts_3d[0, :]/pts_3d[2, :]
        pts_2d[1, :] = pts_3d[1, :]/pts_3d[2, :]
        return pts_2d.T#n*2
        
    root = 'BOP2020datasets-core/doorholder_v2'
    rgb_path = './vis_result/{im_id:06d}.png'
    model_points = np.array(o3d.read_point_cloud(
        root + '/models_pc/obj_000001.ply').points) / 1000.0

    cam_cx = 979.327
    cam_cy = 582.283
    cam_fx = 2077.04
    cam_fy = 2078.04
    K = np.array([[cam_fx, 0.0, cam_cx],\
                    [0.0, cam_fy, cam_cy],\
                    [0.0, 0.0, 1]])
    rotate_flag = np.zeros(21)    

    for i in range(len(results)):    
        im_id = results.infos.loc[i]['im_id']
        ins_id = results.infos.loc[i]['ins_id']
        pose = results.poses[i].numpy()# (4, 4)
        pred_R = pose[0:3, 0:3]# 3*3
        pred_t = pose[0:3, 3].reshape((1, 3))# 1*3

        img = Image.open(rgb_path.format(im_id=im_id))
        if rotate_flag[im_id] == 0:
            img = img.rotate(180)
            rotate_flag[im_id] = 1
        img = np.array(img)[:, :, 0:3]
        pts_2d = compute_proj(model_points, pred_R, pred_t, K).astype(np.int)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        mask = np.zeros((img.shape), dtype=np.uint8)  
        for i in np.arange(len(pts_2d)):
            mask = cv2.circle(mask, (pts_2d[i][0], pts_2d[i][1]), 1, [r,g,b], -1, cv2.LINE_AA)
        img = cv2.addWeighted(img, 1, mask, 0.4, 0)  

        im = Image.fromarray(img.astype(np.uint8))
        im.save(f'./vis_result/{im_id:06d}.png')

# def save_result(all_predictions):
#     ##  analysis predictions to save
#     results = {}
#     for key in all_predictions.keys():
#         if key == 'maskrcnn_detections/detections':
#             # detection
#             det_ret = all_predictions[key]
#             bboxes = det_ret.bboxes # (2339, 4)
#             scores = torch.Tensor(det_ret.infos["score"].values) # (2339)   
#             scene_id = torch.Tensor(det_ret.infos["scene_id"].values) # (2339)   
#             view_id = torch.Tensor(det_ret.infos["view_id"].values) # (2339)   
#             label = det_ret.infos["label"].values.tolist() # (2339)
#             results[key] = {
#                 "bboxes": bboxes,
#                 "scores": scores,
#                 "scene_id": scene_id,
#                 "view_id": view_id,
#                 "label": label,
#                 }
#         else:
#             # pose
#             pose_ret = all_predictions[key]
#             poses = pose_ret.poses # [N, 4, 4]
#             poses_input = pose_ret.poses_input # [N, 4, 4]
#             K_crop = pose_ret.K_crop # [N, 3, 3]
#             boxes_rend = pose_ret.boxes_rend # [N, 4]
#             boxes_crop = pose_ret.boxes_crop # [N, 4]
#             results[key] = {
#                 "poses": poses,
#                 "poses_input": poses_input,
#                 "K_crop": K_crop,
#                 "boxes_rend": boxes_rend,
#                 "boxes_crop": boxes_crop,
#                 }
#     torch.save(results, "tless_scene20_result.pth")

def main():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if 'MAST' in logger.name:
            logger.setLevel(logging.DEBUG)

    coarse_dir = './local_data/experiments/bop-doorholder-pbr-coarse-transnoise-zxyavg-370970'
    refiner_dir = './local_data/experiments/bop-doorholder-pbr-refiner--192197'

    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--config', default='bop-pbr', type=str)
    parser.add_argument("--coarse_dir", default=coarse_dir, type=str)
    parser.add_argument("--refiner_dir", default=refiner_dir, type=str)
    parser.add_argument("--save_dir", default='./', type=str)
    args = parser.parse_args()
    run_inference(args)
    
if __name__ == '__main__':
    main()
    # all_predictions = torch.load('./results.pth')
    # vis_result(all_predictions['coarse/iteration=1'])