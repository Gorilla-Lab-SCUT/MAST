import argparse
import functools
import logging
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import MAST.utils.tensor_collection as tc
import numpy as np
import torch
import torch.multiprocessing
import yaml
from MAST.bop_config import (BOP_CONFIG, PBR_COARSE, PBR_DETECTORS,
                                 PBR_REFINER, SYNT_REAL_COARSE,
                                 SYNT_REAL_DETECTORS, SYNT_REAL_REFINER)
from MAST.config import EXP_DIR, LOCAL_DATA_DIR, MEMORY, RESULTS_DIR
from MAST.datasets.datasets_cfg import make_object_dataset
from MAST.evaluation.runner_utils import format_results, gather_predictions
from MAST.integrated.detector import Detector
from MAST.integrated.icp_refiner import ICPRefiner
from MAST.integrated.pose_predictor import CoarseRefinePosePredictor
# Pose estimator
from MAST.lib3d.rigid_mesh_database import MeshDataBase
from MAST.rendering.bullet_batch_renderer import BulletBatchRenderer
# Detection
from MAST.training.detector_models_cfg import \
    check_update_config as check_update_config_detector
from MAST.training.detector_models_cfg import create_model_detector
from MAST.training.pose_models_cfg import \
    check_update_config as check_update_config_pose
from MAST.training.pose_models_cfg import (create_model_coarse,
                                               create_model_refiner)
from MAST.utils.distributed import (get_rank, get_tmp_dir, get_world_size,
                                        init_distributed_mode)
from MAST.utils.logging import get_logger
from tqdm import tqdm


# def patch_tqdm():
#     tqdm = sys.modules['tqdm'].tqdm
#     sys.modules['tqdm'].tqdm = functools.partial(tqdm, file=sys.stdout)
#     return
# patch_tqdm()

logger = get_logger(__name__)

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_detector(run_id):
    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_detector(cfg)
    label_to_category_id = cfg.label_to_category_id
    model = create_model_detector(cfg, len(label_to_category_id))
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    model.cfg = cfg
    model.config = cfg
    model = Detector(model)
    return model


def load_pose_models(coarse_run_id, refiner_run_id=None, n_workers=8, coarse_epoch=None, refiner_epoch=None):
    run_dir = EXP_DIR / coarse_run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_pose(cfg)

    object_ds = make_object_dataset(cfg.object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_set=cfg.urdf_ds_name, n_workers=n_workers)
    mesh_db_batched = mesh_db.batched().cuda()

    def load_model(run_id, epoch):
        if run_id is None:
            return
        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config_pose(cfg)
        if cfg.train_refiner:
            model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        else:
            model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        ckpt = torch.load(run_dir / (f'checkpoint-epoch={epoch}.pth.tar'))
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        model.config = cfg
        return model

    coarse_model = load_model(coarse_run_id, coarse_epoch)
    refiner_model = load_model(refiner_run_id, refiner_epoch)
    model = CoarseRefinePosePredictor(coarse_model=coarse_model,
                                      refiner_model=refiner_model)
    return model, mesh_db



class PredictionRunner():
    def __init__(self, scene_ds, batch_size=1, cache_data=False, n_workers=4):
        
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.tmp_dir = get_tmp_dir()
        self.load_depth = None
        self.sampler = None

        dataloader = scene_ds.dataloader(batch_size=batch_size,
                                n_workers=n_workers,
                                sampler=self.sampler,
                                use_collate_fn=True)
        if cache_data:
            self.dataloader = list(tqdm(dataloader))
        else:
            self.dataloader = dataloader

    def get_predictions(self,
                        detector,
                        pose_predictor,
                        icp_refiner=None,
                        n_coarse_iterations=1,
                        n_refiner_iterations=1,
                        detection_th=0.0):

        predictions = defaultdict(list)
        use_icp = icp_refiner is not None
        for n, data in enumerate(tqdm(self.dataloader)):
            images = data['images'].cuda().float()
            cameras = data['cameras'].cuda().float()
            im_infos = data['im_infos']
            transes = data['transes']
            depth = None
            if self.load_depth:
                depth = data['depth'].cuda().float()
            # logger.info(f"{'-'*80}")
            # logger.info(f"Predictions on {data['im_infos']}")
            
            def get_preds():
                torch.cuda.synchronize()
                start = time.time()
                this_batch_detections = detector.get_detections(
                    images=images, one_instance_per_class=False, detection_th=detection_th,
                    output_masks=use_icp, mask_th=0.9)

                for key in ('scene_id', 'view_id', 'group_id'):
                    this_batch_detections.infos[key] = this_batch_detections.infos['batch_im_id'].apply(lambda idx: im_infos[idx][key])

                all_preds = dict()
                if len(this_batch_detections) > 0:
                    final_preds, all_preds = pose_predictor.get_predictions(
                        images, cameras.K, detections=this_batch_detections,
                        n_coarse_iterations=n_coarse_iterations,
                        n_refiner_iterations=n_refiner_iterations, transes=transes
                    )

                    if use_icp:
                        all_preds['icp'] = icp_refiner.refine_poses(final_preds, this_batch_detections.masks, depth, cameras)

                torch.cuda.synchronize()
                duration = time.time() - start
                n_dets = len(this_batch_detections)

                # logger.info(f'Full predictions: {n_dets} detections + pose estimation in {duration:.3f} s')
                # logger.info(f"{'-'*80}")
                return this_batch_detections, all_preds, duration

            if n == 0:
                get_preds()
            this_batch_detections, all_preds, duration = get_preds()

            if use_icp:
                this_batch_detections.delete_tensor('masks')  # Saves memory when saving

            # NOTE: time isn't correct for n iterations < max number of iterations
            for k, v in all_preds.items():
                v.infos = v.infos.loc[:, ['scene_id', 'view_id', 'label', 'score']]
                v.infos['time'] = duration
                predictions[k].append(v.cpu())
            predictions['detections'].append(this_batch_detections.cpu())

        predictions = dict(predictions)
        for k, v in predictions.items():
            predictions[k] = tc.concatenate(v)
        return predictions



def run_inference(args, ds_cfg):
    """ inference on a single dataset"""

    # print configs
    logger.info(f"{'-'*80}")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    logger.info(f"{'-'*80}")

    scene_ds = ds_cfg.dataset

    if args.icp:
        scene_ds.load_depth = args.icp

    pred_kwargs = dict()

    pred_runner = PredictionRunner(scene_ds, batch_size=args.pred_bsz,
                                    cache_data=False, n_workers=args.n_workers)

    detector = load_detector(args.detector_run_id)

    pose_predictor, mesh_db = load_pose_models(
        coarse_run_id=args.coarse_run_id,
        refiner_run_id=args.refiner_run_id,
        n_workers=args.n_workers,
        coarse_epoch=args.coarse_epoch,
        refiner_epoch=args.refiner_epoch
    )

    icp_refiner = None
    if args.icp:
        renderer = pose_predictor.coarse_model.renderer
        icp_refiner = ICPRefiner(mesh_db,
                                 renderer=renderer,
                                 resolution=pose_predictor.coarse_model.cfg.input_resize)

    pred_kwargs.update({
        'maskrcnn_detections': dict(
            detector=detector,
            pose_predictor=pose_predictor,
            n_coarse_iterations=args.n_coarse_iterations,
            n_refiner_iterations=args.n_refiner_iterations,
            icp_refiner=icp_refiner
            )})

    all_predictions = dict()
    for pred_prefix, pred_kwargs_n in pred_kwargs.items():
        logger.info(f"Prediction: {pred_prefix}")
        preds = pred_runner.get_predictions(**pred_kwargs_n)
        for preds_name, preds_n in preds.items():
            all_predictions[f'{pred_prefix}/{preds_name}'] = preds_n

    logger.info("Done with inference.")
    torch.distributed.barrier()

    for k, v in all_predictions.items():
        all_predictions[k] = v.gather_distributed(tmp_dir=get_tmp_dir()).cpu()

    if get_rank() == 0:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f'Finished inference on {args.ds_name}')
        results = format_results(all_predictions, dict(), dict())
        torch.save(results, save_dir / (f'c_epoch={args.coarse_epoch}-r_epoch={args.refiner_epoch}.pth.tar'))
        (save_dir / 'config.yaml').write_text(yaml.dump(args))
        logger.info(f'Saved predictions in {save_dir}/c_epoch={args.coarse_epoch}-r_epoch={args.refiner_epoch}.pth.tar')

    torch.distributed.barrier()
    return


class LM:
    def __init__(self, category: str = '01') -> None:
        from .lm_dataset import LM_Dataset
        self.dataset = LM_Dataset(split='test', category=category)
        self.ds_name = 'lmo'
        self.save_name = f'lm/{category}'


class LMO:
    def __init__(self) -> None:
        from .lmo_dataset import LMO_Dataset
        self.dataset = LMO_Dataset()
        self.save_name = 'lmo'
        self.ds_name = 'lmo'


class LMO_BOP:
    def __init__(self) -> None:
        from .lmo_bop_dataset import LMO_bop_Dataset
        self.dataset = LMO_bop_Dataset()
        self.save_name = 'lmo_bop'
        self.ds_name = 'lmo'


class YCBV:
    def __init__(self) -> None:
        from .ycbv_bop_dataset import YCBV_bop_Dataset
        self.dataset = YCBV_bop_Dataset()
        self.save_name = 'ycbv_bop'
        self.ds_name = 'ycbv'


class HB:
    def __init__(self) -> None:
        from .hb_dataset import HB_Dataset
        self.dataset = HB_Dataset()
        self.save_name = 'hb'
        self.ds_name = 'lmo'


def dataset_cfg(ds='LM', category=None):
    if ds == 'LM':
        return LM(category=category)
    elif ds == 'LMO':
        return LMO()
    elif ds == 'LMO_BOP':
        return LMO_BOP()
    elif ds == 'HB':
        return HB()
    elif ds == 'YCBV':
        return YCBV()


if __name__ =="__main__":

    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if 'MAST' in logger.name:
            logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--config', default='bop-pbr', type=str)
    parser.add_argument('--icp', action='store_true')
    parser.add_argument('--coarse_epoch', default=0, type=int)
    parser.add_argument('--refiner_epoch', default=0, type=int)
    parser.add_argument('--ds', type=str)
    parser.add_argument('--obj_id', default='01', type=str)
    parser.add_argument('--coarse_run_id', type=str)
    args = parser.parse_args()

    init_distributed_mode()

    cfg = argparse.ArgumentParser('').parse_args([])

    cfg.coarse_epoch = args.coarse_epoch
    cfg.refiner_epoch = args.refiner_epoch
    cfg.n_workers = 8
    cfg.pred_bsz = 1
    cfg.n_frames = None
    cfg.n_groups = None
    cfg.skip_evaluation = False
    cfg.external_predictions = True

    cfg.n_coarse_iterations = 1
    cfg.n_refiner_iterations = 0
    cfg.icp = args.icp

    if args.icp:
        args.comment = f'icp-{args.comment}'

    if args.config == 'bop-pbr':
        MODELS_DETECTORS = PBR_DETECTORS
        MODELS_COARSE = PBR_COARSE
        MODELS_REFINER = PBR_REFINER

    elif args.config == 'bop-synt+real':
        MODELS_DETECTORS = SYNT_REAL_DETECTORS
        MODELS_COARSE = SYNT_REAL_COARSE
        MODELS_REFINER = SYNT_REAL_REFINER

    ds_cfg = dataset_cfg(args.ds, category=args.obj_id)
    ds_name = ds_cfg.ds_name

    this_cfg = deepcopy(cfg)
    this_cfg.ds_name = BOP_CONFIG[ds_name]['inference_ds_name'][0]

    this_cfg.detector_run_id = MODELS_DETECTORS.get(ds_name)
    # this_cfg.coarse_run_id = MODELS_COARSE.get(ds_name)
    this_cfg.coarse_run_id = args.coarse_run_id
    this_cfg.refiner_run_id = MODELS_REFINER.get(ds_name)
    if len(args.comment) == 0:
        save_dir = RESULTS_DIR / f'{args.config}-{this_cfg.coarse_run_id}'
    else:
        save_dir = RESULTS_DIR / f'{args.config}-{this_cfg.coarse_run_id}-{args.comment}'
    logger.info(f'Save dir: {save_dir}')
    this_cfg.save_dir = save_dir / f'dataset={ds_cfg.save_name}'
    run_inference(this_cfg, ds_cfg)
