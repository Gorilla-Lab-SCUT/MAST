import operator as op
from torch.utils.data import ConcatDataset
from MAST.datasets.bop import BOPDataset
from .lm_dataset import LM_Dataset


class ST_config:
    pass


def st_cfg(ds: str, obj_id: str):
    if ds == 'LMO':
        stcfg.occ_aug_ds = LM_Dataset('train')

        if obj_id == '01':  # ape
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            stcfg.scene_target_ds_train = BOPDataset(ds_dir=ds_dir, split='test_nobop', single_obj_name=stcfg.obj_name)  # tgt train
            stcfg.scene_ds_val = BOPDataset(ds_dir=ds_dir, split='test_bop', single_obj_name=stcfg.obj_name)  # tgt test
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.0
            stcfg.EMA_alpha = 0.0
            stcfg.model_coarse = 1
            stcfg.model_refiner = 1
            stcfg.self_train_lr = 3e-5
            stcfg.self_train_epoch = 10
            stcfg.obj_threshold = {
            'obj_000001': {'name': 'Ape',       'th': 0.5, 'op': op.gt},
            'obj_000002': {'name': 'Benchvise', 'th': 1e5, 'op': op.gt},
            'obj_000004': {'name': 'Camera',    'th': 1e5, 'op': op.gt},
            'obj_000005': {'name': 'Can',       'th': 1e5, 'op': op.gt},
            'obj_000006': {'name': 'Cat',       'th': 1e5, 'op': op.gt},
            'obj_000008': {'name': 'Driller',   'th': 1e5, 'op': op.gt},
            'obj_000009': {'name': 'Duck',      'th': 1e5, 'op': op.gt},
            'obj_000010': {'name': 'Eggbox',    'th': 1e5, 'op': op.gt},
            'obj_000011': {'name': 'Glue',      'th': 1e5, 'op': op.gt},
            'obj_000012': {'name': 'Holepunch', 'th': 1e5, 'op': op.gt},
            'obj_000013': {'name': 'Iron',      'th': 1e5, 'op': op.gt},
            'obj_000014': {'name': 'Lamp',      'th': 1e5, 'op': op.gt},
            'obj_000015': {'name': 'Phone',     'th': 1e5, 'op': op.gt},
            }


        if obj_id == '05':  # can
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            target_train_1 = BOPDataset(ds_dir=ds_dir, split='test_nobop', single_obj_name=stcfg.obj_name)
            target_train_2 = LM_Dataset(split="train", category=[obj_id])
            stcfg.scene_target_ds_train = ConcatDataset([target_train_1, target_train_2])  # tgt train
            stcfg.scene_target_ds_train = target_train_1
            stcfg.scene_ds_val = BOPDataset(ds_dir=ds_dir, split='test_bop', single_obj_name=stcfg.obj_name)  # tgt test
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = True
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.0
            stcfg.EMA_alpha = 0.0
            stcfg.model_coarse = 1
            stcfg.model_refiner = 1 #None
            stcfg.self_train_lr = 3e-5
            stcfg.self_train_epoch = 10
            stcfg.obj_threshold = {
            'obj_000001': {'name': 'Ape',       'th': 1e5, 'op': op.gt},
            'obj_000002': {'name': 'Benchvise', 'th': 1e5, 'op': op.gt},
            'obj_000004': {'name': 'Camera',    'th': 1e5, 'op': op.gt},
            'obj_000005': {'name': 'Can',       'th': 0.5, 'op': op.gt},
            'obj_000006': {'name': 'Cat',       'th': 1e5, 'op': op.gt},
            'obj_000008': {'name': 'Driller',   'th': 1e5, 'op': op.gt},
            'obj_000009': {'name': 'Duck',      'th': 1e5, 'op': op.gt},
            'obj_000010': {'name': 'Eggbox',    'th': 1e5, 'op': op.gt},
            'obj_000011': {'name': 'Glue',      'th': 1e5, 'op': op.gt},
            'obj_000012': {'name': 'Holepunch', 'th': 1e5, 'op': op.gt},
            'obj_000013': {'name': 'Iron',      'th': 1e5, 'op': op.gt},
            'obj_000014': {'name': 'Lamp',      'th': 1e5, 'op': op.gt},
            'obj_000015': {'name': 'Phone',     'th': 1e5, 'op': op.gt},
            }


        if obj_id == '06':  # cat
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            target_train_1 = BOPDataset(ds_dir=ds_dir, split='test_nobop', single_obj_name=stcfg.obj_name)
            target_train_2 = LM_Dataset(split="train", category=[obj_id])
            stcfg.scene_target_ds_train = ConcatDataset([target_train_1, target_train_2])  # tgt train
            stcfg.scene_ds_val = BOPDataset(ds_dir=ds_dir, split='test_bop', single_obj_name=stcfg.obj_name)  # tgt test
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.0
            stcfg.EMA_alpha = 0.0
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 3e-5
            stcfg.self_train_epoch = 15
            stcfg.obj_threshold = {
            'obj_000001': {'name': 'Ape',       'th': 1e5, 'op': op.gt},
            'obj_000002': {'name': 'Benchvise', 'th': 1e5, 'op': op.gt},
            'obj_000004': {'name': 'Camera',    'th': 1e5, 'op': op.gt},
            'obj_000005': {'name': 'Can',       'th': 1e5, 'op': op.gt},
            'obj_000006': {'name': 'Cat',       'th': 0.55, 'op': op.gt},
            'obj_000008': {'name': 'Driller',   'th': 1e5, 'op': op.gt},
            'obj_000009': {'name': 'Duck',      'th': 1e5, 'op': op.gt},
            'obj_000010': {'name': 'Eggbox',    'th': 1e5, 'op': op.gt},
            'obj_000011': {'name': 'Glue',      'th': 1e5, 'op': op.gt},
            'obj_000012': {'name': 'Holepunch', 'th': 1e5, 'op': op.gt},
            'obj_000013': {'name': 'Iron',      'th': 1e5, 'op': op.gt},
            'obj_000014': {'name': 'Lamp',      'th': 1e5, 'op': op.gt},
            'obj_000015': {'name': 'Phone',     'th': 1e5, 'op': op.gt},
            }


        if obj_id == '08':  # driller
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            target_train_1 = BOPDataset(ds_dir=ds_dir, split='test_nobop', single_obj_name=stcfg.obj_name)
            target_train_2 = LM_Dataset(split="train", category=[obj_id])
            stcfg.scene_target_ds_train = ConcatDataset([target_train_1, target_train_2])  # tgt train
            stcfg.scene_ds_val = BOPDataset(ds_dir=ds_dir, split='test_bop', single_obj_name=stcfg.obj_name)  # tgt test
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.0
            stcfg.EMA_alpha = 0.0
            stcfg.model_coarse = 1
            stcfg.model_refiner = 1
            stcfg.self_train_lr = 3e-5
            stcfg.self_train_epoch = 10
            stcfg.obj_threshold = {
            'obj_000001': {'name': 'Ape',       'th': 1e5, 'op': op.gt},
            'obj_000002': {'name': 'Benchvise', 'th': 1e5, 'op': op.gt},
            'obj_000004': {'name': 'Camera',    'th': 1e5, 'op': op.gt},
            'obj_000005': {'name': 'Can',       'th': 1e5, 'op': op.gt},
            'obj_000006': {'name': 'Cat',       'th': 1e5, 'op': op.gt},
            'obj_000008': {'name': 'Driller',   'th': 0.34, 'op': op.gt},
            'obj_000009': {'name': 'Duck',      'th': 1e5, 'op': op.gt},
            'obj_000010': {'name': 'Eggbox',    'th': 1e5, 'op': op.gt},
            'obj_000011': {'name': 'Glue',      'th': 1e5, 'op': op.gt},
            'obj_000012': {'name': 'Holepunch', 'th': 1e5, 'op': op.gt},
            'obj_000013': {'name': 'Iron',      'th': 1e5, 'op': op.gt},
            'obj_000014': {'name': 'Lamp',      'th': 1e5, 'op': op.gt},
            'obj_000015': {'name': 'Phone',     'th': 1e5, 'op': op.gt},
            }


        if obj_id == '09':  # duck
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            stcfg.scene_target_ds_train = BOPDataset(ds_dir=ds_dir, split='test_nobop', single_obj_name=stcfg.obj_name)  # tgt train
            stcfg.scene_ds_val = BOPDataset(ds_dir=ds_dir, split='test_bop', single_obj_name=stcfg.obj_name)  # tgt test
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.0
            stcfg.EMA_alpha = 0.0
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 3e-5
            stcfg.self_train_epoch = 10
            stcfg.obj_threshold = {
            'obj_000001': {'name': 'Ape',       'th': 1e5, 'op': op.gt},
            'obj_000002': {'name': 'Benchvise', 'th': 1e5, 'op': op.gt},
            'obj_000004': {'name': 'Camera',    'th': 1e5, 'op': op.gt},
            'obj_000005': {'name': 'Can',       'th': 1e5, 'op': op.gt},
            'obj_000006': {'name': 'Cat',       'th': 1e5, 'op': op.gt},
            'obj_000008': {'name': 'Driller',   'th': 1e5, 'op': op.gt},
            'obj_000009': {'name': 'Duck',      'th': 0.5, 'op': op.gt},
            'obj_000010': {'name': 'Eggbox',    'th': 1e5, 'op': op.gt},
            'obj_000011': {'name': 'Glue',      'th': 1e5, 'op': op.gt},
            'obj_000012': {'name': 'Holepunch', 'th': 1e5, 'op': op.gt},
            'obj_000013': {'name': 'Iron',      'th': 1e5, 'op': op.gt},
            'obj_000014': {'name': 'Lamp',      'th': 1e5, 'op': op.gt},
            'obj_000015': {'name': 'Phone',     'th': 1e5, 'op': op.gt},
            }


        if obj_id == '10':  # eggbox
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            # stcfg.scene_target_ds_train = BOPDataset(ds_dir=ds_dir, split='test_nobop', single_obj_name=stcfg.obj_name)  # tgt train
            stcfg.scene_target_ds_train = LM_Dataset(split="train", category=[obj_id])
            stcfg.scene_ds_val = BOPDataset(ds_dir=ds_dir, split='test_bop', single_obj_name=stcfg.obj_name)  # tgt test
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = True
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.0
            stcfg.EMA_alpha = 0.0
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 3e-5
            stcfg.self_train_epoch = 10
            stcfg.obj_threshold = {
            'obj_000001': {'name': 'Ape',       'th': 1e5, 'op': op.gt},
            'obj_000002': {'name': 'Benchvise', 'th': 1e5, 'op': op.gt},
            'obj_000004': {'name': 'Camera',    'th': 1e5, 'op': op.gt},
            'obj_000005': {'name': 'Can',       'th': 1e5, 'op': op.gt},
            'obj_000006': {'name': 'Cat',       'th': 1e5, 'op': op.gt},
            'obj_000008': {'name': 'Driller',   'th': 1e5, 'op': op.gt},
            'obj_000009': {'name': 'Duck',      'th': 1e5, 'op': op.gt},
            'obj_000010': {'name': 'Eggbox',    'th': 0.2, 'op': op.gt},
            'obj_000011': {'name': 'Glue',      'th': 1e5, 'op': op.gt},
            'obj_000012': {'name': 'Holepunch', 'th': 1e5, 'op': op.gt},
            'obj_000013': {'name': 'Iron',      'th': 1e5, 'op': op.gt},
            'obj_000014': {'name': 'Lamp',      'th': 1e5, 'op': op.gt},
            'obj_000015': {'name': 'Phone',     'th': 1e5, 'op': op.gt},
            }


        if obj_id == '11':  # glue
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            stcfg.scene_target_ds_train = BOPDataset(ds_dir=ds_dir, split='test_nobop', single_obj_name=stcfg.obj_name)  # tgt train
            stcfg.scene_ds_val = BOPDataset(ds_dir=ds_dir, split='test_bop', single_obj_name=stcfg.obj_name)  # tgt test
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = True
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.0
            stcfg.EMA_alpha = 0.0
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 3e-5
            stcfg.self_train_epoch = 10
            stcfg.obj_threshold = {
            'obj_000001': {'name': 'Ape',       'th': 1e5, 'op': op.gt},
            'obj_000002': {'name': 'Benchvise', 'th': 1e5, 'op': op.gt},
            'obj_000004': {'name': 'Camera',    'th': 1e5, 'op': op.gt},
            'obj_000005': {'name': 'Can',       'th': 1e5, 'op': op.gt},
            'obj_000006': {'name': 'Cat',       'th': 1e5, 'op': op.gt},
            'obj_000008': {'name': 'Driller',   'th': 1e5, 'op': op.gt},
            'obj_000009': {'name': 'Duck',      'th': 1e5, 'op': op.gt},
            'obj_000010': {'name': 'Eggbox',    'th': 1e5, 'op': op.gt},
            'obj_000011': {'name': 'Glue',      'th': 0.43, 'op': op.gt}, # 0.35
            'obj_000012': {'name': 'Holepunch', 'th': 1e5, 'op': op.gt},
            'obj_000013': {'name': 'Iron',      'th': 1e5, 'op': op.gt},
            'obj_000014': {'name': 'Lamp',      'th': 1e5, 'op': op.gt},
            'obj_000015': {'name': 'Phone',     'th': 1e5, 'op': op.gt},
            }


        if obj_id == '12':  # holepunch
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            stcfg.scene_target_ds_train = BOPDataset(ds_dir=ds_dir, split='test_nobop', single_obj_name=stcfg.obj_name)  # tgt train
            stcfg.scene_ds_val = BOPDataset(ds_dir=ds_dir, split='test_bop', single_obj_name=stcfg.obj_name)  # tgt test
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.0
            stcfg.EMA_alpha = 0.0
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 3e-5
            stcfg.self_train_epoch = 10
            stcfg.obj_threshold = {
                'obj_000001': {'name': 'Ape',       'th': 1e5, 'op': op.gt},
                'obj_000002': {'name': 'Benchvise', 'th': 1e5, 'op': op.gt},
                'obj_000004': {'name': 'Camera',    'th': 1e5, 'op': op.gt},
                'obj_000005': {'name': 'Can',       'th': 1e5, 'op': op.gt},
                'obj_000006': {'name': 'Cat',       'th': 1e5, 'op': op.gt},
                'obj_000008': {'name': 'Driller',   'th': 1e5, 'op': op.gt},
                'obj_000009': {'name': 'Duck',      'th': 1e5, 'op': op.gt},
                'obj_000010': {'name': 'Eggbox',    'th': 1e5, 'op': op.gt},
                'obj_000011': {'name': 'Glue',      'th': 1e5, 'op': op.gt},
                'obj_000012': {'name': 'Holepunch', 'th': 0.4, 'op': op.gt}, # use LM setting no post aug
                'obj_000013': {'name': 'Iron',      'th': 1e5, 'op': op.gt},
                'obj_000014': {'name': 'Lamp',      'th': 1e5, 'op': op.gt},
                'obj_000015': {'name': 'Phone',     'th': 1e5, 'op': op.gt},
                }


    elif ds == 'LM':
        if obj_id == '01':
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            stcfg.ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=stcfg.ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            stcfg.scene_target_ds_train = LM_Dataset(split="train", category=[obj_id])  # tgt train
            stcfg.scene_ds_val = LM_Dataset(split="train", category=[obj_id])  # tgt val
            stcfg.occ_aug_ds = None
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.001
            stcfg.EMA_alpha = 0.7
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 1e-5
            stcfg.self_train_epoch = 5

            # LM coarse  1 obj 1 model
            stcfg.obj_threshold = {
            'obj_000001': {'name': 'Ape',       'th': 0.42, 'op': op.gt}
            }


        if obj_id == '02':
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            stcfg.ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=stcfg.ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            stcfg.scene_target_ds_train = LM_Dataset(split="train", category=[obj_id])  # tgt train
            stcfg.scene_ds_val = LM_Dataset(split="train", category=[obj_id])  # tgt val
            stcfg.occ_aug_ds = None
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.001
            stcfg.EMA_alpha = 0.7
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 3e-5
            stcfg.self_train_epoch = 5

            # LM coarse  1 obj 1 model
            stcfg.obj_threshold = {
            'obj_000002': {'name': 'Benchvise', 'th': 0.45, 'op': op.gt}
            }


        if obj_id == '04':
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            stcfg.ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=stcfg.ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            stcfg.scene_target_ds_train = LM_Dataset(split="train", category=[obj_id])  # tgt train
            stcfg.scene_ds_val = LM_Dataset(split="train", category=[obj_id])  # tgt val
            stcfg.occ_aug_ds = None
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.0
            stcfg.EMA_alpha = 0.0
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 1e-5
            stcfg.self_train_epoch = 5

            # LM coarse  1 obj 1 model
            stcfg.obj_threshold = {
            'obj_000004': {'name': 'Camera',    'th': 0.48, 'op': op.gt}
            }


        if obj_id == '05':
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            stcfg.ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=stcfg.ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            stcfg.scene_target_ds_train = LM_Dataset(split="train", category=[obj_id])  # tgt train
            stcfg.scene_ds_val = LM_Dataset(split="train", category=[obj_id])  # tgt val
            stcfg.occ_aug_ds = None
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.0
            stcfg.EMA_alpha = 0.0
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 1e-5
            stcfg.self_train_epoch = 5

            # LM coarse  1 obj 1 model
            stcfg.obj_threshold = {
            'obj_000005': {'name': 'Can',       'th': 0.45, 'op': op.gt}
            }


        if obj_id == '06':
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            stcfg.ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=stcfg.ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            stcfg.scene_target_ds_train = LM_Dataset(split="train", category=[obj_id])  # tgt train
            stcfg.scene_ds_val = LM_Dataset(split="train", category=[obj_id])  # tgt val
            stcfg.occ_aug_ds = None
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.0
            stcfg.EMA_alpha = 0.7
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 1e-5
            stcfg.self_train_epoch = 5

            # LM coarse  1 obj 1 model
            stcfg.obj_threshold = {
            'obj_000006': {'name': 'Cat',       'th': 0.49, 'op': op.gt}
            }

        if obj_id == '08':
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            stcfg.ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=stcfg.ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            stcfg.scene_target_ds_train = LM_Dataset(split="train", category=[obj_id])  # tgt train
            stcfg.scene_ds_val = LM_Dataset(split="train", category=[obj_id])  # tgt val
            stcfg.occ_aug_ds = None
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.001
            stcfg.EMA_alpha = 0.7
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 3e-5
            stcfg.self_train_epoch = 5

            # LM coarse  1 obj 1 model
            stcfg.obj_threshold = {
            'obj_000008': {'name': 'Driller',   'th': 0.45, 'op': op.gt}
            }

        if obj_id == '09':
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            stcfg.ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=stcfg.ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            stcfg.scene_target_ds_train = LM_Dataset(split="train", category=[obj_id])  # tgt train
            stcfg.scene_ds_val = LM_Dataset(split="train", category=[obj_id])  # tgt val
            stcfg.occ_aug_ds = None
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.007  # Nobj1model 0.007
            stcfg.EMA_alpha = 0.5
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 3e-5
            stcfg.self_train_epoch = 5

            # LM coarse  1 obj 1 model
            stcfg.obj_threshold = {
            'obj_000009': {'name': 'Duck',      'th': 0.30, 'op': op.gt}
            }

        if obj_id == '10':
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            stcfg.ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=stcfg.ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            stcfg.scene_target_ds_train = LM_Dataset(split="train", category=[obj_id])  # tgt train
            stcfg.scene_ds_val = LM_Dataset(split="train", category=[obj_id])  # tgt val
            stcfg.occ_aug_ds = None
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.0005
            stcfg.EMA_alpha = 0.7
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 3e-5
            stcfg.self_train_epoch = 10

            # LM coarse  1 obj 1 model
            stcfg.obj_threshold = {
            'obj_000010': {'name': 'Eggbox',    'th': 0.40, 'op': op.gt}
            }

        if obj_id == '11':
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            stcfg.ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=stcfg.ds_dir, split='train_pbr')  # src
            stcfg.scene_target_ds_train = LM_Dataset(split="train", category=[obj_id])  # tgt train
            stcfg.scene_ds_val = LM_Dataset(split="train", category=[obj_id])  # tgt val
            stcfg.occ_aug_ds = None
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.007
            stcfg.EMA_alpha = 0.45
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 1e-5
            stcfg.self_train_epoch = 5

            # LM coarse  1 obj 1 model
            stcfg.obj_threshold = {
            'obj_000011': {'name': 'Glue',      'th': 0.35, 'op': op.gt}, # src rgb aug=False
            }

        if obj_id == '12':
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            stcfg.ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=stcfg.ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            stcfg.scene_target_ds_train = LM_Dataset(split="train", category=[obj_id])  # tgt train
            stcfg.scene_ds_val = LM_Dataset(split="train", category=[obj_id])  # tgt val
            stcfg.occ_aug_ds = None
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.0
            stcfg.EMA_alpha = 0.7
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 3e-5
            stcfg.self_train_epoch = 10

            # LM coarse  1 obj 1 model
            stcfg.obj_threshold = {
            'obj_000012': {'name': 'Holepunch', 'th': 0.40, 'op': op.gt}
            }

        if obj_id == '13':
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            stcfg.ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=stcfg.ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            stcfg.scene_target_ds_train = LM_Dataset(split="train", category=[obj_id])  # tgt train
            stcfg.scene_ds_val = LM_Dataset(split="train", category=[obj_id])  # tgt val
            stcfg.occ_aug_ds = None
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.05
            stcfg.EMA_alpha = 0.0
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 3e-5
            stcfg.self_train_epoch = 5

            # LM coarse  1 obj 1 model
            stcfg.obj_threshold = {
            'obj_000013': {'name': 'Iron',      'th': 0.45, 'op': op.gt}, # src rgb aug=False
            }

        if obj_id == '14':
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            stcfg.ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=stcfg.ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            stcfg.scene_target_ds_train = LM_Dataset(split="train", category=[obj_id])  # tgt train
            stcfg.scene_ds_val = LM_Dataset(split="train", category=[obj_id])  # tgt val
            stcfg.occ_aug_ds = None
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.0
            stcfg.EMA_alpha = 0.7
            stcfg.model_coarse = 1
            stcfg.model_refiner = None
            stcfg.self_train_lr = 3e-5
            stcfg.self_train_epoch = 15

            # LM coarse  1 obj 1 model
            stcfg.obj_threshold = {
            'obj_000014': {'name': 'Lamp',      'th': 0.40, 'op': op.gt}
            }

        if obj_id == '15':
            stcfg = ST_config()
            stcfg.obj_name = f'obj_0000{obj_id}'
            stcfg.ds_dir = 'local_data/bop_datasets/lmo'
            stcfg.scene_ds_train = BOPDataset(ds_dir=stcfg.ds_dir, split='train_pbr', single_obj_name=stcfg.obj_name)  # src
            stcfg.scene_target_ds_train = LM_Dataset(split="train", category=[obj_id])  # tgt train
            stcfg.scene_ds_val = LM_Dataset(split="train", category=[obj_id])  # tgt val
            stcfg.occ_aug_ds = None
            stcfg.source_loss_weight = 0.0
            stcfg.target_loss_weight = 1.0
            stcfg.post_occ_aug_tgt = False
            stcfg.post_rgb_aug_tgt = False
            stcfg.pseudo_label_selection_threshold_decay_rate = 0.01
            stcfg.EMA_alpha = 0.0
            stcfg.model_coarse = 1
            stcfg.model_refiner = 1
            stcfg.self_train_lr = 3e-5
            stcfg.self_train_epoch = 5

            # LM coarse  1 obj 1 model
            stcfg.obj_threshold = {
            'obj_000015': {'name': 'Phone',     'th': 0.45, 'op': op.gt}, # src rgb aug=False
            }

    return stcfg
