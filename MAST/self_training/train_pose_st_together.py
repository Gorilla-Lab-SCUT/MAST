import functools
import time
from collections import defaultdict

import numpy as np
import simplejson as json
import torch
import torch.distributed as dist
import torch.nn.utils
import yaml
from torch.backends import cudnn
from torch.utils.data import ConcatDataset, DataLoader
from torchnet.meter import AverageValueMeter
from tqdm import tqdm

from MAST.config import EXP_DIR
from MAST.datasets.datasets_cfg import (make_object_dataset,
                                            make_scene_dataset)
from MAST.datasets.pose_dataset_w_origin_img_msk import PoseDataset
from MAST.datasets.samplers import ListSampler, PartialSampler
from MAST.lib3d.rigid_mesh_database import MeshDataBase
from MAST.rendering.bullet_batch_renderer import BulletBatchRenderer
from MAST.training.pose_models_cfg import (check_update_config,
                                               create_model_pose)
from MAST.utils.distributed import (get_rank, get_world_size,
                                        init_distributed_mode, reduce_dict,
                                        sync_model)
from MAST.utils.logging import get_logger
from MAST.utils.multiepoch_dataloader import MultiEpochDataLoader
from MAST.utils.random_seed import (dataloader_generator, seed_worker,
                                        set_seed)

from .make_selftrain_dataset import make_selftrain_dataset
from .pose_forward_loss_select_tgt import h_pose_select_tgt
from .pose_forward_loss_src import h_pose_src
from .pose_forward_loss_tgt import h_pose_tgt
from .pose_forward_loss_ub_val import h_pose_ub_val
from .st_utils import update_ema_variables

cudnn.benchmark = True
logger = get_logger(__name__)


def log(config, model, optimizer,
        log_dict, epoch):
    save_dir = config.save_dir
    save_dir.mkdir(exist_ok=True)
    log_dict.update(epoch=epoch)
    if not (save_dir / 'config.yaml').exists():
        (save_dir / 'config.yaml').write_text(yaml.dump(config))

    def save_checkpoint(model):
        ckpt_name = 'checkpoint'+'-epoch='+str(epoch)
        ckpt_name += '.pth.tar'
        path = save_dir / ckpt_name
        torch.save({'state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch}, path)

    save_checkpoint(model)
    with open(save_dir / 'log.txt', 'a') as f:
        f.write(json.dumps(log_dict, ignore_nan=True) + '\n')

    logger.info(config.run_id)
    logger.info(log_dict)


def train_pose(args):
    torch.set_num_threads(1)
    set_seed(get_rank())

    if args.resume_run_id:
        resume_dir = EXP_DIR / args.resume_run_id
        resume_args = yaml.load((resume_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        keep_fields = set(['resume_run_id', ])
        vars(args).update({k: v for k, v in vars(resume_args).items() if k not in keep_fields})

    args.train_refiner = args.TCO_input_generator == 'gt+noise'
    args.train_coarse = not args.train_refiner
    args.save_dir = EXP_DIR / args.resume_run_id  # args.run_id
    args = check_update_config(args)

    logger.info(f"{'-'*80}")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    logger.info(f"{'-'*80}")

    # Initialize distributed
    device = torch.cuda.current_device()
    init_distributed_mode()
    world_size = get_world_size()
    args.n_gpus = world_size
    args.global_batch_size = world_size * args.batch_size
    logger.info(f'Connection established with {world_size} gpus.')

    # Make train/val datasets
    def make_source_datasets(dataset_names):
        datasets = []
        for (ds_name, n_repeat) in dataset_names:
            assert 'test' not in ds_name
            ds = make_scene_dataset(ds_name)
            logger.info(f'Loaded {ds_name} with {len(ds)} images.')
            for _ in range(n_repeat):
                datasets.append(ds)
        return ConcatDataset(datasets)


    from .per_obj_config import st_cfg
    stcfg = st_cfg(args.ds, args.resume_run_id.split('/')[-1])
    occ_aug_ds = stcfg.occ_aug_ds
    scene_ds_train = stcfg.scene_ds_train
    scene_target_ds_train = stcfg.scene_target_ds_train
    scene_ds_val = stcfg.scene_ds_val


    ds_kwargs = dict(
        resize=args.input_resize,
        rgb_augmentation=args.rgb_augmentation,
        background_augmentation=args.background_augmentation,
        min_area=args.min_area,
        gray_augmentation=args.gray_augmentation,
    )
    ds_train = PoseDataset(scene_ds_train, **ds_kwargs)
    target_ds_train = PoseDataset(scene_target_ds_train, resize=args.input_resize, min_area=args.min_area)  # w/o pre-aug
    ds_val = PoseDataset(scene_ds_val, resize=args.input_resize, min_area=args.min_area)


    train_sampler = PartialSampler(ds_train, epoch_size=args.epoch_size)
    ds_iter_train = DataLoader(ds_train, sampler=train_sampler, batch_size=args.batch_size,
                               num_workers=args.n_dataloader_workers, collate_fn=ds_train.collate_fn,
                               drop_last=False, pin_memory=True)
    ds_iter_train = MultiEpochDataLoader(ds_iter_train)

    # val_sampler = PartialSampler(ds_val, epoch_size=int(0.1 * args.epoch_size))
    ds_iter_val = DataLoader(ds_val, batch_size=args.batch_size,
                             num_workers=args.n_dataloader_workers, collate_fn=ds_val.collate_fn,
                             drop_last=False, pin_memory=True)
    # ds_iter_val = MultiEpochDataLoader(ds_iter_val)

    target_ds_iter_train =  DataLoader(target_ds_train, batch_size=args.batch_size,
                            num_workers=args.n_dataloader_workers,
                            collate_fn=target_ds_train.collate_fn,
                            drop_last=False, pin_memory=True, shuffle=False)  # target dataloader

    # Make model
    renderer = BulletBatchRenderer(object_set=args.urdf_ds_name, n_workers=args.n_rendering_workers)
    object_ds = make_object_dataset(args.object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds).batched(n_sym=args.n_symmetries_batch).cuda().float()

    model = create_model_pose(cfg=args, renderer=renderer, mesh_db=mesh_db).cuda()

    if args.resume_run_id:
        resume_dir = EXP_DIR / args.resume_run_id
        path = resume_dir / ('checkpoint' + '-epoch=' + args.epoch + '.pth.tar')
        logger.info(f'Loading checkpoing from {path}')
        save = torch.load(path)
        state_dict = save['state_dict']
        model.load_state_dict(state_dict)
        start_epoch = save['epoch'] + 1
    else:
        start_epoch = 0
    end_epoch = args.n_epochs

    if args.run_id_pretrain is not None:
        pretrain_path = EXP_DIR / args.run_id_pretrain / 'checkpoint.pth.tar'
        logger.info(f'Using pretrained model from {pretrain_path}.')
        model.load_state_dict(torch.load(pretrain_path)['state_dict'])

    # Synchronize models across processes.
    model = sync_model(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device,
                                                      find_unused_parameters=False, broadcast_buffers=False)


    ### create coarse ####
    model_coarse = stcfg.model_coarse
    if model_coarse is not None:
        model_coarse = create_model_pose(cfg=args, renderer=renderer, mesh_db=mesh_db).cuda()
        resume_dir = EXP_DIR / args.resume_run_id
        pretrain_path = resume_dir / (f'checkpoint-epoch={args.epoch}.pth.tar')
        logger.info(f'Using pretrained model from {pretrain_path}.')
        model_coarse.load_state_dict(torch.load(pretrain_path)['state_dict'])
        model_coarse = sync_model(model_coarse)
        model_coarse = torch.nn.parallel.DistributedDataParallel(model_coarse, device_ids=[device], output_device=device,
                                                        find_unused_parameters=False, broadcast_buffers=False)
        model_coarse.eval()
        for param in model_coarse.parameters():
            param.detach_()

    #### create refiner ####
    model_refiner = stcfg.model_refiner
    if model_refiner is not None:
        model_refiner = create_model_pose(cfg=args, renderer=renderer, mesh_db=mesh_db).cuda()
        pretrain_path = EXP_DIR / 'bop-lmo-pbr-refiner-transnoise-zxyavg-decoupled-bindelta' / 'checkpoint-epoch=660.pth.tar'
        logger.info(f'Using pretrained model from {pretrain_path}.')
        model_refiner.load_state_dict(torch.load(pretrain_path)['state_dict'])
        model_refiner = sync_model(model_refiner)
        model_refiner = torch.nn.parallel.DistributedDataParallel(model_refiner, device_ids=[device], output_device=device,
                                                        find_unused_parameters=False, broadcast_buffers=False)
        model_refiner.eval()
        for param in model_refiner.parameters():
            param.detach_()
    #### create refiner ####


    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_run_id:
        if 'optimizer_state_dict' in save.keys():
            optimizer.load_state_dict(save['optimizer_state_dict'])
            logger.info("Loaded optimizer state_dict")
    optimizer.param_groups[0]['lr'] = stcfg.self_train_lr

    START_FLAG = True
    args.th_decay = 0.0
    for epoch in range(start_epoch, start_epoch + stcfg.self_train_epoch):
        meters_train = defaultdict(lambda: AverageValueMeter())
        meters_val = defaultdict(lambda: AverageValueMeter())
        meters_time = defaultdict(lambda: AverageValueMeter())

        h_src = functools.partial(h_pose_src, model=model, cfg=args, n_iterations=args.n_iterations,
                              mesh_db=mesh_db, input_generator=args.TCO_input_generator)
        h_tgt = functools.partial(h_pose_tgt, model=model, cfg=args, n_iterations=args.n_iterations,
                              mesh_db=mesh_db, input_generator=args.TCO_input_generator)
        h_ub_val = functools.partial(h_pose_ub_val, model=model, cfg=args, n_iterations=args.n_iterations,
                              mesh_db=mesh_db, input_generator=args.TCO_input_generator)
        h_select_tgt = functools.partial(h_pose_select_tgt, model_coarse=model_coarse, model_refiner=model_refiner, 
                              cfg=args, n_iterations=args.n_iterations,
                              mesh_db=mesh_db, input_generator=args.TCO_input_generator, self_train_cfg=stcfg)

        @torch.no_grad()
        def select_tgt():
            model.eval()
            FIRST_DATA = True
            iterator = tqdm(target_ds_iter_train, ncols=80, postfix='collecting pseudo label')
            for n, data in enumerate(iterator):
                selected_data = h_select_tgt(data=data, meters=meters_train)
                if selected_data is not None:
                    if FIRST_DATA:
                        images_selected = selected_data['images']
                        K_selected = selected_data['K']
                        TCO_gt_selected = selected_data['TCO_gt']
                        TCO_pred_selected = selected_data['TCO_pred']
                        labels_selected = selected_data['labels']
                        bboxes_selected = selected_data['bboxes']
                        zb_pseudo_selected = selected_data['zb_pseudo']
                        masks_selected = selected_data['masks']
                        FIRST_DATA = False
                    else:
                        images_selected = torch.cat((images_selected, selected_data['images']), dim=0)
                        K_selected = torch.cat((K_selected, selected_data['K']), dim=0)
                        TCO_gt_selected = torch.cat((TCO_gt_selected, selected_data['TCO_gt']), dim=0)
                        TCO_pred_selected = torch.cat((TCO_pred_selected, selected_data['TCO_pred']), dim=0)
                        labels_selected = np.concatenate((labels_selected, selected_data['labels']), axis=0)
                        bboxes_selected = torch.cat((bboxes_selected, selected_data['bboxes']), dim=0)
                        zb_pseudo_selected = torch.cat((zb_pseudo_selected, selected_data['zb_pseudo']), dim=0)
                        masks_selected = torch.cat((masks_selected, selected_data['masks']), dim=0)

            full_selected_data = dict(
                images=images_selected,
                K=K_selected,
                TCO_gt=TCO_gt_selected,
                TCO_pred=TCO_pred_selected,
                labels=labels_selected,
                bboxes=bboxes_selected,
                zb_pseudo=zb_pseudo_selected,
                masks=masks_selected
            )
            return full_selected_data


        def train_epoch(full_selected_data):
            model.train()

            # def set_bn_eval(module):
            #     if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            #         module.eval() # ;module.track_running_stats = False
            # model.apply(set_bn_eval)

            iterator = {"tgt": tqdm(full_selected_data, ncols=80),
                        "src": tqdm(ds_iter_train, ncols=80)}
            t = time.time()
            zfeats = []
            zfeatt = []
            Rfeats = []
            Rfeatt = []
            zlbls = []
            zlblt = []
            rlbls = []
            rlblt = []
            for n, (target, source) in enumerate(zip(iterator['tgt'], iterator['src'])):
                optimizer.zero_grad()

                loss_t, rft, zft, rlt, zlt = h_tgt(data=target, meters=meters_train)
                loss_t = loss_t * stcfg.target_loss_weight
                loss_t.backward()
                iterator['tgt'].set_postfix(dict(loss_t=loss_t.item(), lr=optimizer.param_groups[0]['lr']))
                meters_train['loss_t_total'].add(loss_t.item())

                loss_s, rfs, zfs, rls, zls = h_src(data=source, meters=meters_train)
                loss_s = loss_s * stcfg.source_loss_weight
                loss_s.backward()
                iterator['src'].set_postfix(dict(loss_s=loss_s.item(), lr=optimizer.param_groups[0]['lr']))
                meters_train['loss_s_total'].add(loss_s.item())

                # zfeats.append(zfs.detach().cpu())
                # zfeatt.append(zft.detach().cpu())
                # Rfeats.append(rfs.detach().cpu())
                # Rfeatt.append(rft.detach().cpu())
                # zlbls.append(zls)
                # zlblt.append(zlt)
                # rlbls.append(rls)
                # rlblt.append(rlt)
                # if n ==80:    
                #     zfeats = torch.stack(zfeats).reshape(-1,zfs.size(-1))
                #     zfeatt = torch.stack(zfeatt).reshape(-1,zft.size(-1))
                #     Rfeats = torch.stack(Rfeats).reshape(-1,rfs.size(-1))
                #     Rfeatt = torch.stack(Rfeatt).reshape(-1,rft.size(-1))
                #     zls = torch.stack(zlbls).reshape(-1)
                #     zlt = torch.stack(zlblt).reshape(-1)
                #     rls = torch.stack(rlbls).reshape(-1)
                #     rlt = torch.stack(rlblt).reshape(-1)
                #     dis = dict(
                #         zfs=zfeats,
                #         zft=zfeatt,
                #         rfs=Rfeats,
                #         rft=Rfeatt,
                #         zls=zls,
                #         zlt=zlt,
                #         rls=rls,
                #         rlt=rlt
                #     )
                #     torch.save(dis,'dis.pth')
                #     exit()
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.clip_grad_norm, norm_type=2)
                meters_train['grad_norm'].add(torch.as_tensor(total_grad_norm).item())
                optimizer.step()


        @torch.no_grad()
        def validation():
            model.eval()
            R_acc = 0
            x_acc = 0
            y_acc = 0
            z_acc = 0
            sample_num = 0
            for sample in tqdm(ds_iter_val, ncols=80, postfix='validating'):
                loss, accs = h_ub_val(data=sample, meters=meters_val)
                R_acc += accs['R_acc']
                x_acc += accs['x_acc']
                y_acc += accs['y_acc']
                z_acc += accs['z_acc']
                sample_num += accs['batch_size']
                meters_val['loss_total'].add(loss.item())
            print('\ntest_R_acc: {}\ntest_x_acc:{}\ntest_y_acc:{}\ntest_z_acc:{}'.format(R_acc/sample_num,
                                                                                        x_acc/sample_num,
                                                                                        y_acc/sample_num,
                                                                                        z_acc/sample_num))
            with open(args.save_dir / 'log.txt', 'a') as f:
                f.write("test: Racc: {}, xacc: {}, yacc: {}, zacc: {}, loss_val: {}".format(
                    R_acc/sample_num, 
                    x_acc/sample_num,
                    y_acc/sample_num,
                    z_acc/sample_num,
                    meters_val['loss_total'].mean) + '\n')

        if START_FLAG:
            validation()
            START_FLAG = False
        full_selected_data = select_tgt()
        if occ_aug_ds is not None:
            selftrain_cfg = dict(occ_aug=stcfg.post_occ_aug_tgt,
                                 rgb_aug=stcfg.post_rgb_aug_tgt,
                                 occ_rgbs=occ_aug_ds.rgbs, 
                                 occ_masks=occ_aug_ds.masks, 
                                 occ_id=occ_aug_ds.obj_ids)
        else:
            selftrain_cfg = dict(occ_aug=stcfg.post_occ_aug_tgt,
                                 rgb_aug=stcfg.post_rgb_aug_tgt,)
        selected_dataset = make_selftrain_dataset(full_selected_data, **selftrain_cfg)
        selected_dataloader = DataLoader(selected_dataset, batch_size=args.batch_size,collate_fn=selected_dataset.collate_fn,
                                        num_workers=args.n_dataloader_workers, drop_last=False, pin_memory=True, shuffle=True)
        train_epoch(selected_dataloader)
        args.th_decay += stcfg.pseudo_label_selection_threshold_decay_rate
        update_ema_variables(model, model_coarse, a=stcfg.EMA_alpha, global_step=1e5)
        if epoch % args.val_epoch_interval == 0:
            validation()

        log_dict = dict()
        log_dict.update({
            'grad_norm': meters_train['grad_norm'].mean,
            'grad_norm_std': meters_train['grad_norm'].std,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'time': time.time(),
            'n_iterations': (epoch + 1) * len(ds_iter_train),
            'n_datas': (epoch + 1) * args.global_batch_size * len(ds_iter_train),
        })

        for string, meters in zip(('train', 'val'), (meters_train, meters_val)):
            for k in dict(meters).keys():
                log_dict[f'{string}_{k}'] = meters[k].mean

        log_dict = reduce_dict(log_dict)
        if get_rank() == 0:
            log(config=args, model=model, optimizer=optimizer, epoch=epoch,
                log_dict=log_dict)
        dist.barrier()
