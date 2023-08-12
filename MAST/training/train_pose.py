import functools
import time
from collections import defaultdict

import simplejson as json
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim.lr_scheduler
import yaml
from torch.backends import cudnn
from torch.utils.data import ConcatDataset, DataLoader
from torchnet.meter import AverageValueMeter
from tqdm import tqdm

from MAST.config import EXP_DIR
from MAST.datasets.datasets_cfg import (make_object_dataset,
                                            make_scene_dataset)
from MAST.datasets.pose_dataset import PoseDataset
from MAST.datasets.samplers import ListSampler, PartialSampler
from MAST.lib3d.rigid_mesh_database import MeshDataBase
from MAST.rendering.bullet_batch_renderer import BulletBatchRenderer
from MAST.utils.distributed import (get_rank, get_world_size,
                                        init_distributed_mode, reduce_dict,
                                        sync_model)
from MAST.utils.logging import get_logger
from MAST.utils.multiepoch_dataloader import MultiEpochDataLoader
from MAST.utils.random_seed import (dataloader_generator, seed_worker,
                                        set_seed)
from MAST.datasets.bop import BOPDataset
from .pose_forward_loss import h_pose
from .pose_models_cfg import check_update_config, create_model_pose

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
        ckpt_name = f'checkpoint-epoch={epoch}.pth.tar'
        path = save_dir / ckpt_name
        torch.save({'state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch}, path)
    if epoch % 1 == 0:
        save_checkpoint(model)
    with open(save_dir / 'log.txt', 'a') as f:
        f.write(json.dumps(log_dict, ignore_nan=True) + '\n')

    logger.info(config.run_id)
    logger.info(log_dict)


def train_pose(args):
    torch.set_num_threads(1)
    # set_seed(get_rank())  do not use seed due to dataloader sampling method

    if args.resume_run_id:
        resume_dir = EXP_DIR / args.resume_run_id
        resume_args = yaml.load((resume_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        keep_fields = set(['resume_run_id', 'epoch_size', ])
        vars(args).update({k: v for k, v in vars(resume_args).items() if k not in keep_fields})
        args.save_dir = EXP_DIR / args.resume_run_id
    else:
        args.save_dir = EXP_DIR / args.run_id

    args.train_refiner = args.TCO_input_generator == 'gt+noise'
    args.train_coarse = not args.train_refiner
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
    def make_datasets(dataset_names):
        datasets = []
        for (ds_name, n_repeat) in dataset_names:
            assert 'test' not in ds_name
            ds = make_scene_dataset(ds_name)
            logger.info(f'Loaded {ds_name} with {len(ds)} images.')
            for _ in range(n_repeat):
                datasets.append(ds)
        return ConcatDataset(datasets)

    if args.resume_run_id is not None and args.resume_run_id.split('/')[-1].isdigit():  # if resume id end with digits
        ds_dir = 'local_data/bop_datasets/lmo'
        obj_name = f"obj_0000{args.resume_run_id.split('/')[-1]}"
        scene_ds_train = BOPDataset(ds_dir=ds_dir, split='train_pbr', single_obj_name=obj_name)
        scene_ds_val = BOPDataset(ds_dir=ds_dir, split='train_pbr', single_obj_name=obj_name)
    else:
        scene_ds_train = make_datasets(args.train_ds_names)
        scene_ds_val = make_datasets(args.val_ds_names)

    ds_kwargs = dict(
        resize=args.input_resize,
        rgb_augmentation=args.rgb_augmentation,
        background_augmentation=args.background_augmentation,
        min_area=args.min_area,
        gray_augmentation=args.gray_augmentation,
    )
    ds_train = PoseDataset(scene_ds_train, **ds_kwargs)
    ds_val = PoseDataset(scene_ds_val, **ds_kwargs)

    train_sampler = PartialSampler(ds_train, epoch_size=args.epoch_size)
    ds_iter_train = DataLoader(ds_train, sampler=train_sampler, batch_size=args.batch_size,
                               num_workers=args.n_dataloader_workers, collate_fn=ds_train.collate_fn,
                               drop_last=False, pin_memory=True)
    ds_iter_train = MultiEpochDataLoader(ds_iter_train)

    val_sampler = PartialSampler(ds_val, epoch_size=int(0.1 * args.epoch_size))
    ds_iter_val = DataLoader(ds_val, sampler=val_sampler, batch_size=args.batch_size,
                             num_workers=args.n_dataloader_workers, collate_fn=ds_val.collate_fn,
                             drop_last=False, pin_memory=True)
    ds_iter_val = MultiEpochDataLoader(ds_iter_val)

    # Make model
    renderer = BulletBatchRenderer(object_set=args.urdf_ds_name, n_workers=args.n_rendering_workers)
    object_ds = make_object_dataset(args.object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds).batched(n_sym=args.n_symmetries_batch).cuda().float()

    model = create_model_pose(cfg=args, renderer=renderer, mesh_db=mesh_db).cuda()


    if args.resume_run_id:
        resume_dir = EXP_DIR / args.resume_run_id
        path = resume_dir / f'checkpoint-epoch={args.epoch}.pth.tar'
        logger.info(f'Loading checkpoint from {path}')
        save = torch.load(path)
        pretrained_state_dict = save['state_dict']
        try: 
            model.load_state_dict(pretrained_state_dict)
        except:
            ## this block is used for loading pretrain weights of partial model
            input('Pretrained weights does not match the model, load weights? [Enter to continue]')
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
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
                                                    find_unused_parameters=False, broadcast_buffers=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_run_id:
        if 'optimizer_state_dict' in save.keys():
            try:
                optimizer.load_state_dict(save['optimizer_state_dict'])
                # optimizer.param_groups[0]['lr'] = args.lr  # set lr
                logger.info("Loaded optimizer state_dict")
            except:
                logger.info("Didn't load optimizer state_dict")

    # Warmup
    if start_epoch < args.n_epochs_warmup:
        if args.n_epochs_warmup == 0:
            lambd = lambda epoch: 1
        else:
            n_batches_warmup = args.n_epochs_warmup * (args.epoch_size // args.batch_size)
            lambd = lambda batch: (batch + 1) / n_batches_warmup
        lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambd)
        lr_scheduler_warmup.last_epoch = start_epoch * args.epoch_size // args.batch_size

    # LR schedulers
    # Divide LR by 10 every args.lr_epoch_decay
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_epoch_decay, gamma=0.1,
    )
    lr_scheduler.last_epoch = start_epoch - 1

    if args.gs_loss:
        feats_collection = {cls:[] for cls in range(40)}
        feat_centers = torch.load('x_feat_mat.pth').cuda()

    for epoch in range(start_epoch, end_epoch):
        meters_train = defaultdict(lambda: AverageValueMeter())
        meters_val = defaultdict(lambda: AverageValueMeter())
        meters_time = defaultdict(lambda: AverageValueMeter())

        h = functools.partial(h_pose, model=model, cfg=args, n_iterations=args.n_iterations,
                              mesh_db=mesh_db, input_generator=args.TCO_input_generator)

        def train_epoch():
            model.train()
            iterator = tqdm(ds_iter_train, ncols=80)
            t = time.time()
            feats, lbls = [], []
            if args.gs_loss:
                confusion = build_confusion_mat_by_ecs(feat_centers)
            for n, sample in enumerate(iterator):
                if n > 0:
                    meters_time['data'].add(time.time() - t)

                optimizer.zero_grad()

                t = time.time()
                loss, feat, lbl = h(data=sample, meters=meters_train)

                if args.gs_loss:
                    # ocloss = online_center_L2_loss(feat, lbl, feat_centers)
                    # loss += ocloss * 1e-3
                    sim_vec = similarity_vec(feat, feat_centers)
                    gsloss = graph_similarity_L2_loss(sim_vec, lbl, confusion)
                    loss += gsloss * 2.0
                    for i, lb in enumerate(lbl):# collect feat for cal centers
                        feats_collection[lb.item()].append(feat[i].detach().cpu())

                # feats.append(feat.detach().cpu())
                # lbls.append(lbl)
                # if n == 100:
                #     zfeat = torch.stack(feats).reshape(-1,feat.size(-1))
                #     zl = torch.stack(lbls).reshape(-1)
                #     dis = dict(
                #         zf=zfeat,
                #         zl=zl,
                #     )
                #     torch.save(dis,'dis.pth')
                #     exit()
                meters_time['forward'].add(time.time() - t)
                iterator.set_postfix(dict(loss=loss.item(), lr=optimizer.param_groups[0]['lr']))
                meters_train['loss_total'].add(loss.item())

                t = time.time()
                loss.backward()
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.clip_grad_norm, norm_type=2)
                meters_train['grad_norm'].add(torch.as_tensor(total_grad_norm).item())

                optimizer.step()
                meters_time['backward'].add(time.time() - t)
                meters_time['memory'].add(torch.cuda.max_memory_allocated() / 1024. ** 2)

                if epoch < args.n_epochs_warmup:
                    lr_scheduler_warmup.step()
                t = time.time()
            
            if args.gs_loss:
                for k, v in feats_collection.items():
                    print(f'{k}: {len(v)}')
                    try: feat_centers[k] = torch.stack(v).mean(0)
                    except: print(f'Bin{k} has no sample!!')
                print('Update feature centers')


            if epoch >= args.n_epochs_warmup:
                lr_scheduler.step()

        @torch.no_grad()
        def validation():
            model.eval()
            for sample in tqdm(ds_iter_val, ncols=80):
                loss, feat, lbl = h(data=sample, meters=meters_val)
                meters_val['loss_total'].add(loss.item())

        train_epoch()
        if epoch % args.val_epoch_interval == 0:
            validation()

        log_dict = dict()
        log_dict.update({
            'grad_norm': meters_train['grad_norm'].mean,
            'grad_norm_std': meters_train['grad_norm'].std,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'time_forward': meters_time['forward'].mean,
            'time_backward': meters_time['backward'].mean,
            'time_data': meters_time['data'].mean,
            'gpu_memory': meters_time['memory'].mean,
            'time': time.time(),
            'n_iterations': (epoch + 1) * len(ds_iter_train),
            'n_datas': (epoch + 1) * args.global_batch_size * len(ds_iter_train),
        })

        for string, meters in zip(('train', 'val'), (meters_train, meters_val)):
            for k in dict(meters).keys():
                log_dict[f'{string}_{k}'] = meters[k].mean

        log_dict = reduce_dict(log_dict)
        if get_rank() == 0:
            log(config=args, model=model, optimizer=optimizer,epoch=epoch,
                log_dict=log_dict)
        dist.barrier()
