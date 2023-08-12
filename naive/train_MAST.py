import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import lib.model as model
import lib.transform as tran
from lib.MAST import (ExplicitInterClassGraphLoss, bins_deltas_to_ts_batch,
                      grid_xyz, t_to_bin_delta_batch, xentropy)
from lib.read_data import ImageList_r as ImageList

parser = argparse.ArgumentParser(description='PyTorch DAregre experiment')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--src', type=str, default='c', metavar='S',
                    help='source dataset')
parser.add_argument('--tgt', type=str, default='n', metavar='S',
                    help='target dataset')
parser.add_argument('--lr', type=float, default=0.03,
                        help='init learning rate for fine-tune')
parser.add_argument('--gamma', type=float, default=0.0001,
                        help='learning rate decay')
parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ctr = grid_xyz()

data_transforms = {
    'train': tran.rr_train(resize_size=224),
    'val': tran.rr_train(resize_size=224),
    'test': tran.rr_eval(resize_size=224),
}
# set dataset
batch_size = {"train": 36, "val": 36, "test": 36}
c="data/image_list/color_train.txt"
n="data/image_list/noisy_train.txt"
s="data/image_list/scream_train.txt"

c_t="data/image_list/color_test.txt"
n_t="data/image_list/noisy_test.txt"
s_t="data/image_list/scream_test.txt"

if args.src =='c':
    source_path = c
elif args.src =='n':
    source_path = n
elif args.src =='s':
    source_path = s

if args.tgt =='c':
    target_path = c
elif args.tgt =='n':
    target_path = n
elif args.tgt =='s':
    target_path = s

if args.tgt =='c':
    target_path_t = c_t
elif args.tgt =='n':
    target_path_t = n_t
elif args.tgt =='s':
    target_path_t = s_t

dsets = {"train": ImageList(open(source_path).readlines(), transform=data_transforms["train"]),
         "val": ImageList(open(target_path).readlines(),transform=data_transforms["val"]),
         "test": ImageList(open(target_path_t).readlines(),transform=data_transforms["test"])}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                               shuffle=True, num_workers=16)
                for x in ['train', 'val']}
dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"],
                                                   shuffle=False, num_workers=16)

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val','test']}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def Regression_test(loader, model, optimizer=None, save=False, iter_num=None):
    model.eval()
    MSE = [0, 0, 0, 0]
    MAE = [0, 0, 0, 0]
    number = 0
    with torch.no_grad():
        for (imgs, labels) in loader:
            imgs = imgs.to(device)
            labels_source = labels.to(device)
            labels1 = labels_source[:, 2]
            labels3 = labels_source[:, 4]
            labels4 = labels_source[:, 5]
            labels1 = labels1.unsqueeze(1)
            labels3 = labels3.unsqueeze(1)
            labels4 = labels4.unsqueeze(1)
            # labels_source = torch.cat((labels1, labels3, labels4), dim=1)
            # labels = labels_source.float()
            labels = labels1.float()
            bins, deltas, f = model(imgs)
            pred = bins_deltas_to_ts_batch(bins, deltas, ctr)
            # MSE[0] += torch.nn.MSELoss(reduction='sum')(pred[:, 0], labels[:, 0])
            # MAE[0] += torch.nn.L1Loss(reduction='sum')(pred[:, 0], labels[:, 0])
            # MSE[1] += torch.nn.MSELoss(reduction='sum')(pred[:, 1], labels[:, 1])
            # MAE[1] += torch.nn.L1Loss(reduction='sum')(pred[:, 1], labels[:, 1])
            # MSE[2] += torch.nn.MSELoss(reduction='sum')(pred[:, 2], labels[:, 2])
            # MAE[2] += torch.nn.L1Loss(reduction='sum')(pred[:, 2], labels[:, 2])
            MSE[3] += torch.nn.MSELoss(reduction='sum')(pred, labels)
            MAE[3] += torch.nn.L1Loss(reduction='sum')(pred, labels)
            number += imgs.size(0)
    for j in range(4):
        MSE[j] = MSE[j] / number
        MAE[j] = MAE[j] / number
    # print(f"\tMSE: {MSE[0]},{MSE[1]},{MSE[2]}")
    # print(f"\tMAE: {MAE[0]},{MAE[1]},{MAE[2]}")
    print(f"\tMSEall : {MSE[3]}")
    print(f"\tMAEall : {MAE[3]}")
    if save:
        torch.save({'model':model.state_dict(), 'optim': optimizer.state_dict()},
         f'checkpoints/{args.src}->{args.tgt}-it_{iter_num}-MAE_{MAE[3]:.3f}.pth')
        print(f'checkpoints/{args.src}->{args.tgt}-it_{iter_num}-MAE_{MAE[3]:.3f}.pth')


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer


class Model_Regression(nn.Module):
    def __init__(self):
        super(Model_Regression,self).__init__()
        self.model_fc = model.Resnet18Fc()
        self.cls_layer = nn.Linear(512, 6)
        self.reg_layer = nn.Linear(512, 6)


    def forward(self,x):
        feature = self.model_fc(x)
        cls= self.cls_layer(feature)
        reg = self.reg_layer(feature)
        return cls, reg, feature


def pretrain_on_src(Model_R):
    criterion = {"cls": xentropy, "reg": nn.MSELoss(), "icg": ExplicitInterClassGraphLoss()}
    optimizer_dict = [{"params": filter(lambda p: p.requires_grad, Model_R.model_fc.parameters()), "lr": 0.1},
                    {"params": filter(lambda p: p.requires_grad, Model_R.cls_layer.parameters()), "lr": 0.01},
                    {"params": filter(lambda p: p.requires_grad, Model_R.reg_layer.parameters()), "lr": 0.01}]
    optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)

    train_cross_loss = train_mse_loss = train_icg_loss = train_total_loss = 0.0

    len_source = len(dset_loaders["train"]) - 1
    iter_source = iter(dset_loaders["train"])

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])

    test_interval = 500
    num_iter = 20002
    print(args)
    for iter_num in range(1, num_iter + 1):
        Model_R.train()
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=args.gamma, power=0.75,
                                    weight_decay=0.0005)
        optimizer.zero_grad()

        # initialize a new one after enumerated the whole datasets
        if iter_num % len_source == 0:
            iter_source = iter(dset_loaders["train"])
        data_source = iter_source.next()
        inputs_source, labels_source = data_source
        
        # label of dSprites: ('none', 'shape', 'scale', 'orientation', 'position x', 'position y')
        labels1 = labels_source[:, 2]
        # labels3 = labels_source[:, 4]
        # labels4 = labels_source[:, 5]

        labels1 = labels1.unsqueeze(1)
        # labels3 = labels3.unsqueeze(1)
        # labels4 = labels4.unsqueeze(1)

        # labels_source = torch.cat((labels1,labels3,labels4),dim=1)
        labels_source = labels1
        labels_source = labels_source.float().to(device)

        inputs = inputs_source
        inputs = inputs.to(device)

        bins, deltas = t_to_bin_delta_batch(labels_source, ctr)
        inputs_s = inputs.narrow(0, 0, batch_size["train"])
        # inputs_t = inputs.narrow(0, batch_size["train"], batch_size["train"])
        cls, reg, f = Model_R(inputs_s)
        
        classifier_loss = criterion["cls"](cls, bins)
        regressor_loss = criterion['reg'](reg, deltas)
        icg_loss = criterion['icg'](torch.max(bins, dim=-1)[1], f) * 1.0
        total_loss = classifier_loss + regressor_loss + icg_loss
        total_loss.backward()
        optimizer.step()
        train_cross_loss += classifier_loss.item()
        train_mse_loss += regressor_loss.item()
        train_icg_loss += icg_loss.item()
        train_total_loss += total_loss.item()
        if iter_num % 500 == 0:
            print((f"Iter {iter_num:05d}, "
                  f"Average Cross Entropy Loss: {train_cross_loss / float(test_interval):.4f}; "
                  f"Average MSE Loss: {train_mse_loss / float(test_interval):.4f}; "
                  f"Average ICG Loss: {train_icg_loss / float(test_interval):.4f}; "
                  f"Average Training Loss: {train_total_loss / float(test_interval):.4f}"))
            train_cross_loss = train_mse_loss = train_icg_loss = train_total_loss = 0.0
        if (iter_num % test_interval) == 0:
            Regression_test(dset_loaders['test'], Model_R, optimizer=optimizer, save=True, iter_num=iter_num)


def collect_samples_with_pseudo_label(loader, model, threshold):
    '''Pseudo label generation and selection on target dataset'''
    model.eval()
    img_selected = []
    labels_selected = []
    with torch.no_grad():
        for (imgs, labels) in tqdm(loader['val']):
            if len(img_selected) >501:
                break
            labels_source = labels.to(device)
            labels1 = labels_source[:, 2]
            labels3 = labels_source[:, 4]
            labels4 = labels_source[:, 5]

            labels1 = labels1.unsqueeze(1)
            labels3 = labels3.unsqueeze(1)
            labels4 = labels4.unsqueeze(1)

            # labels_source = torch.cat((labels1, labels3, labels4), dim=1)
            labels_source = labels1
            labels = labels_source.float()

            imgs = imgs.to(device)
            cls, reg, f = model(imgs)
            sel = F.softmax(cls, dim=-1).max(-1)[0] > threshold
            preds = bins_deltas_to_ts_batch(cls, reg, ctr)
            img_selected.append(imgs[sel].cpu())
            labels_selected.append(preds[sel].cpu())
    return img_selected, labels_selected


def selftrain_t(Model_R, img_selected, labels_selected):
    criterion = {"cls": xentropy, "reg": nn.MSELoss(), "icg": ExplicitInterClassGraphLoss()}
    optimizer_dict = [{"params": filter(lambda p: p.requires_grad, Model_R.model_fc.parameters()), "lr": 0.03},
                    {"params": filter(lambda p: p.requires_grad, Model_R.cls_layer.parameters()), "lr": 0.01},
                    {"params": filter(lambda p: p.requires_grad, Model_R.reg_layer.parameters()), "lr": 0.01}]
    optimizer = optim.SGD(optimizer_dict, lr=0.03, momentum=0.9, weight_decay=0.0005, nesterov=True)
    train_cross_loss = train_mse_loss = train_icg_loss = train_total_loss = 0.0
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    test_interval = 100

    for iter_num, (img, labelt) in enumerate(zip(img_selected, labels_selected)):
        Model_R.train()
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=args.gamma, power=0.75,
                                    weight_decay=0.0005)
        optimizer.zero_grad()
        cls, reg, f = Model_R(img.cuda())
        bins, deltas = t_to_bin_delta_batch(labelt.cuda(), ctr)
        classifier_loss = criterion["cls"](cls, bins)
        regressor_loss = criterion['reg'](reg, deltas)
        icg_loss = criterion['icg'](torch.max(bins, dim=-1)[1], f) * 1.0
        total_loss = classifier_loss + regressor_loss + icg_loss
        total_loss.backward()
        optimizer.step()
        train_cross_loss += classifier_loss.item()
        train_mse_loss += regressor_loss.item()
        train_icg_loss += icg_loss.item()
        train_total_loss += total_loss.item()
        if iter_num % test_interval == 0:
            print((f"Iter {iter_num:05d}, "
                  f"Average Cross Entropy Loss: {train_cross_loss / float(test_interval):.4f}; "
                  f"Average MSE Loss: {train_mse_loss / float(test_interval):.4f}; "
                  f"Average ICG Loss: {train_icg_loss / float(test_interval):.4f}; "
                  f"Average Training Loss: {train_total_loss / float(test_interval):.4f}"))
            train_cross_loss = train_mse_loss = train_icg_loss = train_total_loss = 0.0
            Regression_test(dset_loaders['test'], Model_R)



Model_R = Model_Regression().to(device)
# pretrain_on_src(Model_R)
Model_R.load_state_dict(torch.load('checkpoints/s->n-it_6000-MAE_0.072.pth')['model'])
threshold = 0.8
for _ in range(10):
    threshold -= 0.05
    img_selected, labels_selected = collect_samples_with_pseudo_label(dset_loaders, Model_R, threshold)
    selftrain_t(Model_R, img_selected, labels_selected)
    img_selected, labels_selected
