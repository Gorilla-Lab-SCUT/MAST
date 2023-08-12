import argparse
import os.path as osp

import numpy as np
import torch
from MAST.config import RESULTS_DIR
from MAST.datasets.datasets_cfg import make_object_dataset
from MAST.lib3d.rigid_mesh_database import MeshDataBase
from tqdm import tqdm

from .test_utils import read_results, cal_auc
from .ycbv_bop_dataset import YCBV_bop_Dataset


def filtre_preds(one_frame_preds):
    """
    note: only valid for LineMod Occlusion
    arg: 
        list, one frame pred
    return: 
        dict(key=obj_id, value=pred),
        filtred one frame pred, remove duplicates.
    """
    obj_statics=[]
    new_pred_dict = {}
    for i in range(len(one_frame_preds)):
        obj = one_frame_preds[i]
        if (obj['obj_id'] not in obj_statics):
            obj_statics.append(obj['obj_id'])
            new_pred_dict[str(obj['obj_id'])] = obj
        elif obj['score'] > new_pred_dict[str(obj['obj_id'])]['score']:
            new_pred_dict[str(obj['obj_id'])] = obj
    return new_pred_dict


def cal_add_cuda(pred_RT, gt_RT, model, diameter, percentage=0.1):
    """ ADD metric
    1. compute the average of the 3d distances between the transformed vertices
    2. pose_pred is considered correct if the distance is less than 10% of the object's diameter
    args:
        pred_RT:(pred_R: (3,3), pred_t: (1, 3))
        model: (N, 3) vertices point cloud
        diameter: float diameter of the model
        percentage: ADD < percentage * diameter to be positive sample
    return:
        int, 1 or 0
    """
    pred_R, pred_t = pred_RT
    gt_R, gt_t = gt_RT
    diameter = diameter * percentage
    pred_model = torch.mm(model, pred_R.transpose(1, 0)) + pred_t
    gt_model = torch.mm(model, gt_R.transpose(1, 0)) + gt_t
    mean_dist = torch.mean(torch.norm(pred_model - gt_model, dim=1))
    # return int(mean_dist < diameter)
    return mean_dist



def cal_adds_cuda(pred_RT, gt_RT, model, diameter, percentage=0.1):
    """ ADD-S metric
    1. compute the average of the 3d distances between the transformed vertices
    2. pose_pred is considered correct if the distance is less than 10% of the object's diameter
    args:
        pred_RT:(pred_R: (3,3), pred_t: (1, 3))
        model: (N, 3) vertices point cloud
        diameter: float diameter of the model
        percentage: ADD < percentage * diameter to be positive sample
    return:
        int, 1 or 0
    """
    pred_R, pred_t = pred_RT
    gt_R, gt_t = gt_RT
    diameter = diameter * percentage
    N, _ = model.size()

    pred_model = torch.mm(model, pred_R.transpose(1, 0)) + pred_t
    pred_model = pred_model.view(1, N, 3).repeat(N, 1, 1)

    gt_model = torch.mm(model, gt_R.transpose(1, 0)) + gt_t
    gt_model = gt_model.view(N, 1, 3).repeat(1, N, 1)

    dis = torch.norm(pred_model - gt_model, dim=2)
    mean_dist = torch.mean(torch.min(dis, dim=1)[0])

    # return int(mean_dist < diameter)
    return mean_dist


def main(args):
    results_path = osp.join(RESULTS_DIR, args.results_id,
                   f"dataset=ycbv_bop/c_epoch={args.coarse_epoch}-r_epoch={args.refiner_epoch}.pth.tar")
    object_ds = make_object_dataset('ycbv.bop')
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    results = read_results(results_path, args.method, frame_num=900)
    dataloader = YCBV_bop_Dataset().dataloader(batch_size=1, n_workers=0, use_collate_fn=True)

    cls_id_map = {
            '000001': 'master_chef_can',
            '000002': 'cracker_box',
            '000003': 'sugar_box',
            '000004': 'tomato_soup_can',
            '000005': 'mustard_bottle',
            '000006': 'tuna_fish_can',
            '000007': 'pudding_box',
            '000008': 'gelatin_box',
            '000009': 'potted_meat_can',
            '000010': 'banana',
            '000011': 'pitcher_base',
            '000012': 'bleach_cleanser',
            '000013': 'bowl',
            '000014': 'mug',
            '000015': 'power_drill',
            '000016': 'wood_block',
            '000017': 'scissors',
            '000018': 'large_marker',
            '000019': 'large_clamp',
            '000020': 'extra_large_clamp',
            '000021': 'foam_brick'
        }

    ADD_S = {'000001': [], '000002': [], '000003': [], '000004': [],
             '000005': [], '000006': [], '000007': [], '000008': [],
             '000009': [], '000010': [], '000011': [], '000012': [],
             '000013': [], '000014': [], '000015': [], '000016': [],
             '000017': [], '000018': [], '000019': [], '000020': [],
             '000021': []
        }  # ADD(-S)

    ADDS = {'000001': [], '000002': [], '000003': [], '000004': [],
             '000005': [], '000006': [], '000007': [], '000008': [],
             '000009': [], '000010': [], '000011': [], '000012': [],
             '000013': [], '000014': [], '000015': [], '000016': [],
             '000017': [], '000018': [], '000019': [], '000020': [],
             '000021': []
        }  # ADD-S

    OBJ_COUNTS = {'000001': 0, '000002': 0, '000003': 0, '000004': 0,
                  '000005': 0, '000006': 0, '000007': 0, '000008': 0,
                  '000009': 0, '000010': 0, '000011': 0, '000012': 0,
                  '000013': 0, '000014': 0, '000015': 0, '000016': 0,
                  '000017': 0, '000018': 0, '000019': 0, '000020': 0,
                  '000021': 0
            }  # count obj labels in all image

    for im_id, batch in enumerate(tqdm(dataloader)):
        # each batch is an image
        labels = batch['transes'][0]  # one image label
        preds = filtre_preds(results[str(im_id)])  # one image preds

        for obj_id, pose_label in labels.items():
            
            OBJ_COUNTS[obj_id] +=1  ## count objects labels

            if str(int(obj_id)) in preds.keys():  # check whether predicted
                assert preds[str(int(obj_id))]['im_id'] == im_id == batch['im_infos'][0]['view_id']
                assert preds[str(int(obj_id))]['obj_id'] == int(obj_id)

                label_R = torch.as_tensor(pose_label[:3, :3]).type(torch.float).cuda()
                label_t = torch.as_tensor(pose_label[:3, -1]).type(torch.float).cuda()
                pred_R = torch.as_tensor(preds[str(int(obj_id))]['R']).type(torch.float).cuda()
                pred_t = torch.as_tensor(preds[str(int(obj_id))]['t']).type(torch.float).cuda()
                model = torch.as_tensor(mesh_db.meshes['obj_'+obj_id].vertices/1e3).type(torch.float).cuda()
                diameter_m = torch.as_tensor(mesh_db.infos['obj_'+obj_id]['diameter_m']).type(torch.float).cuda()

                if obj_id == "000013" or obj_id == "000016" or obj_id == "000019" or obj_id == "000020" or obj_id == "000021":
                    ADD_S[obj_id].append(cal_adds_cuda((pred_R, pred_t),(label_R,label_t),model, diameter_m).cpu())
                else:
                    ADD_S[obj_id].append(cal_add_cuda((pred_R, pred_t),(label_R,label_t),model, diameter_m).cpu())

                ADDS[obj_id].append(cal_adds_cuda((pred_R, pred_t),(label_R,label_t),model, diameter_m).cpu())
                # print(obj_id,'\n', label_R,"\n", pred_R,'\n', label_t,'\n', pred_t, "\n")

    for obj, add in ADD_S.items():
        ADD_S[obj] = cal_auc(add)

    for obj, add in ADDS.items():
        ADDS[obj] = cal_auc(add)

    # category level average
    # for obj, add in ADD_S.items():
    #     ADD_S[obj] = ADD_S[obj]/OBJ_COUNTS[obj]

    # all category mean
    mean_ADD_S = np.mean(np.array(list(ADD_S.values())))
    mean_ADDS = np.mean(np.array(list(ADDS.values())))

    # change names to real class name, print percentage
    print('AUC of ADD(-S)')
    for key in ADD_S.keys():
        print(f'{cls_id_map[key]}: {ADD_S[key] * 100}')
    print(f'Mean: {mean_ADD_S * 100}')

    print('\nAUC of ADD-S')
    for key in ADDS.keys():
        print(f'{cls_id_map[key]}: {ADDS[key] * 100}')
    print(f'Mean: {mean_ADDS * 100}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--method', default='maskrcnn_detections/coarse/iteration=1', type=str)
    parser.add_argument('--results_id', default="", type=str)
    parser.add_argument('--coarse_epoch', default=0, type=int)
    parser.add_argument('--refiner_epoch', default=0, type=int)
    args = parser.parse_args()
    main(args)
