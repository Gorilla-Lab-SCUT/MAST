import argparse
import os.path as osp

import numpy as np
import torch
from MAST.config import RESULTS_DIR
from MAST.datasets.datasets_cfg import make_object_dataset
from MAST.lib3d.rigid_mesh_database import MeshDataBase
from tqdm import tqdm

from .hb_dataset import HB_Dataset
from .test_utils import read_results


def filtre_preds(one_frame_preds):
    """
    note: only valid for LineMod Occlusion
    arg: 
        list, one frame pred
    return: 
        dict(key=obj_id, value=pred),
        filtred one frame pred, remove duplicates.
    """
    hb_obj_in_lm_id = [2, 8, 15]
    obj_statics=[]
    new_pred_dict = {}
    for i in range(len(one_frame_preds)):
        obj = one_frame_preds[i]
        if (obj['obj_id'] in hb_obj_in_lm_id):
            if (obj['obj_id'] not in obj_statics):
                obj_statics.append(obj['obj_id'])
                new_pred_dict[str(obj['obj_id'])] = obj
            elif obj['score'] > new_pred_dict[str(obj['obj_id'])]['score']:
                new_pred_dict[str(obj['obj_id'])] = obj
    return new_pred_dict


def cal_add_cuda(pred_RT, gt_RT, model, model_hb, diameter, percentage=0.1):
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
    gt_model = torch.mm(model_hb, gt_R.transpose(1, 0)) + gt_t
    mean_dist = torch.mean(torch.norm(pred_model - gt_model, dim=1))
    return int(mean_dist < diameter)



def cal_adds_cuda(pred_RT, gt_RT, model, model_hb, diameter, percentage=0.1):
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

    gt_model = torch.mm(model_hb, gt_R.transpose(1, 0)) + gt_t
    gt_model = gt_model.view(N, 1, 3).repeat(1, N, 1)

    dis = torch.norm(pred_model - gt_model, dim=2)
    mean_dist = torch.mean(torch.min(dis, dim=1)[0])

    return int(mean_dist < diameter)



def main(args):
    results_path = osp.join(RESULTS_DIR, args.results_id,
                   f"dataset=hb/c_epoch={args.coarse_epoch}-r_epoch={args.refiner_epoch}.pth.tar")
    object_ds = make_object_dataset('lm.eval')
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    results = read_results(results_path, args.method, frame_num=340)
    dataloader = HB_Dataset().dataloader(batch_size=1, n_workers=0, use_collate_fn=True)

    cls_id_map = {
            '000002': 'Bvise',
            '000008': 'Driller',
            '000015': 'Phone'
        }

    ADD_S = {'000002': 0, '000008': 0, '000015': 0}  # ADD(-S)

    OBJ_COUNTS = {'000002': 0, '000008': 0, '000015': 0}  # count obj labels in all image


    lm_model2hb_model_T = {
        '000002': [torch.tensor([[ 7.5606e-02, -9.9692e-01, -2.0967e-02],
                                 [ 9.6066e-01,  7.8459e-02, -2.6641e-01],
                                 [ 2.6724e-01,  7.7392e-09,  9.6363e-01]]).float().cuda(),
                   torch.tensor([ 0.0105, -0.0124, -0.0017]).float().cuda()],
        '000008': [torch.tensor([[-3.9281e-08, -9.9255e-01, -1.2187e-01],
                                [ 9.9863e-01, -6.3782e-03,  5.1946e-02],
                                [-5.2336e-02, -1.2170e-01,  9.9119e-01]]).float().cuda(),
                   torch.tensor([-0.0041, -0.0039,  0.0045]).float().cuda()],
        '000015': [torch.tensor([[ 9.9452e-01, -3.6947e-09,  1.0453e-01],
                                [-9.8628e-10,  1.0000e+00,  4.4730e-08],
                                [-1.0453e-01, -4.4588e-08,  9.9452e-01]]).float().cuda(),
                   torch.tensor([7.9696e-03, 1.2831e-09, 6.9724e-04]).float().cuda()]
    }
    

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
                R, t = lm_model2hb_model_T[obj_id]
                model_hb = model @ R.T + t
                if obj_id == "000010" or obj_id == "000011":
                    ADD_S[obj_id] += cal_adds_cuda((pred_R, pred_t),(label_R,label_t),model, model_hb, diameter_m)
                else:
                    ADD_S[obj_id] += cal_add_cuda((pred_R, pred_t),(label_R,label_t),model, model_hb, diameter_m)
                # print(obj_id,'\n', label_R,"\n", pred_R,'\n', label_t,'\n', pred_t, "\n")


    # category level average recall
    for obj, add in ADD_S.items():
        ADD_S[obj] = ADD_S[obj]/OBJ_COUNTS[obj]

    # all category mean
    mean = np.mean(np.array(list(ADD_S.values())))

    # change names to real class name
    for key in ADD_S.keys():
        print(f'{cls_id_map[key]}: {ADD_S[key] * 100}')
    print(f'Mean: {mean * 100}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--method', default='maskrcnn_detections/coarse/iteration=1', type=str)
    parser.add_argument('--results_id', default="", type=str)
    parser.add_argument('--coarse_epoch', default=0, type=int)
    parser.add_argument('--refiner_epoch', default=0, type=int)
    args = parser.parse_args()
    main(args)
