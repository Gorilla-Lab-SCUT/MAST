import argparse
import os.path as osp

import numpy as np
import torch
from MAST.config import RESULTS_DIR
from MAST.datasets.datasets_cfg import make_object_dataset
from MAST.lib3d.rigid_mesh_database import MeshDataBase
from tqdm import tqdm

from .lm_dataset import LM_Dataset
from .test_utils import cal_add_cuda, cal_adds_cuda, read_results


def filtre_preds(one_frame_preds, lm_obj_id):
    """
    note: only valid for LineMod
    args: 
        list, one frame pred
        lm_obj_id: int, linemod object id
    return: 
        dict(key=obj_id, value=pred),
        filtred one frame pred, remove duplicates.
    """

    obj_statics=[]
    new_pred_dict = {}
    for i in range(len(one_frame_preds)):
        obj = one_frame_preds[i]
        if obj['obj_id'] ==int(lm_obj_id):
            if (obj['obj_id'] not in obj_statics):
                obj_statics.append(obj['obj_id'])
                new_pred_dict[str(obj['obj_id'])] = obj
            elif obj['score'] > new_pred_dict[str(obj['obj_id'])]['score']:
                new_pred_dict[str(obj['obj_id'])] = obj
    return new_pred_dict



def main(args, category):
    results_path = osp.join(RESULTS_DIR, args.results_id,
                   f"dataset=lm/{category}/c_epoch={args.coarse_epoch}-r_epoch={args.refiner_epoch}.pth.tar")
    obj_id = '0000' + category
    object_ds = make_object_dataset('lm.eval')
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    results = read_results(results_path, args.method, frame_num=1214)
    dataloader = LM_Dataset(split="test",category=category).dataloader(batch_size=1, n_workers=0, use_collate_fn=True)

    cls_id_map = {
        '000001': 'Ape',
        '000002': 'Benchvise',
        '000004': 'Camera',
        '000005': 'Can',
        '000006': 'Cat',
        '000008': 'Driller',
        '000009': 'Duck',
        '000010': 'Eggbox',
        '000011': 'Glue',
        '000012': 'Holepuncher',
        '000013': 'Iron',
        '000014': 'Lamp',
        '000015': 'Phone'
    }

    ADD_S = {'000001': 0, '000002': 0, '000003': 0, '000004': 0,'000005': 0, '000006': 0, '000007': 0,'000008': 0,
             '000009': 0, '000010': 0, '000011': 0, '000012': 0, '000013': 0, '000014': 0, '000015': 0,}  # ADD(-S)

    OBJ_COUNTS = {'000001': 0, '000002': 0, '000003': 0, '000004': 0,'000005': 0, '000006': 0, '000007': 0,'000008': 0,
             '000009': 0, '000010': 0, '000011': 0, '000012': 0, '000013': 0, '000014': 0, '000015': 0,}  # count obj labels in all image

    for im_id, batch in enumerate(tqdm(dataloader)):
        # each batch is an image
        label_R, label_t = batch['transes'][0]
        
        preds = filtre_preds(results[str(im_id)], lm_obj_id=obj_id)  # one image preds

        OBJ_COUNTS[obj_id] += 1  ## count objects labels

        if str(int(obj_id)) in preds.keys():  # check whether predicted
            assert preds[str(int(obj_id))]['im_id'] == im_id == batch['im_infos'][0]['view_id']
            assert preds[str(int(obj_id))]['obj_id'] == int(obj_id)

            label_R = torch.as_tensor(label_R).type(torch.float).cuda()
            label_t = torch.as_tensor(label_t).type(torch.float).cuda()
            pred_R = torch.as_tensor(preds[str(int(obj_id))]['R']).type(torch.float).cuda()
            pred_t = torch.as_tensor(preds[str(int(obj_id))]['t']).type(torch.float).cuda()
            model = torch.as_tensor(mesh_db.meshes['obj_'+obj_id].vertices/1000.0).type(torch.float).cuda()
            diameter_m = torch.as_tensor(mesh_db.infos['obj_'+obj_id]['diameter_m']).type(torch.float).cuda()

            if obj_id == "000010" or obj_id == "000011" or obj_id == "000003":
                ADD_S[obj_id] += cal_adds_cuda((pred_R, pred_t),(label_R,label_t),model, diameter_m)
            else:
                ADD_S[obj_id] += cal_add_cuda((pred_R, pred_t),(label_R,label_t),model, diameter_m)
            # print(obj_id,'\n', label_R,"\n", pred_R,'\n', label_t,'\n', pred_t, "\n")

    # recall
    ADD_S[obj_id] = ADD_S[obj_id]/OBJ_COUNTS[obj_id]

    print(f'{cls_id_map[obj_id]}: {ADD_S[obj_id]}')
    return f'{cls_id_map[obj_id]}: ', ADD_S[obj_id]


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--method', default='maskrcnn_detections/coarse/iteration=1', type=str)
    parser.add_argument('--results_id', default='', type=str)
    parser.add_argument('--coarse_epoch', default=0, type=int)
    parser.add_argument('--refiner_epoch', default=0, type=int)
    parser.add_argument('--obj_id', type=str)
    args = parser.parse_args()
    output = {}
    # evaluate all classes
    cls, adds = main(args, args.obj_id)
    output[cls] = adds

    # print out percentage
    for k, v in output.items():
        print(f'{k}{v * 100}')
    print(f'Mean: {np.mean(np.array(list(output.values()))) * 100}')
