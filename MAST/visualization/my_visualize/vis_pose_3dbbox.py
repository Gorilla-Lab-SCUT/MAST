import argparse
import os.path as osp

import numpy as np
import torch
from MAST.config import RESULTS_DIR
from MAST.datasets.datasets_cfg import make_object_dataset
from MAST.lib3d.rigid_mesh_database import MeshDataBase
from tqdm import tqdm

from .lmo_bop_dataset import LMO_bop_Dataset
from .test_utils import cal_add_cuda, cal_adds_cuda, read_results
import cv2
import json


def filtre_preds(one_frame_preds):
    """
    note: only valid for LineMod Occlusion
    arg: 
        list, one frame pred
    return: 
        dict(key=obj_id, value=pred),
        filtred one frame pred, remove duplicates.
    """
    lmo_obj = [1,5,6,8,9,10,11,12]
    obj_statics=[]
    new_pred_dict = {}
    for i in range(len(one_frame_preds)):
        obj = one_frame_preds[i]
        if (obj['obj_id'] in lmo_obj):
            if (obj['obj_id'] not in obj_statics):
                obj_statics.append(obj['obj_id'])
                new_pred_dict[str(obj['obj_id'])] = obj
            elif obj['score'] > new_pred_dict[str(obj['obj_id'])]['score']:
                new_pred_dict[str(obj['obj_id'])] = obj
    return new_pred_dict



def main(args):
    bboxp = json.load(open('local_data/bop_datasets/lmo/models/models_info.json'))
    object_id = str(int(args.results_id.split('/')[-1]))
    box = bboxp[object_id]
    min_x = box['min_x']; min_y= box['min_y']; min_z= box['min_z']; size_x= box['size_x']; size_y= box['size_y']; size_z= box['size_z']
    ptmin = np.stack([min_x, min_y, min_z])[None,:]
    pts = np.repeat(ptmin,8,0)
    pts[1][0] += size_x
    pts[3][1] += size_y
    pts[2][0] += size_x; pts[2][1] += size_y
    pts[4][2] += size_z
    pts[5][2] += size_z; pts[5][0] += size_x
    pts[7][2] += size_z; pts[7][1] += size_y
    pts[6][2] += size_z; pts[6][0] += size_x; pts[6][1] += size_y
    bbox = pts / 1000

    results_path = osp.join(RESULTS_DIR, args.results_id,
                   f"dataset=lmo_bop/c_epoch={args.coarse_epoch}-r_epoch={args.refiner_epoch}.pth.tar")
    object_ds = make_object_dataset('lm.eval')
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    results = read_results(results_path, args.method, frame_num=200)
    dataloader = LMO_bop_Dataset().dataloader(batch_size=1, n_workers=0, use_collate_fn=True)

    cls_id_map = {
            '000001': 'Ape',
            '000004': 'Camera',
            '000005': 'Can',
            '000006': 'Cat',
            '000008': 'Driller',
            '000009': 'Duck',
            '000010': 'Eggbox',
            '000011': 'Glue',
            '000012': 'Holepuncher',
        }

    ADD_S = {'000001': 0, '000005': 0, '000006': 0, '000008': 0,
             '000009': 0, '000010': 0, '000011': 0, '000012': 0}  # ADD(-S)

    OBJ_COUNTS = {'000001': 0, '000005': 0, '000006': 0, '000008': 0,
                  '000009': 0, '000010': 0, '000011': 0, '000012': 0}  # count obj labels in all image

    for im_id, batch in enumerate(tqdm(dataloader)):
        # each batch is an image
        labels = batch['transes'][0]  # one image label
        preds = filtre_preds(results[str(im_id)])  # one image preds
        img = batch['images'][0].permute(1,2,0) * 255
        img = img.numpy().astype(np.uint8)[:, :, ::-1]
        img = img.copy()

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

                if obj_id == "000010" or obj_id == "000011":
                    cal = cal_adds_cuda((pred_R, pred_t),(label_R,label_t),model, diameter_m)
                    ADD_S[obj_id] += cal
                else:
                    cal = cal_add_cuda((pred_R, pred_t),(label_R,label_t),model, diameter_m)
                    ADD_S[obj_id] += cal
                    objectis='ape'
                    method='base'
                    if obj_id == f"0000{args.results_id.split('/')[-1]}" and osp.isfile(f'vis_{objectis}/{im_id}_MAST.png') :# cal == 1: #
                        # try:
                        #     img = cv2.imread(f'vis/{im_id}_lb.png')
                        #     print(img.shape)
                        # except:
                        #     continue

                        proj_box = np.around(compute_proj(bbox, label_R.cpu().numpy(), label_t.cpu().numpy()[:,None].T, batch['cameras'].K[0])).astype(np.int)#8*2
                        color = (0,0,255)  # red is gt
                        for i in range(0, proj_box.shape[0]):
                            img = cv2.circle(img, (proj_box[i][0], proj_box[i][1]), 2, color, -1, cv2.LINE_AA)
                        img = cv2.line(img, tuple(proj_box[0]), tuple(proj_box[1]), color, 2)
                        img = cv2.line(img, tuple(proj_box[1]), tuple(proj_box[2]), color, 2)
                        img = cv2.line(img, tuple(proj_box[2]), tuple(proj_box[3]), color, 2)
                        img = cv2.line(img, tuple(proj_box[3]), tuple(proj_box[0]), color, 2)
                        img = cv2.line(img, tuple(proj_box[4]), tuple(proj_box[5]), color, 2)
                        img = cv2.line(img, tuple(proj_box[5]), tuple(proj_box[6]), color, 2)
                        img = cv2.line(img, tuple(proj_box[6]), tuple(proj_box[7]), color, 2)
                        img = cv2.line(img, tuple(proj_box[7]), tuple(proj_box[4]), color, 2)
                        img = cv2.line(img, tuple(proj_box[0]), tuple(proj_box[4]), color, 2)
                        img = cv2.line(img, tuple(proj_box[1]), tuple(proj_box[5]), color, 2)
                        img = cv2.line(img, tuple(proj_box[2]), tuple(proj_box[6]), color, 2)
                        img = cv2.line(img, tuple(proj_box[3]), tuple(proj_box[7]), color, 2)

                        proj_box = np.around(compute_proj(bbox, pred_R.cpu().numpy(), pred_t.cpu().numpy()[:,None].T, batch['cameras'].K[0])).astype(np.int)#8*2
                        color = (0,255,0)
                        for i in range(0, proj_box.shape[0]):
                            img = cv2.circle(img, (proj_box[i][0], proj_box[i][1]), 2, color, -1, cv2.LINE_AA)
                        img = cv2.line(img, tuple(proj_box[0]), tuple(proj_box[1]), color, 2)
                        img = cv2.line(img, tuple(proj_box[1]), tuple(proj_box[2]), color, 2)
                        img = cv2.line(img, tuple(proj_box[2]), tuple(proj_box[3]), color, 2)
                        img = cv2.line(img, tuple(proj_box[3]), tuple(proj_box[0]), color, 2)
                        img = cv2.line(img, tuple(proj_box[4]), tuple(proj_box[5]), color, 2)
                        img = cv2.line(img, tuple(proj_box[5]), tuple(proj_box[6]), color, 2)
                        img = cv2.line(img, tuple(proj_box[6]), tuple(proj_box[7]), color, 2)
                        img = cv2.line(img, tuple(proj_box[7]), tuple(proj_box[4]), color, 2)
                        img = cv2.line(img, tuple(proj_box[0]), tuple(proj_box[4]), color, 2)
                        img = cv2.line(img, tuple(proj_box[1]), tuple(proj_box[5]), color, 2)
                        img = cv2.line(img, tuple(proj_box[2]), tuple(proj_box[6]), color, 2)
                        img = cv2.line(img, tuple(proj_box[3]), tuple(proj_box[7]), color, 2)
                        cv2.imwrite(f'vis_{objectis}/{im_id}_{method}.png', img)
                # print(obj_id,'\n', label_R,"\n", pred_R,'\n', label_t,'\n', pred_t, "\n")


    # category level average recall
    for obj, add in ADD_S.items():
        ADD_S[obj] = ADD_S[obj]/OBJ_COUNTS[obj]

    # all category mean
    mean = np.mean(np.array(list(ADD_S.values())))

    # change names to real class name, print percentage
    for key in ADD_S.keys():
        print(f'{cls_id_map[key]}: {ADD_S[key] * 100}')
    print(f'Mean: {mean * 100}')


def compute_proj(pts_3d, R, t, K):
    pts_2d = np.zeros((2, pts_3d.shape[0]), dtype=np.float32)#2*n
    pts_3d = np.dot(K, np.add(np.dot(R, pts_3d.T), t.T))#3*n
    pts_2d[0, :] = pts_3d[0, :]/pts_3d[2, :]
    pts_2d[1, :] = pts_3d[1, :]/pts_3d[2, :]
    return pts_2d.T#n*2


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--method', default='maskrcnn_detections/coarse/iteration=1', type=str)
    parser.add_argument('--results_id', default="", type=str)
    parser.add_argument('--coarse_epoch', default=0, type=int)
    parser.add_argument('--refiner_epoch', default=0, type=int)
    args = parser.parse_args()
    main(args)
