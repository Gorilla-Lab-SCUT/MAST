import torch
from tqdm import tqdm
import numpy as np


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
    return int(mean_dist < diameter)


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

    return int(mean_dist < diameter)


def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap


def cal_auc(dis_list, max_dis=0.1):
    D = np.array(dis_list)
    D[np.where(D > max_dis)] = np.inf
    D = np.sort(D)
    n = len(dis_list)
    acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
    aps = VOCap(D, acc)
    return aps


def read_results(results_path, method, frame_num):
    """
    args:
        results_path: inference results path
        method: 'maskrcnn_detections/refiner/iteration=4'
    return:
        total infernce result dict(key=view_id, value=list(preds))
    """
    predictions = torch.load(results_path)['predictions']
    predictions = predictions[method]
    print(f"Predictions from: {results_path}")
    print(f"Method: {method}")
    print(f"Number of predictions: {len(predictions)}")
    preds = {str(i):[] for i in range(frame_num)}
    for n in tqdm(range(len(predictions))):
        TCO_n = predictions.poses[n]
        t = TCO_n[:3, -1]
        R = TCO_n[:3, :3]
        row = predictions.infos.iloc[n]
        obj_id = int(row.label.split('_')[-1])
        score = row.score
        time = row.time
        pred = dict(scene_id=row.scene_id,
                    im_id=row.view_id,
                    obj_id=obj_id,
                    score=score,
                    t=t, R=R, time=time)
        preds[str(row.view_id)].append(pred)
    return preds
