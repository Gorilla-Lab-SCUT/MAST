from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F


def cls2onehot(labels: np.array([str])):
    map = { 'obj_000001': 0, 'obj_000002': 1,
            'obj_000003': 2, 'obj_000004': 3,
            'obj_000005': 4, 'obj_000006': 5,
            'obj_000007': 6, 'obj_000008': 7,
            'obj_000009': 8, 'obj_000010': 9,
            'obj_000011': 10,'obj_000012': 11,
            'obj_000013': 12,'obj_000014': 13,
            'obj_000015': 14,
            }
    bs = labels.shape[0]
    labels = [map[i] for i in labels]
    onehots = torch.zeros(bs, 15)
    onehots[range(bs), labels] = 1
    return onehots.cuda()


def cls2idx(labels: np.array([str])):
    map = { 'obj_000001': 0, 'obj_000002': 1,
            'obj_000003': 2, 'obj_000004': 3,
            'obj_000005': 4, 'obj_000006': 5,
            'obj_000007': 6, 'obj_000008': 7,
            'obj_000009': 8, 'obj_000010': 9,
            'obj_000011': 10,'obj_000012': 11,
            'obj_000013': 12,'obj_000014': 13,
            'obj_000015': 14,
            }
    bs = labels.shape[0]
    labels = [map[i] for i in labels]
    return torch.as_tensor(labels).contiguous().cuda()


def shannon_entropy(pred):
    Lsoftmax = F.log_softmax(pred, dim=-1)
    sftm = F.softmax(pred, dim=-1)
    loss = - (sftm * Lsoftmax).sum(dim=1).mean()
    return loss


def refine_TCO_by_mask_deprecated(TCO_pred, mask_crop, labels, renderer, K_crop, render_size):
    r""" refine the xyz by compare rendered mask and gt mask according to pinhole imaging
         TODO: better to remove the for loop
         NOTE: bad, wait and see
    """
    renders_by_pred = renderer.render(obj_infos=[dict(name=l) for l in labels],
                            TCO=TCO_pred,
                            K=K_crop, resolution=render_size).contiguous()

    bsz = len(mask_crop)
    TCO_refined = deepcopy(TCO_pred)
    for i in range(bsz):
        # xmax xmin, ymax, ymin of gt mask on pixel coordinate
        xmap_masked, ymap_masked = torch.nonzero(mask_crop[i], as_tuple=True)
        deltax = xmap_masked.max() - xmap_masked.min()
        deltay = ymap_masked.max() - ymap_masked.min()

        # xmax xmin, ymax, ymin of predicted TCO rendered mask on pixel coordinate
        xmap_masked_pred, ymap_masked_pred = torch.nonzero(renders_by_pred[i][0], as_tuple=True)
        deltax_pred = xmap_masked_pred.max() - xmap_masked_pred.min()
        deltay_pred = ymap_masked_pred.max() - ymap_masked_pred.min()

        z_pred = TCO_refined[i, 2, 3]
        z_pred_refined_from_x = z_pred * (deltax/deltax_pred)
        z_pred_refined_from_y = z_pred * (deltay/deltay_pred)
        z_pred_refined = (z_pred_refined_from_x + z_pred_refined_from_y)/ 2
        TCO_refined[i, 2, 3] = z_pred_refined
        # TCO_refined[i, :2, 3] = (TCO_refined[i, :2, 3] / z_pred.repeat(1, 2)) * z_pred_refined.repeat(1, 2)

    return TCO_refined


def select_batch_idx_by_z_for_self_regression(dR_bin, dx_bin, dy_bin, dz_bin, cfg, labels, obj_threshold):
    r"""select samples that meet z larger/smaller than threshold"""
    bsz = len(dz_bin)

    def select(batch_bins, labels):
        r"""select top threshold percentage of a batch by probability as pseudo label training samples
            set various thresholds for various objects
        """
        full_batch_prob, full_batch_label = torch.max(F.softmax(batch_bins, dim=-1), dim=-1)
        ## this 2 lines are old method of selection
        # selected_batch_idx = torch.topk(full_batch_prob, k=int(cfg.target_select_threshold * len(full_batch_prob)))[1]
        # selected_batch_idx = torch.where(full_batch_prob < cfg.target_select_threshold)[0]
        selected_batch_idx = []
        for i in range(bsz):
            if obj_threshold[labels[i]]['op'](full_batch_prob[i], obj_threshold[labels[i]]['th'] - cfg.th_decay):
                selected_batch_idx.append(i)
        return torch.tensor(selected_batch_idx).cuda(), full_batch_label


    def logits2label(batch_bins):
        ## do not need softmax
        # full_batch_prob, full_batch_label = torch.max(F.softmax(batch_bins, dim=-1), dim=-1)
        full_batch_prob, full_batch_label = torch.max(batch_bins, dim=-1)
        return full_batch_label

    R_full_batch_label = logits2label(dR_bin)
    x_full_batch_label = logits2label(dx_bin)
    y_full_batch_label = logits2label(dy_bin)
    z_selected_batch_idx, z_full_batch_label = select(dz_bin, labels)

    if len(z_selected_batch_idx)>0:
        return (z_selected_batch_idx, 
                R_full_batch_label,
                x_full_batch_label,
                y_full_batch_label,
                z_full_batch_label)
    else:
        return None


def update_ema_variables(student_model, teacher_ema_model, a, global_step):
    '''
    Exponential Mean Average Mean Teacher
    Modified from https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/main.py#L189
                  https://arxiv.org/abs/1703.01780

    if ema_mean_teacher:
        for param in model_teacher.parameters():
            param.detach_()
    '''
    # Use the true average until the exponential average is more correct
    print('EMA parameters updating')
    a = min(1 - 1 / (global_step + 1), a)
    for ema_param, param in zip(teacher_ema_model.parameters(), student_model.parameters()):
        ema_param.data.mul_(a).add_(param.data, alpha=1 - a)
