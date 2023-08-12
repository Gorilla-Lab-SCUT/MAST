from mmap import MAP_DENYWRITE
from operator import mod
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from MAST.visualization.singleview import render_result
from MAST.config import DEBUG_DATA_DIR
from MAST.lib3d.camera_geometry import get_K_crop_resize, boxes_from_uv

from MAST.lib3d.cropping import deepim_crops_robust as deepim_crops
from MAST.lib3d.camera_geometry import project_points_robust as project_points

from MAST.lib3d.rotations import (
    compute_rotation_matrix_from_ortho6d, compute_rotation_matrix_from_quaternions)
from MAST.models.loss_ops import apply_imagespace_predictions

from MAST.utils.logging import get_logger
from MAST.models.bin_delta_Rt import bin_delta_to_Rts_batch, sample_rotations_60, grid_xyz

logger = get_logger(__name__)


class RHeadNet(nn.Module):
    def __init__(self, in_channels, num_layers=2, num_filters=1024, kernel_size=3, output_dim=[60, 360], freeze=False):
        super(RHeadNet, self).__init__()
        self.freeze = freeze
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 2:
            padding = 0
        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(nn.Conv2d(_in_channels, num_filters, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

        self.R_bin = nn.Sequential(nn.Linear(num_filters, 512), nn.ReLU(inplace=True),
                                   nn.Linear(512, output_dim[0]))
        self.R_delta = nn.Sequential(nn.Linear(num_filters, 512), nn.ReLU(inplace=True),
                                     nn.Linear(512, output_dim[1]))


    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                for i, l in enumerate(self.features):
                    x = l(x)
                x = x.flatten(2).mean(dim=-1)
                R_bin = self.R_bin(x).detach()
                R_delta = self.R_delta(x).detach().reshape(-1, 60, 6)
                outputs = dict(
                    R_bin=R_bin,
                    R_delta=R_delta,
                    R_feat=x
                )
                return outputs
        else:
            for i, l in enumerate(self.features):
                x = l(x)
            x = x.flatten(2).mean(dim=-1)
            R_bin = self.R_bin(x)
            R_delta = self.R_delta(x).reshape(-1, 60, 6)
            outputs = dict(
                    R_bin=R_bin,
                    R_delta=R_delta,
                    R_feat=x
            )
            return outputs


class xyHeadNet(nn.Module):
    def __init__(self, in_channels, num_layers=2, num_filters=512, kernel_size=3, output_dim=[20, 20], freeze=False):
        super(xyHeadNet, self).__init__()
        self.freeze = freeze
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 2:
            padding = 0
        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(nn.Conv2d(_in_channels, num_filters, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

        self.x_bin = nn.Sequential(nn.Linear(num_filters, output_dim[0]))
        self.x_delta = nn.Sequential(nn.Linear(num_filters, output_dim[1]))
        self.y_bin = nn.Sequential(nn.Linear(num_filters, output_dim[0]))
        self.y_delta = nn.Sequential(nn.Linear(num_filters, output_dim[1]))


    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                for i, l in enumerate(self.features):
                    x = l(x)
                x = x.flatten(2).mean(dim=-1)
                x_bin = self.x_bin(x).detach()
                x_delta = self.x_delta(x).detach()
                y_bin = self.y_bin(x).detach()
                y_delta = self.y_delta(x).detach()
                outputs = dict(
                    x_bin=x_bin,
                    x_delta=x_delta,
                    y_bin=y_bin,
                    y_delta=y_delta,
                )
                return outputs
        else:
            for i, l in enumerate(self.features):
                x = l(x)
            x = x.flatten(2).mean(dim=-1)
            x_bin = self.x_bin(x)
            x_delta = self.x_delta(x)
            y_bin = self.y_bin(x)
            y_delta = self.y_delta(x)
            outputs = dict(
                x_bin=x_bin,
                x_delta=x_delta,
                y_bin=y_bin,
                y_delta=y_delta,
            )
            return outputs


class zHeadNet(nn.Module):
    def __init__(self, in_channels, num_layers=2, num_filters=512, kernel_size=3, output_dim=[40, 40], freeze=False):
        super(zHeadNet, self).__init__()
        self.freeze = freeze
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 2:
            padding = 0
        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(nn.Conv2d(_in_channels, num_filters, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

        self.z_bin = nn.Sequential(nn.Linear(num_filters, output_dim[0]))
        self.z_delta = nn.Sequential(nn.Linear(num_filters, output_dim[1]))


    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                for i, l in enumerate(self.features):
                    x = l(x)
                x = x.flatten(2).mean(dim=-1)
                x = F.normalize(x, p=2, dim=1)
                z_bin = self.z_bin(x).detach()
                z_delta = self.z_delta(x).detach()
                outputs = dict(
                    z_bin=z_bin,
                    z_delta=z_delta,
                    z_feat=x
                )
                return outputs
        else:
            for i, l in enumerate(self.features):
                x = l(x)
            x = x.flatten(2).mean(dim=-1)
            x = F.normalize(x, p=2, dim=1)
            z_bin = self.z_bin(x)
            z_delta = self.z_delta(x)
            outputs = dict(
                    z_bin=z_bin,
                    z_delta=z_delta,
                    z_feat=x
            )
            return outputs



class PosePredictor(nn.Module):
    def __init__(self, backbone, renderer,
                 mesh_db, render_size=(240, 320),
                 pose_dim=9):
        super().__init__()

        self.backbone = backbone
        self.renderer = renderer
        self.mesh_db = mesh_db
        self.render_size = render_size
        self.pose_dim = pose_dim

        n_features = backbone.n_features

        self.R_bin_ctrs = torch.from_numpy(sample_rotations_60("matrix")).float().cuda()
        self.xy_bin_ctrs, self.z_bin_ctrs = grid_xyz()
        self.xy_bin_ctrs = torch.from_numpy(self.xy_bin_ctrs).float().cuda()
        self.z_bin_ctrs = torch.from_numpy(self.z_bin_ctrs).float().cuda()

        self.Rhead = RHeadNet(in_channels=n_features)
        self.xyhead = xyHeadNet(in_channels=n_features)
        self.zhead = zHeadNet(in_channels=n_features)

        self.debug = False
        self.tmp_debug = dict()

    def enable_debug(self):
        self.debug = True

    def disable_debug(self):
        self.debug = False

    def crop_inputs(self, images, K, TCO, labels):
        bsz, nchannels, h, w = images.shape
        assert K.shape == (bsz, 3, 3)
        assert TCO.shape == (bsz, 4, 4)
        assert len(labels) == bsz
        meshes = self.mesh_db.select(labels)
        points = meshes.sample_points(2000, deterministic=True)
        uv = project_points(points, K, TCO)  # project pointcloud to image coords
        boxes_rend = boxes_from_uv(uv)
        boxes_crop, images_cropped = deepim_crops(
            images=images, obs_boxes=boxes_rend, K=K,
            TCO_pred=TCO, O_vertices=points, output_size=self.render_size, lamb=1.4
        )
        # cosypose only use (b) the bounding box defined by T, k, CO and 
        # the vertices of the object l to define the size and location of 
        # the crop in the real input image during training testing, thus obs_boxes=boxes_rend
        K_crop = get_K_crop_resize(K=K.clone(), boxes=boxes_crop,
                                   orig_size=images.shape[-2:], crop_resize=self.render_size)
        if self.debug:
            self.tmp_debug.update(
                boxes_rend=boxes_rend,
                rend_center_uv=project_points(torch.zeros(bsz, 1, 3).to(K.device), K, TCO),
                uv=uv,
                boxes_crop=boxes_crop,
            )
        return images_cropped, K_crop.detach(), boxes_rend, boxes_crop

    def update_pose(self, TCO, K_crop, pose_outputs):
        if self.pose_dim == 9:
            # dR = compute_rotation_matrix_from_ortho6d(pose_outputs[:, 0:6])
            # vxvyvz = pose_outputs[:, 6:9]
            pose_outputs['R_delta'] = compute_rotation_matrix_from_ortho6d(pose_outputs['R_delta'])
            dR, vxvyvz = self.get_pose_from_bin_delta(pose_outputs)

        elif self.pose_dim == 7:
            dR = compute_rotation_matrix_from_quaternions(pose_outputs[:, 0:4])
            vxvyvz = pose_outputs[:, 4:7]
        else:
            raise ValueError(f'pose_dim={self.pose_dim} not supported')
        TCO_updated = apply_imagespace_predictions(TCO, K_crop, vxvyvz, dR)
        return TCO_updated, dR, vxvyvz

    def net_forward(self, x):
        x = self.backbone(x)  # [28, 1536, 7, 10]
        # x = x.flatten(2).mean(dim=-1)
        R_bin_delta = self.Rhead(x)
        xy_bin_delta = self.xyhead(x)
        z_bin_delta = self.zhead(x)
        outputs = {**R_bin_delta, **xy_bin_delta, **z_bin_delta}
        return outputs


    def get_pose_from_bin_delta(self,outputs):

        dR, vxvyvz = bin_delta_to_Rts_batch(outputs['R_bin'].detach(), outputs['R_delta'], self.R_bin_ctrs, outputs['x_bin'].detach(),
                                    outputs['y_bin'].detach(), outputs['z_bin'].detach(), outputs['x_delta'], outputs['y_delta'],
                                    outputs['z_delta'], self.xy_bin_ctrs, self.z_bin_ctrs)
        return dR, vxvyvz


    def forward(self, images, K, labels, TCO, n_iterations=1, TCO_gt=None, transes=None):
        bsz, nchannels, h, w = images.shape
        assert K.shape == (bsz, 3, 3)
        assert TCO.shape == (bsz, 4, 4)
        assert len(labels) == bsz

        outputs = dict()
        TCO_input = TCO
        for n in range(n_iterations):
            TCO_input = TCO_input.detach()
            images_crop, K_crop, boxes_rend, boxes_crop = self.crop_inputs(images, K, TCO_input, labels)
            renders = self.renderer.render(obj_infos=[dict(name=l) for l in labels],
                                           TCO=TCO_input,
                                           K=K_crop, resolution=self.render_size).contiguous()

            x = torch.cat((images_crop, renders), dim=1).contiguous()

            model_outputs = self.net_forward(x)


            ## replace pred bin with gt bin
            # for label in transes[0].keys():
            #     try: idx = list(labels).index('obj_'+label)
            #     except: continue
            #     gt = torch.tensor(transes[0][label.strip('obj_')]).float().cuda()
            #     gt_dR, gt_vxvyvz = cal_gt_dR_vxyz(TCO_input[[idx]], gt, K_crop[[idx]])
            #     Rb_gt, Rd_gt, xb_gt, yb_gt, zb_gt, xd_gt, yd_gt, zd_gt = Rts_to_bin_delta_batch(gt_dR, 
            #     gt_vxvyvz, self.R_bin_ctrs, self.xy_bin_ctrs, self.z_bin_ctrs)
            #     model_outputs['R_bin'][idx] = Rb_gt
            #     # model_outputs['x_bin'][idx] = xb_gt
            #     # model_outputs['y_bin'][idx] = yb_gt
            #     # model_outputs['z_bin'][idx] = zb_gt

            TCO_output, dR, vxvyvz = self.update_pose(TCO_input, K_crop, model_outputs)
            if TCO_gt is not None:
                render_result(self.renderer, TCO_output, TCO_gt, K_crop, labels, self.render_size, images_crop, model_outputs, TCO_input)

            bins = model_outputs['R_bin'], model_outputs['x_bin'], model_outputs['y_bin'], model_outputs['z_bin']
            deltas = model_outputs['R_delta'], model_outputs['x_delta'], model_outputs['y_delta'], model_outputs['z_delta']
            R_feat = model_outputs['R_feat']
            z_feat = model_outputs['z_feat']
            model_outputs = (dR, vxvyvz)

            outputs[f'iteration={n+1}'] = {
                'TCO_input': TCO_input,
                'TCO_output': TCO_output,
                'K_crop': K_crop,
                'model_outputs': model_outputs,
                'boxes_rend': boxes_rend,
                'boxes_crop': boxes_crop,
                'bins': bins,
                'deltas': deltas,
                'R_feat': R_feat,
                'z_feat': z_feat,
            }

            TCO_input = TCO_output

            if self.debug:
                self.tmp_debug.update(outputs[f'iteration={n+1}'])
                self.tmp_debug.update(
                    images=images,
                    images_crop=images_crop,
                    renders=renders,
                )
                path = DEBUG_DATA_DIR / f'debug_iter={n+1}.pth.tar'
                logger.info(f'Wrote debug data: {path}')
                torch.save(self.tmp_debug, path)

        return outputs
