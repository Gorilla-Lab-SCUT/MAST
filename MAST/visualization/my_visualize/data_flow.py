from MAST.models.loss_ops import TCO_init_from_boxes
from MAST.models.pose.PosePredictor import crop_inputs, update_pose, network
import torch.cat as concatenate

# TRAINING
import MAST.scripts.run_pose_training   # parameter configurations
import MAST.training.train_pose         # training procedure
import MAST.training.pose_forward_loss  # initalize TCO_input and calculate loss

# INFERENCE & TRAINING
## Pose Predictor
import MAST.training.pose_models_cfg        # configure single coarse/refiner backbone (coarse/refiner are the same structure)
from MAST.models.pose import PosePredictor  # single coarse/refiner model definition
import MAST.integrated.pose_predictor       # config coarse & refiner together for inference

# INFERENCE
from MAST.evaluation.pred_runner.bop_predictions import BopPredictionRunner    # inference procedure

# OTHERS
from MAST.lib3d.rigid_mesh_database import MeshDataBase                        # mesh database
import MAST.datasets.bop_object_datasets                                       # load objects model
from MAST.rendering.bullet_batch_renderer import BulletBatchRenderer as render # batched renderer
from MAST.integrated.icp_refiner import ICPRefiner                             # ICP


################## NOTES ##################
# self:
#     inference & evaluation: MAST.scripts.run_cosypose_eval
#     training: MAST.scripts.run_pose_training {coarse, refiner}
# BOP:
#     inference: MAST.scripts.run_bop_inference
#     evaluation: MAST.scripts.run_bop20_eval_multi
#     training: 
#         detector: MAST.scripts.run_detector_training
#         pose estimator: MAST.scripts.run_pose_training {coarse, refiner}


# debug print:
#     print("\n\n!!!!!Warnning: debug@[%s@%s]:\n" % (__file__, sys._getframe().f_lineno), predictions.keys(), "\n\n")

# lmo labels:
#     'maskrcnn_detections/refiner/iteration=4'
#     poses: torch.Size([4622, 4, 4]) torch.float32 cpu
################## NOTES ##################


# input = images_uncropped, obj_mesh_model, K, bbox, obj_label
# supervising labels = TCO_gt, obj_label, gt_bbox, symmetries

def TRAINING_INFERENCE():
     images_uncropped = '480x640 original images'
     K = 'camera intrinsics'
     gt_bbox = '4 floats'
     bbox = '4 floats from mask rcnn'
     obj_label = 'label depicts which category'
     boxes_rend = 'bbox rendered from TCO_input and K'
     boxes_crop = 'bbox slightly enlarged from boxes_rend'
     
     TCO_input = TCO_init_from_boxes(z_range=(1.0, 1.0),gt_bbox or bbox, K)  # @MAST.training.pose_forward_loss.h_pose

     for i in range(iteration_num=4):
          model(images_uncropped, K, obj_label, TCO_input)
          def model(images_uncropped, K, obj_label, TCO_input):  # @MAST.models.pose.PosePredictor.forward
               images_crop, K_crop, boxes_rend, boxes_crop = crop_inputs(images_uncropped, K, TCO_input, obj_label)
               rendered_crop = render(obj_label, TCO_input, K_crop, resolution=(240, 320))
               input = concatenate(images_crop, rendered_crop)
               output = network(input) x
               TCO_output = update_pose(TCO_input, K_crop, output('delta T'))
               TCO_input = TCO_output



# NOTE: bin and delta adaptation for UDA
# for detailed data flow, please see figure in the phone
# 1. bin.detach() + delta ==bin_delta_to_Rts_batch (argmax(bin))==> dR, vxvyvz ==> origin CO_disentangled loss for delta regression
# 2. T_gt + T_in + K_crop ==> gt_dR, gt_vxvyvz ==> gt_bin
# 3. bin + gt_bin ==> soft cross entropy loss