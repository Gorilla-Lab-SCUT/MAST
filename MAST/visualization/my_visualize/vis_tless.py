import torch
from MAST.config import LOCAL_DATA_DIR
from MAST.datasets.datasets_cfg import make_scene_dataset
from MAST.rendering.bullet_scene_renderer import BulletSceneRenderer
from MAST.visualization.singleview import make_singleview_prediction_plots, filter_predictions
from MAST.visualization.singleview import filter_predictions
from bokeh.plotting import gridplot
from bokeh.io import show, output_notebook; output_notebook()
import os
import ipdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import numpy as np
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

result_id = 'bop-pbr--194797/dataset=tless'
ds_name, urdf_ds_name = 'tless.bop19', 'tless.cad'
# pred_key = 'maskrcnn_detections/refiner/iteration=4'
pred_key = 'maskrcnn_detections/coarse/iteration=1'

results = LOCAL_DATA_DIR / 'results' / result_id / 'results.pth.tar'
scene_ds = make_scene_dataset(ds_name)
results = torch.load(results)['predictions']
results[pred_key].infos.loc[:, ['scene_id', 'view_id']].groupby('scene_id').first()

scene_id, view_id = 20, 3

this_preds = filter_predictions(results[pred_key], scene_id, view_id)
renderer = BulletSceneRenderer(urdf_ds_name)
figures = make_singleview_prediction_plots(scene_ds, renderer, this_preds)
renderer.disconnect()
# print(this_preds)

# img = images[0].permute(1,2,0).cpu().numpy()
# x1,y1,x2,y2 = boxes[0].cpu().numpy().astype(int)
# cv2.rectangle(img, (x1,y1), (x2,y2), (1.0,0,0), 5)
# im = Image.fromarray((img*255).astype(np.uint8))
# im.save('./3.jpg')  

show(figures['input_im'])
show(figures['pred_overlay'])