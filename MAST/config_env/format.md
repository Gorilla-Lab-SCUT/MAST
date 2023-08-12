# details of BOP dataset format
## Directory structure
```sh
BOP_dataset_root
├── models (3D object models)
│   ├── ...
│   └── xxx.ply
|
├── models_eval (resampled)
│   ├── ...
│   └── xxx.ply
|
├── train_{object_type} (training images of object ${object_type})
│   ├── rgb/gray (images)
│   │   ├── 000000.png
│   │   ├── ...
│   │   └── xxxxxx.png
│   ├── depth (16-bit unsigned short depth images)
│   │   ├── 000000.png
│   │   ├── ...
│   │   └── xxxxxx.png
│   ├── mask
│   │   ├── 000000_000000.png
│   │   ├── 000000_000001.png
│   │   ├── ...
│   │   └── xxxxxx_yyyyyy.png
│   ├── mask_visib
│   │   ├── 000000_000000.png
│   │   ├── 000000_000001.png
│   │   ├── ...
│   │   └── xxxxxx_yyyyyy.png
│   ├── scene_gt.json
│   ├── scene_gt_info.json (*generate by using `bop_toolkit/scripts/calc_gt_info.py`)
|   └── scene_camera.json
|
|
├── val_{scene_type} (validation images of scene ${scene_type})
│   ├── rgb/gray (images)
│   │   ├── 000000.png
│   │   ├── ...
│   │   └── xxxxxx.png
│   ├── depth (16-bit unsigned short depth images)
│   │   ├── 000000.png
│   │   ├── ...
│   │   └── xxxxxx.png
│   ├── mask
│   │   ├── 000000_000000.png
│   │   ├── 000000_000001.png
│   │   ├── ...
│   │   └── xxxxxx_yyyyyy.png
│   ├── mask_visib
│   │   ├── 000000_000000.png
│   │   ├── 000000_000001.png
│   │   ├── ...
│   │   └── xxxxxx_yyyyyy.png
│   ├── scene_gt.json
│   ├── scene_gt_info.json (*generate by using `bop_toolkit/scripts/calc_gt_info.py`)
|   └── scene_camera.json
|
├── test_{scene_type} (test images of scene ${scene_type})
│   ├── rgb/gray (images)
│   │   ├── 000000.png
│   │   ├── ...
│   │   └── xxxxxx.png
│   ├── depth (16-bit unsigned short depth images)
│   │   ├── 000000.png
│   │   ├── ...
│   │   └── xxxxxx.png
│   ├── mask
│   │   ├── 000000_000000.png
│   │   ├── 000000_000001.png
│   │   ├── ...
│   │   └── xxxxxx_yyyyyy.png
│   ├── mask_visib
│   │   ├── 000000_000000.png
│   │   ├── 000000_000001.png
│   │   ├── ...
│   │   └── xxxxxx_yyyyyy.png
│   ├── scene_gt.json
│   ├── scene_gt_info.json (*generate by using `bop_toolkit/scripts/calc_gt_info.py`)
|   └── scene_camera.json
|
├── camera.json (camera paramters)
├── test_target_bop16.json (A list of test targets used for evaltion)
└── dataset_info.md (dataset specific information)
```

