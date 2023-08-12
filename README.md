<h1 align="center">Manifold-Aware Self-Training for Unsupervised Domain Adaptation on Regressing 6D Object Pose</h1>

## Installation
```Shell
git clone --recurse-submodules git@github.com:Gorilla-Lab-SCUT/MAST.git
cd MAST
conda env create -n MAST --file environment.yaml
conda activate MAST
python setup.py develop  # install locally
runjob-config MAST/config_env/job-runner-config.yaml  # config runjob
```
## Downloading and preparing data
* create a folder `local_data`
* download [bop_datasets](https://bop.felk.cvut.cz/datasets/)
* download [Linemod_preprocessed](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7)
* download OCCLUSION_LINEMOD
* download [VOCdevkit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
* go to [CosyPose google drive](https://drive.google.com/drive/folders/1JmOYbu1oqN81Dlj2lh6NCAMrC8pEdAtD)
download URDF files and put them in local_data/urdfs, download detector-bop-lmo-pbr--517542 and put it in local_data/experiments

## Training
* step 1: To train on synthesis datasets, using scripts in `train_src.sh`
* step 2: To train on both real and synthesis datasets, using scripts in `train_st.sh`

## Testing
Please see the scripts in `test.sh`

## Acknowledgements
Our implementation leverages the code from [CosyPose](https://github.com/ylabbe/cosypose.git).