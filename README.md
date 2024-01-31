<h1 align="center">Manifold-Aware Self-Training for Unsupervised Domain Adaptation on Regressing 6D Object Pose</h1>

## Installation
```Shell
git clone git@github.com:Gorilla-Lab-SCUT/MAST.git
mkdir deps && cd deps
git clone https://github.com/ylabbe/bop_toolkit_cosypose.git
git clone https://github.com/ylabbe/bullet3.git
git clone https://github.com/ylabbe/job-runner
git clone https://github.com/ylabbe/bop_toolkit_challenge20
cd ..
conda env create -n MAST --file MAST/config_env/environment.yaml
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

## Citation
```bibtex
@inproceedings{ijcai2023p193,
  title     = {Manifold-Aware Self-Training for Unsupervised Domain Adaptation on Regressing 6D Object Pose},
  author    = {Zhang, Yichen and Lin, Jiehong and Chen, Ke and Xu, Zelin and Wang, Yaowei and Jia, Kui},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {1740--1748},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/193},
  url       = {https://doi.org/10.24963/ijcai.2023/193},
}

```
