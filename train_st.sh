ds=LM # LMO
CUDA_VISIBLE_DEVICES=0 \
python -m MAST.self_training.run_pose_self_training\
    --resume bop-lmo-pbr-coarse-transnoise-zxyavg-decoupled-bindelta-icg-selft/01\
    --icg_loss\
    --ds $ds\
    --epoch 0
