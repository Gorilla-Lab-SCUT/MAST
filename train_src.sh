CUDA_VISIBLE_DEVICES=0 \
runjob --ngpus=4 \
python -m MAST.scripts.run_pose_training\
    --resume bop-lmo-pbr-coarse-transnoise-zxyavg-decoupled-bindelta-icg\
    --icg_loss\
    --epoch 0
    # --config bop-lmo-pbr-coarse\
    # --resume bop-lmo-pbr-coarse-transnoise-zxyavg-decoupled-bindelta-icg/01\
