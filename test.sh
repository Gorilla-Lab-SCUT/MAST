id=01  # 01 02 04 05 06 08 09 10 11 12 13 14 15
ds=LM  # LM LMO LMO_BOP HB YCBV
run_id=bop-lmo-pbr-coarse-transnoise-zxyavg-decoupled-bindelta-icg-selft/$id
for epoch in 0
do
CUDA_VISIBLE_DEVICES=0 \
python -m MAST.testing.inference\
    --config bop-pbr\
    --ds $ds\
    --obj_id $id\
    --coarse_run_id $run_id\
    --coarse_epoch $epoch\
    --refiner_epoch 660
done

# LM
for epoch in 0
do
CUDA_VISIBLE_DEVICES=0 \
python -m MAST.testing.lm_evaluation \
--results_id bop-pbr-bop-lmo-pbr-coarse-transnoise-zxyavg-decoupled-bindelta-icg-selft/$id \
--coarse_epoch $epoch \
--refiner_epoch 660 \
--obj_id $id
done

# Homebrewed
for epoch in 0
do
CUDA_VISIBLE_DEVICES=0 \
python -m MAST.testing.hb_evaluation \
--results_id bop-pbr-bop-lmo-pbr-coarse-transnoise-zxyavg-decoupled-bindelta-icg/$id \
--coarse_epoch $epoch \
--refiner_epoch 660
done

# LMO
for epoch in 0
do
CUDA_VISIBLE_DEVICES=0 \
python -m MAST.testing.lmo_evaluation \
--results_id bop-pbr-bop-lmo-pbr-coarse-transnoise-zxyavg-decoupled-bindelta-icg-selft/$id \
--coarse_epoch $epoch \
--refiner_epoch 660
done

# LMO-BOP 200 images
for epoch in 0
do
CUDA_VISIBLE_DEVICES=0 \
python -m MAST.testing.lmo_bop_evaluation \
--results_id bop-pbr-bop-lmo-pbr-coarse-transnoise-zxyavg-decoupled-bindelta-icg/$id \
--coarse_epoch $epoch \
--refiner_epoch 660
done

# YCBV
for epoch in 0
do
CUDA_VISIBLE_DEVICES=0 \
python -m MAST.testing.ycbv_evaluation \
--results_id bop-pbr-bop-ycbv-pbr-coarse-transnoise-zxyavg-decoupled-bindelta-icg \
--coarse_epoch $epoch \
--refiner_epoch $epoch
done
