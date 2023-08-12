python train_rsd.py --gpu_id 0 --src s --tgt n --tradeoff 0.001 --tradeoff2 0.1 | tee ./s2n_0001_01.log
python train_rsd.py --gpu_id 0 --src c --tgt n --tradeoff 0.001 --tradeoff2 0.1 | tee ./c2n_0001_01.log
python train_rsd.py --gpu_id 0 --src s --tgt c --tradeoff 0.001 --tradeoff2 0.1 | tee ./s2c_0001_01.log
python train_rsd.py --gpu_id 0 --src n --tgt c --tradeoff 0.001 --tradeoff2 0.1 | tee ./n2c_0001_01.log
python train_rsd.py --gpu_id 0 --src n --tgt s --tradeoff 0.001 --tradeoff2 0.1 | tee ./n2s_0001_01.log
python train_rsd.py --gpu_id 0 --src c --tgt s --tradeoff 0.001 --tradeoff2 0.01 | tee ./c2s_0001_001.log

python train_MAST.py --gpu_id 0 --src s --tgt n | tee ./s2n.log
python train_MAST.py --gpu_id 0 --src c --tgt n | tee ./c2n.log
python train_MAST.py --gpu_id 0 --src s --tgt c | tee ./s2c.log
python train_MAST.py --gpu_id 0 --src n --tgt c | tee ./n2c.log
python train_MAST.py --gpu_id 0 --src n --tgt s | tee ./n2s.log
python train_MAST.py --gpu_id 0 --src c --tgt s | tee ./c2s.log
