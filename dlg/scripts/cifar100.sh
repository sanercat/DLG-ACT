#CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='cifar100' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=500. --epsilon=0.01
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='cifar100' --defense='none'
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='cifar100' --defense='dp' --scale=1e-3
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='cifar100' --defense='cp' --percent_num=90
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='cifar100' --defense='soteria' --percent_num=60 --layer_num=32
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='cifar100' --defense='ats' --aug_list='21-13-3+7-4-15+1-2-5-8-10'
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='cifar100' --defense='precode' --precode_size=256 --beta=1e-3