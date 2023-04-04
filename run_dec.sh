# LPN  
# python train_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.003_lpn4_balance_decouple_e11_lambd768_v1' \
# --data_dir='/home/lvbinbin/wty/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --LPN \
# --block=4 \
# --lr=0.003 \
# --fp16 \
# --experiment_name='two_view_long_share_d0.75_256_s1_lr0.003_lpn4_balance_decouple_e11_lambd768_v1' \
# --gpu_ids='1' \
# --g=0.9 \
# --lambd=0.0013 \
# --balance \
# --decouple \
# --e1=1 \
# --e2=1 \
# --seed=3

# python test_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.003_lpn4_balance_decouple_e11_lambd768_v1' \
# --test_dir='/home/lvbinbin/wty/datasets/University-Release/test' \
# --gpu_ids='1'

# Baseline expand ID
# python train_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_google_lr0.01_expand-id' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.01 \
# --experiment_name='two_view_long_share_d0.75_256_s1_google_lr0.01_expand-id' \
# --gpu_ids='4' \
# --fp16 \
# --expand_id \
# --seed=3

# python test_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_google_lr0.01_expand-id' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --gpu_ids='4'

# Baseline expand batchsize 
# python train_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_google_lr0.01_batchsize16' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --batchsize=16 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.01 \
# --experiment_name='two_view_long_share_d0.75_256_s1_google_lr0.01_batchsize16' \
# --gpu_ids='1' \
# --fp16 \
# --normal \
# --seed=3

# python test_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_google_lr0.01_batchsize16' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --gpu_ids='1'

# baseline
python train_dec.py \
--name='two_view_long_share_d0.75_256_s1_lr0.01_balance_intra-inter_decouple_lambd-768_g0.8_e00_lambd1-4000_sacle1-5_0deloss_0.1aug' \
--data_dir='/home/wangtingyu/datasets/University-Release/train' \
--views=3 \
--droprate=0.75 \
--share \
--stride=1 \
--h=256 \
--w=256 \
--lr=0.01 \
--balance \
--decouple \
--e1=0 \
--e2=0 \
--g=0.9 \
--fp16 \
--lambd=0.0013 \
--lambd1=0.00025 \
--scale1=0.1 \
--experiment_name='two_view_long_share_d0.75_256_s1_lr0.01_balance_intra-inter_decouple_lambd-768_g0.8_e00_lambd1-4000_sacle1-5_0deloss_0.1aug' \
--gpu_ids='1' \
--seed=3

python test_mul_gpu.py \
--name='two_view_long_share_d0.75_256_s1_lr0.01_balance_intra-inter_decouple_lambd-768_g0.8_e00_lambd1-4000_sacle1-5_0deloss_0.1aug' \
--test_dir='/homewangtingyu/datasets/University-Release/test' \
--gpu_ids='1'
