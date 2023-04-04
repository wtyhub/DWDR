## baseline + DWDR ##
# python train_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.01_decouple_balance_lambda768_e11' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.01 \
# --balance \
# --decouple \
# --fp16 \
# --e1=1 \
# --e2=1 \
# --g=0.9 \
# --lambd=0.0013 \
# --experiment_name='two_view_long_share_d0.75_256_s1_lr0.01_decouple_balance_lambda768_e11' \
# --gpu_ids='1' \

# python test_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.01_decouple_balance_lambda768_e11' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --gpu_ids='1'

## baseline + Barlow Twins ##
python train_mul_gpu.py \
--name='two_view_long_share_d0.75_256_s1_lr0.01_decouple_balance_lambda768_e00' \
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
--fp16 \
--e1=0 \
--e2=0 \
--g=0.9 \
--lambd=0.0013 \
--experiment_name='two_view_long_share_d0.75_256_s1_lr0.01_decouple_balance_lambda768_e00' \
--gpu_ids='0' \

python test_mul_gpu.py \
--name='two_view_long_share_d0.75_256_s1_lr0.01_decouple_balance_lambda768_e00' \
--test_dir='/home/wangtingyu/datasets/University-Release/test' \
--gpu_ids='0'

## LPN + DWDR ##
# python train_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.003_lpn4_balance_decouple_e11_lambd768' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
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
# --experiment_name='two_view_long_share_d0.75_256_s1_lr0.003_lpn4_balance_decouple_e11_lambd768' \
# --gpu_ids='1' \
# --g=0.9 \
# --lambd=0.0013 \
# --balance \
# --decouple \
# --e1=1 \
# --e2=1 \

# python test_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.003_lpn4_balance_decouple_e11_lambd768' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --gpu_ids='1'

## swin-b + DWDR ##
# python train_mul_gpu.py \
# --name='two_view_swin-b_long_share_d0.75_224_s1_lr0.005_balance_decouple_lambda512_e11_g0.9' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --share \
# --stride=1 \
# --h=224 \
# --w=224 \
# --lr=0.005 \
# --balance \
# --decouple \
# --e1=1 \
# --e2=1 \
# --g=0.9 \
# --fp16 \
# --lambd=0.002 \
# --experiment_name='two_view_swin-b_long_share_d0.75_224_s1_lr0.005_balance_decouple_lambda512_e11_g0.9' \
# --gpu_ids='0' \
# --swin 

# python test_mul_gpu.py \
# --name='two_view_swin-b_long_share_d0.75_224_s1_lr0.005_balance_decouple_lambda512_e11_g0.9' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --gpu_ids='0'



# Baseline expand ID
# python train_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_google_lr0.01_expand-3id_batch16' \
# --data_dir='/home/lvbinbin/wty/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.01 \
# --experiment_name='two_view_long_share_d0.75_256_s1_google_lr0.01_expand-3id_batch16' \
# --gpu_ids='3' \
# --fp16 \
# --expand_id \
# --seed=3 \
# --batchsize=16 

# python test_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_google_lr0.01_expand-3id_batch16' \
# --test_dir='/home/lvbinbin/wty/datasets/University-Release/test' \
# --gpu_ids='3'

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



## Baseline + satellite based sampling ##
# python train_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.01_batch8_sat_leading' \
# --data_dir='/home/lvbinbin/wty/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.01 \
# --balance \
# --e1=1 \
# --e2=1 \
# --g=0.9 \
# --fp16 \
# --lambd=0.0013 \
# --experiment_name='two_view_long_share_d0.75_256_s1_lr0.01_batch8_sat_leading' \
# --gpu_ids='3' \
# --sat_lead \
# --seed=3 \
# --batchsize=8

# python test_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.01_batch8_sat_leading' \
# --test_dir='/home/lvbinbin/wty/datasets/University-Release/test' \
# --gpu_ids='3'

# train swin
# python train_mul_gpu.py \
# --name='two_view_swin-b_long_share_d0.75_224_s1_lr0.005' \
# --data_dir='/home/lvbinbin/wty/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --share \
# --stride=1 \
# --h=224 \
# --w=224 \
# --lr=0.005 \
# --e1=1 \
# --e2=1 \
# --g=0.9 \
# --fp16 \
# --lambd=0.0013 \
# --experiment_name='two_view_swin-b_long_share_d0.75_224_s1_lr0.005' \
# --gpu_ids='3' \
# --swin \
# --seed=3

# python test_mul_gpu.py \
# --name='two_view_swin-b_long_share_d0.75_224_s1_lr0.005' \
# --test_dir='/home/lvbinbin/wty/datasets/University-Release/test' \
# --gpu_ids='3'



# train half channel dwdr
# python train_mul_gpu_half.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.01_balance_decouple_lambda256_e11_g0.9_0.5channel' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.01 \
# --balance \
# --decouple \
# --e1=1 \
# --e2=1 \
# --g=0.9 \
# --fp16 \
# --lambd=0.0039 \
# --experiment_name='two_view_long_share_d0.75_256_s1_lr0.01_balance_decouple_lambda256_e11_g0.9_0.5channel' \
# --gpu_ids='0' \
# --seed=3 

# python test_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.01_balance_decouple_lambda256_e11_g0.9_0.5channel' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --gpu_ids='0'
