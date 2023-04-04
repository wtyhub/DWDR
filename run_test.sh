#  python test_cvact.py \
#  --name='act_vgg_noshare_warm5_lr0.05_decouple_e_1_1' \
#  --test_dir='/home/wangtingyu/datasets/CVACT/val' \
#  --gpu_ids='1'

# python test.py \
# --name='three_view_long_share_d0.75_256_s1_google_PCB4_lr0.001' \
# --test_dir='/home/wangtyu/datasets/University-Release/test' \
# --batchsize=64 \
# --gpu_ids='1'

python test_mul_gpu.py \
--name='two_view_long_share_d0.75_256_s1_lr0.01_decouple_balance_lambda768_e11' \
--test_dir='/home/wangtingyu/datasets/University-Release/test' \
--gpu_ids='1'

# python test_mul_gpu_pca.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.01_decouple_balance_lambd768_g0.9_alpha1_1_v1' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --gpu_ids='5'