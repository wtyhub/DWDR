
# python train_cvact.py \
# --name='act_vgg_noshare_warm5_8LPN-s-r_lr0.1' \
# --data_dir='/home/wangtyu/datasets/CVACT/train' \
# --warm_epoch=5 \
# --batchsize=16 \
# --use_vgg16 \
# --h=256 \
# --w=256 \
# --fp16 \
# --LPN \
# --lr=0.1 \
# --block=8 \
# --gpu_ids='2'

# python test_cvact.py \
# --name='act_vgg_noshare_warm5_8LPN-s-r_lr0.1' \
# --test_dir='/home/wangtyu/datasets/CVACT/val' \
# --gpu_ids='2'

# LPN resnet50
# python train_cvact.py \
# --name='act_res50_noshare_warm5_8LPN-s-r_lr0.05' \
# --data_dir='/home/wangtyu/datasets/CVACT/train' \
# --warm_epoch=5 \
# --batchsize=16 \
# --h=256 \
# --w=256 \
# --fp16 \
# --LPN \
# --lr=0.05 \
# --block=8 \
# --stride=1 \
# --gpu_ids='2'

# python test_cvact.py \
# --name='act_res50_noshare_warm5_8LPN-s-r_lr0.05' \
# --test_dir='/home/wangtyu/datasets/CVACT/val' \
# --gpu_ids='2'

# python train_cvact.py \
# --name='act_res50_noshare_warm5_lr0.06_block8_decouple_e11_pt_lambd768' \
# --data_dir='/home/lvbinbin/wty/datasets/CVACT/train_pt' \
# --warm_epoch=5 \
# --batchsize=16 \
# --h=112 \
# --w=616 \
# --pt \
# --lr=0.06 \
# --block=8 \
# --stride=1 \
# --pool='lpn' \
# --balance \
# --decouple \
# --only_decouple \
# --e1=1 \
# --e2=1 \
# --fp16 \
# --lambd=0.0013 \
# --dataset='cvact' \
# --gpu_ids='3'

# python test_cvact.py \
# --name='act_res50_noshare_warm5_lr0.06_block8_decouple_e11_pt_lambd768' \
# --test_dir='/home/lvbinbin/wty/datasets/CVACT/val_pt' \
# --gpu_ids='3' 

# backbone vgg16
# python train_cvact.py \
# --name='act_vgg_noshare_warm5_lr0.05' \
# --data_dir='/home/wangtyu/datasets/CVACT/train' \
# --warm_epoch=5 \
# --batchsize=16 \
# --h=256 \
# --w=256 \
# --use_vgg16 \
# --lr=0.05 \
# --dataset='cvact' \
# --gpu_ids='3'

# python test_cvact.py \
# --name='act_vgg_noshare_warm5_lr0.05' \
# --test_dir='/home/wangtyu/datasets/CVACT/val' \
# --gpu_ids='3'

# backbone decouple vgg16
# python train_cvact.py \
# --name='act_vgg_noshare_warm5_lr0.1_decouple_e_0_0' \
# --data_dir='/home/wangtyu/datasets/CVACT/train' \
# --warm_epoch=5 \
# --batchsize=16 \
# --h=256 \
# --w=256 \
# --use_vgg16 \
# --lr=0.1 \
# --balance \
# --decouple \
# --only_decouple \
# --dataset='cvact' \
# --e1=0 \
# --e2=0 \
# --fp16 \
# --gpu_ids='3'

# python test_cvact.py \
# --name='act_vgg_noshare_warm5_lr0.1_decouple_e_0_0' \
# --test_dir='/home/wangtyu/datasets/CVACT/val' \
# --gpu_ids='3'

# swin
python train_cvact.py \
--name='act_swin-b_noshare_warm5_lr0.005_decouple_e11_pt_lambd512' \
--data_dir='/home/wangtingyu/datasets/CVACT/train_pt' \
--warm_epoch=5 \
--batchsize=16 \
--h=224 \
--w=224 \
--pt \
--lr=0.005 \
--stride=1 \
--pool='avg' \
--balance \
--decouple \
--only_decouple \
--e1=1 \
--e2=1 \
--fp16 \
--lambd=0.002 \
--dataset='cvact' \
--swin \
--gpu_ids='0'

python test_cvact.py \
--name='act_swin-b_noshare_warm5_lr0.005_decouple_e11_pt_lambd512' \
--test_dir='/home/wangtingyu/datasets/CVACT/val_pt' \
--gpu_ids='0' 
