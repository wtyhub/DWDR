# lpn vgg16
#python train_cvusa.py \
#--name='usa_vgg_noshare_warm5_8LPN-s-r_lr0.1' \
#--data_dir='/home/wangtyu/datasets/CVUSA/train' \
#--warm_epoch=5 \
#--batchsize=16 \
#--h=256 \
#--w=256 \
#--use_vgg16 \
#--fp16 \
#--LPN \
#--lr=0.1 \
#--block=8 \
#--gpu_ids='3'
#
#python test_cvusa.py \
#--name='usa_vgg_noshare_warm5_8LPN-s-r_lr0.1' \
#--test_dir='/home/wangtyu/datasets/CVUSA/val' \
#--gpu_ids='3' \

# lpn resnet50 pt
python train_cvusa.py \
--name='usa_res50_noshare_warm5_lr0.07_block8_decouple_e11_pt_lambd768' \
--data_dir='/home/wangtingyu/datasets/CVUSA/train_pt' \
--warm_epoch=5 \
--batchsize=16 \
--h=112 \
--w=616 \
--pt \
--lr=0.07 \
--block=8 \
--stride=1 \
--pool='lpn' \
--balance \
--decouple \
--only_decouple \
--e1=1 \
--e2=1 \
--fp16 \
--lambd=0.0013 \
--gpu_ids='2'

python test_cvusa.py \
--name='usa_res50_noshare_warm5_lr0.07_block8_decouple_e11_pt_lambd768' \
--test_dir='/home/wangtingyu/datasets/CVUSA/val_pt' \
--gpu_ids='2' \

# baseline vgg16
# python train_cvusa.py \
# --name='usa_vgg_noshare_warm5_lr0.02_balance_decouple' \
# --data_dir='/home/wangtingyu/datasets/CVUSA/train' \
# --warm_epoch=5 \
# --batchsize=16 \
# --h=256 \
# --w=256 \
# --use_vgg16 \
# --lr=0.02 \
# --balance \
# --decouple \
# --gpu_ids='5'

# python test_cvusa.py \
# --name='usa_vgg_noshare_warm5_lr0.02_balance_decouple' \
# --test_dir='/home/wangtingyu/datasets/CVUSA/val' \
# --gpu_ids='5'