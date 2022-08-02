##  sudo /usr/local/MATLAB/R2017b/bin/matlab
# EDSR baseline model (x2) + JPEG augmentation
#CUDA_VISIBLE_DEVICES=0 python main.py --model EDSR --scale 2 --patch_size 48 --batch_size 4  --n_feats 256 --n_resblocks 32 --save EDSRx2 --reset --epochs 100 --data_range 1-800/801-801
#    --pre_train /home/shiyanshi/项目代码/雷鹏程/遥感论文/遥感论文实验/experiment/对比实验/EDSR5_256x2/model/model_best.pt
#CUDA_VISIBLE_DEVICES=1 python main.py --model EDSR --scale 3 --patch_size 144  --n_feats 256 --n_resblocks 5 --save EDSR5_retrain_256x3 --reset --epochs 100 \
#    --pre_train /home/shiyanshi/项目代码/雷鹏程/遥感论文/遥感论文实验/experiment/对比实验/EDSR5_256x3/model/model_best.pt
#CUDA_VISIBLE_DEVICES=0 python main.py --model EDSR --scale 4 --patch_size 192  --n_feats 256 --n_resblocks 5 --save 111EDSR5_retrain_256x4 --reset --epochs 100 \
#    --pre_train /home/shiyanshi/项目代码/雷鹏程/遥感论文/遥感论文实验/experiment/对比实验/EDSR5_256x4/model/model_best.pt

#CUDA_VISIBLE_DEVICES=0 python main.py --model MSRN --scale 2 --patch_size 96  --save MSRNx2 --reset --epochs 200
#CUDA_VISIBLE_DEVICES=0 python main.py --model MSRN --scale 3 --patch_size 144   --save MSRNx3 --reset --epochs 200
#CUDA_VISIBLE_DEVICES=0 python main.py --model MSRN --scale 4 --patch_size  192  --save MSRNx4 --reset  --epochs 200

#--pre_train /home/shiyanshi/项目代码/雷鹏程/遥感论文/遥感论文实验/experiment/OURS_10_sa_naturex2/model/model_best.pt

#CUDA_VISIBLE_DEVICES=1 python main.py --model CARN --scale 2 --patch_size 96   --save CARNepoch200x2 --reset --epochs 200
#CUDA_VISIBLE_DEVICES=1 python main.py --model CARN --scale 3 --patch_size 144   --save CARNepoch200x3 --reset --epochs 200
#python main.py --model CARN --scale 4 --patch_size 192   --save CARN_epoch_800x4 --reset \
#   --epochs 800  --decay 300-500-600-700

#python main.py --model DRRN --scale 2 --patch_size 48   --save DRRNx2 --reset  --epochs 200
#python main.py --model DRRN --scale 3 --patch_size 48   --save DRRNx3 --reset  --epochs 200
#python main.py --model DRRN --scale 4 --patch_size  48   --save DRRNx4 --reset  --epochs 200
#
#python main.py --model VDSROUR --scale 2 --patch_size 48   --save VDSRx2 --reset  --epochs 200
#python main.py --model VDSROUR --scale 3 --patch_size 48   --save VDSRx3 --reset  --epochs 200
#python main.py --model VDSROUR --scale 4 --patch_size  48   --save VDSRx4 --reset  --epochs 200

#
#CUDA_VISIBLE_DEVICES=0 python main.py --model SRCNN --scale 2 --patch_size 48   --save SRCNNx2 --reset  --epochs 200
#CUDA_VISIBLE_DEVICES=0 python main.py --model SRCNN --scale 3 --patch_size 48   --save SRCNNx3 --reset  --epochs 200
#CUDA_VISIBLE_DEVICES=0 python main.py --model SRCNN --scale 4 --patch_size  48   --save SRCNNx4 --reset  --epochs 200
#
#CUDA_VISIBLE_DEVICES=0 python main.py --model RCAN --scale 2 --patch_size 96   --save RCANx2 --reset \
#--epochs 800  --decay 300-500-600-700  --data_test Set5 --batch_size 16
#CUDA_VISIBLE_DEVICES=0 python main.py --model IDN --scale 3 --patch_size 144   --save IDNx3 --reset --epochs 200
#CUDA_VISIBLE_DEVICES=0 python main.py --model IDN --scale 4 --patch_size  192   --save IDNx4 --reset --epochs 200

# RDN BI model (x2)
#CUDA_VISIBLE_DEVICES=0 python3.6 main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 4  --patch_size 48 --reset --data_range 1-800/801-801
# RDN BI model (x3)
#CUDA_VISIBLE_DEVICES=1 python3.6 main.py --scale 3 --save RDN_D4C8G64_BIx3 --model RDN --epochs 200 --batch_size 16  --patch_size 144 --reset
# RDN BI model (x4)
#CUDA_VISIBLE_DEVICES=1 python3.6 main.py --scale 4 --save RDN_D4C8G64_BIx4 --model RDN --epochs 200 --batch_size 16  --patch_size 192 --reset


### 选项---expend_channel
## efficient 版本
 python main.py --model GFFNET_M --scale 2 --patch_size 96  --n_resblocks 4  \
--epochs 800  --decay 300-500-600-700 --save GFFNET4_L_epoch800x2 --reset --data_test DIV2K --batch_size 16
#CUDA_VISIBLE_DEVICES=0  python main.py --model GFFNET_M --scale 3 --patch_size 144  --n_resblocks 16  --epochs 800  --decay 300-500-600-700 --save GFFNET16_L_epoch800x3 --reset
#CUDA_VISIBLE_DEVICES=0  python main.py --model GFFNET_M --scale 4 --patch_size 192 --n_resblocks 16  --epochs 800  --decay 300-500-600-700 --save GFFNET16_L_epoch800x4 --reset

## full_train 版本
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 2 --patch_size 96  --n_resblocks 12 --save GFFNET12_epoch800x2 --reset  --expend_channel 256  \
#   --epochs 800  --decay 300-500-600-700
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 3 --patch_size 144  --n_resblocks 12 --save GFFNET12_epoch800x3 --reset  --expend_channel 256  \
#   --epochs 800  --decay 300-500-600-700
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 4 --patch_size 192  --n_resblocks 10 --save GFFNET10_L_GFF_epoch800x4 --reset  --expend_channel 128  \
#   --epochs 800  --decay 300-500-600-700

#CUDA_VISIBLE_DEVICES=0 python main.py --model SRFBN --scale 2 --patch_size 96 --batch_size 8   --save SRFBNx2 --reset  \
#   --epochs 200  --decay 200 --data_test DIV2K

#CUDA_VISIBLE_DEVICES=0 python main.py --model GSCN --scale 2 --patch_size 96  --n_resblocks 6 --n_resgroups 6  --save GSCN_r6g6_x2 --reset   \
#   --epochs 100  --data_test DIV2K  --batch_size 8

#python main.py --model GSCN --scale 2 --patch_size 96  --n_resblocks 6 --n_resgroups 6  --save GSCN_r6g6_CA_SAx2 --reset   \
#   --epochs 200   --data_test DIV2K  --batch_size 8
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 3 --patch_size 144  --n_resblocks 12 --save GFFNET12_epoch800x3 --reset  --expend_channel 256  \
#   --epochs 800  --decay 300-500-600-700
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 4 --patch_size 192  --n_resblocks 10 --save GFFNET10_L_GFF_epoch800x4 --reset  --expend_channel 128  \
#   --epochs 800  --decay 300-500-600-700



## 残差块实验
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 3 --patch_size 144  --n_resblocks 4 --save GFFNET4x3 --reset  --expend_channel 256  \
#   --epochs 100
#
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 3 --patch_size 144  --n_resblocks 8 --save GFFNET8x3 --reset  --expend_channel 256  \
#   --epochs 100
#
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 4 --patch_size 192  --n_resblocks 4 --save GFFNET4x4 --reset  --expend_channel 256  \
#   --epochs 100
#
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 4 --patch_size 192  --n_resblocks 8 --save GFFNET8x4 --reset  --expend_channel 256  \
#   --epochs 100
#
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 2 --patch_size 96  --n_resblocks 16 --save GFFNET16x2 --reset  --expend_channel 256  \
#   --epochs 100
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 3 --patch_size 144  --n_resblocks 16 --save GFFNET16x3 --reset  --expend_channel 256  \
#   --epochs 100
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 4 --patch_size 192  --n_resblocks 16 --save GFFNET16x4 --reset  --expend_channel 256  \
#   --epochs 100





#CUDA_VISIBLE_DEVICES=1 python main.py --model TEMP_M --scale 2 --patch_size 96  --n_resblocks 6 --save GFFNET6_onepathx2 --reset  --expend_channel 128  \
#   --epochs 800  --decay 300-500-600-700
# CUDA_VISIBLE_DEVICES=0 python main.py --model TEMP_M --scale 3 --patch_size 144  --n_resblocks 6 --save GFFNET6_onepathx3 --reset  --expend_channel 128  \
#   --epochs 800  --decay 300-500-600-700
# CUDA_VISIBLE_DEVICES=0 python main.py --model TEMP_M --scale 4 --patch_size 192  --n_resblocks 6 --save GFFNET6_onepathx4 --reset  --expend_channel 128  \
#   --epochs 800  --decay 300-500-600-700


#CUDA_VISIBLE_DEVICES=0 python main.py --model TEMP --scale 2 --patch_size 96  --n_resblocks 36 --save GSCN36_GFFx2 --reset  --expend_channel 256  \
#   --epochs 200  --batch_size 8 --decay 300-500-600-700

#CUDA_VISIBLE_DEVICES=0 python main.py --model TEMP --scale 2 --patch_size 96  --n_resblocks 40 --save GFFNET40_WDSR_retrainx2 --reset  --expend_channel 256  \
#    --pre_train /home/shiyanshi/项目代码/雷鹏程/遥感论文/遥感论文实验/experiment/GFFNET40_WDSRx2/model/model_latest.pt  \
#   --epochs 350   --decay 200-250-300
#CUDA_VISIBLE_DEVICES=0 python main.py --model TEMP --scale 3 --patch_size 144  --n_resblocks 40 --save GFFNET40_WDSRx3 --reset  --expend_channel 256  \
#   --epochs 800  --decay 300-500-600-700
#CUDA_VISIBLE_DEVICES=0 python main.py --model EDSR --scale 4 --patch_size 192  --n_resblocks 20 --save GFFNET40_EDSRx4 --reset  --expend_channel 256  \
#   --epochs 800  --decay 300-500-600-700 --n_feats 64

#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 4 --patch_size 192  --n_resblocks 12 --save GFFNET12_WDSRx4 --reset  --expend_channel 256  \
#   --epochs 200

#python main.py --model DDBPN --scale 4 --patch_size 128  --save DDBPNx4 --reset \
#   --epochs 300  --decay 200