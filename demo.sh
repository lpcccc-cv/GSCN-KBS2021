
### ---expend_channel
## efficient version
 python main.py --model GFFNET_M --scale 2 --patch_size 96  --n_resblocks 4  \
--epochs 800  --decay 300-500-600-700 --save GFFNET4_L_epoch800x2 --reset --data_test DIV2K --batch_size 16
#CUDA_VISIBLE_DEVICES=0  python main.py --model GFFNET_M --scale 3 --patch_size 144  --n_resblocks 16  --epochs 800  --decay 300-500-600-700 --save GFFNET16_L_epoch800x3 --reset
#CUDA_VISIBLE_DEVICES=0  python main.py --model GFFNET_M --scale 4 --patch_size 192 --n_resblocks 16  --epochs 800  --decay 300-500-600-700 --save GFFNET16_L_epoch800x4 --reset

## full_train version
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 2 --patch_size 96  --n_resblocks 12 --save GFFNET12_epoch800x2 --reset  --expend_channel 256  \
#   --epochs 800  --decay 300-500-600-700
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 3 --patch_size 144  --n_resblocks 12 --save GFFNET12_epoch800x3 --reset  --expend_channel 256  \
#   --epochs 800  --decay 300-500-600-700
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 4 --patch_size 192  --n_resblocks 10 --save GFFNET10_L_GFF_epoch800x4 --reset  --expend_channel 128  \
#   --epochs 800  --decay 300-500-600-700


#CUDA_VISIBLE_DEVICES=0 python main.py --model GSCN --scale 2 --patch_size 96  --n_resblocks 6 --n_resgroups 6  --save GSCN_r6g6_x2 --reset   \
#   --epochs 100  --data_test DIV2K  --batch_size 8

#python main.py --model GSCN --scale 2 --patch_size 96  --n_resblocks 6 --n_resgroups 6  --save GSCN_r6g6_CA_SAx2 --reset   \
#   --epochs 200   --data_test DIV2K  --batch_size 8
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 3 --patch_size 144  --n_resblocks 12 --save GFFNET12_epoch800x3 --reset  --expend_channel 256  \
#   --epochs 800  --decay 300-500-600-700
#CUDA_VISIBLE_DEVICES=0 python main.py --model GFFNET --scale 4 --patch_size 192  --n_resblocks 10 --save GFFNET10_L_GFF_epoch800x4 --reset  --expend_channel 128  \
#   --epochs 800  --decay 300-500-600-700


