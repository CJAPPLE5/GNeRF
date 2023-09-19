#!/bin/bash  

TRAIN_NUM=8

# 定义要遍历的文件夹路径  
folder_path="./data/nerf_llff_data"  

# echo ${TRAIN_NUM}

python scripts/llff2nerf.py data/nerf_llff_data/room --images images_4 --downscale 4 --train_num ${TRAIN_NUM}

# 遍历文件夹并输出文件  
# for file in "$folder_path"/*  
# do  
#     python scripts/llff2nerf.py data/nerf_llff_data/$(basename $file) --images images_4 --downscale 4 --train_num ${TRAIN_NUM}
#     # rm -r ./log/spaseview/llff/$(basename $file)/ngp
#     # rm -r ./log/spaseview/llff/$(basename $file)/view_${TRAIN_NUM}/ngp
#     # python main_nerf.py data/nerf_llff_data/$(basename $file)/ --workspace ./log/spaseview/llff/$(basename $file)/view_${TRAIN_NUM}/ngp -O --iters 2500 --num_rays 4096
# done  