#!/bin/bash  
  
# 定义要遍历的文件夹路径  
folder_path="./data/nerf_llff_data"  
  
# 遍历文件夹并输出文件  
for file in "$folder_path"/*  
do  
    # python scripts/llff2nerf.py data/nerf_llff_data/$(basename $file) --images images_4 --downscale 4
    # rm -r ./log/spaseview/llff/$(basename $file)/ngp
    python main_tensoRF.py data/nerf_llff_data/$(basename $file)/ --workspace ./log/spaseview/llff/$(basename $file)/tensorf -O --iters 5000 --num_rays 4096
done  