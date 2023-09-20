#!/bin/bash  

EXP_NAME="supervise_pixel"
NETWORK="ngp"
TRAIN_NUM=8
DATA_BASE="data/nerf_llff_data"
TEST_SCENE="room"
DOWNSCALE=4
RESUME=0

# 测试单场景
 

python scripts/llff2nerf.py ${DATA_BASE}/${TEST_SCENE} --images images_${DOWNSCALE} --downscale ${DOWNSCALE} --train_num ${TRAIN_NUM}

if [ ${RESUME} -eq 0 ]; then  
    rm -r ./log/spaseview/${EXP_NAME}/${DATA_BASE}/${TEST_SCENE}/view_${TRAIN_NUM}/${NETWORK}
fi

python main_nerf.py ${DATA_BASE}/${TEST_SCENE} --workspace ./log/spaseview/${EXP_NAME}/${DATA_BASE}/${TEST_SCENE}/view_${TRAIN_NUM}/${NETWORK} -O --iters 2500 --num_rays 4096

# 测试多场景

# 定义要遍历的文件夹路径  
# folder_path="./data/nerf_llff_data" 

# 遍历文件夹并输出文件  
# for file in ${DATA_BASE}/*  
# do  
#     python scripts/llff2nerf.py ${DATA_BASE}/$(basename $file) --images images_${DOWNSCALE} --downscale ${DOWNSCALE} --train_num ${TRAIN_NUM}
#     if [ ${RESUME} -eq 0 ]; then  
#         rm -r ./log/spaseview/${EXP_NAME}/${DATA_BASE}/$(basename $file)/view_${TRAIN_NUM}/${NETWORK}
#     fi
#     python main_nerf.py ${DATA_BASE}/${TEST_SCENE} --workspace ./log/spaseview/${EXP_NAME}/${DATA_BASE}/$(basename $file)/view_${TRAIN_NUM}/${NETWORK} -O --iters 2500 --num_rays 4096
# done  