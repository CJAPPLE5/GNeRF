#!/bin/bash  

TEST_PREFIX="srun -p aigc-video --nodes 1 --ntasks-per-node 1 --cpus-per-task 2 --gres=gpu:1 --quotatype=auto"
SUBMIT_PREFIX="srun -p aigc-video --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 --gres=gpu:8"

function train_raw(){
    EXP_NAME="raw"
    NETWORK="ngp"
    TRAIN_NUM=-1
    DATASET="nerf_llff_data"
    DATA_BASE="data/${DATASET}"
    TEST_SCENE="room"
    DOWNSCALE=4
    RESUME=0
    
    $TEST_PREFIX python scripts/llff2nerf.py ${DATA_BASE}/${TEST_SCENE} --images images_${DOWNSCALE} --downscale ${DOWNSCALE} --train_num ${TRAIN_NUM}

    if [ ${RESUME} -eq 0 ]; then  
        rm -r ./log/spaseview/${EXP_NAME}/${DATASET}/${TEST_SCENE}/view_${TRAIN_NUM}/${NETWORK}
    fi

    $TEST_PREFIX python main_nerf.py ${DATA_BASE}/${TEST_SCENE} \
    --workspace ./log/spaseview/${EXP_NAME}/${DATASET}/${TEST_SCENE}/view_${TRAIN_NUM}/${NETWORK} \
    -O --iters 10000 --num_rays 4096

}
function train_ibr(){
    EXP_NAME="supervise_pixel"
    NETWORK="ngp"
    TRAIN_NUM=9
    DATASET="nerf_llff_data"
    DATA_BASE="data/${DATASET}"
    TEST_SCENE="room"
    DOWNSCALE=4
    RESUME=0    
    $TEST_PREFIX python scripts/llff2nerf.py ${DATA_BASE}/${TEST_SCENE} --images images_${DOWNSCALE} --downscale ${DOWNSCALE} --train_num ${TRAIN_NUM}

    if [ ${RESUME} -eq 0 ]; then  
        rm -r ./log/spaseview/${EXP_NAME}/${DATASET}/${TEST_SCENE}/view_${TRAIN_NUM}_ibr/${NETWORK}
    fi

    $TEST_PREFIX python main_nerf.py ${DATA_BASE}/${TEST_SCENE} --workspace ./log/spaseview/${EXP_NAME}/${DATASET}/${TEST_SCENE}/view_${TRAIN_NUM}_ibr/${NETWORK} -O --iters 2500 --num_rays 4096 --use_ibr

}
function test_ibr(){
    # add root path
    export PYTHONPATH=.:$PYTHONPATH
    EXP_NAME="raw"
    NETWORK="ngp"
    TRAIN_NUM=-1
    DATASET="nerf_llff_data"
    DATA_BASE="data/${DATASET}"
    TEST_SCENE="room"
    DOWNSCALE=4
    RESUME=0 
    $TEST_PREFIX python testing/test_ibrnet.py ${DATA_BASE}/${TEST_SCENE} --workspace ./log/spaseview/${EXP_NAME}/${DATASET}/${TEST_SCENE}/view_${TRAIN_NUM}_ibr/${NETWORK} \
    -O --iters 2500 --num_rays 4096\
    --ckpt log/spaseview/raw/nerf_llff_data/room/view_-1/ngp/checkpoints/ngp_ep0286.pth \
    --use_ibr

}
function data(){
    EXP_NAME="supervise_pixel"
    NETWORK="ngp"
    TRAIN_NUM=7
    DATASET="nerf_llff_data"
    DATA_BASE="data/${DATASET}"
    TEST_SCENE="room"
    DOWNSCALE=4
    RESUME=0    
    $TEST_PREFIX python scripts/llff2nerf.py ${DATA_BASE}/${TEST_SCENE} --images images_${DOWNSCALE} --downscale ${DOWNSCALE} --train_num ${TRAIN_NUM}
}
$1