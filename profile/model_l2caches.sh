#!/bin/bash
if [ -z "$1" ]
then
    echo "please specify task to execute"
    exit 1
fi

RES_DIR=$PWD/../resource/ # beware! '/' at the end is required!
BUILD_DIR=$PWD/../bin/
task=$1
req=1
batch=1
mean=100
res=100

SERVER_PID=$(echo get_server_list | nvidia-cuda-mps-control)

sleep 1

sleep 1
echo set_active_thread_percentage "$SERVER_PID" 10 | nvidia-cuda-mps-control
sleep 1
output1="$task"_l2caches_1_10
rm ./data/"$output1".ncu-rep
ncu -o ./data/"$output1" --metrics \
    gpu__time_duration.sum,lts__t_sectors.avg.pct_of_peak_sustained_elapsed \
    $BUILD_DIR/inference_wtwarm --task $task --taskfile $RES_DIR/models/$task.pt --requests $req \
    --batch $batch --mean $mean --input $RES_DIR/$input_tx --input_config_json $RES_DIR/input_config.json
ncu --import data/"$output1".ncu-rep --csv > data/"$output1".csv

sleep 1
echo set_active_thread_percentage "$SERVER_PID" 50 | nvidia-cuda-mps-control
sleep 1
output2="$task"_l2caches_16_50
rm ./data/"$output2".ncu-rep
ncu -o ./data/"$output2" --metrics \
    gpu__time_duration.sum,lts__t_sectors.avg.pct_of_peak_sustained_elapsed \
    $BUILD_DIR/inference_wtwarm --task $task --taskfile $RES_DIR/models/$task.pt --requests $req \
    --batch $batch --mean $mean --input $RES_DIR/$input_tx --input_config_json $RES_DIR/input_config.json
ncu --import data/"$output2".ncu-rep --csv > data/"$output2".csv

sleep 1
echo set_active_thread_percentage "$SERVER_PID" 100 | nvidia-cuda-mps-control
sleep 1
output3="$task"_l2caches_32_100
rm ./data/"$output3".ncu-rep
ncu -o ./data/"$output3" --metrics \
    gpu__time_duration.sum,lts__t_sectors.avg.pct_of_peak_sustained_elapsed \
    $BUILD_DIR/inference_wtwarm --task $task --taskfile $RES_DIR/models/$task.pt --requests $req \
    --batch $batch --mean $mean --input $RES_DIR/$input_tx --input_config_json $RES_DIR/input_config.json
ncu --import data/"$output3".ncu-rep --csv > data/"$output3".csv


python3 model_l2caches.py data/"$output1".csv data/"$output2".csv data/"$output3".csv $task
