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
#echo set_active_thread_percentage "$SERVER_PID" 50 | nvidia-cuda-mps-control
for i in {6,12,18,24,30,36,42,48,54,60,66,72,78,84,90,96,100}
do
#{6,30,54,78,100}

echo set_active_thread_percentage "$SERVER_PID" $i | nvidia-cuda-mps-control
sleep 1

output="$task"_l2caches_"$i"
rm ./data/"$output".ncu-rep
ncu -o ./data/"$output" --metrics \
    gpu__time_duration.sum,lts__t_sectors.avg.pct_of_peak_sustained_elapsed \
    $BUILD_DIR/inference_wtwarm --task $task --taskfile $RES_DIR/models/$task.pt --requests $req \
    --batch $batch --mean $mean --input $RES_DIR/$input_tx --input_config_json $RES_DIR/input_config.json
ncu --import data/"$output".ncu-rep --csv > data/"$output".csv

done
sleep 1

#python3 model_l2caches.py data/"$task"_l2caches_6.csv data/"$task"_l2caches_30.csv data/"$task"_l2caches_54.csv data/"$task"_l2caches_78.csv data/"$task"_l2caches_100.csv $task

