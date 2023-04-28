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
res=20


#for i in {1,2,4,8,16,32}
#do

SERVER_PID=$(echo get_server_list | nvidia-cuda-mps-control)
echo set_active_thread_percentage $SERVER_PID $res | nvidia-cuda-mps-control

sleep 1

sudo rm ./data/"$task"_"$batch".ncu-rep

#for i in {1,2,4,8,16,32}
#do

if true
then
#nsys profile -t cuda -s none --export=sqlite --force-overwrite=true -o data/"$task"_"$batch"\
ncu -o data/"$task"_"$batch".ncu-rep --metrics \
    gpu__time_duration.sum,lts__t_sectors.avg.pct_of_peak_sustained_elapsed \
    $BUILD_DIR/inference_wtwarm --task $task --taskfile $RES_DIR/models/$task.pt --requests $req \
    --batch $batch --mean $mean --input $RES_DIR/$input_tx --input_config_json $RES_DIR/input_config.json\
    
fi

sleep 1
#ncu  --metrics \
 #   gpu__time_duration.sum,lts__t_sectors.avg.pct_of_peak_sustained_elapsed \
#$BUILD_DIR/standalone_inference --task $task --taskfile $RES_DIR/models/$task.pt --requests $req \
#--batch $batch --mean $mean --input $RES_DIR/$input_tx --input_config_json $RES_DIR/input_config.json

ncu --import  data/"$task"_"$batch".ncu-rep --csv >  data/"$task"_"$batch".csv
python3 l2cache.py data/"$task"_"$batch".csv $task
#done

