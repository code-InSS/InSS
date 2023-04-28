#!/bin/bash

if [ -z "$1" ]
then
    echo "please specify task to execute"
    exit 1
fi

RES_DIR=$PWD/../resource/ # beware! '/' at the end is required!
BUILD_DIR=$PWD/../bin/
task=$1
req=100
batch=8
mean=100
res=20


#for i in {1,2,4,8,16,32}
#do

SERVER_PID=$(echo get_server_list | nvidia-cuda-mps-control)
echo set_active_thread_percentage $SERVER_PID $res | nvidia-cuda-mps-control

sleep 1


a=1
while(($a<=5))
do
    nsys profile -t cuda -s none --export=sqlite --force-overwrite=true -o data/2060s_sysidletime_"$a" ./conperf $a $task  $batch
    let a+=1
done

python3 sepper.py sys $task
#$BUILD_DIR/standalone_inference --task $task --taskfile $RES_DIR/models/$task.pt --requests $req \
#--batch $batch --mean $mean --input $RES_DIR/$input_tx --input_config_json $RES_DIR/input_config.json\
    

