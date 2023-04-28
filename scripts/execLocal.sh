#!/bin/bash

if [ -z "$1" ]
then
    echo "please specify task to execute"
    exit 1
fi
if [ -z "$2" ]
then
    echo "please specify # of reqeusts to provide"
    exit 1
fi
if [ -z "$3" ]
then
    echo "please specify batch size"
    exit 1
fi



i=$4

SERVER_PID=$(echo get_server_list | nvidia-cuda-mps-control)
echo set_active_thread_percentage $SERVER_PID $i | nvidia-cuda-mps-control


RES_DIR=$PWD/../resource/ # beware! '/' at the end is required!
BUILD_DIR=$PWD/../bin/
task=$1
req=$2
batch=$3
mean=100



$BUILD_DIR/standalone_inference --task $task --taskfile $RES_DIR/models/$task.pt --requests $req \
--batch $batch --mean $mean --input $RES_DIR/$input_tx --input_config_json $RES_DIR/input_config.json


