#!/bin/bash

function random(){
    for i in $@; do
        echo "$i $RANDOM";
    done | sort -k2n | cut -d " " -f1;
}


function checkGPU(){
    for gpu in $(random $@); do
        gpu_stat=($(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu));
        if [ $gpu_stat -lt 1000 ]
        then
            return $gpu;
        fi;
    done;
    return 255;
}


function sub(){
    result=""
    for a in $1; do
        if echo $2 | grep -wq $a 
        then
            :
        else
            result="$result $a";
        fi
    done;
    echo $result
}

## main

USE_GPU=(0 1 2 3);
Tasks=(TASK-1 TASK-2 TASK-3)
running_gpu=""
running_pid=""

for TASK in "${Tasks[@]}"; do
    while [ 1 == 1 ];
    do
        check_gpu_id=($( sub "$( echo "${USE_GPU[@]}" )" "$( echo "${running_gpu[@]}" )" ))
        if [ ${#check_gpu_id[@]} -gt 0 ]
        then
            checkGPU $( echo "${check_gpu_id[@]}" )
            avaible_gpu=$?
            if [ $avaible_gpu != 255 ]
            then
                running_gpu="$running_gpu $avaible_gpu"
                echo running_gpu: $running_gpu;
                # main task
                echo $TASK $avaible_gpu &
                bash ./test.sh $TASK $avaible_gpu > ${TASK}.log &   # $1, $2 in tesh.sh for $TASK, $avaible_gpu
                #
                running_pid="$running_pid $!"
                echo running_pid: $running_pid;
                break;
            fi
        fi
        
        idx=0
        tmp_gpu=$running_gpu;
        for pid in $running_pid; do
            kill -s 0 $pid 2>/dev/null;  # 2>/dev/null silence error
            if [ $? == 1 ]
            then
                stop_gpu=($tmp_gpu);
                running_gpu=$( sub "$( echo "${running_gpu[@]}" )" "${stop_gpu[$idx]}" )
                running_pid=$( sub "$( echo "${running_pid[@]}" )" "$pid" )
                let idx+=1;
            fi
        done;

        sleep 1;
    done;
done;

: 'example outputs:
running_gpu: 3
running_pid: 36660
TASK-1 3
running_gpu: 3 2
running_pid: 36660 36680
TASK-2 2
running_gpu: 3 2 0
running_pid: 36660 36680 36700
TASK-3 0
'