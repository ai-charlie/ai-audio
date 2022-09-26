#!/bin/bash

if [ $# != 1 ];then
    echo "specify the dir to do the wave slow conversion."
    exit
fi


for file in $1
do
    if [ "${file##*.}" = "wav" ]; then
        echo "sox $file $file speed 0.9"
        sox $file $file speed 0.9
        # 修正音高
        # sox input.wav output.wav track tempo 1.33 
    fi 
done
