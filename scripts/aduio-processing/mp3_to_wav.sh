#!/bin/bash

if [ $# != 1 ];then
    echo "specify the dir to do the conversion."
    exit
fi


for file in $1/*.mp3
do
    echo "sox -b 16 -r 8000 -t mp3  $file -t wav $1/$(basename $file .mp3).wav"
    sox -b 16 -r 8000 -t mp3  $file -t wav $1/$(basename $file .mp3).wav
    # 如果需要剥离声道可以在末尾添加remix 1
    # sox -b 16 -r 8000 -t mp3  $file -t wav $1/$(basename $file .mp3).wav remix 1
done