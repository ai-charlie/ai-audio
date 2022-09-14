#!/bin/bash

if [ $# != 1 ];then
    echo "specify the dir to do the conversion."
    exit
fi


for file in $1/*.mp4
do
    echo "ffmpeg -i  $file -ac 1 -ar 16k $1/$(basename $file .mp4).wav"
    ffmpeg -i  $file -ac 1 -ar 16k $1/$(basename $file .mp4).wav
done