#!/bin/bash

if [ $# != 1 ];then
    echo "specify the dir to do the calculation."
    exit
fi

output_tmp="$1.tmp"
output="$1.file_duration"

for file in $1/*.wav
do
    dur=`soxi -d ${file}`
    Dur=`soxi -D ${file}`
    echo "$file $dur $Dur" >> ${output_tmp}
done


sort -t ' ' -k3 ${output_tmp} -u > ${output}
rm ${output_tmp}

