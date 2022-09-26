#!/bin/bash
# https://pytorch.org/docs/1.9.1/distributed.html#distributed-launch

ps -ef | grep "main" | grep -v grep | awk '{print "kill -9 "$2}'

source /opt/conda/bin/activate base

# wordsize here is the numbers of every process in each gpu 
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --loss CenterLoss --optimizer sgd --arch resnet50 --num-classes 5749 --print-freq 10 --lr 0.0001 --multiprocessing-distributed --world-size 1 --rank 0  /mnt/pci-0000:00:1f.2-ata-1-part1/ZLQ/data/TAL-SER/

ps -ef | grep "main" | grep -v grep | awk '{print "kill -9 "$2}'