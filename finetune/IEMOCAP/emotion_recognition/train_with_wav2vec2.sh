#!/bin/bash
# https://pytorch.org/docs/1.9.1/distributed.html#distributed-launch

ps -ef | grep "train_with_wav2vec2" | grep -v grep | awk '{print "kill -9 "$2}'

source /opt/conda/bin/activate base

# wordsize here is the numbers of every process in each gpu 
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml --data_folder=your_data_folder

ps -ef | grep "train_with_wav2vec2" | grep -v grep | awk '{print "kill -9 "$2}'