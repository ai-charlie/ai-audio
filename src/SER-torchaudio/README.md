# README
## 项目文件说明
- README.md
    - 本项目说明书
    
- VIDEO50 文件夹 
    - 一师一优课2019年50个初中语文教学视频
    - 教学视频字幕文件
    - 教学视频音频文件——通过ffmepg 提取的MP3
    - 对字幕进行的词频统计图
  
- video-srt-gui-ffmpeg-0.3.3-x64 文件夹
    - go语言编写的阿里云的智能语音服务
    - 包括对象存储oss服务和语音文件识别
  
- getWAV.py
    - 将VIDEO50文件夹中的视频提取音频，存入AUDIO50文件夹
  
- word_counts.py
    - 对教学视频的字幕进行词云图绘制
  
- simsun.ttc
    word_counts.py中进行词云绘制的字体文件
  
- ffmpeg.exe
    getWAV.py 中进行音频提取的ffmepg软件

# dataset 
- Publicly Available Emotional Speech Dataset (ESD) for Speech Synthesis and Voice Conversion 用于语音合成和语音转换的公开情感语音数据集
    - 这个数据集包含了10个以普通话为母语的人和10个以英语为母语的人所说的350个平行的话语，这些话语有5种情绪状态(中性、快乐、愤怒、悲伤和惊讶)。提供了文字记录。https://github.com/HLTSingapore/Emotional-Speech-Data

```
@inproceedings{zhou2021seen,
  title={Seen and unseen emotional style transfer for voice conversion with a new emotional speech dataset},
  author={Zhou, Kun and Sisman, Berrak and Liu, Rui and Li, Haizhou},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={920--924},
  year={2021},
  organization={IEEE}
}
@article{zhou2021emotional,
title = {Emotional voice conversion: Theory, databases and ESD},
journal = {Speech Communication},
volume = {137},
pages = {1-18},
year = {2022},
issn = {0167-6393}
}
```

    



## 具体实现
- 音频
    - 【数据】语音转文字，形成字幕 （✔）
      - 使用阿里云api 完成了50个视频的字幕
      - 存在一定误差
          - 目前的算法在实际应用中都存在误差，无法解决
          - 实际教学视频数据集不能保证环境完全安静，客观存在噪声
    - 说话人识别/分离
    - 【数据】字幕语言文字标注并分类
      - 标注软件
      -
  - 【数据】好未来教师语音情感数据集
    - 【方法】不确定
  
- 图像
  - 举手识别
  - 站立识别
  - 走动识别
  


## TODO




## ISSUES
- mffc 提取出来是二维向量，如何合并为一维向量
- 