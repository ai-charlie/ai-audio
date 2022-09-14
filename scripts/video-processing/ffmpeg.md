# FFMPEG

## 官方文档 

[https://ffmpeg.org/documentation.html](https://ffmpeg.org/documentation.html)

### Video and Audio file format conversion
[https://ffmpeg.org/ffmpeg.html#Video-and-Audio-file-format-conversion](https://ffmpeg.org/ffmpeg.html#Video-and-Audio-file-format-conversion)

1. mp3 转 wav (默认格式)
    ```bash
    ffmpeg -i xxx.mp3 -f wav xxx.wav
    ```
1. mp3 转 pcm （采样率16000hz，分辨率16bits，单声道）
    ```bash
    ffmpeg -i xxx.mp3 -acodec pcm_s16le -f s16le -ac 1 -ar 16000 xxx.pcm 
    ```

1. mp3 转 ogg
    ```bash
    ffmpeg -i xxx.mp3 -acodec libvorbis -ab 128k xxx.ogg
    ```

1. mp4 转 wav
    ```bash
    ffmpeg -i  xxx.mp4 -ac 1 -ar 16k xxx.wav
    ```