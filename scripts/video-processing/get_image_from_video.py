import subprocess
import os
import subprocess
import shutil


# 目录文件不存在则自动创建,存在则清空并创建
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)


# 视频提取功能
def video_extract(source_video, to_path, speed):
    to_path = to_path + '%05d.jpg'
    strcmd = 'ffmpeg -i "%s" -filter:v "select=not(mod(n\,%d)),setpts=N/(25*TB)" -qscale:v 1 "%s"' % (
    source_video, speed, to_path)
    # strcmd = 'ffmpeg -i %s -ss 1 -f image2 %s'%(source_video,to_path)
    # print(strcmd)
    subprocess.call(strcmd, shell=True)


# 处理流程
def deal_process(src_path, to_base_path, speed):
    if not os.path.exists(src_path):
        return
    if os.path.isdir(src_path):
        for dir in os.listdir(src_path):
            path = os.path.join(src_path, dir)
            deal_process(path, to_base_path, speed)
    else:
        video_name = os.path.splitext(os.path.basename(src_path))[0]
        to_path = os.path.join(to_base_path, video_name) + '/'
        # to_path = os.path.join(to_base_path,video_name+'.jpg')
        check_dir(to_path)
        # to_path = to_path+video_name+'_'
        video_extract(src_path, to_path, speed)


if __name__ == '__main__':
    src_path = r'/VIDEO50/MP4/'  # 原始视频目录
    to_base_path = r'/VIDEO50/images/'  # 抽帧存放目录
    speed = 10  # 视频抽帧间隔   25帧/秒
    deal_process(src_path, to_base_path, speed)