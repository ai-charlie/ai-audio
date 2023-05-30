# ========================================================================
# 1.处理CASIA_database的数据，让它能被使用
# ========================================================================
from utils import remove, move, rename, mkdirs, waveform, spectrogram


path = 'CASIA_database/'
names = ['liuchanhg', 'wangzhe', 'zhaoquanyin', 'ZhaoZuoxiang']
emotions = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def remove_not_wav():
    for name in names:
        for emotion in emotions:
            remove(path + name + '/' + emotion)


def rename_all():
    for name in names:
        for emotion in emotions:
            rename(path + name + '/' + emotion)


def move_all():
    for name in names:
        for emotion in emotions:
            move(path + name + '/' + emotion)


def mkdir_all():
    for emotion in emotions:
        mkdirs(path + emotion)

