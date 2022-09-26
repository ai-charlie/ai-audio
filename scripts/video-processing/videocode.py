# -*- coding: utf-8 -*-
import os


# 获取视频文件
def file_name(file_dir, vedio_type):
    L = []
    Video_type_List = [".mp4", ".m4v"]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in Video_type_List:
                input_file = ""+os.path.join(root, file)
                print(input_file)
                L.append(os.path.join(root, file))
                # ffmpeg
                #视频压缩 ffmpeg -i in -b:v 400k -s 960x540 newfiles/learner-demo.mp4

                command = "ffmpeg -i " + input_file + " -r 15 " + file_dir + "/smallmp4/" + os.path.splitext(file)[
                    0] + vedio_type

                # 执行命令方式一
                d = os.popen(command)
                d.read()

                # 执行命令方式二
                # v = os.chdir(command)
                # res = os.popen(v.testcomman)
                # tempstream = res._stream
                # print(tempstream.buffer.read().decode(encoding='utf-8', errors ='ignore'))

                # 命令执行方式三
                # 从mkv视频中提取字幕文件
                # cmd = "ffmpeg" + ' -i ' + infile + ' -map 0:s:0 ' + outfile
                # subprocess.call(cmd, shell=True)

                #os.system("pause")
    return L


input_file_dir = "./VIDEO50/srt"
output_audio_type = ".mp4"
video_file_list = file_name(input_file_dir, output_audio_type)


