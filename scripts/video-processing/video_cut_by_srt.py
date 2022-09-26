# -*- coding: utf-8 -*-
import os
import jieba

# 获取文件夹下的字幕文件
def file_name(file_dir):
    L = []
    type_List = [".srt"]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in type_List:
                input_file = "" + os.path.join(root, file)
                print(input_file)
                L.append(os.path.join(root, file))
    return L


input_file_dir = "./VIDEO50/srt"
file_list = file_name(input_file_dir)

for filename in file_list:
    with open(filename, encoding='UTF-8') as file_obj:
        mytime = ""
        for line in file_obj:
            line = line.strip()
            if len(line) and not line.isdigit():
                first_str = line[0:1]
                if not first_str.isdigit():
                    # print(line)
                    mytext = mytext + "，" + line


                # ffmpeg
                #视频压缩 ffmpeg -i in -b:v 400k -s 960x540 newfiles/learner-demo.mp4

                command = "ffmpeg -i " + input_file + " -r 15 " + file_dir + "/flips/" + os.path.splitext(file)[
                    0] + vedio_type

                # 执行命令方式一
                d = os.popen(command)
                d.read()
