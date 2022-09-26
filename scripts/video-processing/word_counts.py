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


input_file_dir = "/home/tx/AI/data/2019-M-CHN/srt/"
output_file_dir ='/home/tx/AI/data/2019-M-CHN/png/'
file_list = file_name(input_file_dir)

for filename in file_list:
# filename = "./VIDEO50/2019juniorChinese1.srt"
    with open(filename, encoding='UTF-8') as file_obj:
        mytext = ""
        for line in file_obj:
            line = line.strip()
            if len(line) and not line.isdigit():
                first_str = line[0:1]
                if not first_str.isdigit():
                    # print(line)
                    mytext = mytext + "，" + line

    # 分词
    import jieba
    mytext = " ".join(jieba.cut(mytext))

    # 绘制词云
    from wordcloud import WordCloud
    wordcloud = WordCloud(font_path="simsun.ttc").generate(mytext)
    # %pylab inline

    from matplotlib import pyplot as plt
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(output_file_dir+filename.split('.')[0]+".png")