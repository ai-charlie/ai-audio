# -*- coding: utf-8 -*-
import os
import numpy as np
import csv


# 获取wav文件
def get_file_name(file_dir, lable):
    file_dir = file_dir + lable
    L = []
    type_list = [".wav"]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in type_list:
                input_file = root + "/" + file
                # data = input_file+","+lable
                data =[input_file,lable]
                # print(data)
                # L+=data
                L.append(data)
    return L


def make_rand_train_test_csv(input_file_dir,_train_data_rate, _header):
    data = []
    len_list = []
    for label in label_List:
        ll = get_file_name(input_file_dir, label)
        data += ll
        len_list.append(len(ll))
    data = load_data("data.csv")
    seed = 3
    np.random.seed(seed)
    data_length = len(data)
    print("data_len:", data_length)
    # 训练集测试集划分
    train_length = int(data_length * _train_data_rate)

    # 按类别比例随机生成训练集
    start=0
    train_indices=set()
    for i in len_list:
        a = np.arange(start, start+i, 1)
        start += i
        b = np.random.choice(a, int(i*_train_data_rate), replace=False)
        # 合并集合set
        train_indices=train_indices|set(b)
        #print(train_indices)
    test_indices = np.array(list(set(range(data_length)) - set(train_indices)))
    train_indices = np.array(list(set(range(data_length)) - set(test_indices)))

    print("train_length:", train_length,len(train_indices))
    print("train_indices: ", len(train_indices), train_indices)
    print("test_indices:", len(test_indices))
    print("test_indices:", len(test_indices), test_indices)

    # 生成训练数据集
    with open("train.csv", "w", newline='', encoding='UTF-8') as f:
        writer = csv.writer(f)
        writer.writerow(_header)  # 第一行为header标签行
        writer.writerows(np.array(data)[train_indices])
        f.close()

    # 生成测试数据集
    with open("test.csv", "w", newline='', encoding='UTF-8')as a_test:
        writer = csv.writer(a_test)
        writer.writerow(_header)  # 第一行为header标签行
        writer.writerows(np.array(data)[test_indices])
        a_test.close()


def write_data_csv(_input_file_dir_,header, _label_List_):
    data=[]
    for label in _label_List_:
        ll = get_file_name(_input_file_dir_, label)
        data+=ll
    with open("data.csv", "w", newline='', encoding='UTF-8') as file_obj:
        writer = csv.writer(file_obj)
        # 写入头
        writer.writerow(header)
        i=1
        for data_i in data:
            # 写入数据
            data_i = [i,]+data_i
            writer.writerow(data_i)
            i+=1
        file_obj.close()


def load_data(a_csv_file):
    data = []
    with open(a_csv_file)as afile:
        a_reader = csv.reader(afile)  # 从原始数据集中将所有数据读取出来并保存到a_reader中
        labels = next(a_reader)  # 提取第一行设置为labels
        for row in a_reader:  # 将a_reader中每一行的数据提取出来并保存到data的列表中
            if row == "\n":pass
            data.append(row)
            # print(data)
    return data


if __name__ == '__main__':
    header = ["", "path", "emotion"]
    input_file_dir = "E:/PyCharmProject/Speech-Emotion-Recognition-master/datastet/hwl/"
    label_List = ["negative", "positive", "neutral"]
    train_data_rate = 0.7
    wav_file_list = []
    write_data_csv(input_file_dir, header, label_List)
    make_rand_train_test_csv(input_file_dir,train_data_rate, header)
    # load_data("data.csv")
