# ========================================================================
# 1.将好未来数据集划分为3类
# ========================================================================
from utils import move
import os, shutil

label_path = 'F:/ZLQ/数据集/TAL_SER语音情感数据集/TAL-SER/label/label.txt'
sex_path = 'F:/ZLQ/数据集/TAL_SER语音情感数据集/TAL-SER/label/utt2gen.txt'
root_path = 'F:/ZLQ/数据集/TAL_SER语音情感数据集/TAL-SER/'

# 读取label.txt
with open(label_path, 'r') as f:
    label_data = f.readlines()
f.close()
ID = []
P = []
A = []
f = 0
for p_a in label_data:
    if f == 0:
        f = 1
        continue
    tempt = p_a.split(' ')
    ID.append(tempt[0])
    P.append(float(tempt[1]))
    A.append(float(tempt[2].strip('\n')))

# 读取utt2gen.txt,即性别标签
with open(sex_path, 'r') as f:
    sex_data = f.readlines()
f.close()
sex_ID = []
sex_label = []
for sex in sex_data:
    tempt = sex.split(' ')
    sex_ID.append(tempt[0])
    sex_label.append(tempt[1].strip('\n'))


# 将对应条件的音频复制到对应文件夹, 条件为P值，A值
def move_to_class():
    for i in range(len(P)):
        print(i)
        if P[i] > 0 and A[i] > 0:
            old_path = root_path + 'all/' + ID[i] + '.wav'
            new_path = root_path + 'all/positive/' + ID[i] + '.wav'
            shutil.copy(old_path, new_path)
        elif P[i] > 0 and A[i] < 0:
            old_path = root_path + 'all/' + ID[i] + '.wav'
            new_path = root_path + 'all/neutral/' + ID[i] + '.wav'
            shutil.copy(old_path, new_path)
        elif P[i] < 0 and A[i] > 0:
            old_path = root_path + 'all/' + ID[i] + '.wav'
            new_path = root_path + 'all/negative/' + ID[i] + '.wav'
            shutil.copy(old_path, new_path)
        # elif P[i] < 0 and A[i] < 0:
        #     old_path = root_path + 'all/' + ID[i] + '.wav'
        #     new_path = root_path + 'all/unpleasant/' + ID[i] + '.wav'
        #     shutil.copy(old_path, new_path)


# 将对应条件的音频复制到对应文件夹, 条件为P值，A值，性别
def move_to_class_f_m():
    positive = [0, 0]
    negative = [0, 0]
    neutral = [0, 0]
    unpleasant = [0, 0]
    for i in range(len(P)):print(i)
        if P[i] > 0 and A[i] > 0:
            if sex_label[i] == 'male' and positive[0] < 200:
                old_path = root_path + 'all/' + ID[i] + '.wav'
                new_path = root_path + 'all/positive/' + ID[i] + '.wav'
                shutil.copy(old_path, new_path)
                positive[0] += 1
            elif sex_label[i] == 'female' and positive[1] < 500:
                old_path = root_path + 'all/' + ID[i] + '.wav'
                new_path = root_path + 'all/positive/' + ID[i] + '.wav'
                shutil.copy(old_path, new_path)
                positive[1] += 1
        elif P[i] > 0 and A[i] < 0:
            if sex_label[i] == 'male' and negative[0] < 200:
                old_path = root_path + 'all/'
        + ID[i] + '.wav'
                new_path = root_path + 'all/negative/' + ID[i] + '.wav'
                shutil.copy(old_path, new_path)
                negative[0] += 1
            elif sex_label[i] == 'female' and negative[1] < 500:
                old_path = root_path + 'all/' + ID[i] + '.wav'
                new_path = root_path + 'all/negative/' + ID[i] + '.wav'
                shutil.copy(old_path, new_path)
                negative[1] += 1
        elif P[i] < 0 and A[i] > 0:
            if sex_label[i] == 'male' and neutral[0] < 200:
                old_path = root_path + 'all/' + ID[i] + '.wav'
                new_path = root_path + 'all/neutral/' + ID[i] + '.wav'
                shutil.copy(old_path, new_path)
                neutral[0] += 1
            elif sex_label[i] == 'female' and neutral[1] < 500:
                old_path = root_path + 'all/' + ID[i] + '.wav'
                new_path = root_path + 'all/neutral/' + ID[i] + '.wav'
                shutil.copy(old_path, new_path)
                neutral[1] += 1
        # elif P[i] < 0 and A[i] < 0:
        #     if sex_label[i] == 'male' and unpleasant[0] < 200:
        #         old_path = root_path + 'all/' + ID[i] + '.wav'
        #         new_path = root_path + 'all/unpleasant/' + ID[i] + '.wav'
        #         shutil.copy(old_path, new_path)
        #         unpleasant[0] += 1
        #     elif sex_label[i] == 'female' and unpleasant[1] < 500:
        #         old_path = root_path + 'all/' + ID[i] + '.wav'
        #         new_path = root_path + 'all/unpleasant/' + ID[i] + '.wav'
        #         shutil.copy(old_path, new_path)
        #         unpleasant[1] += 1


# 将所有的音频文件复制到‘all’这个文件夹
def move_all():
    for dir in os.listdir(root_path):
        if dir[:3] == 'SER':
            for wave in os.listdir(root_path + dir):
                old_path = root_path + dir + '/' + wave
                new_path = root_path + 'all/' + wave
                shutil.copy(old_path, new_path)


# 统计性别数量
def count_f_m():
    positive = [0, 0]
    negative = [0, 0]
    neutral = [0, 0]
    unpleasant = [0, 0]
    for i in range(len(P)):
        if P[i] > 0 and A[i] > 0:
            if sex_label[i] == 'male':
                positive[0] += 1
            else:
                positive[1] += 1
        elif P[i] > 0 and A[i] < 0:
            if sex_label[i] == 'male':
                neutral[0] += 1
            else:
                neutral[1] += 1
        elif P[i] < 0 and A[i] > 0:
            if sex_label[i] == 'male':
                negative[0] += 1
            else:
                negative[1] += 1
        elif P[i] < 0 and A[i] < 0:
            if sex_label[i] == 'male':
                unpleasant[0] += 1
            else:
                unpleasant[1] += 1
    print('positive male:', positive[0], ' positive female:', positive[1])
    print('neutral male:', neutral[0], ' neutral female:', neutral[1])
    print('negative male:', negative[0], ' negative female:', negative[1])
    print('unpleasant male:', unpleasant[0], ' unpleasant female:', unpleasant[1])
    return positive, neutral, negative, unpleasant


move_to_class_f_m()
# count_f_m()
