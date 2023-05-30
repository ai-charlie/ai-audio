# ========================================================================
# 1.用于建立并训练cnn_lstm
# ========================================================================
import os

from tensorflow.keras.utils import to_categorical
import extract_feats.opensmile as of
import extract_feats.librosa as lf
import tensorflow as tf
import models
from utils import parse_opt
from tensorflow.keras.layers import Flatten, Conv1D, \
    Activation, BatchNormalization
from tensorflow.keras.layers import LSTM as KERAS_LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Permute, Multiply
from tensorflow.python.keras.models import Model, load_model
from sklearn.metrics import f1_score, recall_score, precision_score


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return


def reshape_input(data: np.ndarray) -> np.ndarray:
    """二维数组转三维"""
    # (n_samples, n_feats) -> (n_samples, time_steps = 1, input_size = n_feats)
    # time_steps * input_size = n_feats
    data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
    return data


def attention_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(1, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def cnn_lstm(input_shape: int,
             rnn_size: int,
             dropout: float = 0.5,
             n_classes: int = 3,
             lr: float = 0.001):
    model = Sequential()
    input = Input([1, input_shape])
    x = Conv1D(filters=128, kernel_size=5, padding='same')(input)  # 卷积层
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=128, kernel_size=5, padding='same')(x)  # 卷积层
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # x = attention_block(x)
    x = KERAS_LSTM(rnn_size, return_sequences=True)(x)  # (time_steps = 1, n_feats)
    x = attention_block(x)
    x = Flatten()(x)
    x = Dropout(dropout)(x)
    x = Dense(rnn_size, activation='relu')(x)
    # x = Dense(rnn_size, activation='tanh')(x)

    x = Dense(n_classes, activation='softmax')(x)  # 分类层
    model.add(Model(input, x))
    optimzer = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimzer, metrics=['accuracy'])
    return model


def train(config):
    # 加载被 preprocess.py 预处理好的特征
    x_train, x_test, y_train, y_test = lf.load_feature(config, train=True)

    # 搭建模型
    model = cnn_lstm(input_shape=x_train.shape[1], rnn_size=config.rnn_size)
    # print('************************************************')
    # print(model.model.summary())
    # print('************************************************')
    # 训练模型
    print('----- start training', config.model, '-----')
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # 独热编码

    x_train, x_test = reshape_input(x_train), reshape_input(x_test)
    ck_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('checkpoints', 'CNN_LSTM_LIBROSA_hwl.h5'),
                                                     # monitor='val_f1',
                                                     monitor='val_accuracy',
                                                     # mode='max', verbose=2,
                                                     save_best_only=True)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', profile_batch=0)
    history = model.fit(
        x_train, y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        shuffle=True,  # 每个 epoch 开始前随机排列训练数据
        validation_data=(x_test, y_test),
        # callbacks=[Metrics(valid_data=(x_test, y_test)),
        #            ck_callback,
        #            tb_callback]
        callbacks=[ck_callback,
                   tb_callback]
    )

    # 训练集上的损失和准确率
    acc = history.history["accuracy"]
    loss = history.history["loss"]
    # 验证集上的损失和准确率
    val_acc = history.history["val_accuracy"]
    val_loss = history.history["val_loss"]

    # 绘制acc和loss图
    # curve(acc, val_acc, "Accuracy", "acc")
    # curve(loss, val_loss, "Loss", "loss")

    max_val_acc = max(val_acc)
    print('验证集最高准确率：', max_val_acc)
    print('----- end training ', config.model, ' -----')

    # 验证模型
    model = load_model(os.path.join('checkpoints', 'CNN_LSTM_LIBROSA_hwl.h5'))
    loss, test_acc = model.evaluate(x_test, y_test)
    print('测试集准确率：', test_acc)
    return max_val_acc


def fineTune(config):
    # 加载被 preprocess.py 预处理好的特征
    x_train, x_test, y_train, y_test = lf.load_feature(config, train=True)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # 独热编码
    x_train, x_test = reshape_input(x_train), reshape_input(x_test)

    # 加载预训练好的模型
    model = load_model(os.path.join('checkpoints', 'CNN_LSTM_LIBROSA_hwl.h5'))
    loss, test_acc = model.evaluate(x_test, y_test)
    print('测试集准确率：', test_acc)

    ck_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('checkpoints', 'CNN_LSTM_LIBROSA_hwl.h5'),
                                                     # monitor='val_f1',
                                                     monitor='val_accuracy',
                                                     # mode='max', verbose=2,
                                                     save_best_only=True)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', profile_batch=0)
    history = model.fit(
        x_train, y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        shuffle=True,  # 每个 epoch 开始前随机排列训练数据
        validation_data=(x_test, y_test),
        # callbacks=[Metrics(valid_data=(x_test, y_test)),
        #            ck_callback,
        #            tb_callback]
        callbacks=[ck_callback,
                   tb_callback]
    )
    # 验证集上的准确率
    val_acc = history.history["val_accuracy"]
    max_val_acc = max(val_acc)
    print('验证集最高准确率：', max_val_acc)
    model = load_model(os.path.join('checkpoints', 'CNN_LSTM_LIBROSA_hwl.h5'))
    loss, test_acc = model.evaluate(x_test, y_test)
    print('训练后测试集准确率：', test_acc)
    return max_val_acc


if __name__ == '__main__':
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    acc = 0
    max_acc = 0
    train_num = 10.0
    for i in range(int(train_num)):
        config = parse_opt()
        # tempt = train(config)
        tempt = fineTune(config)
        acc += tempt
        if max_acc < tempt:
            max_acc = tempt
    print('验证集平均准确率：', acc / train_num)
    print('10次最高准确率：', max_acc)
