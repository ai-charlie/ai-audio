# ========================================================================
# 1.
# ========================================================================
from tensorflow.keras.layers import Flatten, Conv1D, \
    Activation, BatchNormalization
from tensorflow.keras.layers import LSTM as KERAS_LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Permute, Multiply
from tensorflow.python.keras.models import Model
import keras
from .dnn import DNN


def attention_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(1, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


class CNN_LSTM(DNN):
    def __init__(self, model: Sequential, trained: bool = False) -> None:
        super(CNN_LSTM, self).__init__(model, trained)

    @classmethod
    def make(
        cls,
        input_shape: int,
        n_kernels: int,
        kernel_sizes: int,
        rnn_size: int,
        hidden_size: int,
        dropout: float = 0.5,
        n_classes: int = 6,
        lr: float = 0.001
    ):
        """
        搭建模型

        Args:
            input_shape (int): 特征维度
            rnn_size (int): LSTM 隐藏层大小
            hidden_size (int): 全连接层大小
            dropout (float, optional, default=0.5): dropout
            n_classes (int, optional, default=6): 标签种类数量
            lr (float, optional, default=0.001): 学习率
        """
        model = Sequential()
        # model.add(Conv1D(filters=128, kernel_size=5, padding='same', input_shape=(1, input_shape)))  # 卷积层
        # model.add(BatchNormalization(axis=-1))
        # model.add(Activation('relu'))
        # # model.add(Conv1D(filters=128, kernel_size=5, padding='same', input_shape=(1, input_shape)))  # 卷积层
        # # model.add(BatchNormalization(axis=-1))
        # # model.add(Activation('relu'))
        # model.add(KERAS_LSTM(rnn_size))  # (time_steps = 1, n_feats)
        # model.add(Dropout(dropout))
        # model.add(Dense(rnn_size, activation='relu'))
        # # model.add(Dense(rnn_size, activation='tanh'))
        #
        # model.add(Dense(n_classes, activation='softmax'))  # 分类层
        # optimzer = Adam(lr=lr)
        # model.compile(loss='categorical_crossentropy', optimizer=optimzer, metrics=['accuracy'])
        # return cls(model)

        input = Input([1, input_shape])
        x = Conv1D(filters=128, kernel_size=5, padding='same')(input)  # 卷积层
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = Conv1D(filters=128, kernel_size=5, padding='same')(x)  # 卷积层
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        # x = attention_block(x)
        x = KERAS_LSTM(rnn_size, return_sequences= True)(x)  # (time_steps = 1, n_feats)
        x = attention_block(x)
        x = Flatten()(x)
        x = Dropout(dropout)(x)
        x = Dense(rnn_size, activation='relu')(x)
        # x = Dense(rnn_size, activation='tanh')(x)

        x = Dense(n_classes, activation='softmax')(x)  # 分类层
        model.add(Model(input, x))
        optimzer = Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimzer, metrics=['accuracy'])
        return cls(model)

    def reshape_input(self, data: np.ndarray) -> np.ndarray:
        """二维数组转三维"""
        # (n_samples, n_feats) -> (n_samples, time_steps = 1, input_size = n_feats)
        # time_steps * input_size = n_feats
        data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
        return data
