import os
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from ..base import BaseModel
from utils import curve
import tensorflow as tf
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


class DNN(BaseModel, ABC):
    """
    所有基于 Keras 的深度学习模型的基类

    Args:
        n_classes (int): 标签种类数量
        lr (float): 学习率
    """

    def __init__(self, model: Sequential, trained: bool = False) -> None:
        super(DNN, self).__init__(model, trained)
        print(self.model.summary())

    def save(self, path: str, name: str) -> None:
        """
        保存模型

        Args:
            path (str): 模型路径
            name (str): 模型文件名
        """
        h5_save_path = os.path.join(path, name + ".h5")
        self.model.save_weights(h5_save_path)

        save_json_path = os.path.join(path, name + ".json")
        with open(save_json_path, "w") as json_file:
            json_file.write(self.model.to_json())

    @classmethod
    def load(cls, path: str, name: str):
        """
        加载模型

        Args:
            path (str): 模型路径
            name (str): 模型文件名
        """
        # 加载 json
        model_json_path = os.path.abspath(os.path.join(path, name + ".json"))
        json_file = open(model_json_path, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # 加载权重
        model_path = os.path.abspath(os.path.join(path, name + ".h5"))
        model.load_weights(model_path)

        return cls(model, True)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: int = 32,
        n_epochs: int = 20
    ) -> None:
        """
        训练模型

        Args:
            x_train (np.ndarray): 训练集样本
            y_train (np.ndarray): 训练集标签
            x_val (np.ndarray, optional): 测试集样本
            y_val (np.ndarray, optional): 测试集标签
            batch_size (int): 批大小
            n_epochs (int): epoch 数
        """
        if x_val is None or y_val is None:
            x_val, y_val = x_train, y_train

        x_train, x_val = self.reshape_input(x_train), self.reshape_input(x_val)
        ck_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('checkpoints', 'CNN_LSTM_LIBROSA_CASIA'),
                                                         monitor='val_f1',
                                                         mode='max', verbose=2,
                                                         save_best_only=True)
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', profile_batch=0)
        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            shuffle=True,  # 每个 epoch 开始前随机排列训练数据
            validation_data=(x_val, y_val),
            callbacks=[Metrics(valid_data=(x_val, y_val)),
                       ck_callback,
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

        self.trained = True
        return max_val_acc

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """
        预测音频的情感

        Args:
            samples (np.ndarray): 需要识别的音频特征

        Returns:
            results (np.ndarray): 识别结果
        """
        # 没有训练和加载过模型
        if not self.trained:
            raise RuntimeError("There is no trained model.")

        samples = self.reshape_input(samples)
        return np.argmax(self.model.predict(samples), axis=1)

    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        """
        预测音频的情感的置信概率

        Args:
            samples (np.ndarray): 需要识别的音频特征

        Returns:
            results (np.ndarray): 每种情感的概率
        """
        if not self.trained:
            raise RuntimeError('There is no trained model.')

        if hasattr(self, 'reshape_input'):
            samples = self.reshape_input(samples)
        return self.model.predict(samples)[0]

    @abstractmethod
    def reshape_input(self):
        pass
