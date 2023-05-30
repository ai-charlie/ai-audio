from tensorflow.keras.utils import to_categorical
import extract_feats.opensmile as of
import extract_feats.librosa as lf
import models
from utils import parse_opt


def train(config):
    """
    训练模型

    Args:
        config: 配置项

    Returns:
        model: 训练好的模型
    """

    # 加载被 preprocess.py 预处理好的特征
    max_val_acc = 0
    if config.feature_method == 'o':
        x_train, x_test, y_train, y_test = of.load_feature(config, train=True)

    elif config.feature_method == 'l':
        x_train, x_test, y_train, y_test = lf.load_feature(config, train=True)

    # x_train, x_test (n_samples, n_feats)
    # y_train, y_test (n_samples)

    # 搭建模型
    model = models.make(config=config, n_feats=x_train.shape[1])
    # print('************************************************')
    print(model.model.summary())
    # print('************************************************')
    # 训练模型
    print('----- start training', config.model, '-----')
    if config.model in ['lstm', 'cnn1d', 'cnn2d', 'cnn_lstm']:
        y_train, y_val = to_categorical(y_train), to_categorical(y_test)  # 独热编码
        max_val_acc = model.train(
            x_train, y_train,
            x_test, y_val,
            batch_size=config.batch_size,
            n_epochs=config.epochs
        )
    else:
        model.train(x_train, y_train)
    print('----- end training ', config.model, ' -----')

    # 验证模型
    model.evaluate(x_test, y_test)
    # 保存训练好的模型
    model.save(config.checkpoint_path, config.checkpoint_name)

    return max_val_acc


if __name__ == '__main__':
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    acc = 0
    max_acc = 0
    train_num = 1.0
    for i in range(int(train_num)):
        config = parse_opt()
        tempt = train(config)
        acc += tempt
        if max_acc < tempt:
            max_acc = tempt
    print('验证集平均准确率：', acc/train_num)
    print('10次最高准确率：', max_acc)
