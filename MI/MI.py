import numpy as np
import keras as K
import argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import mnist, cifar10, cifar100

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a Dropout model on a image classification task")
    parser.add_argument("--method", type=str, default=None, help="The method of Dropout.")
    parser.add_argument("--epoch", type=int, default=None, help="The train epochs.")

    args = parser.parse_args()

    return args

args = parse_args()

# 加载数据
if args.method == "MNIST":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
elif args.method == "CIFAR10":
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
else:
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

# 处理数据
if args.method == "MNIST":
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
elif args.method == "CIFAR10":
    num_pixels = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
else:
    num_pixels = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# 标准化数据
X_train /= 255
X_test /= 255

# 对标签进行one-hot编码
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# 定义模型
model = Sequential()
model.add(Dense(512, input_dim=num_pixels, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=int(args.epoch), batch_size=200)

# 计算每个神经元互信息
from sklearn.feature_selection import mutual_info_classif
import pandas as pd

# 获取中间层的输出
get_layer_output = K.backend.function([model.layers[0].input],
                                  [model.layers[1].output])
layer_output = get_layer_output([X_train])[0]

# 计算互信息
mi = mutual_info_classif(layer_output, y_train[:,0])
mi_df = pd.DataFrame({'mi': mi})

# 对互信息较低的神经元进行剪枝
low_mi_indices = mi_df[mi_df['mi'] < 0.05].index
pruned_model = Sequential()
pruned_model.add(Dense(512, input_dim=num_pixels, activation='relu'))

# 删除互信息较低的神经元对应的列，并随机初始化一些新的列
old_weights = model.layers[1].get_weights()[0]
old_biases = model.layers[1].get_weights()[1]
new_weights = np.delete(old_weights, low_mi_indices, axis=1)
num_new_cols = old_weights.shape[1] - new_weights.shape[1]
new_cols = np.random.randn(new_weights.shape[0], num_new_cols) * 0.01
new_weights = np.concatenate([new_weights, new_cols], axis=1)

pruned_model.add(Dense(new_weights.shape[1], activation='relu'))
pruned_model.add(Dense(num_classes, activation='softmax'))

# 拷贝训练好的参数
for i, layer in enumerate(pruned_model.layers):
    if i == 0:
        layer.set_weights(model.layers[i].get_weights())
    elif i == 1:
        layer.set_weights([new_weights, old_biases])
    else:
        layer.set_weights(model.layers[i].get_weights())

# 重新编译模型
pruned_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 重新训练
pruned_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)