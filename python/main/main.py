## python3 main.py

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np

input_dim = 26  # 28-2
output_dim = 1  # 価格
batch_size = 5  # 説明がめんどくさいので省略。一度に処理するデータ（画像枚）数
epochs = 3  # デバッグ用。本学習では増やす。いっぱいあるほど基本的に良い。過学習に気をつける。


class DummyGenerator:
    """
    https://www.meti.go.jp/press/2021/07/20210716001/20210716001.html
    ここらへんを読みましょう。
    https://developers.goalist.co.jp/entry/keras-fit-generator
    """

    def __init__(self, file_path):
        self._file_path = file_path

    def generate(self, batch_size):
        count = 0
        while count < 100:
            count += 1
            # ここでfile_pathから必要なデータを引っ張り出す
            # 入力のaugmentを入れましょう。
            yield np.ones((batch_size, input_dim)), np.ones((batch_size, output_dim))


train_generator = DummyGenerator("train.csv")
test_generator = DummyGenerator("test.csv")


def mame_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim))
    return model


model = mame_model(input_dim, output_dim)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
history = model.fit_generator(
    train_generator.generate(batch_size),
    epochs=epochs,     # エポック数の指定
    verbose=1,         # ログ出力の指定. 0だとログが出ない
    validation_steps=4,
    validation_data=test_generator.generate(batch_size)
)

score = model.evaluate_generator(test_generator.generate(batch_size), steps=3, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
