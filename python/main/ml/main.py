import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow_hub as hub

input_dim = [240, 240, 3]  # とりあえずpretrainモデルのドキュメントにあわせた。
output_dim = 5  # 今回はご種類のペットボトル
batch_size = 5  # 説明がめんどくさいので省略。一度に処理するデータ（画像枚）数
epochs = 3  # デバッグ用。本学習では増やす。いっぱいあるほど基本的に良い。過学習に気をつける。
model_path = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/feature_vector/2"


class DummyGenerator:
    def __init__(self, file_path):
        self._file_path = file_path

    def __call__(self, batch_size):
        while True:
            # ここでfile_pathから必要なデータを引っ張り出す
            # 入力のaugmentを入れましょう。
            yield np.ones((batch_size, *input_dim)), np.tile(np.arange(output_dim) / np.sum(np.arange(output_dim)), (batch_size, 1))


train_generator = DummyGenerator("train.csv")
test_generator = DummyGenerator("test.csv")


def mame_model(output_dim):
    model = keras.Sequential()
    model.add(hub.KerasLayer(model_path, trainable=True))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(output_dim, activation='softmax'))
    return model


model = mame_model(output_dim)
model.build([None, *input_dim])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
history = model.fit_generator(
    train_generator(batch_size),
    epochs=epochs,     # エポック数の指定
    steps_per_epoch=10,
    verbose=1,         # ログ出力の指定. 0だとログが出ない
    validation_steps=4,
    validation_data=test_generator(batch_size)
)

score = model.evaluate(test_generator(batch_size), steps=3, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
