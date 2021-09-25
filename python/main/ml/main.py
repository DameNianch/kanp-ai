# -*- coding: utf-8 -*-

import os
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow_hub as hub
import glob
import random
from data_util import load_images

input_dim = [240, 240, 3]  # とりあえずpretrainモデルのドキュメントにあわせた。
output_dim = 2  # 今回はご種類のペットボトル
batch_size = 5  # 説明がめんどくさいので省略。一度に処理するデータ（画像枚）数
epochs = 3  # デバッグ用。本学習では増やす。いっぱいあるほど基本的に良い。過学習に気をつける。
model_path = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/feature_vector/2"

mldata_dir = "mldata/images"
target_class = ["cola", "calpico"]
train_dir = [os.path.join(mldata_dir, "train", i_target) for i_target in target_class]
test_dir = train_dir  # TODO: なんとかする。


class DummyGenerator:
    def __init__(self, target_dir):
        class_vector = np.eye(len(target_dir))
        self._image_paths = []
        self._image_classes = []
        for i, i_dir in enumerate(target_dir):
            for j_path in glob.glob(os.path.join(i_dir, "*")):
                self._image_paths.append(j_path)
                self._image_classes.append(class_vector[i])
        self._image_indices = list(range(len(self._image_paths)))

    def __call__(self, batch_size):
        while True:
            batch_indices = random.choices(self._image_indices, k=batch_size)
            batch_classes = np.array([self._image_classes[i] for i in batch_indices])
            batch_input = load_images([self._image_paths[i] for i in batch_indices])
            yield batch_input, batch_classes


train_generator = DummyGenerator(train_dir)
test_generator = DummyGenerator(test_dir)


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
