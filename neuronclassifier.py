#!/usr/bin/env python
from skimage import io
import keras
from keras.models import Sequential, load_model
from keras.layers import (Dense, Dropout, Activation, Flatten,
                          Conv3D, MaxPooling3D)
import argparse


class NeuronModel:
    """
    """
    def __init__(self, input_shape=(8, 16, 16, 1)):
        self.input_shape = input_shape

        self._model = Sequential()
        self._model.add(Conv3D(16, (1, 5, 5), padding='valid',
                               input_shape=input_shape))
        self._model.add(Activation('relu'))
        self._model.add(Conv3D(16, (3, 3, 3), padding='valid'))
        self._model.add(Activation('relu'))
        self._model.add(Conv3D(16, (1, 5, 5), padding='valid'))
        self._model.add(Activation('relu'))
        self._model.add(Conv3D(16, (3, 3, 3), padding='valid'))
        self._model.add(Activation('relu'))
        self._model.add(Flatten())
        self._model.add(Dense(16))
        self._model.add(Activation('relu'))
        self._model.add(Dense(1, activation='sigmoid'))
        self._model.summary()
        self._model.compile(loss='binary_crossentropy', optimizer='Adam',
                            metrics=['accuracy'])

    def fit(self, generator, steps_per_epoch=32):
        self._model.fit_generator(generator, steps_per_epoch,
                                  epochs=100,
                                  validation_split=0.1,
                                  class_weight={0: 1, 1: 1.5},
                                  verbose=1)

    def evaluate(self, generator, steps_per_epoch):
        self._model.evaluate_generator(generator, steps_per_epoch)

    def load(self, filepath):
        self._model.load_model(filepath)

    def save(self, filepath):
        self._model.save(filepath)

    def generator(self, inputs_filenames, targets_filenames,
                  rotation_range=0, brightness_range=0,
                  contrast_range=0):
        while True:
            for inputs_filename, targets_filename in zip(inputs_filenames,
                                                         targets_filenames):
                inputs = self.load_tif(inputs_filename)
                targets = self.load_tif(targets_filename)
                bounding_box_size = self.input_shape[0:3]
                for box in self.boxes(targets, bounding_box_size):
                    ([x1, y1, z1], [x2, y2, z2]) = box
                    subtargets = targets[z1:z2, y1:y2, x1:x2]
                    mean = subtargets.mean()
                    if mean < 0.95 and mean > 0.05:
                        subtargets = targets[z1:z2, y1:y2, x1:x2]
                        subinputs = inputs[z1:z2, y1:y2, x1:x2]
                        yield (subinputs, subtargets)

    def load_tif(self, filename, dtype=None):
        if dtype is None:
            return io.imread(filename)
        else:
            return io.imread(filename).astype(dtype)

    def boxes(self, data, bounding_box_size):
        [x, y, z] = [0, 0, 0]
        (L, H, W) = bounding_box_size
        while True:
            x += W
            if x + W > data.shape[2]:
                x = 0
                y += H
            if y + H > data.shape[1]:
                y = 0
                z += L
            if z + L > data.shape[0]:
                z = 0
                raise StopIteration
            yield ([x, y, z], [x+W, y+H, z+L])


def test():
    TRAINING_INPUTS = ['training/inputs.tif']
    TRAINING_TARGETS = ['training/targets.tif']

    TEST_INPUTS = ['test/inputs.tif']
    TEST_TARGETS = ['test/targets.tif']

    nn = NeuronModel()

    train_generator = nn.generator(TRAINING_INPUTS, TRAINING_TARGETS)
    nn.fit(train_generator)

    test_generator = nn.generator(TEST_INPUTS, TEST_TARGETS)
    nn.evaluate(test_generator)

    nn.save('model.h5')


def main():
    test()


if __name__ == '__main__':
    main()
