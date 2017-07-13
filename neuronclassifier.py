from skimage import io
import keras
from keras.models import Sequential, load_model
from keras.layers import (Dense, Dropout, Activation, Flatten,
                          Conv3D, MaxPooling3D)
from keras import backend
import argparse
import numpy as np


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
        self._model.compile(loss='binary_crossentropy',
                            optimizer='Adam',
                            metrics=['accuracy'])

    def fit(self, generator, steps_per_epoch=64):
        return self._model.fit_generator(generator, steps_per_epoch,
                                         epochs=50,
                                         class_weight={0: 1, 1: 1.5},
                                         verbose=2)

    def evaluate(self, generator, steps):
        return self._model.evaluate_generator(generator, steps)

    def load(self, filepath):
        self._model.load_model(filepath)

    def save(self, filepath):
        self._model.save(filepath)


def generator(inputs_filenames, targets_filenames,
              bounding_box_size,
              rotation_range=0, brightness_range=0,
              contrast_range=0, background_value=1,
              neuron_value=0):
    while True:
        for inputs_filename, targets_filename in zip(inputs_filenames,
                                                     targets_filenames):
            inputs = load_tif(inputs_filename)
            inputs = inputs.reshape(1, inputs.shape[0],
                                    inputs.shape[1],
                                    inputs.shape[2], 1)
            targets = load_tif(targets_filename)
            targets = targets.reshape(1, targets.shape[0],
                                      targets.shape[1],
                                      targets.shape[2], 1)
            (H, W, L) = bounding_box_size
            [x, y, z] = [0, 0, 0]

            for neuron_coord in coords(targets, value=neuron_value):
                [x, y, z] = neuron_coord
                [x1, y1, z1] = [x - L/2, y - W/2, z - H/2]
                [x2, y2, z2] = [x + L/2, y + W/2, z + H/2]
                bounding_coords = [[x1, y1, z1], [x2, y2, z2]]
                if in_bounds(targets, bounding_coords):
                    subinputs = inputs[:, z1:z2, y1:y2, x1:x2, :]
                    subtargets = np.array([neuron_value])
                    yield (subinputs, subtargets)
                    neuron_neighborhood = targets[:, (z-2):(z+2),
                                                  (y-2):(y+2), (x-2):(x+2), :]
                    for background_coord in coords(neuron_neighborhood,
                                                   value=background_value):
                        [x, y, z] = background_coord
                        [x1, y1, z1] = [x - L/2, y - W/2, z - H/2]
                        [x2, y2, z2] = [x + L/2, y + W/2, z + H/2]
                        bounding_coords = [[x1, y1, z1], [x2, y2, z2]]
                        if in_bounds(targets, bounding_coords):
                            subinputs = inputs[:, z1:z2, y1:y2, x1:x2, :]
                            subtargets = np.array([background_value])
                            yield (subinputs, subtargets)


def in_bounds(data, bounding_coords):
    [x1, y1, z1] = bounding_coords[0]
    [x2, y2, z2] = bounding_coords[1]
    in_bounds = (x1 > 0 and y1 > 0 and z1 > 0) and \
                (x2 < data.shape[1] and y2 < data.shape[2] and
                 z2 < data.shape[3])
    return in_bounds


def load_tif(filename, dtype=None):
    if dtype is None:
        return io.imread(filename)
    else:
        return io.imread(filename).astype(dtype)


def coords(data, value):
    [x, y, z] = [0, 0, 0]
    while True:
        x += 1
        if x >= data.shape[3]:
            x = 0
            y += 1
        if y >= data.shape[2]:
            y = 0
            z += 1
        if z >= data.shape[1]:
            z = 0
            raise StopIteration
        while data[0, z, y, x, 0] != value:
            x += 1
            if x >= data.shape[3]:
                x = 0
                y += 1
            if y >= data.shape[2]:
                y = 0
                z += 1
            if z >= data.shape[1]:
                z = 0
                raise StopIteration
        yield [x, y, z]


def test():
    TRAINING_INPUTS = ['training/inputs.tif']
    TRAINING_TARGETS = ['training/targets.tif']

    TEST_INPUTS = ['test/inputs.tif']
    TEST_TARGETS = ['test/targets.tif']

    input_shape = (8, 16, 16, 1)

    nn = NeuronModel(input_shape=input_shape)
    bounding_box_size = input_shape[0:3]

    train_generator = generator(TRAINING_INPUTS, TRAINING_TARGETS,
                                bounding_box_size)
    nn.fit(train_generator)

    test_generator = generator(TEST_INPUTS, TEST_TARGETS, bounding_box_size)
    loss, accuracy = nn.evaluate(test_generator, steps=64)
    print('''
Evaluation
==========''')
    print('Loss: {}'.format(loss))
    print('Accuracy: {}'.format(accuracy))

    nn.save('model.h5')


def main():
    test()


if __name__ == '__main__':
    main()
