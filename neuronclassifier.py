import numpy as np
from skimage import io
import keras
from keras.models import Sequential, load_model
from keras.layers import (Dense, Dropout, Activation, Flatten,
                          Conv3D, MaxPooling3D)


class NeuronModel:
    """
    """
    input_shape = (50, 50, 50, 1)

    def __init__(self, input_shape=(50, 50, 50, 1)):
        self._model = Sequential()
        self._model.add(Conv3D(16, (5, 5, 1), padding='valid',
                               input_shape=input_shape))
        self._model.add(Activation('relu'))
        self._model.add(Conv3D(16, (3, 3, 3), padding='valid'))
        self._model.add(Activation('relu'))
        self._model.add(Conv3D(16, (5, 5, 1), padding='valid'))
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

    def fit(self, generator, steps_per_epoch):
        self._model.fit_generator(generator, steps_per_epoch)

    def evaluate(self, generator, steps_per_epoch):
        self._model.evaluate_generator(generator, steps_per_epoch)

    def load(self, filepath):
        self._model.load_model(filepath)

    def save(self, filepath):
        self._model.save(filepath)

    def generator(self, inputs, targets,
                  bounding_box_size=input_shape,
                  search_box_size=(5, 5, 5),
                  rotation_range=0, brightness_range=0,
                  contrast_range=0):
        (L, W, H) = bounding_box_size
        (a, b, c) = search_box_size
        [x, y, z] = [0, 0, 0]
        while True:
            [x, y, z] = self.get_box(targets, [x, y, z], bounding_box_size)
            subtargets = targets[z:z+a, y:y+a, x:x+c]
            mean = subtargets.mean()
            if mean < 0.90 and mean > 0.10:
                subtargets = targets[z:z+L, y:y+H, x:x+W]
                subinputs = inputs[z:z+L, y:y+H, x:x+W]
                yield (subinputs, subtargets)

    def load_tif(self, filename, dtype=None):
        if dtype is None:
            return io.imread(filename)
        else:
            return io.imread(filename).astype(dtype)

    def get_box(self, data, coordinates, bounding_box_size):
        [x, y, z] = coordinates
        (L, H, W) = bounding_box_size
        x += W
        if x + W > data.shape[2]:
            x = 0
            y += H
        if y + H > data.shape[1]:
            y = 0
            z += L
        if z + L > data.shape[0]:
            z = 0
        return [x, y, z]


def main():
    NEURON_TRAINING_PATH = []
    NEURON_TEST_PATH = []

    nn = NeuronModel()

    for dataset_path in NEURON_TRAINING_PATH:
        inputs = nn.load_tif(filename=dataset_path, dtype=np.float)
        targets = nn.load_tif(filename=dataset_path, dtype=np.int)
        generator = nn.generator(inputs, targets)
        nn.fit(generator)
    for dataset_path in NEURON_TEST_PATH:
        inputs = nn.load_tif(filename=dataset_path, dtype=np.float)
        targets = nn.load_tif(filename=dataset_path, dtype=np.int)
        generator = nn.generator(inputs, targets)
        nn.evaluate(generator)
    nn.save('model.h5')
