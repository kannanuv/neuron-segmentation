from skimage import io
import keras
from keras.models import Sequential, load_model
from keras.layers import (Dense, Dropout, Activation, Flatten,
                          Conv3D, MaxPooling3D)
from keras import backend
from keras.callbacks import EarlyStopping
import argparse
import numpy as np
import tifffile
import h5py
from random import randint, choice
import pickle


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
        self._model.add(Dropout(rate=0.5)) # Dropout necessary to
                                           # avoid overfitting
        self._model.add(Activation('relu'))
        self._model.add(Dense(1, activation='sigmoid'))
        self._model.summary()
        self._model.compile(loss='binary_crossentropy',
                            optimizer='Adam',
                            metrics=['accuracy'])

    def fit(self, generator, steps_per_epoch=16):
        """
        Trains the model
        
        Args: 
            generator (generator): A generator to generate test data from
            steps_per_epoch (int): Number of step per epoch
        """
        callbacks = [EarlyStopping(monitor='loss', min_delta=0.0001,
                                   patience=5, mode='min')]
        return self._model.fit_generator(generator, steps_per_epoch,
                                         epochs=500,
                                         class_weight={0: 1, 1: 1.5},
                                         verbose=2, callbacks=callbacks)

    def evaluate(self, generator, steps):
        """
        Evaluates the accuracy of model on test dataset
        
        Args:
            generator (generator): A generator to generate test data from
            steps (int): Number to steps to perform evaluation on
        Returns:
            (float, float): The loss and accuracy, respectively
        """
        return self._model.evaluate_generator(generator, steps)

    def predict(self, tif_filename, bounding_box_size):
        """
        Runs an inference on raw TIFF file
        
        Args:
            tif_filename (string): Name of raw TIFF file
            bounding_box_size (tuple): Tuple of (height, width, length) of
                bounding box
        Returns:
            array: Array of predicted segmentation
        """
        # Load raw image volume
        data = load_tif(tif_filename)
        (H, W, L) = bounding_box_size
        data = np.pad(data, ((H/2,), (W/2,), (L/2,)), mode='symmetric')
        
        # Reshape input tensor to have dimension 5 (Batch, Z, Y, X, Channel)
        inputs = data.reshape(1, data.shape[0],
                              data.shape[1],
                              data.shape[2], 1)
        
        # Normalize input tensor to have unit norm
        max_inputs = float(np.max(inputs))
        min_inputs = float(np.min(inputs))
        inputs = (inputs - min_inputs)/(max_inputs - min_inputs)
        targets = np.zeros(inputs.shape)

        # Iterates through volume coordinates to predict voxel values
        for coord in coords(inputs):
            (H, W, L) = bounding_box_size
            [x, y, z] = coord
            bounding_box = [[x - L/2, y - W/2, z - H/2],
                            [x + L/2, y + W/2, z + H/2]]
            if in_bounds(inputs, bounding_box):
                [[x1, y1, z1], [x2, y2, z2]] = bounding_box
                subinputs = inputs[:, z1:z2, y1:y2, x1:x2, :]
                targets[0, z, y, x, 0] = self._model.predict(subinputs,
                                                             batch_size=1)[0]
        # Reshapes prediction volume to dimension 3 (Z, Y, X)
        targets = targets.reshape(targets.shape[1],
                                  targets.shape[2],
                                  targets.shape[3])
        # Removes padding
        targets = targets[H/2:-H/2, W/2:-W/2, L/2:-L/2]
        return targets

    def load(self, filepath):
        """
        Loads model from filepath

        Args:
            filepath (string): Path to load model from
        """
        self._model = load_model(filepath)

    def save(self, filepath):
        """
        Saves model to filepath
        
        Args:
            filepath (string): Path to save model to
        """
        return self._model.save(filepath)


def generator(inputs_filenames,
              targets_filenames,
              neuron_lookup_table,
              bounding_box_size,
              rotation_range=0, brightness_range=0,
              contrast_range=0, background_value=1,
              neuron_value=False, search_size=16):
    """
    Generates data from raw and ground-truth datasets
    
    Args:
        inputs_filenames (list): List of strings with input file names
        targets_filenames (list): List of strings with target file names
        bounding_box_size (tuple): Tuple of (height, width, length) of 
            bounding box
    Returns:
        generator: A generator that generates a (input, target) output
    """
    while True:
        for inputs_filename, targets_filename in zip(inputs_filenames,
                                                     targets_filenames):
            # Loads raw input and ground-truth TIFFs
            inputs = load_tif(inputs_filename)
            inputs = inputs.reshape(1, inputs.shape[0],
                                    inputs.shape[1],
                                    inputs.shape[2], 1)
            max_inputs = float(np.max(inputs))
            min_inputs = float(np.min(inputs))
            inputs = (inputs - min_inputs)/(max_inputs - min_inputs)
            
            targets = load_tif(targets_filename)
            targets = targets.reshape(1, targets.shape[0],
                                      targets.shape[1],
                                      targets.shape[2], 1)
            (H, W, L) = bounding_box_size
            [x, y, z] = [0, 0, 0]

            # Searches for a random neuron coordinate to return the
            # coordinate along with N random coordinates in
            # surrounding neighborhood
            with open(neuron_lookup_table) as f:
                neuron_lut = pickle.load(f)
                while True:
                    neuron_coord = choice(neuron_lut)
                    [z, y, x] = neuron_coord
                    [x1, y1, z1] = [x - L/2, y - W/2, z - H/2]
                    [x2, y2, z2] = [x + L/2, y + W/2, z + H/2]
                    bounding_coords = [[x1-search_size,
                                        y1-search_size,
                                        z1-search_size],
                                       [x2+search_size,
                                        y2+search_size,
                                        z2+search_size]]
                    if in_bounds(targets, bounding_coords):
                        subinputs = inputs[:, z1:z2, y1:y2,
                                           x1:x2, :]
                        subtargets = np.array([int(targets[0, z,
                                                           y, x, 0])])
                        yield (subinputs, subtargets)
                        for elements in range(1, search_size):
                            hasBackground = False
                            hasNeuron = False
                            while not (hasBackground and hasNeuron):
                                i = randint(-search_size, search_size-1)
                                j = randint(-search_size, search_size-1)
                                k = randint(-search_size, search_size-1)
                                subinputs = np.rot90(inputs[:,
                                                            z1+k:z2+k,
                                                            y1+j:y2+j,
                                                            x1+i:x2+i,
                                                            :],
                                                     k=randint(0, 3),
                                                     axes=(2,3))
                                subtargets = np.array([int(targets[0,
                                                                   z+k,
                                                                   y+j,
                                                                   x+i,
                                                                   0])])
                                if not hasNeuron and subtargets[0] == neuron_value:
                                    hasNeuron = (subtargets[0] == neuron_value)
                                    yield (subinputs, subtargets)
                                if not hasBackground and subtargets[0] != neuron_value:
                                    hasBackground = (subtargets[0] != neuron_value)
                                    yield (subinputs, subtargets)


class AutomatedConvergence(EarlyStopping):
    """
    Stops training when testing loss converges
    """
    def __init__(self, model, test_generator, monitor='val_loss',
                 min_delta=0, patience=0, mode='auto'):
        super(AutomatedConvergence, self).__init__()
        self.test_generator = test_generator
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        x, y = map(list,
                   zip(*[self.test_generator.next() for i in range(1, 64)]))
        self.loss, self.accuracy = self.model.test_on_batch(x, y)
        current = self.loss
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1


def in_bounds(data, bounding_coords):
    """
    Determines if data is in bounding_coords

    Args:
        data (array): Numpy array of data
        bounding_coords (list): Coordinates of bounding box

    Returns:
        bool: True if bounding coordinates are in data; false otherwise

    Examples:
        >>> in_bounds(np.zeros((2,2)), [[0, 0, 0], [1, 1, 1]])
        True
        >>> in_bounds(np.zeros((2,2)), [[0, 0, 0], [4, 4, 4]])
        False
    """
    [x1, y1, z1] = bounding_coords[0]
    [x2, y2, z2] = bounding_coords[1]
    in_bounds = (x1 >= 0 and y1 >= 0 and z1 >= 0) and \
                (x2 < data.shape[3] and y2 < data.shape[2] and
                 z2 < data.shape[1])
    return in_bounds


def load_tif(filename, dtype=None):
    """
    Loads TIFF as a Numpy array casted as dtype

    Args:
        filename (string): Name of TIFF file
        dtype (dtype): Type to cast file as
    """
    if dtype is None:
        return io.imread(filename)
    else:
        return io.imread(filename).astype(dtype)


def save_tif(filename, array):
    """
    Saves array as TIFF

    Args: 
        filename (string): Name of file
        array (array): Numpy array
    """
    return tifffile.imsave(filename, array)


def coords(data, value=None):
    """
    Generates sequential coordinates from data with value

    Args:
        data (array): An array of data
        value (object): Value to retrieve
    """
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
        if value is None:
            yield [x, y, z]
        else:
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


def random_coords(data, value=None):
    """
    Generates random coordinates from data with value

    Args:
        data (array): An array of data
        value (object): Value to retrieve
    """
    [x, y, z] = [0, 0, 0]
    if value is None:
        while True:
            x = randint(0, data.shape[3]-1)
            y = randint(0, data.shape[2]-1)
            z = randint(0, data.shape[1]-1)
            yield [x, y, z]
    else:
        while True:
            x = randint(0, data.shape[3]-1)
            y = randint(0, data.shape[2]-1)
            z = randint(0, data.shape[1]-1)
            if data[0, z, y, x, 0] == value:
                yield [x, y, z]


def test():
    TRAINING_INPUTS = ['training/inputs.tif']
    TRAINING_TARGETS = ['training/targets.tif']

    TEST_INPUTS = ['test/inputs.tif']
    TEST_TARGETS = ['test/targets.tif']

    PREDICTION = 'prediction/prediction.tif'

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
