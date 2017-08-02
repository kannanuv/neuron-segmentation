import tifffile as tf
import numpy as np
import pickle
import argparse

def edge_generator(ground_truth_filename, output_filename):
    target = tf.imread(ground_truth_filename)
    target = (target != 0)
    target_iterator = np.nditer(target, flags=['multi_index'])

    edge_lut = []

    while not target_iterator.finished:
        value = target_iterator[0]
        coord = it.multi_index
        [x, y, z] = coord
        mean = np.mean(targets[z-1:z+1,:
                               y-1:y+1,
                               x-1:x+1])
        if mean != value:
            edge_lut.append(target_iterator.multi_index)
        target_iterator.iternext()

    with open(output_filename, 'w') as f:
        pickle.dump(edge_lut, f)

def main(args):
    parser = argparse.ArgumentParser(description='Generates lookup table of the' +
                                     'ground-truth volume\'s edges')
    parser.add_argument('ground_truth_filename', metavar='ground-truth',
                        help='File name of ground-truth volume')
    parser.add_argument('-o', '--output', dest='output_filename'
                        help='Output file name')
    args = parser.parse_args()

if __name__ == '__main__':
    main()
