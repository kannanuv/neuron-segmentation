from neuron2Dclassifier import Neuron2DModel
import numpy as np
# import tifffile


def main():
    PREDICTION = 'prediction/inputs2D.tif'
    MODEL = 'unit-normalized-model2D.h5'
    
    input_shape = (1, 16, 16, 1)
    nn = Neuron2DModel(input_shape=input_shape)
    nn.load(MODEL)
    
    prediction = nn.predict(PREDICTION, bounding_box_size=(16, 16))
    np.save('prediction/targets.npy', prediction)
    prediction = (prediction*65535).astype(np.uint16)
    print(prediction)
    save_tif('prediction/targets.tif', prediction)


if __name__ == '__main__':
    main()
