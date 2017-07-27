from neuronclassifier import *


def main():
    PREDICTION = 'prediction/prediction.tif'
    MODEL = 'unit-normalized-model.h5'
    
    input_shape = (8, 16, 16, 1)
    nn = NeuronModel(input_shape=input_shape)
    nn.load(MODEL)
    
    prediction = nn.predict(PREDICTION, bounding_box_size=(8, 16, 16))
    np.save('prediction/targets~.npy', prediction)
    prediction = (prediction*65535).astype(np.uint16)
    print(prediction)
    save_tif('prediction/targets~.tif', prediction)


if __name__ == '__main__':
    main()
