import numpy as np
import argparse
import pickle
from keras.models import Sequential
from keras.layers import Flatten,Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def main():
    parser = argparse.ArgumentParser(description='Train Network.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    args = parser.parse_args()

    loadeddata = pickle.load(open(  './' + args.image_folder + '/data.pickle', "rb" ))
    X_train = loadeddata.get('X_train')
    y_train = loadeddata.get('y_train')
    print(X_train.shape)

    # these lines are needed as the GPU memory got full
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    model = Sequential()
    # Preprocessing
    # Normalize and mean center the image values
    model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))

    # add layers
    model.add(Conv2D(6,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(6,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.summary()

    # build architecture
    model.compile(loss='mse', optimizer='adam')
    # schuffle and split data
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

    model.save('model.h5')



if __name__ == '__main__':
    main()
