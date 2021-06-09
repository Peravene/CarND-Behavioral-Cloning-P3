import numpy as np
import argparse
import pickle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description='Train Network.')
    parser.add_argument(
        'pickle_file',
        type=str,
        default='',
        help='Path to pickle file, which will be used to train the model.'
    )
    args = parser.parse_args()

    # as I have 32GB of RAM, I can load such big pickle files
    loadeddata = pickle.load(open(args.pickle_file, "rb" ))
    X_train = loadeddata.get('X_train')
    y_train = loadeddata.get('y_train')
    #print(X_train.shape)
    #print(X_train[1,1,1,0])
    #print(X_train[1,1,1,1])
    #print(X_train[1,1,1,2])

    # these lines are needed as the 4GB GPU memory got full
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    model = Sequential()

    #############################
    # Preprocessing
    #############################
    # Normalize and mean center the image values
    model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
    # Turn to grayscale
    #model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x)))
    #crop at top (sky) and bottom (engine hood) of image
    model.add(Cropping2D(cropping=((70,25),(0,0))))

    #############################
    # Test architecture
    #############################
    # model.add(Flatten(input_shape=(160,320,3)))
    # model.add(Dense(1))

    #############################
    # LeNet architecture
    #############################
    # model.add(Conv2D(6,5,activation="relu"))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(6,5,activation="relu"))
    # model.add(MaxPooling2D())
    # model.add(Flatten())
    # model.add(Dense(120))
    # model.add(Dense(84))
    # model.add(Dense(1))

    #############################
    # NVidia architecture
    #############################
    model.add(Conv2D(24, 5, strides=2, activation="relu"))
    model.add(Conv2D(36, 5, strides=2, activation="relu"))
    model.add(Conv2D(48, 5, strides=2, activation="relu"))
    model.add(Conv2D(64, 3, activation="relu"))
    model.add(Conv2D(64, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    # draw model
    tf.keras.utils.plot_model(model, to_file='./doc/model.png', show_shapes=True)

    model.summary()

    # build architecture
    model.compile(loss='mse', optimizer='adam')
    # shuffle and split data
    history_object  = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3, verbose=1)

    # print the keys contained in the history object
    print(history_object.history.keys())

    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('./doc/loss.png')
    plt.show()

    model.save('model.h5')

if __name__ == '__main__':
    main()
