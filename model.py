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
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import csv
import math
import random
import cv2

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for line in batch_samples:
                # read center image
                # openCV reads the image in BGR format, which needs to be converted to RGB
                image = cv2.cvtColor(cv2.imread(line[0]), cv2.COLOR_BGR2RGB)
                images.append(image)
                # read current steering
                measurement = float(line[3])
                measurements.append(measurement)

                # use also flipped center image
                images.append(cv2.flip(image,1))
                measurements.append(measurement*-1)

                # read left image
                # openCV reads the image in BGR format, which needs to be converted to RGB
                image = cv2.cvtColor(cv2.imread(line[1]), cv2.COLOR_BGR2RGB)
                images.append(image)

                # read right image
                # openCV reads the image in BGR format, which needs to be converted to RGB
                image = cv2.cvtColor(cv2.imread(line[2]), cv2.COLOR_BGR2RGB)
                images.append(image)

                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                measurements.append(measurement + correction) # Left
                measurements.append(measurement - correction) # right

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

def main():
    parser = argparse.ArgumentParser(description='Train Network.')
    parser.add_argument(
        'data_folder',
        type=str,
        default='',
        help='Path to data folder, which is filled within training mode.'
    )
    args = parser.parse_args()

    samples = []
    with open('./' + args.data_folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    # split data in training and validation samples
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # Set our batch size
    batch_size=32

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

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
    history_object  = model.fit_generator(train_generator, \
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=math.ceil(len(validation_samples)/batch_size), \
            epochs=5, verbose=1)

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
