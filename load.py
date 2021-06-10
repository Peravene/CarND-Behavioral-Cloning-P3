import csv
import cv2
import numpy as np
import argparse
import pickle

def main():
    parser = argparse.ArgumentParser(description='Load data from traning mode.')
    parser.add_argument(
        'data_folder',
        type=str,
        default='',
        help='Path to data folder, which is filled within training mode.'
    )
    args = parser.parse_args()

    # read CSV file line by line
    lines = []
    with open('./' + args.data_folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
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

    # save the read image into a pickle
    data = {'X_train':  np.array(images), 'y_train': np.array(measurements)}
    pickle.dump(data, open( './' + args.data_folder + '/dataAug.pickle', "wb" ), protocol=4)

if __name__ == '__main__':
    main()
