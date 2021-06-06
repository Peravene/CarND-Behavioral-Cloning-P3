import csv
import cv2
import numpy as np
import argparse
import pickle

def main():
    parser = argparse.ArgumentParser(description='Load data from traning mode.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    args = parser.parse_args()

    lines = []
    with open('./' + args.image_folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = './' + args.image_folder + '/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
        #flip also the image
        images.append(cv2.flip(image,1))
        measurements.append(measurement*-1)

    data = {'X_train':  np.array(images), 'y_train': np.array(measurements)}
    pickle.dump(data, open( './' + args.image_folder + '/dataAug.pickle', "wb" ))

if __name__ == '__main__':
    main()
