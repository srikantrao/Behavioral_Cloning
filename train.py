#!/usr/bin/env python
"""
 Comma Ai training model - Added a Fully Connected layer as it was going from
"""
import os
import argparse
import json
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD, Adam, RMSprop
import csv
from scipy import misc
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import pdb

# for splitting into training and validation dataset
from sklearn.model_selection import train_test_split


def get_model():
    input_shape = (80, 320, 3)

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=input_shape,
            output_shape=input_shape))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(128))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(lr=0.0001))

    return model

if __name__ == "__main__":

    y_train= []
    X_train= []

    # Add multiple data sources
    file_paths = []
    file_paths.append('/home/srikant/Documents/self_driving_car/simulator-linux/data/data/')
    file_paths.append('/home/srikant/Documents/self_driving_car/simulator-linux/data/data/turn_data/')
    #file_paths.append('/home/srikant/Documents/self_driving_car/simulator-linux/data/data/turn_data2/')

    for file_path in file_paths:
        with open(file_path+'driving_log.csv', 'r') as csvfile:

            reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
            for row in reader:
                img_center_filename = row[0]
                img_left_filename = row[1]
                img_right_filename = row[2]

                img_center = (misc.imread(file_path + img_center_filename))[80:, :, :]
                img_left = (misc.imread(file_path + img_left_filename))[80:, :, :]
                img_right = (misc.imread(file_path + img_right_filename))[80:, :, :]
                steering_angle = float(row[3])

                # Image pre-processing section

                # Flipping images
                #if np.random.choice([True, False]):
                #steering_angle = -1.0 * steering_angle
                # img_center = np.fliplr(img_center)

                # Slack Community  pre-processing suggestion
                left_steering_angle =  0.08 + steering_angle
                right_steering_angle = - 0.08 + steering_angle

                # Add image to the data set
                X_train.append(img_center)
                y_train.append(steering_angle)
                X_train.append(img_left)
                y_train.append(left_steering_angle)
                X_train.append(img_right)
                y_train.append(right_steering_angle)

    X_train= np.asarray(X_train)
    y_train= np.asarray(y_train)

    #Create training and validation split of the data

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.05, random_state=37)

    print("The number of training samples", len(y_train))
    model = get_model()

    #Print the Model Summary
    model.summary()

    # Use generator
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.15,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        horizontal_flip=False,
        vertical_flip=False,
        featurewise_std_normalization=False,
        featurewise_center=False
    )
    model_path = '/home/srikant/Documents/self_driving_car/simulator-linux/data/data/'
    model_file = "model.h5"

    # Build on previous model if required
    if os.path.isfile(model_path+model_file):
        print("Model already exists ...")
        flag = int(input("Enter 1 if you want to use saved weights .. 0 if you want to start fresh"))
        if flag ==1:
            model.load_weights(model_file)

    print("The initial weights of the model have been loaded ...")

    model.fit_generator(
        datagen.flow(X_train , y_train, batch_size=128), samples_per_epoch=len(y_train), nb_epoch=10, validation_data=(X_valid,y_valid))

    # Save weights of the model

    model.save_weights(model_file)
    with open(model_path+'model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
