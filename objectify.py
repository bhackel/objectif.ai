import tensorflow as tf
import numpy as np
from PIL import Image
import sys, glob, os, shutil, math
import requests

from tensorflow import keras

import data_aligner, data_to_numbers, any_to_numbers, any_aligner

class Objector:
    def __init__(self):
        self.d = 'objectif.ai/'
        try:
            self.load_model()
        except IOError:
            self.generate_model(50)
            self.load_model()

        

    def generate_model(self, epochs):
        # Load data from files
        images = np.load(self.d+'data.npy')
        labels = np.load(self.d+'labels.npy')

        # Normalize pixel values to be between 0 and 1
        images = images / 255.0

        # Randomize data
        idx = np.random.permutation(len(images))
        images,labels = images[idx], labels[idx]

        # train test split should replace this
        train_images = np.array(images[0:450])
        train_labels = np.array(labels[0:450])

        test_images = np.array(images[450:])
        test_labels = np.array(labels[450:])
        #train_images, train_labels, test_images, test_labels = train_test_split(images, labels)


        # CNN model I stole
        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
        model.add(keras.layers.MaxPool2D(2,2))
        model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(2,2))
        model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(2,2))
        model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(2,2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512,activation='relu'))
        model.add(keras.layers.Dense(1,activation='sigmoid'))

        model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['mean_squared_error'])

        # Train the model
        model.fit(train_images, train_labels, epochs=epochs, 
                            validation_data=(test_images, test_labels))

        # Save the model
        model.save(self.d+'model')

    def generate_all(self, epochs):
        # Generates aligned images and trains the model
        self.clear_data()
        data_aligner.align()
        data_to_numbers.numbers()
        self.generate_model(epochs)
        self.load_model()
        print('generated new model')

    def train_one(self):
        # trains on one image for one epoch. probably bad idea
        model.fit(train_images, train_labels, epochs=50, 
                            validation_data=(test_images, test_labels))

    def clear_data(self):
        # Removes and recreates the aligned images directory
        for root, dirs, files in os.walk(self.d+'data/aligned'):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

    def load_model(self):
        self.model = keras.models.load_model(self.d+'model')

    def get_array_from_file(self):
        # takes image file with name 'any', aligns it, returns numbers
        if any_aligner.align() == False:
            return
        any_to_numbers.numbers()

        # load the array, then return it
        return np.load(self.d+'any_data.npy')

    def rate(self, image_array):
        # normalize pixel values
        try:
            image_array = image_array / 255.0
        except:
            return None

        # get rating as a decimal from 0 to 1, convert to 0 to 10
        rating = self.model(image_array)
        rating = float(rating.numpy()[0])
        rating = round(rating*10, 2)
        return rating

    def from_url(self, url, image_type):
        # takes a url, saves it, rates it
        r = requests.get(url, stream = True)
        
        if r.status_code == 200:
            filename = 'any.' + image_type
            output_folder = self.d+'temp/'
            file_location = output_folder + filename

            # write byte content to file in temp directory
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            r.raw.decode_content = True
            with open(file_location, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
            print('Image sucessfully Downloaded')

            # open the image to convert to png in current directory
            image = Image.open(file_location)
            image.save(self.d+'any.png', 'PNG')

            # rate the file
            return self.rate(self.get_array_from_file())
        else:
            print('Image Couldn\'t be retreived')





if __name__ == '__main__':
    a = Objector()
    print(a.rate(a.get_array_from_file()))
