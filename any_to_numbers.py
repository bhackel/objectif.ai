from PIL import Image
import glob, os
import numpy as np

def numbers():
    # Converts image to a numpy array, then saves it to file

    size = 128

    data = []

    image_file = 'objectif.ai/any_aligned.png'

    # Open image
    image = Image.open(image_file)
    array = np.array(image)

    # remove alpha channel
    array = array[:,:,:3]

    # add it to the list
    data.append(array)

    data = np.array(data)

    # save the data to files
    np.save('objectif.ai/any_data', data)
