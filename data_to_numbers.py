from PIL import Image
import glob, os
import numpy as np

def numbers():
    # Converts all images in data/aligned/ into a numpy array, then saves it
    # Also makes another array of labels for every image in the data array

    size = 128

    subfolders = os.listdir("objectif.ai/data/aligned/")

    data = []
    labels = []

    # Sort through each folder, getting its number
    for folder in subfolders:
        #print("\n"*2 + "Going through the", folder)
        rating = int(folder)/10
        if rating == 1.0:
            rating = 0.99999

        print(rating)
        
        # Sort through each file in the folder, converting it to numbers
        input_folder = 'objectif.ai/data/aligned/' + folder
        input_files = os.listdir(input_folder)
        for file in input_files:
            image_file = "{}/{}".format(input_folder, file)

            # Open image
            image = Image.open(image_file)
            array = np.array(image)

            # remove alpha channel
            array = array[:,:,:3]

            # add it to the list
            data.append(array)

            labels.append(rating)

    data = np.array(data)
    labels = np.array(labels)

    # save the data to files
    np.save('objectif.ai/data', data)
    np.save('objectif.ai/labels', labels)
