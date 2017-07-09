import os
import csv
import cv2
import numpy as np
import sklearn
import random
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import math
import os.path

# Import the samples
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split test and training data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Validate that there is an image for every entry in log file
"""
    Make sure that for every entry in driving log csv, there is an image present.
"""
def are_files_missing(data=samples):
    unknown = []
    ind = []
    for i in range(len(data)):
        for count in range(0,3):
            im = data[i][count]    
            if not os.path.isfile(im):
                unknown.append(im)
                ind.append(i)
    return len(unknown)

def validate_data():
    count = are_files_missing()
    assert count == 0
    
validate_data()


# Pre-processing functions

LOGGER = False

"""
    Flip a coin randomly with either True or False happening with random probablity.
"""
def flip_coin():
    p = random.uniform(0,1)
    return get_random_value(0,1, p=p)[0] == 1

"""
    Get one random value between supplied values (inclusive). 
    When supplied, probability is also taken into consideration
"""
def get_random_value(min_val, max_val, size=1, p=None):
    return np.random.choice(np.arange(min_val, max_val+1), size, p)

"""
    Convert given image from rgb to yuv 
"""
def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
"""
    Resize a given image into given size. Default is 64x64. 
    Ensure this is used on all images before feeding them into the network.
"""
def resize(image, size=(64,64), interpol=cv2.INTER_LINEAR):
    height = size[0]
    width = size[1]
    return cv2.resize(image, (width, height), interpolation = interpol)

"""
    Translate the given image in both X and Y directions with 
    a random translation value between -10 and 10
"""
def translate(image):
    if not flip_coin():
        return image
    
    if LOGGER:
        print('translating the image')
    rows,cols,channels = image.shape
    tx, ty = random.sample(range(-10, 10), 2)
    M = np.float32([[1,0,tx],[0,1,ty]])
    return cv2.warpAffine(image,M,(cols,rows))

"""
    Flip a given image along Y axis.
"""
def flip(image, angle):
    if not flip_coin():
        return image, angle        
    if LOGGER:
        print('flipping the image')
    return cv2.flip(image, 1), -1*angle
    
"""
    Rotate a given image with a random angle. Modified based on:
    http://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
"""
def rotate(image, steering_angle):
    if not flip_coin():
        return image, steering_angle
    
    steering_correction = math.radians(steering_angle)
    
    angle = random.uniform(-5,5)
    if LOGGER:
        print('rotating image with angle: %f'%angle)
    if angle >= 0:
        steering_angle = steering_angle + steering_correction
    else:
        steering_angle = steering_angle - steering_correction
    
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH)), steering_angle

"""
    Brighten the image. Modified from:
    http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
"""
def brighten(image):
    if not flip_coin():
        return image
    
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    if LOGGER:
        print('brightening image')
    gamma = random.uniform(0.5, 1.5)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

"""
    Take area of interest in given image. 
    These values indicate the area of interest
"""
def crop(image):
    return image[80:, :,:]

"""
    Choose image either from center, left or right camera.
    Adjust steering angle if chosen from left or right camera
    
    -> Center image does not require any steering correction
    -> Left image, correction will be added
    -> Right image, correction will be subtracted
    
    Order of images in the data file: center (0), left (1), right (2)
"""
def get_image(batch_sample):
    STEERING_ANGLE_CORRECTION = 0.22
    
    i = get_random_value(0,2)[0]
    
    name = './IMG/'+batch_sample[i].split('/')[-1]
    image = cv2.imread(name)
    angle = float(batch_sample[3])
    
    if(i == 1):
        angle = angle + STEERING_ANGLE_CORRECTION
    elif(i == 2):
        angle = angle - STEERING_ANGLE_CORRECTION

    return process_image(image, angle)

"""
    Crops the image to get the area of interest and then
    randomly translates it, rotates it, flips it, brightens it 
    and then resizes the output to 64x64 to feed into the neural network
"""
def process_image(image, angle):
    image = crop(image)
    image = translate(image)
    image, angle = rotate(image, angle)
    image, angle = flip(image, angle)
    image = brighten(image)
    image = resize(image)
    return image, angle

"""
    Generator for generating the images and feeding to neural network
"""
def generator(data=train_samples, batch_size=64):
    while True:
        assert batch_size <= len(data)

        indices = get_random_value(0,len(data)-1, batch_size)
        batch = [data[x] for x in indices]
        batch_images = []
        batch_angles = []
        for i in range(0, len(batch)):
            image, angle = get_image(batch[i])
            batch_images.append(image)
            batch_angles.append(angle)
        yield np.array(batch_images), np.array(batch_angles)


# Generate train generator and validation generator
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)


## Model implementation 
# Model is based on: NVIDIA paper
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

nb_samples_per_epoch = 22784
nb_epochs = 15
nb_validation_samples = 4400


from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dropout, Dense, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

model = Sequential()

# Normalize
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

# Five convolutional and maxpooling layers
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(strides=(1, 1)))

model.add(Flatten())

# Five Dense layers
model.add(Dense(1164))
model.add(Activation('relu'))

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# Callback to monitor validation loss and save the best model it finds
filepath="model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

model.fit_generator(train_generator,
                      samples_per_epoch=nb_samples_per_epoch,
                      validation_data=validation_generator,
                      nb_val_samples=nb_validation_samples,
                      nb_epoch=nb_epochs, 
                      callbacks=callbacks_list)
