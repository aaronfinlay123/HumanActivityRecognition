# -*- coding: utf-8 -*-

# Video Classification tutorial
# overview, steps involved in building the model, exploring the dataset
# training the model and evaluating the model
# traditional method of image classification is as follows:
# take images, use extract the features
"""
import os.path
os.path.exists('mydirectory/myfile.txt')
True
os.path.exists('does-not-exist.txt')
False
os.path.exists('mydirectory')
True

"""
import csv

import cv2  # for capturing video
import math  # for mathematical  operations
import os.path
import matplotlib
import matplotlib.pyplot as plt  # for plotting images
import pandas as pd
from keras.preprocessing import image  # for preprocessing img
import numpy as np  # math operations
from keras.utils import np_utils
from skimage.transform import resize  # for image resizing
from sklearn.model_selection import train_test_split
from glob import glob

from tensorflow.python.keras import Sequential
from tqdm import tqdm

import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.preprocessing import image

from keras.applications.vgg16 import VGG16
import os
from scipy import stats as s

from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm

# open the .txt file which have names of training videos
f = open("trainlist01.txt", "r")
temp = f.read()
videos = temp.split('\n')

# creating a dataframe having video names
train = pd.DataFrame()
train['video_name'] = videos
train = train[:-1]
train.head()

# open the .txt file which have names of test videos
f = open("testlist01.txt", "r")
temp = f.read()
videos = temp.split('\n')

# creating a dataframe having video names
test = pd.DataFrame()
test['video_name'] = videos
test = test[:-1]
test.head()

train_video_tag = []
for i in range(train.shape[0]):
    train_video_tag.append(train['video_name'][i].split('/')[0])

train['tag'] = train_video_tag

# creating tags for test videos
test_video_tag = []
for i in range(test.shape[0]):
    test_video_tag.append(test['video_name'][i].split('/')[0])

test['tag'] = test_video_tag

# Mapping video path to tag
mapping = {}
for subdir, dirs, files in os.walk('UCF-101'):
    if subdir == 'UCF-101':
        pass
    else:
        tag = subdir.split('\\')[1]
        mapping[tag] = []
        for file in files:
            if tag in file:
                mapping[tag].append(os.path.join(subdir, file))

with open('ucf_101.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["class", "file"])
    for k, v in mapping.items():
        for f in v:
            writer.writerow([k, f])

ucf_df = pd.read_csv('ucf_101.csv')
percent_split = math.floor((30 * ucf_df.shape[0]) / 100.0)  # percent to split data 70% train 30% test
test_df = ucf_df.iloc[:percent_split, :]
train_df = ucf_df.iloc[percent_split:, :]
train_new_csv_path = '.\\train_1\\train_new.csv'
# storing frames from train videos and creating csv with tag, filename headers
with open(train_new_csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["class", "image"])
    for i in tqdm(range(percent_split, train_df.shape[0])):
        count = 0
        video_file = train_df['file'][i]
        cap = cv2.VideoCapture(video_file)  # cap vid from given path
        frameRate = cap.get(5)  # frame rate
        x = 1
        while cap.isOpened():
            frameId = cap.get(1)  # current frame no.
            ret, frame = cap.read()
            if not ret:
                break
            if frameId % math.floor(frameRate) == 0:
                # storing the frames in a new folder named train_1
                filename = f".\\train_1\\" + video_file.split('\\')[2].split('.')[
                    0] + f"_frame_{count}.jpg"
                count += 1
                cv2.imwrite(filename, frame)
                writer.writerow([train_df['class'][i], filename])
                if os.path.isfile(filename):
                    continue
                else:
                    raise Exception("File does not exist")

    cap.release()


train = pd.read_csv(train_new_csv_path)
train.head()



# getting the names of all the images
images = glob("train_1/*.jpg")
# creating empty list
train_image = []
train_class = []

# for loop to read and store frames
for i in tqdm(range(train.shape[0])):
    # loading img and keeping target size as (224,244,3)
    img = image.load_img(train['image'][i], target_size=(224,224,3))
    # converting to array
    img = image.img_to_array(img)
    # Normalizing pixel value
    img = img / 255
    # appending the img to train_image lisy
    train_image.append(img)





# converting the list to numpy array
X = np.array(train_image)

# shape of the array
X.shape
# separating the target
y = train['class']

# creating the training and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)

# creating dummies of target variable for train and validation set
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# creating the base model of pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# extracting features for training frames
X_train = base_model.predict(X_train)
X_train.shape


# extracting features for validation frames
X_test = base_model.predict(X_test)
X_test.shape

# reshaping the training as well as validation frames in single dimension
X_train = X_train.reshape(59075, 7*7*512)
X_test = X_test.reshape(14769, 7*7*512)

# normalizing the pixel values
max = X_train.max()
X_train = X_train/max
X_test = X_test/max

# shape of images
X_train.shape

#defining the model architecture
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(25088,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(101, activation='softmax'))

# defining a function to save the weights of best model
from keras.callbacks import ModelCheckpoint
mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')

# compiling the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# training the model
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[mcp_save], batch_size=128)


base_model = VGG16(weights='imagenet', include_top=False)


#defining the model architecture
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(25088,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(101, activation='softmax'))

# loading the trained weights
model.load_weights("weights.hdf5")

# compiling the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# getting the test list
f = open("testlist01.txt", "r")
temp = f.read()
videos = temp.split('\n')

# creating the dataframe
test = pd.DataFrame()
test['video_name'] = videos
test = test[:-1]
test_videos = test['video_name']
test.head()


# creating the tags
train = pd.read_csv('UCF-101/train_new.csv')
y = train['class']
y = pd.get_dummies(y)

# creating two lists to store predicted and actual tags
predict = []
actual = []

# for loop to extract frames from each test video
for i in tqdm(range(test_videos.shape[0])):
    count = 0
    videoFile = test_videos[i]
    cap = cv2.VideoCapture('UCF/'+videoFile.split(' ')[0].split('/')[1])   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    x=1
    # removing all other files from the temp folder
    files = glob('temp/*')
    for f in files:
        os.remove(f)
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames of this particular video in temp folder
            filename ='temp/' + "_frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
    cap.release()
    
    # reading all the frames from temp folder
    images = glob("temp/*.jpg")
    
    prediction_images = []
    for i in range(len(images)):
        img = image.load_img(images[i], target_size=(224,224,3))
        img = image.img_to_array(img)
        img = img/255
        prediction_images.append(img)
        
    # converting all the frames for a test video into numpy array
    prediction_images = np.array(prediction_images)
    # extracting features using pre-trained model
    prediction_images = base_model.predict(prediction_images)
    # converting features in one dimensional array
    prediction_images = prediction_images.reshape(prediction_images.shape[0], 7*7*512)
    # predicting tags for each array
    prediction = model.predict_classes(prediction_images)
    # appending the mode of predictions in predict list to assign the tag to the video
    predict.append(y.columns.values[s.mode(prediction)[0][0]])
    # appending the actual tag of the video
    actual.append(videoFile.split('/')[1].split('_')[1])

# checking the accuracy of the predicted tags
from sklearn.metrics import accuracy_score
accuracy_score(predict, actual)*100