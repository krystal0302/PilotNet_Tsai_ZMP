# -*- coding: utf-8 -*-

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import tensorflow as tf
import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import backend as K
K.tensorflow_backend._get_available_gpus()


print('----------------------------------------------- Load Csv -----------------------------------------------')

df_dv_data = pd.read_csv('./data/Res.csv')
print(df_dv_data.head())

df_dv_data = df_dv_data.drop(['Unnamed: 0'], axis =1)
# n_df_list = []
#
# for item in df_dv_data.values:
#   # add = [item[0],(item[1]/30, item[2]/400)]
#   add = item[1]/30
#   n_df_list.append(add)
#
# n_data = pd.DataFrame(n_df_list)
# print(n_data)
#
# df_dv_data = n_df_list

print(df_dv_data.shape)
print(df_dv_data.head())
print(df_dv_data.values/30)

print('----------------------------------------------- Load Csv -----------------------------------------------')


print('----------------------------------------------- def Image process -----------------------------------------------')
def detecte_white(image):
  #  Convert to HSL
  img_lane_HSL = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

  #  Set the L channel -->  White
  Lchannel_w = img_lane_HSL[:,:,1] 
    
  #  Detect white​def detecte_white(image):
  #  Convert to HSL
  img_lane_HSL = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

  #  Set the L channel -->  White
  Lchannel_w = img_lane_HSL[:,:,1] 
    
  #  Detect white
  mask_w = cv2.inRange(Lchannel_w, 160, 255)
  img_w_lane_dection = cv2.bitwise_and(image, image, mask= mask_w)
    
  return img_w_lane_dection

def do_canny(f):
  gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
  blur = cv2.GaussianBlur(gray, (9, 9), 0)
  canny = cv2.Canny(blur, 50, 200, L2gradient=True)
  return canny

def hough(img):
  rho = 1
  theta = np.pi/180
  threshold = 30
  min_line_length = 20
  max_line_gap = 5
  
  line_image = np.copy(img)*0 #creating a blank to draw lines on
  lines = cv2.HoughLinesP(img, rho, theta, threshold, min_line_length, max_line_gap)
  
  #print(lines)
#   for (x1, y1, x2, y2) in lines:
#     cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
  for l in lines:
    cv2.line(line_image, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (255,0,0), 10)
#  detect_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lines]
  
  color_edges = np.dstack((img,))
  
  #print(color_edges.shape, line_image.shape)
  
  combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
  combo = cv2.cvtColor(combo, cv2.COLOR_GRAY2RGB)
  return combo

def do_segment(f):
  y_dir = f.shape[0]
  x_dir = f.shape[1]
  polygons = np.array([[(0, y_dir-150), (0, int(y_dir*6/16)), (int(x_dir/4), int(y_dir/4)), (int(x_dir*3/4), int(y_dir/4)), (x_dir, int(y_dir*6/16)), (x_dir, y_dir-150)]])
  mask = np.zeros_like(f)
  cv2.fillPoly(mask, polygons, 255)
  segment = cv2.bitwise_and(f, mask)
  return segment
  
def preprocessinf(img):
  img_w_lane = img[100:530, 10:1000]
  image_detected = detecte_white(img_w_lane)
  res = hough(do_segment(do_canny(img)))
  
  
  return res

print('----------------------------------------------- def Image process -----------------------------------------------')

print('----------------------------------------------- Load Img -----------------------------------------------')
import fnmatch
import os
print(os.getcwd())
path_curr = os.getcwd()
#Use to get row data
#img_path = '{}/data/datasets/driving_dataset/'.format(path_curr)
#Use to get processed data
img_path = '{}/data/datasets/new_processed_dataset/'.format(path_curr)
images_count = len(fnmatch.filter(os.listdir('{}/data/datasets/new_processed_dataset/'.format(path_curr)), '*.jpg')) # get count .jpg file in current dir

img_list = []
for imgs in range(images_count):
  read_img = '{}{}.jpg'.format(img_path, imgs)
  print('Loading {}'.format(imgs))
  p_img = cv2.imread(read_img)

  # V4
  p_img = cv2.resize(p_img,(200, 66))

  #Use to save preprocess data
  #p_img = preprocessinf(cv2.imread(read_img))
  #cv2.imwrite('{}/data/datasets/driving_dataset_processed/{}.jpg'.format(path_curr, imgs), p_img)
  # Use to save preprocess data
  img_list.append(p_img)
  
np_img_list = np.array(img_list)

print(np_img_list.shape)

#plt.imshow(np_img_list[9]) #show get gray scale img
#plt.show()

#plt.imshow(do_segment(do_canny(np_img_list[9]))) #show get gray scale img
#plt.show()

# plt.imshow(do_canny(np_img_list[9])) #show get gray scale img
# plt.show()

# plt.imshow(hough(do_canny(np_img_list[9]))) #show get gray scale img
# plt.show()

#plt.imshow(np_img_list[0]) #show get gray scale img
#plt.show()

print('----------------------------------------------- Load Img -----------------------------------------------')

print('----------------------------------------------- Set train data -----------------------------------------------')

from sklearn.model_selection  import train_test_split

X_train, X_test, y_train, y_test = train_test_split(np_img_list, df_dv_data.values/30.0, test_size=0.3, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# print(y_train.iloc[0])

#plt.imshow(X_train[0]) #show get gray scale img #check is it right
#plt.show()

print('----------------------------------------------- Set train data -----------------------------------------------')

print('----------------------------------------------- Set start  -----------------------------------------------')

# from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint
from keras import initializers
from keras import layers, models, optimizers
# from keras.layers

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# X_train = X_train / 255
# X_test = X_test / 255

# model = keras.Sequential()
model = Sequential()

# 84 acc model
# model.add(Lambda(lambda x: x/127.5-1.0, input_shape=X_train.shape[1:]))
# model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
# model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
# model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
# model.add(Conv2D(64, 3, 3, activation='elu'))
# model.add(Conv2D(64, 3, 3, activation='elu'))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(100, activation='elu'))
# model.add(Dense(50, activation='elu'))
# model.add(Dense(10, activation='elu'))
# model.add(Dense(1))

# V2 0.8476605
# model.add(Lambda(lambda x: x/127.5-1.0, input_shape=X_train.shape[1:]))
#
# model.add(Conv2D(24, 5, 5, subsample=(2, 2)))
# model.add(Activation('elu'))
#
# model.add(Conv2D(36, 5, 5, subsample=(2, 2)))
# model.add(Activation('elu'))
#
# model.add(Conv2D(48, 5, 5, subsample=(2, 2)))
# model.add(Activation('elu'))
#
# model.add(Conv2D(64, 3, 3))
# model.add(Activation('elu'))
#
# model.add(Conv2D(64, 3, 3))
# model.add(Activation('elu'))
#
# model.add(BatchNormalization())
# model.add(Flatten())
#
# model.add(Dense(100, activation='elu'))
# model.add(Dense(50, activation='elu'))
# model.add(Dense(37, activation='elu'))
# model.add(Dense(1))

# V3 0.84439
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=X_train.shape[1:]))

model.add(Conv2D(24, 5, 5, subsample=(2, 2)))
model.add(BatchNormalization(momentum=0.8))
model.add(keras.layers.LeakyReLU(alpha=0.3))

model.add(Conv2D(36, 5, 5, subsample=(2, 2)))
model.add(BatchNormalization(momentum=0.8))
model.add(keras.layers.LeakyReLU(alpha=0.3))

model.add(Conv2D(48, 5, 5, subsample=(2, 2)))
model.add(BatchNormalization(momentum=0.8))
model.add(keras.layers.LeakyReLU(alpha=0.3))

model.add(Conv2D(64, 3, 3))
model.add(BatchNormalization(momentum=0.8))
model.add(keras.layers.LeakyReLU(alpha=0.3))

model.add(Conv2D(64, 3, 3))
model.add(BatchNormalization(momentum=0.8))
model.add(keras.layers.LeakyReLU(alpha=0.3))

model.add(Flatten())

model.add(Dense(100))
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(Dense(50))
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(Dense(10))
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(Dense(1))

# V4 With resize image
# model.add(Lambda(lambda x: x/127.5-1.0, input_shape=X_train.shape[1:]))
#
#
# model.add(Conv2D(24, 5, 5, subsample=(2, 2)))
# model.add(Activation('elu'))
# keras.layers.MaxPooling2D(pool_size=(2, 2))
# model.add(BatchNormalization())
#
# model.add(Conv2D(36, 5, 5, subsample=(2, 2)))
# model.add(Activation('elu'))
# keras.layers.MaxPooling2D(pool_size=(2, 2))
# model.add(BatchNormalization())
#
# model.add(Conv2D(48, 5, 5, subsample=(2, 2)))
# model.add(Activation('elu'))
# keras.layers.MaxPooling2D(pool_size=(2, 2))
# model.add(BatchNormalization())
#
# model.add(Conv2D(64, 3, 3))
# model.add(Activation('elu'))
# keras.layers.MaxPooling2D(pool_size=(2, 2))
# model.add(BatchNormalization())
#
# model.add(Conv2D(64, 3, 3))
# model.add(Activation('elu'))
# keras.layers.MaxPooling2D(pool_size=(2, 2))
# model.add(BatchNormalization())
#
#
# model.add(Flatten())
#
# model.add(Dense(200, activation='elu'))
# model.add(BatchNormalization())
# model.add(Dense(100, activation='elu'))
# model.add(BatchNormalization())
# model.add(Dense(50, activation='elu'))
# model.add(BatchNormalization())
# model.add(Dense(10, activation='elu'))
# model.add(BatchNormalization())
# model.add(Dense(1))

print(model.summary())

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='mean_squared_error',
              metrics=['accuracy'])

print('----- Befor fit check -----')
print(X_train[0].shape)
print('----- Befor fit check -----')


filepath = './checkpoint/Best_weights_v2.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]

model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=callback_list, verbose=0)

print('Finish Fit')

test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# test_pred = model.predict(X_test)  #inference

# print(test_pred*30)


# model.save_weights('model_save_weights_function.h5')
model.save('model_save_function.h5')
