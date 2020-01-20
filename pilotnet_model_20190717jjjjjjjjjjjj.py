# -*- coding: utf-8 -*-

import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import tensorflow as tf
import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fnmatch
import os
from sklearn.model_selection import train_test_split

from keras import backend as K

K.tensorflow_backend._get_available_gpus()


def load_data(file_path):
    df = pd.read_csv(file_path)
    # print(df.head())
    print(df.shape[0])
    print(len(df))
    print(df.head())
    df.loc[(df["str"] <= 2) & (df["str"] >= -2), "str"] = 0
    df.loc[(df["str"] <= 7) & (df["str"] > 2), "str"] = 5
    df.loc[(df["str"] <= 30) & (df["str"] > 27), "str"] = 30
    df.loc[(df["str"] <= 22) & (df["str"] > 17), "str"] = 20

    print(df.head())
    # for i in len(df):
    #     if -2 <= df["str"] <= 2:
    #         df.set_value(i, "str", 0)



    df = df.drop(['img'], axis=1)

    # df = df.values



    # for x in df:
    #     # -2 <= x <= 2  --> 0
    #     if (-2 <= x[0]) and (x[0] <= 2):
    #         x[0] = 0
    #
    #     # 2 < x <= 7  --> 5
    #     # -7 <= x < -2  --> -5
    #     elif (2 < x[0]) and (x[0] <= 7):
    #         x[0] = 5
    #     elif (-7 <= x[0]) and (x[0] < -2):
    #         x[0] = -5
    #
    #     # 7 < x <= 12  --> 10
    #     # -12 <= x < -7  --> -10
    #     elif (7 < x[0]) and (x[0] <= 12):
    #         x[0] = 10
    #     elif (-12 <= x[0]) and (x[0] < -7):
    #         x[0] = -10
    #
    #     # 12 < x <= 17  --> 15
    #     # -17 <= x < -12  --> -15
    #     elif (12 < x[0]) and (x[0] <= 17):
    #         x[0] = 15
    #     elif (-17 <= x[0]) and (x[0] < -12):
    #         x[0] = -15
    #
    #     # 17 < x <= 22  --> 20
    #     # -22 <= x < -17  --> -20
    #     elif (17 < x[0]) and (x[0] <= 22):
    #         x[0] = 20
    #     elif (-22 <= x[0]) and (x[0] < -17):
    #         x[0] = -20
    #
    #     # 22 < x <= 27  --> 25
    #     # -27 <= x < -22  --> -25
    #     elif (22 < x[0]) and (x[0] <= 27):
    #         x[0] = 25
    #     elif (-27 <= x[0]) and (x[0] < -22):
    #         x[0] = -25
    #
    #     # 27 < x <= 30  --> 30
    #     # -30 <= x < -27  --> -30
    #     elif (27 < x[0]) and (x[0] <= 30):
    #         x[0] = 30
    #     elif (-30 <= x[0]) and (x[0] < -27):
    #         x[0] = -30

        # df = df.tolist()
        # df = pd.DataFrame(df, columns=['str'])

    # df.to_csv('{}/data/csv/Res_ok_version_processed_step3_check.csv'.format(os.getcwd()), index=0)

    return df


def detecte_white(image):
    img_lane_HSL = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    Lchannel_w = img_lane_HSL[:, :, 1]

    img_lane_HSL = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    Lchannel_w = img_lane_HSL[:, :, 1]

    mask_w = cv2.inRange(Lchannel_w, 160, 255)
    img_w_lane_dection = cv2.bitwise_and(image, image, mask=mask_w)

    return img_w_lane_dection


def do_canny(f):
    gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    canny = cv2.Canny(blur, 50, 200, L2gradient=True)
    return canny


def hough(img):
    rho = 1
    theta = np.pi / 180
    threshold = 30
    min_line_length = 20
    max_line_gap = 5

    line_image = np.copy(img) * 0  # creating a blank to draw lines on
    lines = cv2.HoughLinesP(img, rho, theta, threshold, min_line_length, max_line_gap)

    for l in lines:
        cv2.line(line_image, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (255, 0, 0), 10)

    color_edges = np.dstack((img,))

    combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
    combo = cv2.cvtColor(combo, cv2.COLOR_GRAY2RGB)
    return combo


def do_segment(f):
    y_dir = f.shape[0]
    x_dir = f.shape[1]
    polygons = np.array([[(0, y_dir - 150), (0, int(y_dir * 6 / 16)), (int(x_dir / 4), int(y_dir / 4)),
                          (int(x_dir * 3 / 4), int(y_dir / 4)), (x_dir, int(y_dir * 6 / 16)), (x_dir, y_dir - 150)]])
    mask = np.zeros_like(f)
    cv2.fillPoly(mask, polygons, 255)
    segment = cv2.bitwise_and(f, mask)
    return segment


def preprocessing(img):
    img_w_lane = img[100:530, 10:1000]
    image_detected = detecte_white(img_w_lane)
    res = hough(do_segment(do_canny(image_detected)))

    return res


def load_img(img_dir_path):
    images_count = len(fnmatch.filter(os.listdir(img_dir_path), '*.jpg'))
    img_list = []

    for imgs in range(images_count):
        read_img = '{}{}.jpg'.format(img_dir_path, imgs)
        print('Loading {}'.format(imgs))
        original_img = cv2.imread(read_img)
        # V4
        resized_img = cv2.resize(original_img, (200, 66))

        img_list.append(resized_img)

        np_img_list = np.array(img_list)

    return np_img_list


def show_image(img_arr):
    plt.imshow(img_arr)
    plt.show()


def train_test_data_split(np_img_list, df):
    X_train, X_test, y_train, y_test = train_test_split(np_img_list, df.values / 30.0, test_size=0.3, random_state=0)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return (X_train, X_test, y_train, y_test)


def pilot_model(X_train):
    import keras
    from keras.models import Sequential
    from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, Activation
    from keras.callbacks import ModelCheckpoint

    X_train = X_train.astype('float32')

    model = Sequential()

    # V3 0.84439
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=X_train.shape[1:]))

    model.add(BatchNormalization())
    model.add(Conv2D(24, 5, 5, subsample=(2, 2)))
    model.add(Activation('elu'))

    model.add(Conv2D(36, 5, 5, subsample=(2, 2)))
    model.add(Activation('elu'))

    model.add(Conv2D(48, 5, 5, subsample=(2, 2)))
    model.add(Activation('elu'))

    model.add(Conv2D(64, 3, 3))
    model.add(Activation('elu'))

    model.add(Conv2D(64, 3, 3))
    model.add(Activation('elu'))

    model.add(Flatten())

    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    return model


def train_model(model_weight_save_path, model_save_path, X_train, X_test, y_train, y_test):
    from keras.callbacks import ModelCheckpoint

    model = pilot_model(X_train)

    print(model.summary())
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(model_weight_save_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callback_list = [checkpoint]

    model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=callback_list, verbose=0)

    print('Finish Fit')

    test_loss, test_acc = model.evaluate(X_test, y_test)

    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    model.save(model_save_path)


def retrain_model(load_model_path, model_weight_save_path, model_save_path, X_train, X_test, y_train, y_test):
    from keras.models import load_model
    from keras.callbacks import ModelCheckpoint

    model = load_model(load_model_path)

    print(model.summary())
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    # checkpoint = ModelCheckpoint(model_weight_save_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callback_list = [checkpoint]

    model.fit(X_train, y_train, batch_size=32, epochs=500, validation_split=0.2, verbose=0)  # callbacks=callback_list,

    print('Finish Fit')

    test_loss, test_acc = model.evaluate(X_test, y_test)

    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    model.save(model_save_path)


def evaluate_model(evaluate_model_path, X_test, y_test):
    from keras.models import load_model

    model = load_model(evaluate_model_path)

    print(model.summary())

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # test_pred = model.predict(X_test)  #inference
    #
    # print(test_pred[0]*30)
    #
    # plt.imshow(X_test[0]) #show get gray scale img
    # plt.show()


if __name__ == '__main__':
    path_curr = os.getcwd()
    # Use to get row data
    # img_path = '{}/data/datasets/driving_dataset/'.format(path_curr)

    # csv_path = '{}/data/csv/Res_ok_version_processed_step3_final_sample.csv'.format(path_curr)
    # img_dir = '{}/data/datasets/img_ok_version_processed_step3_final_sample/'.format(path_curr)

    test_csv_path = '{}/data/csv/Res_ok_version_processed_step2_t.csv'.format(path_curr)
    test_img_dir = '{}/data/datasets/img_ok_version_processed_step3_final_smallsizetest/'.format(path_curr)

    model_weight_save_path = '{}/checkpoint/retrain_weights.hdf5'.format(path_curr)
    model_save_path = '{}/data/models/retrain_model_save_function_python2.h5'.format(path_curr)

    load_model_path = '{}/data/models/model_save_function_python2_84.h5'.format(path_curr)
    # print(type(df))
    df = load_data(test_csv_path)
    print(df)
    exit()
    np_img_list = load_img(test_img_dir)

    print(df.shape, np_img_list.shape)

    X_train, X_test, y_train, y_test = train_test_data_split(np_img_list, df)

    print(y_train[0])
    show_image(X_train[0])

    exit()
    # evaluate_model(load_model_path, X_test, y_test)
    #
    retrain_model(load_model_path, model_weight_save_path, model_save_path, X_train, X_test, y_train, y_test)

    # train_model(model_weight_save_path, model_save_path, X_train, X_test, y_train, y_test)

    # evaluate_model(load_model_path, X_test, y_test)

    print(path_curr)
