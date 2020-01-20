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

import keras

from keras import backend as K

K.tensorflow_backend._get_available_gpus()
c = 0


def load_data(file_path):
    # df = pd.read_csv(file_path, nrows=2000) # , skiprows=[i for i in range(1, 2001)]
    df = pd.read_csv(file_path)
    print(df.head())
    df = df.drop(['img'], axis=1)

    df.loc[(-30 <= df["str"]) & (df["str"] < -27), "str"] = -30
    df.loc[(-27 <= df["str"]) & (df["str"] < -22), "str"] = -25
    df.loc[(-22 <= df["str"]) & (df["str"] < -17), "str"] = -20
    df.loc[(-17 <= df["str"]) & (df["str"] < -12), "str"] = -15
    df.loc[(-12 <= df["str"]) & (df["str"] < -7), "str"] = -10
    df.loc[(-7 <= df["str"]) & (df["str"] < -2), "str"] = -5
    df.loc[(-2 <= df["str"]) & (df["str"] <= 2), "str"] = 0
    df.loc[(2 < df["str"]) & (df["str"] <= 7), "str"] = 5
    df.loc[(7 < df["str"]) & (df["str"] <= 12), "str"] = 10
    df.loc[(12 < df["str"]) & (df["str"] <= 17), "str"] = 15
    df.loc[(17 < df["str"]) & (df["str"] <= 22), "str"] = 20
    df.loc[(22 < df["str"]) & (df["str"] <= 27), "str"] = 25
    df.loc[(27 < df["str"]) & (df["str"] <= 30), "str"] = 30

    # df = df.values
    #
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
    #
    #     # df = df.tolist()
    #     # df = pd.DataFrame(df, columns=['str'])
    #
    df.to_csv('{}/data/csv/Res_ok_version_processed_step3_check.csv'.format(os.getcwd()), index=0)

    f, l, r = 0, 0, 0
    for x in df.values:
        # print(x[0])
        if int(x[0]) == 0:
            f = f + 1
        elif int(x[0]) < 0:
            l = l + 1
        elif int(x[0]) > 0:
            r = r + 1
    f = float(f)
    l = float(l)
    r = float(r)
    print(len(df.values))
    print('L {:.2%} | R {:.2%} | F{:.2%}'.format(l / len(df.values), r / len(df.values), f / len(df.values)))

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
        original_img = cv2.imread(read_img)  # , cv2.IMREAD_GRAYSCALE
        # V4
        resized_img = cv2.resize(original_img, (200, 66))
        # resized_img = np.reshape(resized_img, (200,66,1))

        img_list.append(resized_img)

        np_img_list = np.array(img_list)

    return np_img_list

    # ct = 0
    # for imgs in range(images_count):
    #     ct = ct + 1
    #     read_img = '{}{}.jpg'.format(img_dir_path, imgs)
    #     print('Loading {}'.format(imgs))
    #     if imgs <= 2000: # 2000<= imgs and
    #         original_img = cv2.imread(read_img) #, cv2.IMREAD_GRAYSCALE
    #         # V4
    #         resized_img = cv2.resize(original_img, (200, 66))
    #         # resized_img = np.reshape(resized_img, (200,66,1))
    #
    #         img_list.append(resized_img)
    #
    #         np_img_list = np.array(img_list)
    #
    #     if ct == 2000:
    #         break
    #
    # return np_img_list


def show_image(img_arr):
    plt.imshow(img_arr)
    plt.show()


def train_test_data_split(np_img_list, df):
    X_train, X_test, y_train, y_test = train_test_split(np_img_list, df.values, test_size=0.3, random_state=1)

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
    #

    model.add(Conv2D(24, 5, 5, subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv2D(36, 5, 5, subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv2D(48, 5, 5, subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv2D(64, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv2D(64, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Flatten())

    model.add(Dense(100))  # , activation='elu'
    # model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Dense(50))
    # model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Dense(26))
    # model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Dense(13, activation='softmax'))

    # model.add(BatchNormalization())
    # model.add(Conv2D(12, 5, 5, subsample=(2, 2)))
    # model.add(Activation('relu'))
    #
    # # if c = 0:
    # #     c=c+1
    # #     print()
    #
    # model.add(BatchNormalization())
    # model.add(Conv2D(24, 5, 5, subsample=(2, 2)))
    # model.add(Activation('relu'))
    #
    # model.add(BatchNormalization())
    # model.add(Conv2D(36, 5, 5, subsample=(2, 2)))
    # model.add(Activation('relu'))
    #
    # model.add(BatchNormalization())
    # model.add(Conv2D(48, 5, 5, subsample=(2, 2)))
    # model.add(Activation('relu'))
    #
    # model.add(BatchNormalization())
    # model.add(Conv2D(64, 3, 3))
    # model.add(Activation('relu'))
    #
    # model.add(BatchNormalization())
    # model.add(Conv2D(64, 3, 3))
    # model.add(Activation('relu'))
    #
    # model.add(Flatten())
    #
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(26, activation='relu'))
    # model.add(Dense(13, activation='softmax'))

    return model


def train_model(model_weight_save_path, model_save_path, X_train, X_test, y_train, y_test):
    from keras.callbacks import ModelCheckpoint

    model = pilot_model(X_train)

    print(model.summary())
    # model.compile(optimizer=tf.train.AdamOptimizer(),
    #               loss='mean_squared_error',
    #               metrics=['accuracy'])
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    checkpoint = ModelCheckpoint(model_weight_save_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callback_list = [checkpoint]

    model.fit(X_train, y_train, batch_size=64, epochs=300, validation_split=0.2, callbacks=callback_list, verbose=0)

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


def one_hot(nparr):
    speed_table = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
    one_hot = np.zeros((len(nparr), (len(speed_table))))

    for i in range(len(nparr)):
        for j in range(len(speed_table)):
            if nparr[i] == speed_table[j]:
                one_hot[i, j] = 1
    return one_hot


if __name__ == '__main__':
    path_curr = os.getcwd()
    # Use to get row data
    # img_path = '{}/data/datasets/driving_dataset/'.format(path_curr)

    # csv_path = '{}/data/csv/Res_ok_version_processed_step3_final_sample.csv'.format(path_curr)
    # img_dir = '{}/data/datasets/img_ok_version_processed_step3_final_sample/'.format(path_curr)

    test_csv_path = '{}/data/csv/Res_ok_version_processed_step3_final_sample.csv'.format(path_curr)
    test_img_dir = '{}/data/datasets/img_ok_version_processed_step3_final_sample/'.format(
        path_curr)  # img_ok_version_processed_step3_final_sample

    model_weight_save_path = '{}/checkpoint/retrain_weights.hdf5'.format(path_curr)
    model_save_path = '{}/data/models/retrain_model_save_function_python2.h5'.format(path_curr)

    load_model_path = '{}/data/models/retrain_model_save_function_python2.h5'.format(path_curr)

    df = load_data(test_csv_path)

    np_img_list = load_img(test_img_dir)

    print(df.shape, np_img_list.shape)

    X_train, X_test, y_train, y_test = train_test_data_split(np_img_list, df)

    y_train = one_hot(y_train)
    y_test = one_hot(y_test)
    # X_train = X_train / 255
    # X_test = X_test /255

    print(np.argmax(y_train[0]))
    speed_table = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
    print(speed_table[np.argmax(y_train[0])])
    # print(y_train[0])
    # print(np.argmax(y_train[0]))

    # evaluate_model(load_model_path, X_test, y_test)
    #
    # retrain_model(load_model_path, model_weight_save_path, model_save_path, X_train, X_test, y_train, y_test)

    train_model(model_weight_save_path, model_save_path, X_train, X_test, y_train, y_test)

    # evaluate_model(load_model_path, X_test, y_test)

    print(path_curr)
