# -*- coding: utf-8 -*-

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fnmatch
import os

def load_data(file_path):
  df = pd.read_csv(file_path)
  print(df.head())
  df = df.drop(['img'], axis =1)


  return df


def combine_data(img_dir_path, csv_file_path, anti_img_dir_path, anti_csv_file_path):
  df = load_data(csv_file_path)
  anti_df = load_data(anti_csv_file_path)

  images_count = len(fnmatch.filter(os.listdir(img_dir_path), '*.jpg'))
  anti_images_count = len(fnmatch.filter(os.listdir(anti_img_dir_path), '*.jpg'))

  print(images_count, anti_images_count)

  df_list = []

  count = 0

  zero = 0
  R = 0
  L = 0

  for imgs in range(images_count):
    if int(df.values[imgs][0]) == 0:
      zero = zero + 1
    elif int(df.values[imgs][0]) > 0:
      R = R + 1
    elif int(df.values[imgs][0]) < 0:
      L = L + 1

    read_img = '{}{}.jpg'.format(img_dir_path, imgs)
    original_img = cv2.imread(read_img)
    cv2.imwrite('{}/data/datasets/img_ok_version_processed_step3_final/{}.jpg'.format(os.getcwd(), count), original_img)
    df_list.append(['{}.jpg'.format(count), df.values[imgs][0]])
    count = count + 1

  for anti_imgs in range(anti_images_count):
    if int(anti_df.values[anti_imgs][0]) == 0:
      zero = zero + 1
    elif int(anti_df.values[anti_imgs][0]) > 0:
      R = R + 1
    elif int(anti_df.values[anti_imgs][0]) < 0:
      L = L + 1

    anti_read_img = '{}{}.jpg'.format(anti_img_dir_path, anti_imgs)
    anti_original_img = cv2.imread(anti_read_img)
    cv2.imwrite('{}/data/datasets/img_ok_version_processed_step3_final/{}.jpg'.format(os.getcwd(), count), anti_original_img)
    df_list.append(['{}.jpg'.format(count), anti_df.values[anti_imgs][0]])
    count = count + 1


  t_r = float(R)/float(count)
  t_l = float(L)/float(count)
  t_zero = float(zero) / float(count)
  print('R : {:.2%} , L : {:.2%} , F : {:.2%} ,'.format(t_r, t_l, t_zero))

  n_df = pd.DataFrame(df_list, columns=['img', 'str'])
  n_df.to_csv('{}/data/csv/Res_ok_version_processed_step3_final.csv'.format(os.getcwd(), count), index=0)

def show_image(img_arr):
  plt.imshow(img_arr)
  plt.show()

if __name__ == '__main__':
  path_curr = os.getcwd()

  csv_path = '{}/data/csv/Res_ok_version_processed_step2.csv'.format(path_curr)
  img_dir = '{}/data/datasets/img_ok_version_processed_step2/'.format(path_curr)

  anti_csv_path = '{}/data/csv/Res_ok_version_anti_processed_step2.csv'.format(path_curr)
  anti_img_dir = '{}/data/datasets/img_ok_version_anti_processed_step2/'.format(path_curr)

  # df = load_data(csv_path)
  np_img_list = combine_data(img_dir, csv_path, anti_img_dir, anti_csv_path)

  # img = cv2.imread('{}/9183.jpg'.format(img_dir))

  # show_image(do_canny(detecte_white(img)))
  # pre = hough(do_segment(do_canny(detecte_white(img))))
  # combo = cv2.cvtColor(pre, cv2.COLOR_GRAY2RGB)
  # show_image(pre)
  # show_image(img)




