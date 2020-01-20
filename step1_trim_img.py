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
  df = df.drop(['img', 'acc'], axis =1)


  return df

def trim_img(img_dir_path, csv_path):
  df = load_data(csv_path)
  images_count = len(fnmatch.filter(os.listdir(img_dir_path), '*.jpg'))
  df_list = []

  print(images_count)

  count = 0
  for imgs in range(images_count):
    read_img = '{}{}.jpg'.format(img_dir_path, imgs)

    original_img = cv2.imread(read_img)
    if imgs <= 1146:
      pass
    elif imgs <= 2054 and imgs >= 1662:
      pass
    elif imgs <= 2504 and imgs >= 2308:
      pass
    elif imgs <= 3038 and imgs >= 2732:
      pass
    else:
      print('Loading {}'.format(count))
      cv2.imwrite('{}/data/datasets/img_ok_version_anti_processed_step1/{}.jpg'.format(os.getcwd(), count), original_img)
      df_list.append(['{}.jpg'.format(count), df.values[imgs][0]])
      count = count + 1

  n_df = pd.DataFrame(df_list, columns=['img', 'str'])
  n_df.to_csv('{}/data/csv/Res_ok_version_anti_processed_step1.csv'.format(os.getcwd(), count), index=0)


def show_image(img_arr):
  plt.imshow(img_arr)
  plt.show()

if __name__ == '__main__':
  path_curr = os.getcwd()

  csv_path = '{}/data/csv/Res_ok_version_anti.csv'.format(path_curr)
  img_dir = '{}/data/datasets/img_ok_version_anti/'.format(path_curr)

  #
  trim_img(img_dir, csv_path)

  # img = cv2.imread('{}/3000.jpg'.format(img_dir))

  # show_image(do_canny(detecte_white(img)))
  # pre = hough(do_segment(do_canny(detecte_white(img))))
  # combo = cv2.cvtColor(pre, cv2.COLOR_GRAY2RGB)
  # show_image(preprocessing(img))
  # show_image(img)




