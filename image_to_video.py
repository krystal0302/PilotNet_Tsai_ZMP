import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import tensorflow as tf
import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import fnmatch
import os
import pandas as pd
import matplotlib.pyplot as plt

path_curr = os.getcwd()
test_img_dir = '{}/data/datasets/img_ok_version_processed_step3_final_sample/'.format(path_curr)

img_array = []
img_array_r = []

images_count = len(fnmatch.filter(os.listdir(test_img_dir), '*.jpg'))

count = 0

for filename in range(images_count):
  read_img = '{}{}.jpg'.format(test_img_dir, filename)
  img = cv2.imread(read_img)
  height, width, layers = img.shape
  size = (width, height)
  print('Loading {}'.format(read_img))

  if count%2 == 0:
    img_array_r.append(img)
  else:
    img_array.append(img)

  count = count +1

out = cv2.VideoWriter('./project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
  out.write(img_array[i])
out.release()