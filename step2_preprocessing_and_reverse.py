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

def road_segmention_img(img):
  segment_img = do_segment(img)
  h, w = segment_img.shape[:2]
  blured = cv2.blur(segment_img, (3, 3))
  mask = np.zeros((h + 2, w + 2), np.uint8)
  cv2.floodFill(blured, mask, (w / 2, 380), (0, 0, 255), (2, 2, 2), (3, 3, 3), 8)

  detected_white = detecte_white(blured)
  # detected_white[:,:,2] = 255
  # detected_white[:, :, 1] = 255
  # detected_white[:, :, 0] = 255
  #
  #
  detected_red = detecte_red(blured)
  #
  g_mix = cv2.addWeighted(detected_red, 1, detected_white, 1, 0)

  g_mix = cv2.cvtColor(g_mix, cv2.COLOR_BGR2GRAY)

  return detected_white

def load_data(file_path):
  df = pd.read_csv(file_path)
  print(df.head())
  df = df.drop(['img'], axis =1)


  return df

def adjust_gamma(image, gamma=1.0):

  invGamma = 1.0/gamma
  table = np.array([((i / 255.0)**invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')

  return cv2.LUT(image, table)


def detecte_white(image):
  img_lane_HSL = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

  Lchannel_w = img_lane_HSL[:,:,1] 

  img_lane_HSL = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

  Lchannel_w = img_lane_HSL[:,:,1]

  mask_w = cv2.inRange(Lchannel_w, 160, 255)
  img_w_lane_dection = cv2.bitwise_and(image, image, mask= mask_w)
    
  return img_w_lane_dection


def detecte_red(image):
  img_lane_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  lower_red = np.array([0, 150, 150])
  upper_red = np.array([10, 255, 255])

  red_mask = cv2.inRange(img_lane_HSV, lower_red, upper_red)

  img_r_lane_dection = cv2.bitwise_and(image, image, mask=red_mask)

  return img_r_lane_dection

def do_canny(f):
  gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
  blur = cv2.GaussianBlur(gray, (9, 9), 0)
  # canny = cv2.Canny(blur, 50, 200, L2gradient=True)
  canny = cv2.Canny(blur, 50, 300, L2gradient=True)
  return canny

def hough(img):
  rho = 1
  theta = np.pi/180
  threshold = 50
  min_line_length = 100
  max_line_gap = 5
  # threshold = 1
  # min_line_length = 0
  # max_line_gap = 0
  
  line_image = np.copy(img)*0 #creating a blank to draw lines on
  lines = cv2.HoughLinesP(img, rho, theta, threshold, min_line_length, max_line_gap)

  if lines is not None:
    print(lines.size/4)
    for  x1,y1,x2,y2 in lines[0]:
      cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
  
    color_edges = np.dstack((img,)) # combine to oringal

    # combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) # combine to oringal
    combo = cv2.cvtColor(line_image, cv2.COLOR_GRAY2RGB)

    return combo
  else:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img

def do_segment(f):
  y_dir = f.shape[0]
  x_dir = f.shape[1]
  # polygons = np.array([[(0, y_dir-150), (0, int(y_dir*6/16)), (int(x_dir/4), int(y_dir/4)), (int(x_dir*3/4), int(y_dir/4)), (x_dir, int(y_dir*6/16)), (x_dir, y_dir-150)]])
  # polygons = np.array([[(0, y_dir-150), (0, 200), (200, 115), ((x_dir-200), 115), (x_dir, 200), (x_dir, y_dir-150)]])
  polygons = np.array(
    [[(0, 390), (0, 200), (200, 115), ((x_dir - 200), 115), (x_dir, 200), (x_dir, 390)]])
  # mask = np.zeros_like(f)
  mask = np.zeros(f.shape, np.uint8)
  cv2.fillPoly(mask, polygons, (255, 255, 255))
  segment = cv2.bitwise_and(f, mask)
  return segment
  
def preprocessing(img):
  # # img_w_lane = img[100:530, 10:1000]
  # image_detected = detecte_white(img)
  # res = hough(do_segment(do_canny(image_detected)))
  res = road_segmention_img(img)
  
  return res

def reverse_and_process_img(img_dir_path):
  df = load_data(csv_path)
  images_count = len(fnmatch.filter(os.listdir(img_dir_path), '*.jpg'))
  df_list = []

  count = 0
  for imgs in range(images_count):
    read_img = '{}{}.jpg'.format(img_dir_path, imgs)
    print('Loading {}'.format(imgs))
    original_img = cv2.imread(read_img)
    original_img = preprocessing(original_img)

    # V4
    resized_img = original_img #cv2.resize(original_img, (200, 66))

    cv2.imwrite('{}/data/datasets/img_ok_version_processed_step2_tttt/{}.jpg'.format(os.getcwd(), count), resized_img)
    df_list.append(['{}.jpg'.format(count), df.values[imgs][0]])
    count = count + 1

    flip = cv2.flip(resized_img, 1)

    cv2.imwrite('{}/data/datasets/img_ok_version_processed_step2_tttt/{}.jpg'.format(os.getcwd(), count), flip)
    if df.values[imgs][0] == 0:
      df_list.append(['{}.jpg'.format(count), df.values[imgs][0]])
    else:
      df_list.append(['{}.jpg'.format(count), -df.values[imgs][0]])

    count = count + 1

  n_df = pd.DataFrame(df_list, columns=['img', 'str'])
  n_df.to_csv('{}/data/csv/Res_ok_version_processed_step2_t.csv'.format(os.getcwd(), count), index=0)

def test_reverse_and_process_img(img_dir_path):
  # df = load_data(csv_path)
  images_count = len(fnmatch.filter(os.listdir(img_dir_path), '*.jpg'))
  df_list = []

  count = 0
  for imgs in range(images_count):
    read_img = '{}{}.jpg'.format(img_dir_path, imgs)
    print('Loading {}'.format(imgs))
    original_img = cv2.imread(read_img)
    original_img = preprocessing(original_img)

    # V4
    resized_img = original_img #cv2.resize(original_img, (200, 66))

    # cv2.imwrite('{}/data/datasets/img_ok_version_anti_processed_step2_t/{}.jpg'.format(os.getcwd(), count), resized_img)
    # df_list.append(['{}.jpg'.format(count), df.values[imgs][0]])

    show_image(resized_img)

    if count == 0:
      break

    count = count + 1



def show_image(img_arr):
  plt.imshow(img_arr)
  plt.show()


if __name__ == '__main__':
  path_curr = os.getcwd()

  csv_path = '{}/data/csv/Res_ok_version_processed_step1.csv'.format(path_curr)
  img_dir = '{}/data/datasets/img_ok_version_processed_step1/'.format(path_curr)

  # df = load_data(csv_path)
  # np_img_list = reverse_and_process_img(img_dir)

  # img = cv2.imread('{}/9183.jpg'.format(img_dir))

  test = cv2.imread('{}/data/datasets/img_ok_version_processed_step1/1540.jpg'.format(path_curr)) #6086
  #
  cv2.imshow('image', road_segmention_img(test))
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  # show_image(road_segmention_img(test))
  # show_image(detecte_white(test))
  # show_image(hough(do_segment(do_canny(detecte_white(test)))))

  # show_image(b)
  # pre = hough(do_segment(do_canny(detecte_white(img))))
  # combo = cv2.cvtColor(pre, cv2.COLOR_GRAY2RGB)
  # show_image(pre)
  # show_image(img)




