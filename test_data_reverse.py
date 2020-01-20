import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import tensorflow as tf
import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('----------------------------------------------- Load Csv -----------------------------------------------')

df_dv_data = pd.read_csv('./data/Res.csv')

df_dv_data = df_dv_data.drop(['img', 'acc'], axis =1)
df_dv_data = df_dv_data.drop(range(0,550))
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
img_path = '{}/data/datasets/driving_dataset_processed/'.format(path_curr)
images_count = len(fnmatch.filter(os.listdir('{}/data/datasets/driving_dataset/'.format(path_curr)), '*.jpg')) # get count .jpg file in current dir

img_list = []
for imgs in range(images_count):
  read_img = '{}{}.jpg'.format(img_path, imgs)
  print('Loading {}'.format(imgs))
  p_img = cv2.imread(read_img)
  #Use to save preprocess data
  #p_img = preprocessinf(cv2.imread(read_img))
  #cv2.imwrite('{}/data/datasets/driving_dataset_processed/{}.jpg'.format(path_curr, imgs), p_img)
  # Use to save preprocess data
  if imgs < 550:
    pass
  else:
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

print('----------------------------------------------- Load Img -----------------------------------------------')

print(len(df_dv_data.values))
def transpose_image(img, steering):
  img = cv2.flip(img, 1)
  return img, -1.0 * steering

# print(df_dv_data.values[0])
# print(np_img_list[0])

new_img_list = []
new_str_list = []
loaded_data = df_dv_data.values.tolist()

for num in range(len(df_dv_data.values)):
  print(num)
  if df_dv_data.values[num] == 0:
    pass
  else:
    rv_img, rv_str = transpose_image(np_img_list[num], df_dv_data.values[num])
    new_img_list.append(rv_img)
    loaded_data.append(rv_str)
    print('Origin {} |  Reverse {}'.format(df_dv_data.values[num], rv_str))

img_list[len(img_list):len(img_list)] = new_img_list
loaded_data[len(loaded_data):len(loaded_data)] = new_str_list

# print(len(img_list))
# print(len(loaded_data))
# print(img_list[0])
# print(loaded_data[0])
# print(df_dv_data.values.shape)
# print(type(df_dv_data.values))
print(df_dv_data.values)

print(type(loaded_data))

de = pd.DataFrame(np.array(loaded_data), columns=['str'])
de.to_csv('tt.csv')

img_num = 0
for img in img_list:
  cv2.imwrite('{}/new_processed_dataset/{}.jpg'.format(path_curr, img_num), img)
  img_num = img_num + 1

print('----------------------------------------------- //////// -----------------------------------------------')
# print(loaded_data)
r, f, l = 0, 0, 0

for x in loaded_data:
  print(x[0])
  if int(x[0]) == 0:
    f = f + 1
  elif int(x[0]) < 0:
    l = l + 1
  elif int(x[0]) > 0:
    r = r + 1

print('L {:.2%} | R {:.2%} | F{:.2%}'.format(l/len(loaded_data), r/len(loaded_data), f/len(loaded_data)))


print('----------------------------------------------- Done -----------------------------------------------')