#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from keras.models import load_model
import tensorflow as tf
import numpy as np
from keras import backend as K
K.tensorflow_backend._get_available_gpus()



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
    canny = cv2.Canny(blur, 150, 300, L2gradient=True)
    cv2.imshow("canny", canny)
    cv2.waitKey(3)
    return canny


def hough(img):
    rho = 1
    theta = np.pi / 180
    threshold = 30
    min_line_length = 20
    max_line_gap = 5

    line_image = np.copy(img) * 0  # creating a blank to draw lines on
    lines = cv2.HoughLinesP(img, rho, theta, threshold, min_line_length, max_line_gap)

    # print(lines)
    #   for (x1, y1, x2, y2) in lines:
    #     cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    if lines is not None:
        for l in lines:
            cv2.line(line_image, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (255, 0, 0), 10)

        color_edges = np.dstack((img,))


        combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
        combo = cv2.cvtColor(combo, cv2.COLOR_GRAY2RGB)

        return combo

    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img


def do_segment(f):
    y_dir = f.shape[0]
    x_dir = f.shape[1]
    polygons = np.array([[(0, y_dir - 150), (0, int(y_dir * 6 / 16)), (int(x_dir / 4), int(y_dir / 4)),
                          (int(x_dir * 3 / 4), int(y_dir / 4)), (x_dir, int(y_dir * 6 / 16)), (x_dir, y_dir - 150)]])
    mask = np.zeros_like(f)
    cv2.fillPoly(mask, polygons, 255)
    segment = cv2.bitwise_and(f, mask)
    return segment


def preprocessinf(img):
    img_w_lane = img[100:530, 10:1000]
    image_detected = detecte_white(img_w_lane)
    res = hough(do_segment(do_canny(image_detected)))


    return res


class image_converter:

  def __init__(self):
    import os
    print('---------------------------------------------------------------')
    print(os.getcwd())
    model_path = '{}/src/pilotnet/model/'.format(os.getcwd())
    model_name = 'model_save_function_python2_84.h5'
    print('---------------------------------------------------------------')
    self.msg_pub = rospy.Publisher('Steering_Angel', Float32, queue_size=10)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/image_raw", Image, self.callback)
    self.model = load_model('{}{}'.format(model_path, model_name))
    self.model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='mean_squared_error',
              metrics=['accuracy'])
    self.graph = tf.get_default_graph()


  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      cv2.imshow("Image window", cv_image)
      cv2.waitKey(3)
    except CvBridgeError as e:
      print(e)

    img_list = []
    p_img = preprocessinf(cv_image)

    cv2.imshow("preprocessing", p_img)
    cv2.waitKey(3)
    p_img = cv2.resize(p_img, (200, 66))
    img_list.append(p_img)
    np_img_list = np.array(img_list)

    steering_angel = 0
    with self.graph.as_default():
        preds = self.model.predict(np_img_list)
        steering_angel = preds[0]*30
        print(steering_angel)

    try:
      self.msg_pub.publish(steering_angel)
    except CvBridgeError as e:
      print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
