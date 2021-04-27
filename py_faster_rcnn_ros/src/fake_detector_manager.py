#!/usr/bin/env python


import numpy as np
import scipy.io as sio
import os, sys, cv2
import argparse
from timeit import default_timer

import rospy
from rospkg import RosPack
import diagnostic_updater
import diagnostic_msgs
import std_msgs.msg
from std_msgs.msg import UInt8
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon, Point32
from cv_bridge import CvBridge, CvBridgeError
# from rcta_object_pose_detection.srv import *
from wm_od_interface_msgs.msg import *


# Detector manager class for ros faster rcnn
class FakeDetectorManager():
    def main(self, class_name="barrier", image_topic="/image"):
        # Our trained classes
        self.class_name = class_name
        self.bridge = CvBridge()

        self.image_topic = image_topic
        self.classes = {}


        self.pub_ = rospy.Publisher("detected_objects", ira_dets, queue_size=10)
        self.pub_viz_ = rospy.Publisher("~object_visualization", Image, queue_size=10)

        
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.imageCb, queue_size = 1, buff_size = 2**24)

        rospy.loginfo("Launched node for object detection and pose estimation")

        self.min_pos = [0.1, 0.1]
        self.max_pos = [0.9, 0.9]
        self.data = None

        while not rospy.is_shutdown():
            key = raw_input()
            print("got key:", key)
            if key == 'w':
                self.min_pos[0] -= 0.025
            elif key == 's':
                self.min_pos[0] += 0.025
            elif key == 'a':
                self.min_pos[1] -= 0.025
            elif key == 'd':
                self.min_pos[1] += 0.025
            elif key == '\x1b[A':
                self.max_pos[0] -= 0.025
            elif key == '\x1b[B':
                self.max_pos[0] += 0.025
            elif key == '\x1b[D':
                self.max_pos[1] -= 0.025
            elif key == '\x1b[C':
                self.max_pos[1] += 0.025
            
            if self.data is not None:
                self.imageCb(self.data)
        rospy.spin()


    def stopStreamCb(self,req):
        self.image_sub.unregister();
        message = ('streaming stopped')
        rospy.logwarn(message)
        return False

    def imageCb(self, data):
        self.data = data
        start = rospy.Time().now()
        rospy.sleep(1)
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8");
        except CvBridgeError as e:
            print(e)

        detection_results = ira_dets()
        detection_results.header = data.header
        detector_out = []
        
        dstart = rospy.Time().now()
        self.objectDetection(self.cv_image, detector_out, data.header)
        dstop = rospy.Time().now()
        detection_results.dets = detector_out
        detection_results.n_dets = len(detector_out)
        self.pub_.publish(detection_results)
        stop = rospy.Time().now()
        # rospy.loginfo('Total processing: {}s, object detection: {}s '.\
        #               format( \
        #                       (stop-start).to_sec(), \
        #                       (dstop-dstart).to_sec()))

        self.visualizeAndPublish(detector_out, self.cv_image)



        return True


    def visualizeAndPublish(self, output, imgIn):
        imgOut = imgIn.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        thickness = 2
        for index in range(len(output)):
            label = output[index].obj_name
            x_p1 = output[index].bbox.points[0].x
            y_p1 = output[index].bbox.points[0].y
            x_p3 = output[index].bbox.points[2].x
            y_p3 = output[index].bbox.points[2].y
            confidence = output[index].confidence
            if label in self.classes.keys():
                color = self.classes[label]
            else:
                #generate a new color if first time seen this label
               color = np.random.randint(0,255,3)

            self.classes[label] = color
            cv2.rectangle(imgOut, (int(x_p1), int(y_p1)), (int(x_p3), int(y_p3)), (color[0],color[1],color[2]),thickness)
            text = ('{:s}: {:.3f}').format(label,confidence)
            cv2.putText(imgOut, text, (int(x_p1), int(y_p1+20)), font, fontScale, (255,255,255), thickness ,cv2.LINE_AA)


        image_msg = self.bridge.cv2_to_imgmsg(imgOut, "bgr8")
        self.pub_viz_.publish(image_msg)


    def objectDetection(self, im, objects, header):
        """Detect object classes in an image using pre-computed object proposals."""

        item = ira_det()
        bbox = [self.min_pos[1]*im.shape[1], self.min_pos[0] * im.shape[0], self.max_pos[1]*im.shape[1], self.max_pos[0]*im.shape[0]]
        bbox = [int(b) for b in bbox]
        item.bbox.points = [Point32(x=bbox[0], y=bbox[1]), Point32(x=bbox[2], y=bbox[1]), Point32(x=bbox[2], y=bbox[3]), Point32(x=bbox[0], y=bbox[3])]
        item.header = header
        item.obj_name = self.class_name
        item.confidence = 0.95
        objects.append(item)



if __name__=="__main__":

    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Fake detector used to test later parts of the pipeline')

    parser.add_argument('--class_name', type=str, default='barrier')
    parser.add_argument('--image_topic', type=str, default='/image')

    args = parser.parse_args((rospy.myargv()[1:]))

    rospy.init_node("detector_manager_node")


    dm = FakeDetectorManager()

    dm.main(class_name=args.class_name, image_topic=args.image_topic)
