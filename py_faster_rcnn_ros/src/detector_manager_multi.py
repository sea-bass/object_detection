#!/usr/bin/env python


import numpy as np
import scipy.io as sio
import os, sys, cv2
import argparse
from timeit import default_timer

import rospy
from rospkg import RosPack
# import diagnostic_updater
# import diagnostic_msgs
import std_msgs.msg
from std_msgs.msg import UInt8
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon, Point32
from cv_bridge import CvBridge, CvBridgeError
# from rcta_object_pose_detection.srv import *
from wm_od_interface_msgs.msg import *

# package = RosPack()
# path = package.get_path('py-faster-rcnn')
path = "/opt"
sys.path.append(os.path.join(path, 'py-faster-rcnn', 'lib'))
sys.path.append(os.path.join(path, 'py-faster-rcnn', 'caffe-fast-rcnn', 'python'))

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import caffe

detections_count = 0
detections_time = 0

# Detector manager class for ros faster rcnn
class DetectorManager():
    def main(self, gpu_id=0, cpu_mode=False):
        self.network_filename = rospy.get_param('~caffe_model',
                                                'box_0_1_ZF_faster_rcnn_final.caffemodel')

        self.network_name = os.path.basename(self.network_filename)

        package = RosPack()
        path = package.get_path('py_faster_rcnn_ros')
        prototxt = os.path.join(path, 'models', 'zf', 'protos', 'faster_rcnn_test-21_classes.pt')
        
        # Our trained classes
        if self.network_name == 'ZF_faster_rcnn_all_three_extra_case.caffemodel':
            self.CLASSES = ('__background__',
                            'unused', 'pelican_case', 'cabinet', 'unused',
                            'unused', 'unused', 'unused', 'unused', 'unused',
                            'unused', 'unused', 'unused', 'unused',
                            'unused', 'unused', 'unused',
                            'unused', 'gascan','unused', 'unused')
        elif self.network_name == 'ZF_faster_rcnn_crate_plus_three.caffemodel':
            self.CLASSES = ('__background__',
                            'unused', 'crate', 'pelican_case', 'cabinet',
                            'unused', 'unused', 'unused', 'unused', 'unused',
                            'unused', 'unused', 'unused', 'unused',
                            'unused', 'unused', 'unused',
                            'unused', 'gascan','unused', 'unused')
        elif self.network_name == 'ZF_faster_rcnn_alley.caffemodel':
            self.CLASSES = ('__background__',
                            'pelican_case', 'person', 'cabinet', 'car',
                            'gate', 'door', 'car2', 'traffic_cone', 'chair',
                            'window', 'gascan', 'traffic barrel', 'horse',
                            'motorbike', 'person2', 'pottedplant',
                            'sheep', 'gascan2','train', 'tvmonitor')
        elif self.network_name == 'ZF_faster_rcnn_final.caffemodel':
            self.CLASSES = ('__background__',
                            'generator', 'crate', 'bench', 'school_bus',
                            'dumpster', 'pelican_case', 'backpack',
                            'suitcase', 'gas_can', 'gravestone',
                            'debris', 'trash_bin', 'wood_pallet',
                            'tower', 'barrier', 'police_truck',
                            'people', 'toilet', 'light_pole',
                            'barrel', 'gas_pump', 'motorcycle',
                            'control_tower', 'tank', 'shop', 'window',
                            'electrical_box', 'bicycle', 'gate', 'chair',
                            'table', 'traffic_sign', 'truss', 'pcv_pipe',
                            'weapons', 'stairs', 'door_ground', 'ingress_ground',
                            'door_upper_floor', 'door_unhinged', 'garage_door',
                            'door_frames', 'car', 'safety_barrier', 'truck')
            prototxt = os.path.join(path, 'models', 'zf', 'protos', 'faster_rcnn_test-46_classes.pt')
        elif self.network_name == 'ZF_crate_briefcase_safety_barrier.caffemodel':
            self.CLASSES = ('__background__',
                            'safety_barrier', 'unused2', 'crate', 'briefcase',
                            'unused5', 'unused6', 'unused7', 'unused8', 'unused9',
                            'unused10', 'unused11', 'unused12', 'unused13',
                            'unused14', 'unused15', 'unused16',
                            'unused17', 'unused18','unused19', 'unused20')
        elif self.network_name == 'box_0_1_ZF_faster_rcnn_final.caffemodel':
            self.CLASSES = ('__background__',
                            '1', '0', 'unused3', 'unused4',
                            'unused5', 'unused6', 'unused7', 'unused8', 'unused9',
                            'unused10', 'unused11', 'unused12', 'unused13',
                            'unused14', 'unused15', 'unused16',
                            'unused17', 'unused18','unused19', 'unused20')
        else:
            rospy.logwarn('Could not find hardcoded class names to match loaded model. \
                           Falling back to all_three_extra_case classes')
            self.CLASSES = ('__background__',
                            'unused1', 'unused2', 'unused3', 'unused4',
                            'unused5', 'unused6', 'unused7', 'unused8', 'unused9',
                            'unused10', 'unused11', 'unused12', 'unused13',
                            'unused14', 'unused15', 'unused16',
                            'unused17', 'unused18','unused19', 'unused20')


        # Load Faster RCNN ZF Network
        self.net = None
        self.gpu_id = gpu_id
        self.cpu_mode = cpu_mode
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        if self.gpu_id >= 0:
            cfg.GPU_ID = self.gpu_id

        rospy.loginfo("Loading caffe_model %s", self.network_filename)

        print("Loading caffee_model: " + self.network_filename)
        print("Using prototxt: " + prototxt)

        caffemodel = self.network_filename
        

        if not os.path.isfile(caffemodel):
            # Try appending path to the filename and see if that exists
            caffemodel = os.path.join(path, 'models', 'caffe', self.network_filename)
            if not os.path.isfile(caffemodel):
                raise IOError(('{:s} not found.').format(caffemodel))

        rospy.loginfo("Loading model from %s", caffemodel)
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        rospy.loginfo("Deep neural network loaded from %s", caffemodel)


        #getting ready for object detection
        self.bridge = CvBridge()
        self.classes = {}


        self.image_topic_A = rospy.get_param('~image_topic_A', '/camera_A/color_image_raw')
        self.image_topic_B = rospy.get_param('~image_topic_B', '/camera_B/color_image_raw')
        self.image_topic_C = rospy.get_param('~image_topic_C', '/camera_C/color_image_raw')

        rospy.loginfo("Autostarting detector_manager")
        self.pubs = {
            "A": rospy.Publisher("/camera_A/detected_objects", ira_dets, queue_size=10),
            "B": rospy.Publisher("/camera_B/detected_objects", ira_dets, queue_size=10),
            "C": rospy.Publisher("/camera_C/detected_objects", ira_dets, queue_size=10),
        }
        self.pub_viz_ = rospy.Publisher("~object_visualization", Image, queue_size=10)
        self.image_sub_A = rospy.Subscriber(self.image_topic_A, Image, 
                                            lambda data: self.imageCb(data, "A"),
                                            queue_size = 1, buff_size = 2**24)
        self.image_sub_B = rospy.Subscriber(self.image_topic_B, Image,
                                            lambda data: self.imageCb(data, "B"),
                                            queue_size = 1, buff_size = 2**24)
        self.image_sub_C = rospy.Subscriber(self.image_topic_C, Image,
                                            lambda data: self.imageCb(data, "C"),
                                            queue_size = 1, buff_size = 2**24)
        self.confidence_th = rospy.get_param("~confidence", 0.5)
        # rospy.Service('start_detection', ImageFrame, self.startStreamCb)
        # rospy.Service('stop_detection', ImageFrame, self.stopStreamCb)

        rospy.loginfo("Launched node for object detection and pose estimation")

        # if rospy.get_param("~autostart", False):
        #     req = ImageFrame()
        #     req.confidence_th = rospy.get_param("~confidence", 0.5)
        #     self.startStreamCb(req)
        # else:
        #     rospy.loginfo("Not not autostarting detector_manager, use service call to start")

        # diagnostics timer
        # diagnostics_frequency = rospy.get_param('~diagnostics_frequency', 1)
        # rospy.Timer(rospy.Duration(1.0 / diagnostics_frequency), self.diagnostics_callback)

        rospy.spin()

    def diagnostics_callback(self,event):
        updater.update()


    # def startStreamCb(self,req):
    #     message = ('streaming started')
    #     self.image_sub = rospy.Subscriber(self.image_topic, Image, self.imageCb, queue_size = 1, buff_size = 2**24)
    #     self.confidence_th = req.confidence_th
    #     return True

    def stopStreamCb(self,req):
        self.image_sub.unregister()
        message = ('streaming stopped')
        rospy.logwarn(message)
        return False

    def imageCb(self, data, camera="A"):
        start = rospy.Time().now()
        if self.cpu_mode:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(self.gpu_id)
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        detection_results = ira_dets()
        detection_results.header = data.header
        detector_out = []
        if self.net == None:
            rospy.logerr("Network not loaded")
            self.visualizeAndPublish(detector_out, self.cv_image)
        else:
            dstart = rospy.Time().now()
            self.objectDetection(self.net, self.cv_image, detector_out, data.header, thresh=self.confidence_th)
            dstop = rospy.Time().now()
            detection_results.dets = detector_out
            detection_results.n_dets = len(detector_out)
            self.pubs[camera].publish(detection_results)
            stop = rospy.Time().now()
            rospy.loginfo('Total processing: {}s, object detection: {}s '.\
                          format( \
                                  (stop-start).to_sec(), \
                                  (dstop-dstart).to_sec()))

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
            id = output[index].header.frame_id
            if label in self.classes.keys():
                color = self.classes[label]
            else:
                #generate a new color if first time seen this label
               color = np.random.randint(0,255,3)

            self.classes[label] = color
            cv2.rectangle(imgOut, (int(x_p1), int(y_p1)), (int(x_p3), int(y_p3)), (color[0],color[1],color[2]),thickness)
            text = ('{:s}: {:.3f}').format(label,confidence)
            cv2.putText(imgOut, text, (int(x_p1), int(y_p1+20)), font, fontScale, (255,255,255), thickness ,cv2.LINE_AA)
            cv2.putText(imgOut, id, (int(x_p1), int(y_p1+40)), font, fontScale, (255,255,255), thickness ,cv2.LINE_AA)

        # save image to file
        image_path = '/curiosity/catkin_ws_docker/detected_objects.jpg'
        cv2.imwrite(image_path, imgOut)
        print('Detected objects visualization saved to just saved file to %s' % image_path)
        image_msg = self.bridge.cv2_to_imgmsg(imgOut, "bgr8")
        self.pub_viz_.publish(image_msg)


    def objectDetection(self, net, im, objects, header, thresh=0.8):
        """Detect object classes in an image using pre-computed object proposals."""
        # Detect all object classes and regress object bounds
        start = default_timer()
        scores, boxes = im_detect(net, im)
        end = default_timer()
        message = ('Detection took {:.3f}s for '
                   '{:d} object proposals').format((end - start), boxes.shape[0])
        rospy.loginfo(message)
        global detections_time, detections_count
        detections_time += (end - start)
        detections_count += 1
        # save detection result to objects
        CONF_THRESH = thresh #default is 0.8
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(self.CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            #filter out low confidence objects
            idx = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(idx) > 0:
                for i in idx:
                    item = ira_det()
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    item.bbox.points = [Point32(x=bbox[0], y=bbox[1]), Point32(x=bbox[2], y=bbox[1]), Point32(x=bbox[2], y=bbox[3]), Point32(x=bbox[0], y=bbox[3])]
                    item.header = header
                    item.obj_name = cls
                    item.confidence = score
                    objects.append(item)

    # def produce_diagnostics(self, stat):
    #     global detections_time, detections_count
    #     if detections_time == 0:
    #         stat.summary(diagnostic_msgs.msg.DiagnosticStatus.WARN, "No detections reported.")
    #     else:
    #         rate = detections_count/detections_time
    #         stat.summary(diagnostic_msgs.msg.DiagnosticStatus.OK, "Frequency of Detections")
    #         stat.add("Detections Rate (hz)", rate)
    #         stat.add("Detections Time (sec)", detections_time)
    #         stat.add("Detections Count", detections_count)
    #         detections_count = 0
    #         detections_time = 0
    #     return stat


if __name__=="__main__":

    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='ROS node of Faster R-CNN Object Detection')

    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        default=False, action='store_true')


    args = parser.parse_args((rospy.myargv()[1:]))

    rospy.init_node("detector_manager_node")


    dm = DetectorManager()

    # updater = diagnostic_updater.Updater()

    # updater.setHardwareID("none")
    # updater.add("time ", dm.produce_diagnostics)

    # updater.force_update()
    dm.main(args.gpu_id, args.cpu_mode)
