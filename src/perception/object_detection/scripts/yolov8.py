#!/usr/local/lib/obj_det_env/bin/python3

from ultralytics import YOLO
import time
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
import rospy
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from rospy.numpy_msg import numpy_msg
import json
from yolo.msg import Detections
import supervision as sv
from std_msgs.msg import Bool

# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)


class ObjectDetectionNode:

    def __init__(self) -> None:
        rospy.init_node('object_detection', anonymous=False)

        self.model = YOLO(rospy.get_param('/object_detection/model_type'))
        self.model.to(rospy.get_param('/object_detection/device'))
        
        rospy.loginfo(f"[{rospy.get_name()}] " + "Loaded model")
        self.visualize = rospy.get_param('/object_detection/visualize')
        self.data_pub = rospy.Publisher(rospy.get_param('/object_detection/publish_topic'), Detections, queue_size=1)
        if self.visualize:
            self.annotated_image_pub = rospy.Publisher(rospy.get_param('/object_detection/visualize_topic'), Image, queue_size=1)

        self.rgb_image_subscriber = message_filters.Subscriber(rospy.get_param('/object_detection/subscribe_topic'), Image)
        self.rgb_image_subscriber.registerCallback(self.callback)
        self.cv_bridge = CvBridge()
        rospy.loginfo(f"[{rospy.get_name()}] " + "Node Ready...")

        self.yolo_status_control_sub = rospy.Subscriber("yolo_status_control", Bool, self.yolo_status_control)
        self.yolo_detection_enabled = False
        self.started_publishing = False
        
    def runModel(self, img):
        results = self.model(img, show = False, verbose= False)
        if self.started_publishing == False:
            self.started_publishing = True
            rospy.loginfo(f"[{rospy.get_name()}] " + "Started publishing data")
        boxes =  results[0].boxes
        detections = sv.Detections.from_yolov8(results[0])
        confs, box_cls, box = detections.confidence, detections.class_id, detections.xyxy
        # box = boxes.xyxy
        # box_cls =boxes.cls
        if self.visualize:
            for (b,cls) in zip(box,box_cls):
                img = cv2.rectangle(img, (int(b[0]),int(b[1])),(int(b[2]),int(b[3])) , (255,0,0), 2)
                
                img = cv2.putText(img, str(self.model.names[cls.item()]) + " | " + str(cls.item()),(int(b[0]),int(b[1])),  cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0,0,255), 1, cv2.LINE_AA)
        return box, box_cls, confs, img

    def callback(self,ros_rgb_image):
        # self.start_time = time.time()


        if self.yolo_detection_enabled:

            # print("Yolo Object Detector will START INFERENCING")

            rgb_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image)
            rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)
            boxes, classes, confs, annotated_img = self.runModel(rgb_image)
            boxes = boxes # shape = (nPredictions, 4)
            classes = classes.flatten()
            boxes = boxes.flatten()
            confs = confs.flatten()

            nPredictions = len(classes)

            msg = Detections()
            msg.nPredictions  = nPredictions
            msg.box_bounding_boxes = boxes
            msg.box_classes = classes
            msg.confidences = confs

            
            # msg = {
            #     'boxes' : np.array(boxes),
            #     'box_classes' : classes.cpu().numpy(),
            # } 
            # self.data_pub.publish(json.dumps(msg, cls = NumpyEncoder))
            self.data_pub.publish(msg)
            
            if self.visualize:
                annotated_img = cv2.resize(annotated_img, (0, 0), fx = 0.5, fy = 0.5)
                # annotated_img = cv2.rotate(annotated_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                self.annotated_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(annotated_img))



    def yolo_status_control(self, msg):

        if msg == True:
            print("ACTIVATING Yolo Object Detector")
            self.yolo_detection_enabled = True
        else:
            print("DEACTIVATING Yolo Object Detector")
            self.yolo_detection_enabled = False


    
if __name__ == "__main__":
    
    node = ObjectDetectionNode()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass