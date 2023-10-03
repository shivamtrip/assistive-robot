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
import threading

# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)


class ObjectDetectionNode:

    def __init__(self) -> None:
        rospy.init_node('object_detection', anonymous=False)

        self.class_list = rospy.get_param('/object_detection/class_list')
        self.class_id_map = {self.class_list[i]: i for i in range(len(self.class_list))}
        rospy.loginfo(f"[{rospy.get_name()}] Loaded class list: {self.class_list}")
        self.model = YOLO(rospy.get_param('/object_detection/base_model'))
        self.model.to(rospy.get_param('/object_detection/device_1'))
        
        self.model_custom = YOLO(rospy.get_param('/object_detection/custom_model'))
        self.model_custom.to(rospy.get_param('/object_detection/device_2'))
        
        rospy.loginfo(f"[{rospy.get_name()}] " + "Loaded model")
        self.visualize = rospy.get_param('/object_detection/visualize')
        self.data_pub = rospy.Publisher(rospy.get_param('/object_detection/publish_topic'), Detections, queue_size=1)
        if self.visualize:
            self.annotated_image_pub = rospy.Publisher(rospy.get_param('/object_detection/visualize_topic'), Image, queue_size=1)

        self.rgb_image_subscriber = message_filters.Subscriber(rospy.get_param('/object_detection/subscribe_topic'), Image)
        self.rgb_image_subscriber.registerCallback(self.callback)
        self.cv_bridge = CvBridge()
        rospy.loginfo(f"[{rospy.get_name()}] " + "Node Ready...")
        self.last_image_time = time.time()
        self.started_publishing = False
    
    def runModel(self, img, model, outputs):
        results = model(img, show = False, verbose= False)
        
        if self.started_publishing == False:
            self.started_publishing = True
            rospy.loginfo(f"[{rospy.get_name()}] " + "Started publishing data")
        
        boxes =  results[0].boxes
        detections = sv.Detections.from_yolov8(results[0])
        confs, box_cls, box = detections.confidence, detections.class_id, detections.xyxy

        final_results = [
            [],
            [],
            [],
            []
        ]

        
        for (b,cls, conf) in zip(box,box_cls, confs):
            
            if model.names[cls.item()] in self.class_list:
                
                new_class_id = self.class_id_map[model.names[cls.item()]]
                final_results[0].append(b)
                final_results[1].append(new_class_id) #remapped class index
                final_results[2].append(conf)
                if self.visualize:
                    img = cv2.rectangle(img, (int(b[0]),int(b[1])),(int(b[2]),int(b[3])) , (255,0,0), 2)
                    
                    img = cv2.putText(img, str(self.class_list[new_class_id]) + " | " + str(new_class_id),(int(b[0]),int(b[1])),  cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0,0,255), 1, cv2.LINE_AA)
                
        for i in range(len(final_results)-1):
            outputs[i] = final_results[i]

        outputs[3] = img
    def mergeResults(self, results1, results2):
        boxes1, classes1, confs1, annotated_img1 = results1
        boxes2, classes2, confs2, annotated_img2 = results2
        boxes = np.concatenate((boxes1, boxes2))
        classes = np.concatenate((classes1, classes2))
        confs = np.concatenate((confs1, confs2))
        return boxes, classes, confs, annotated_img1
    

    def callback(self,ros_rgb_image):
        self.last_image_time = time.time()
        rgb_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image)
        rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)
        
        r1 = [None, None, None, None]
        r2 = [None, None, None, None]
        
        t1 = threading.Thread(target = self.runModel, args = (rgb_image, self.model, r1))
        t2 = threading.Thread(target = self.runModel, args = (rgb_image, self.model_custom, r2))
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()   

        boxes, classes , confs, annotated_img = self.mergeResults(r1, r2)
        
        classes = classes.flatten()
        boxes = boxes.flatten()
        confs = confs.flatten()

        nPredictions = len(classes)

        msg = Detections()
        msg.nPredictions  = nPredictions
        msg.box_bounding_boxes = boxes
        msg.box_classes = classes
        msg.confidences = confs

        self.data_pub.publish(msg)
        
        if self.visualize:
            annotated_img = cv2.resize(annotated_img, (0, 0), fx = 0.5, fy = 0.5)
            annotated_img = cv2.rotate(annotated_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.annotated_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(annotated_img))



    
if __name__ == "__main__":
    
    node = ObjectDetectionNode()
    try:
        while True:
            if time.time() - node.last_image_time > 60:
                break
            rospy.sleep(1)
    except rospy.ROSInterruptException:
        pass