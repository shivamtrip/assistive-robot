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
        
        print(self.model.names)
        
        rospy.loginfo(f"[{rospy.get_name()}] " + "Loaded model")
        self.visualize = rospy.get_param('/object_detection/visualize')
        self.data_pub = rospy.Publisher(rospy.get_param('/object_detection/publish_topic'), Detections, queue_size=1)
        if self.visualize:
            self.annotated_image_pub = rospy.Publisher(rospy.get_param('/object_detection/visualize_topic'), Image, queue_size=1)


        self.upscale_fac = 1/rospy.get_param('/image_shrink/downscale_ratio')

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
        ]

        h, w = img.shape[:2]
        
        upscale_fac = self.upscale_fac
        h *= upscale_fac
        w *= upscale_fac
        for (b,cls, conf) in zip(box,box_cls, confs):
            if model.names[cls.item()] in self.class_list:
                
                new_class_id = self.class_id_map[model.names[cls.item()]]
                x1, y1, x2, y2 = b
                
                y_new1 = w - x2 * upscale_fac
                x_new1 = y1 * upscale_fac
                y_new2 = w - x1 * upscale_fac
                x_new2 = y2 * upscale_fac
                
                b = [x_new1, y_new1, x_new2, y_new2]
                
                final_results[0].append(b)
                final_results[1].append(new_class_id) #remapped class index
                final_results[2].append(conf)
                
                
        for i in range(len(final_results)):
            outputs[i] = final_results[i]

        
        
    def mergeResults(self, results_coco, resultscustom, img):
        boxes1, classes1, confs1 = results_coco
        boxes2, classes2, confs2 = resultscustom
        # print(classes1, classes2)
        
        if len(boxes2) == 0:
            boxes = np.array(boxes1)
            classes = np.array(classes1)
            confs = np.array(confs1)
        elif len(boxes1) == 0:
            boxes = np.array(boxes2)
            classes = np.array(classes2)
            confs = np.array(confs2)
        else:
            boxes = np.concatenate((boxes1, boxes2))
            classes = np.concatenate((classes1, classes2))
            confs = np.concatenate((confs1, confs2))
        
        
        if self.visualize:
            for i in range(len(boxes)):
                b = boxes[i] / self.upscale_fac
                cls = classes[i]
                img = cv2.rectangle(img, (int(b[0]),int(b[1])),(int(b[2]),int(b[3])) , (255,0,0), 2)
                
                img = cv2.putText(img, str(self.class_list[cls]),(int(b[0]),int(b[1])),  cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0,0,255), 1, cv2.LINE_AA)
                
        
        
        return boxes, classes, confs, img
    

    def callback(self,ros_rgb_image):
        self.last_image_time = time.time()
        rgb_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image)
        # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rotated_image = cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)
        
        image_time = ros_rgb_image.header.stamp
        
        r1 = [[], [], []]
        r2 = [[], [], []]
        
        t2 = threading.Thread(target = self.runModel, args = (rotated_image, self.model, r1))
        t1 = threading.Thread(target = self.runModel, args = (rotated_image, self.model_custom, r2))
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()   

        
        boxes, classes, confs, annotated_img = self.mergeResults(r1, r2, rgb_image)
        # print("Boxes=", boxes)
        # print("Classes=", classes)
        # print("Confs=", confs)
        classes = classes.flatten()
        boxes = boxes.flatten()
        confs = confs.flatten().astype(np.float32)

        nPredictions = len(classes)

        msg = Detections()
        # print(image_time)
        msg.header.stamp = image_time
        msg.nPredictions  = nPredictions
        msg.box_bounding_boxes = boxes.tolist()
        msg.box_classes = classes.tolist()
        msg.confidences = confs.tolist()

        self.data_pub.publish(msg)
        
        if self.visualize:
            annotated_img = cv2.resize(annotated_img, (0, 0), fx = 0.5, fy = 0.5)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            self.annotated_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv2.rotate(annotated_img, cv2.ROTATE_90_CLOCKWISE)))



    
if __name__ == "__main__":
    
    node = ObjectDetectionNode()
    try:
        while True:
            if time.time() - node.last_image_time > 60:
                break
            rospy.sleep(1)
    except rospy.ROSInterruptException:
        pass