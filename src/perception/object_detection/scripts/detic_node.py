#!/usr/local/lib/detic_det_env/bin/python3
import cv2


import numpy as np
import rospy
# from yolo.srv import DeticDetections, DeticDetectionsResponse
from yolo.msg import DeticDetectionsAction, DeticDetectionsActionFeedback, DeticDetectionsActionResult, DeticDetectionsFeedback, DeticDetectionsGoal, DeticDetectionsResult, DeticDetectionsActionGoal
import message_filters
from cv_bridge import CvBridge
import time
from sensor_msgs.msg import Image
import torch
import json
import sys
import actionlib
sys.path.append('/usr/local/lib/detic_det_env/Detic/')
sys.path.append('/usr/local/lib/detic_det_env/detectron2/')
sys.path.insert(0, '/usr/local/lib/detic_det_env/Detic/third_party/CenterNet2/')

from detic.predictor import VisualizationDemo
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder
from detic.config import add_detic_config

from centernet.config import add_centernet_config

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog


class DeticNode:
    
    def __init__(self) -> None:
        rospy.init_node('detic_node', anonymous=False)
        
        self.annotated_image_pub = rospy.Publisher(rospy.get_param('/object_detection/detic/annotation_topic'), Image, queue_size=1)
        self.cv_bridge = CvBridge()
        
        # self.upscale_fac = 1/rospy.get_param('/image_shrink/downscale_ratio') # we will send full size images to detic.
        self.upscale_fac = 1
        self.server = actionlib.SimpleActionServer('detic_predictions', DeticDetectionsAction, self.callback, auto_start=False)

        self.setup_detectron()
        self.warmup()

        # self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback, queue_size=1)
        # self.server = rospy.Service('detic_predictions', DeticDetections, self.callback)
        self.server.start()
        rospy.loginfo(f"[{rospy.get_name()}] " + "Detic Ready...")

    def image_callback(self, msg):
        mmsg = DeticDetections()
        mmsg.image = msg
        starttime = time.time()
        response = self.callback(mmsg)
        rospy.loginfo("Took" + str(time.time() - starttime) + " seconds...")



    def setup_detectron(self):
        rospy.loginfo("Building model.")
        BUILDIN_CLASSIFIER = {
            'lvis': '/usr/local/lib/detic_det_env/Detic/datasets/metadata/lvis_v1_clip_a+cname.npy',
            'objects365': '/usr/local/lib/detic_det_env/Detic/datasets/metadata/o365_clip_a+cnamefix.npy',
            'openimages': '/usr/local/lib/detic_det_env/Detic/datasets/metadata/oid_clip_a+cname.npy',
            'coco': '/usr/local/lib/detic_det_env/Detic/datasets/metadata/coco_clip_a+cname.npy',
        }

        BUILDIN_METADATA_PATH = {
            'lvis': 'lvis_v1_val',
            'objects365': 'objects365_v2_val',
            'openimages': 'oid_val_expanded',
            'coco': 'coco_2017_val',
        }
        
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(rospy.get_param('/object_detection/detic/config_path'))
        cfg.MODEL.WEIGHTS = rospy.get_param('/object_detection/detic/weight_path')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = False # For better visualization purpose. Set to False for all classes.
        cfg.MODEL.DEVICE= rospy.get_param('/object_detection/detic/device')
        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = '/usr/local/lib/detic_det_env/Detic/datasets/metadata/lvis_v1_train_cat_info.json'
        self.predictor = DefaultPredictor(cfg)
        
        if rospy.get_param('/object_detection/detic/custom_vocab'):
            metadata = MetadataCatalog.get("__unused")
            metadata.thing_classes = rospy.get_param('/object_detection/class_list')
            classifier = self.get_clip_embeddings(metadata.thing_classes)
            print("Using custom vocab")
        else:
            vocabulary = rospy.get_param('/object_detection/detic/vocab_name')
            metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
            classifier = BUILDIN_CLASSIFIER[vocabulary]

        
        num_classes = len(metadata.thing_classes)
        self.class_list = metadata.thing_classes
        reset_cls_test(self.predictor.model, classifier, num_classes)
        
        self.class_names = self.predictor.metadata.get("thing_classes", None)
        

    def warmup(self):
        rospy.loginfo(f"[{rospy.get_name()}]" + "Warming up...")
        img = np.zeros((640, 480, 3), dtype=np.uint8)
        for i in range(3):
            starttime = time.time()
            self.run_model(img)
            rospy.loginfo(f"Time taken: {time.time() - starttime}")
        rospy.loginfo(f"[{rospy.get_name()}] " + "Warmed up!")

    def get_clip_embeddings(self, vocabulary, prompt='a '):
        text_encoder = build_text_encoder(pretrain=True)
        text_encoder.eval()
        texts = [prompt + x for x in vocabulary]
        emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return emb

    def run_model(self, img):
        predictions = self.predictor(img)
        instances = predictions['instances'].to(torch.device("cpu"))
        
        pred_masks = np.array(instances.pred_masks)
        scores = np.array(instances.scores)
        class_indices = np.array(instances.pred_classes)
        boxes = np.array(instances.pred_boxes.tensor)
        
        h, w = img.shape[:2]
        
        upscale_fac = self.upscale_fac
        h *= upscale_fac
        w *= upscale_fac
        new_boxes = []
        for b in boxes:
            
            x1, y1, x2, y2 = b
            y_new1 = w - x2 * upscale_fac
            x_new1 = y1 * upscale_fac
            y_new2 = w - x1 * upscale_fac
            x_new2 = y2 * upscale_fac
            b = [x_new1, y_new1, x_new2, y_new2]
            new_boxes.append(b)
            
        boxes = np.array(new_boxes.copy())
        
        seg_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)

        # largest to smallest order to reduce occlusion.
        sorted_index = np.argsort([-mask.sum() for mask in pred_masks])
        for i in sorted_index:
            mask = pred_masks[i]
            # label 0 is reserved for background label, so starting from 1
            seg_mask[mask] = (i + 1)
        
        return seg_mask, boxes, class_indices, scores
            
    def callback(self, msg: DeticDetectionsActionGoal):
        starttime = time.time()
        ros_rgb_image = msg.image
        rgb_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, 'passthrough')
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rotated_image = cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)
        seg_mask, boxes, classes, confs = self.run_model(rotated_image)

        vizimg = np.array(rgb_image).copy()
        for i in range(len(boxes)):
            b = boxes[i] / self.upscale_fac
            cls = classes[i]
            vizimg = cv2.rectangle(vizimg, (int(b[0]),int(b[1])),(int(b[2]),int(b[3])) , (255,0,0), 2)
            
            vizimg = cv2.putText(vizimg, str(self.class_list[cls]),(int(b[0]),int(b[1])),  cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0,0,255), 1, cv2.LINE_AA)

        
        classes = classes.flatten()
        boxes = boxes.flatten()
        confs = confs.flatten().astype(np.float32)
        nPredictions = len(classes)
        seg_mask = cv2.rotate(seg_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        response = DeticDetectionsActionResult()
        response.seg_mask = self.cv_bridge.cv2_to_imgmsg(seg_mask, )
        response.nPredictions  = nPredictions
        response.box_bounding_boxes = boxes.tolist()
        response.box_classes = classes.tolist()
        response.confidences = confs.tolist()


        # seg_mask = seg_mask.astype(np.float32)
        # seg_mask /= seg_mask.max()
        # seg_mask *= 255
        # seg_mask = seg_mask.astype(np.uint8)
        # seg_mask = cv2.rotate(seg_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # vizimg[:, :, 2][seg_mask != 0] = seg_mask.copy()[seg_mask != 0]
        print(response.box_classes)
        print(response.box_bounding_boxes)

        vizimg = cv2.rotate(vizimg, cv2.ROTATE_90_CLOCKWISE)
        self.annotated_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(vizimg, 'passthrough'))
        
        rospy.loginfo(f"[{rospy.get_name()}] " + f"DETIC ran in {time.time() - starttime} seconds.")
        
        self.server.set_succeeded(result=response)

    
if __name__ == "__main__":
    
    node = DeticNode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass