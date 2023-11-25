#!/home/abhinav/miniconda3/envs/vlmaps/bin/python3

from landmark_index import *
from lseg_map import *
from vis_tools import *
import rospy
import actionlib
import argparse
from collections import OrderedDict 
import os
import rospy
import cv2
from cv_bridge import CvBridge
import sensor_msgs
import json
from utils.clip_mapping_utils import *
from scipy.spatial.transform import Rotation as R
from vlmaps_ros.msg import VLMapsAction, VLMapsResult, VLMapsFeedback

class VLMapsGenerator():

    def __init__(self, ):

        rospy.init_node('vlmaps_node', anonymous=True)

        self.action_server = actionlib.SimpleActionServer(
            'lseg_server', VLMapsAction, execute_cb=self.execute_cb, auto_start=False)
        self.bridge = CvBridge()

        self.root_dir = rospy.get_param('/vlmaps_brain/root_dir')
        self.data_dir = rospy.get_param('/vlmaps_brain/data_dir')
        self.cs =   rospy.get_param('/vlmaps_brain/cs')
        self.gs =  rospy.get_param('/vlmaps_brain/gs')
        self.camera_height = rospy.get_param('/vlmaps_brain/camera_height')
        self.max_depth_value = rospy.get_param('/vlmaps_brain/max_depth_value')
        self.depth_sample_rate = rospy.get_param('/vlmaps_brain/depth_sample_rate')
        self.use_self_built_map = rospy.get_param('/vlmaps_brain/use_self_built_map')
        self.inference = rospy.get_param('/vlmaps_brain/inference')
        self.cuda_device = rospy.get_param('/vlmaps_brain/cuda_device')

        # Sanity checks
        if(self.root_dir is None):
            self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            self.data_dir = os.path.join(self.root_dir, "data", "rosbag_data/")

        # Check if checkpoints exists
        checkpoints_dir = os.path.join(self.root_dir, "scripts/lseg", "checkpoints")
        if not os.path.exists(checkpoints_dir):
            raise Exception("Checkpoints not found in {}. Please download them and place them in the checkpoints directory".format(checkpoints_dir))

        map_save_dir = os.path.join(self.data_dir, "map_correct")
        if self.use_self_built_map:
            map_save_dir = os.path.join(self.data_dir, "map")
        os.makedirs(map_save_dir, exist_ok=True)

        # Set the save paths
        self.color_top_down_save_path = os.path.join(map_save_dir, f"color_top_down_1.npy")
        self.grid_save_path = os.path.join(map_save_dir, f"grid_lseg_1.npy")
        self.obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")

        if(self.inference):
            device = self.cuda_device if torch.cuda.is_available() else "cpu"
            clip_version = "ViT-B/32"
            self.clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                            'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
            self.clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
            self.clip_model.to(device).eval()
            rospy.loginfo(f"[{rospy.get_name()}] " + "CLIP model loaded on cuda device {self.cuda_device}!!")

        rospy.loginfo(f"[{rospy.get_name()}] " + "Waiting to start server")
        self.start()
        rospy.loginfo(f"[{rospy.get_name()}] " + "Server is up")

    def generate_vlmap(self,):
        """Generates the vlmap and saves it in the data_dir"""

        create_lseg_map_batch(
            img_save_dir=self.data_dir,
            use_self_built_map=self.use_self_built_map,
            camera_height=self.camera_height,
            cs=self.cs,
            gs=self.gs,
            depth_sample_rate=self.depth_sample_rate,
            max_depth_value=self.max_depth_value,
            device = self.cuda_device if torch.cuda.is_available() else "cpu",
        )
    
    def execute_cb(self, goal):

        """ Receives labels from action goals and returns segmented masks for each label
        
        Args:
            goal: action goal containing labels
        Returns:
            masks: List[ndarray] of masks for each label
        """

        labels = goal.labels
        rospy.logdebug(f"[{rospy.get_name()}] " + "Received labels {}".format(labels))

        assert isinstance(labels, list), "Labels should be a list but found {}".format(type(labels))
        assert len(labels) > 0, "Labels should not be empty"

        mask_list = []
        # For visualizing zoomed in results
        crop_res = False
        obstacles = load_map(self.obstacles_save_path)
        xmin = 0; xmax = obstacles.shape[0]-1; ymin = 0; ymax = obstacles.shape[1]-1
        x_indices, y_indices = np.where(obstacles == 0)

        if not x_indices.shape[0] or not y_indices.shape[0]:
            rospy.logwarn("No obstacles found in the map. Please run the generate_vlmap function first.")
            x_indices,y_indices = np.where(obstacles)

        if(crop_res):
            xmin = np.min(x_indices)
            xmax = np.max(x_indices)
            ymin = np.min(y_indices)
            ymax = np.max(y_indices)
  
        # Free space mask
        no_map_mask = obstacles[xmin:xmax+1, ymin:ymax+1] > 0
        predictions = get_seg_mask(labels,self.clip_model,self.clip_feat_dim,
                                    self.grid_save_path, obstacles,
                                    xmin, xmax, ymin, ymax)

        for i in range(len(labels)):
            label_mask = np.array(predictions == i)
            label_mask[no_map_mask] = 0
            mask_list.append(label_mask)

        # Publish masks
        self.publish_masks(mask_list,labels)

    def cv2_to_imgmsg(self,cv_image):

        img_msg = sensor_msgs.msg.Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.data = cv_image.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height

        return img_msg

    def publish_masks(self,masks:list,labels:list):

        """ Publishes the masks for each label """

        vlmaps_msg = VLMapsResult()
        vlmaps_msg.header = rospy.Header()

        for i in range(len(labels)):
            mask = masks[i]
            # use numpy.buffer to convert to ros msg
            mask_msg = self.cv2_to_imgmsg(mask)
            vlmaps_msg.masks.append(mask_msg)
            rospy.logdebug(f"[{rospy.get_name()}] " + "Published mask for label {}".format(labels[i]))

        vlmaps_msg.success = True
        self.action_server.set_succeeded(vlmaps_msg)
        rospy.logdebug(f"[{rospy.get_name()}] " + "Published all masks successfully")

    def start(self):
      
        self.action_server.start()
        rospy.loginfo(f"[{rospy.get_name()}] " + " VLMaps Server Started")

if __name__ == "__main__":

    vlmaps = VLMapsGenerator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo(f"[{rospy.get_name()}] " + "Shutting down")

    

