#!/usr/local/lib/robot_env/bin/python3

#
# Copyright (C) 2023 Auxilio Robotics
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

# ROS imports
import actionlib

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionResult, MoveBaseResult, MoveBaseFeedback

import os
import json
from enum import Enum
import tf
import tf.transformations
import time
import math
from call_action import vlmaps_fsm
from costmap_callback import ProcessCostmap
from landmark_index import *
from utils.clip_mapping_utils import *
from vis_tools import *
from cv_bridge import CvBridge
from geometry_msgs.msg import Quaternion

class TestTaskPlanner:
    def __init__(self):
        rospy.init_node('Nav_primitive_tester')

        # Initialize all paths and directories
        self.bridge = CvBridge()
        self.root_dir = rospy.get_param('/vlmaps_robot/root_dir')
        self.data_dir = rospy.get_param('/vlmaps_robot/data_dir')
        self.cs =   rospy.get_param('/vlmaps_robot/cs')
        self.gs =  rospy.get_param('/vlmaps_robot/gs')
        self.camera_height = rospy.get_param('/vlmaps_robot/camera_height')
        self.max_depth_value = rospy.get_param('/vlmaps_robot/max_depth_value')
        self.depth_sample_rate = rospy.get_param('/vlmaps_robot/depth_sample_rate')
        self.use_self_built_map = rospy.get_param('/vlmaps_robot/use_self_built_map')
        self.inference = rospy.get_param('/vlmaps_robot/inference')
        self.cuda_device = rospy.get_param('/vlmaps_robot/cuda_device')
        self.show_vis = rospy.get_param('/vlmaps_robot/show_vis')
        self.data_config_dir = os.path.join(self.data_dir, "config/data_configs")

        # map_save_dir = os.path.join(self.data_dir, "map_correct")
        # if self.use_self_built_map:
        #     map_save_dir = os.path.join(self.data_dir, "map")
        # os.makedirs(map_save_dir, exist_ok=True)

        # self.color_top_down_save_path = os.path.join(map_save_dir, f"color_top_down_1.npy")
        # self.grid_save_path = os.path.join(map_save_dir, f"grid_lseg_1.npy")
        # self.obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")

        # Stores update time
        self.last_update_time = time.time()
        
        # Initialize navigation services and clients
        self.startNavService = rospy.ServiceProxy('/switch_to_navigation_mode', Trigger)

        self.navigation_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
 
        self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for stow robot service...")
        self.stow_robot_service.wait_for_service()

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for move_base server...")
        self.navigation_client.wait_for_server()

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Initializing costmap processor node...")
        self.costmap_processor = ProcessCostmap()

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Initializing vlmaps node...")

        self.labels= "table, chair, floor, sofa, bed, other"
        self.labels = self.labels.split(",")
        self.vlmaps_caller = vlmaps_fsm(self.labels)

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Node Ready.")
        
    def executeTask(self):

        self.stow_robot_service()
        self.startNavService()
        rospy.sleep(0.5)

        # Implement navigation primitive - go_to_object(obj A)
        self.go_to_object("sofa")

        navSuccess = self.navigate_to_location(self.navigationGoal)
        if not navSuccess:
            rospy.loginfo("Failed to navigate to table, adding intermediate waypoints")
            navSuccess = self.navigate_to_location(self.navigationGoal)
   
        self.task_requested = False
        self.time_since_last_command = rospy.Time.now()

    def go_to_object(self, objectName):
        """ Inferences over vlmaps and navigates to object location """

        assert objectName in self.labels, "Object not in vlmaps labels"

        # Call vlmaps
        self.vlmaps_caller.send_goal(self.labels)
        rospy.sleep(0.5)
        mask_list = self.vlmaps_caller.results
        masks = np.array(mask_list)

        rospy.loginfo(f"[{rospy.get_name()}]:" +"Received {} masks from vlmaps".format(masks.shape[0]))

        # Postprocess the results
        outputs = postprocess_masks(masks,self.labels)
        safe_movebase_goal = self.get_movebase_goals_rtabmap(outputs,label=' '+'sofa')

        rospy.loginfo(f"[{rospy.get_name()}]:" +"Sending goal to movebase {}".format(safe_movebase_goal))
        
        # Send goal to movebase
        self.navigate_to_location(safe_movebase_goal)
        
        if (self.show_vis):
            self.show_vlmaps_results(mask_list,outputs,self.labels)

    def navigation_feedback(self, msg : MoveBaseFeedback):
        self.position = msg.base_position.pose.position
        self.orientation = msg.base_position.pose.orientation
        
        if time.time() - self.last_update_time > 1:
            self.last_update_time = time.time()
            rospy.logdebug(f"[{rospy.get_name()}]:" +"Current position: {}".format(self.position))
            rospy.logdebug(f"[{rospy.get_name()}]:" +"Current orientation: {}".format(self.orientation))

    def get_initial_rtabmap_pose(self,):
        """Returns the initial pose of the base in /rtabmap/odom frame """

        pose_dir = os.path.join(self.data_config_dir,"pose")
        assert os.path.exists(pose_dir), "Pose directory not found at {}".format(pose_dir)
        pose_list = sorted(os.listdir(pose_dir), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        pose_list = [os.path.join(pose_dir, x) for x in pose_list]

        pos, rot = load_pose(pose_list[0])
        
        base_pose = np.eye(4)
        base_pose[:3, :3] = rot
        base_pose[:3, 3] = pos.reshape(-1)

        return base_pose
    
    def get_tf_base2cam(self,):
        """Returns the transformation matrix from base to camera frame"""

        add_params_dir = os.path.join(self.data_config_dir, "add_params")
        assert os.path.exists(add_params_dir), "Additional params directory not found at {}".format(add_params_dir)
        tf_base2cam_path = os.path.join(add_params_dir, "rosbag_data_0tf_base2cam.txt")
        tf_base2cam = load_tf(tf_base2cam_path)
        return tf_base2cam

    def get_movebase_goals_rtabmap(self,results:dict,label:str)-> MoveBaseGoal:
        """Returns the movebase goals from the vlmap"""

        ########### WARNING: This is a hacky way to get the initial pose of the robot. Get Tfs from VLMaps ros action server instead ###########
        #TODO: @ag6 - Get the initial pose of the robot and associated transforms from the vlmaps action server

        # ensure that file exists
        initial_pose_path = os.path.join(self.data_config_dir, 'amcl_pose.json')
        assert os.path.exists(initial_pose_path), "Initial pose json file not found at {}".format(initial_pose_path)
        
        f = open(initial_pose_path)
        initial_pose_dict = json.load(f)
        pos = initial_pose_dict['position']
        pos = np.array([pos['x'],pos['y'],pos['z']])
        rot = initial_pose_dict['orientation']
        rot = np.array([rot['x'],rot['y'],rot['z'],rot['w']])

        tf_world2rtabmap = np.eye(4)
        tf_world2rtabmap[:3,:3] = R.from_quat(rot).as_matrix()
        tf_world2rtabmap[:3,3] = np.array(pos)

        tf_rtabmap2base = self.get_initial_rtabmap_pose()
        tf_camera_optical2pcd = get_tf_camera_optical2pcd()
        tf_base2cam = self.get_tf_base2cam()

        assert label in results.keys(), "Label {} not found in the results {}".format(label, results.keys())
        mask = results[label]['mask'].astype(np.uint8)

        bboxes = np.array(results[label]['bboxes'])
        bboxes = np.reshape(bboxes,(-1,4))

        assert bboxes.shape[1] == 4, "bboxes should be of shape (N,4) but found {}".format(bboxes.shape)

        # find areas of all bboxes
        areas = (bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1])
        print("areas: ", areas)
        max_id = np.argmax(areas)
        # find the largest bbox
        max_bbox = bboxes[max_id]

        # find centroid of the bbox
        x,y  = (max_bbox[0] + max_bbox[2])/2, (max_bbox[1] + max_bbox[3])/2

        print("x,y: ", x,y)

        # Vlmap to rtabmap world coordinate
        YY =0 # 2d navigation
        XX,ZZ = grid_id2pos(self.gs,self.cs,x,y)
        print("XX,YY,ZZ: ", XX,YY,ZZ)
        pos_des_cam_frame = np.array([XX,YY,ZZ,1]).reshape(-1,1)

        # find euler angles from tf_base2cam
        euler_angles = R.from_matrix(tf_base2cam[:3,:3]).as_euler('xyz', degrees=True)

        # Position in world frame
        pos_des_world_frame = tf_world2rtabmap @ tf_rtabmap2base @ \
                                tf_base2cam @ tf_camera_optical2pcd @ pos_des_cam_frame
        
        assert pos_des_world_frame.shape == (4,1), "pos_des_world_frame should be of shape (4,1) but found {}".format(pos_des_world_frame.shape)

        X, Y ,Z = pos_des_world_frame[:3,0]

        print("X,Y,Z (unfiltered): ", X,Y,Z)

        X, Y, Z = self.costmap_processor.findNearestSafePoint(X,Y,Z)
        rospy.loginfo(f"[{rospy.get_name()}]:" +" Computed safe goal point: {}, {}, {}".format(X,Y,Z))

        # make movebase goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.pose.position.x = X
        goal.target_pose.pose.position.y = Y
        theta = 0
        quaternion = Quaternion(*tf.transformations.quaternion_from_euler(0.0, 0.0, math.radians(theta)))
        goal.target_pose.pose.orientation = quaternion

        return goal


    def navigate_to_location(self, goal:MoveBaseGoal):
        """Navigates to the given location"""
        self.navigation_client.send_goal(goal, feedback_cb = self.navigation_feedback)
        wait = self.navigation_client.wait_for_result()

        if self.navigation_client.get_state() != actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo(f"[{rospy.get_name()}]:" +"Failed to reach goal at X = {}, Y = {}".format(goal.target_pose.pose.position.x, goal.target_pose.pose.position.y))
            # cancel navigation
            self.navigation_client.cancel_goal()
            return False
        
        rospy.loginfo(f"[{rospy.get_name()}]:" +"Reached goal at X = {}, Y = {}".format(goal.target_pose.pose.position.x, goal.target_pose.pose.position.y))
       
        return True
    
    def show_vlmaps_results(self, mask_list, outputs, labels):

        crop_res = False # Set to true for visualizing zoomed in results
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

        # Visualize label masks
        show_label_masks(mask_list,labels)

        # Visualize all bounding boxes over masks
        show_bounding_boxes(outputs,labels)
        
        seg, patches = get_colored_seg_mask(labels,self.clip_model,self.clip_feat_dim,
                                self.grid_save_path, obstacles,
                                xmin, xmax, ymin, ymax) 

        # More visualizations
        color_top_down = load_map(self.color_top_down_save_path)
        show_color_top_down(color_top_down,obstacles=obstacles)
        show_seg_mask(seg, patches)
        show_obstacle_map(obstacles)


if __name__ == "__main__":
    task_planner = TestTaskPlanner()

    # go to object
    task_planner.go_to_object(" sofa")

    # rospy.sleep(2)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    
    

