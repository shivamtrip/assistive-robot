#!/home/abhinav/miniconda3/envs/vlmaps/bin/python3
#Author: Abhinav Gupta
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
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, PoseArray
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
from geometry_msgs.msg import Quaternion, PoseWithCovarianceStamped
from vlmaps_ros.msg import VLMaps_primitiveAction, VLMaps_primitiveResult, VLMaps_primitiveFeedback
from alfred_navigation.msg import NavManAction, NavManGoal, NavManResult, NavManFeedback
from sensor_msgs.msg import Image
import sensor_msgs.msg

class TestTaskPlanner:
    def __init__(self):
        rospy.init_node('Nav_primitive_tester')

        # Initialize all paths and directories
        self.action_server = actionlib.SimpleActionServer(
            'vlmaps_primitive_server', VLMaps_primitiveAction, execute_cb=self.execute_cb, auto_start=False)
        self.bridge = CvBridge()
        self.root_dir = rospy.get_param('/vlmaps_robot/root_dir')
        self.data_dir = rospy.get_param('/vlmaps_robot/data_dir')
        self.cs =   rospy.get_param('/vlmaps_robot/cs')
        self.gs =  rospy.get_param('/vlmaps_robot/gs')
        self.area_threshold_to_match = rospy.get_param('/vlmaps_robot/area_threshold_to_match')
        self.inference = rospy.get_param('/vlmaps_robot/inference')
        self.cuda_device = rospy.get_param('/vlmaps_robot/cuda_device')
        self.show_vis = rospy.get_param('/vlmaps_robot/show_vis')
        self.data_config_dir = os.path.join(self.data_dir, "config/data_configs")

        # Stores update time
        self.last_update_time = time.time()

        # Stores home location
        self.home_pos = read_home_pos(self.data_config_dir)

        # Defines navigation primitives
        self.primitives_mapping = {"go_to_object": self.go_to_object,
                                    "move_between_objects": self.move_between_objects,
                                    "move_object_closest_to": self.move_object_closest_to}
        
        # Initialize navigation services and clients
        self.startNavService = rospy.ServiceProxy('/switch_to_navigation_mode', Trigger)

        self.navigation_client = actionlib.SimpleActionClient('nav_man', NavManAction)
 
        self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for stow robot service...")
        self.stow_robot_service.wait_for_service()

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for Navigation Manager server...")
        self.navigation_client.wait_for_server()

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Initializing costmap processor node...")
        self.costmap_processor = ProcessCostmap()

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Initializing amcl pose subscriber...")
        self.curr_pos = None
        self.RECEIVED_AMCL_POSE = False
        self.amcl_pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_callback)

        while not self.RECEIVED_AMCL_POSE:
            rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for amcl pose...")
            rospy.sleep(0.5)

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Initializing vlmaps node...")

        self.labels= "table, chair, floor, sofa, bed, other"
        self.labels = self.labels.split(",")
        self.vlmaps_caller = vlmaps_fsm(self.labels)

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Starting Server...")
        self.start()
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Server Started.")
        rospy.loginfo(f"[{rospy.get_name()}]:" + "VLMaps Primitive Node Ready.")

    def execute_cb(self, goal):
        """ Parses the goal and calls the appropriate function to get 2d nav goals!"""
        rospy.loginfo(f"[{rospy.get_name()}]:" +"Received goal: {}".format(goal))

        self.goal_primitive = goal.primitive
        self.goal_obj1 = goal.obj1
        self.goal_obj2 = goal.obj2
        self.task_requested = True

        if self.goal_primitive not in self.primitives_mapping.keys():
            rospy.logerr(f"[{rospy.get_name()}]:" +"Primitive {} not in primitive list {}".format(self.goal_primitive, self.primitives_mapping.keys()))
            rospy.loginfo(f"[{rospy.get_name()}]:" +"Sending result to action server")
            self.action_server.set_aborted(VLMaps_primitiveResult())
            return
        
        else:
            safe_movebase_goal = self.primitives_mapping[self.goal_primitive](self.goal_obj1, self.goal_obj2)
            
        self.publish_goal(safe_movebase_goal)

    def amcl_pose_callback(self, msg):
        """Callback for amcl pose subscriber"""
        self.curr_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        rospy.loginfo(f"[{rospy.get_name()}]:" +"Current position: {}".format(self.curr_pos))
        self.RECEIVED_AMCL_POSE = True

    def publish_goal(self, status):

        msg = VLMaps_primitiveResult()
        msg.header = rospy.Header()
        msg.success = status
        if(status):
            self.action_server.set_succeeded(msg)
            rospy.logdebug(f"[{rospy.get_name()}] " + "Published goals for primitive successfully")
        else:
            self.action_server.set_aborted(msg)
            rospy.logdebug(f"[{rospy.get_name()}] " + "Failed to go to goals as desired")

    def parse_results(self, results:dict):
        """ Parses results by VLMap into masks and heatmap"""
        mask_list = results['masks']
        heatmap_list = results['heatmaps']
        masks = np.array(mask_list)
        heatmaps = np.array(heatmap_list)

        return masks, heatmaps

    def go_to_object(self, obj1, obj2):
        """ Inferences over vlmaps and navigates to object location """

        assert obj1 in self.labels, "Object not in vlmaps labels"

        # Call vlmaps
        goals_vis = []
        self.vlmaps_caller.send_goal(self.labels)
        rospy.sleep(0.5)
        results = self.vlmaps_caller.results
        masks, heatmaps = self.parse_results(results)
        
        rospy.loginfo(f"[{rospy.get_name()}]:" +"Received {} masks from vlmaps".format(masks.shape[0]))

        # Postprocess the results
        outputs = postprocess_masks(masks,self.labels)
        safe_goal2D = self.get_2D_goal_rtabmap(outputs,label=obj1,safe=True)
        safe_movebase_goal = convertToMovebaseGoals(safe_goal2D)
        goals_vis.append(safe_goal2D)

        rospy.loginfo(f"[{rospy.get_name()}]:" +"Sending goal to movebase {}".format(safe_movebase_goal))
        
        # Send goal to movebase
        # self.navigate_to_location(safe_movebase_goal)
        
        if (self.show_vis):
            publish_markers(goals_vis)
            # labels of interest
            USEFUL_LABELS = [obj1]
            publish_heatmap(heatmaps, self.labels, USEFUL_LABELS)
            # self.show_vlmaps_results(mask_list,outputs,self.labels)

    def move_between_objects(self, obj1, obj2,safe=False):
        """ Inferences over vlmaps and navigates between the two objects in the scene."""

        assert obj1 in self.labels, "Object not in vlmaps labels"
        assert obj2 in self.labels, "Object not in vlmaps labels"
        goals2D = []
        goals_vis = []

        # Call vlmaps
        self.vlmaps_caller.send_goal(self.labels)
        rospy.sleep(0.5)
        results = self.vlmaps_caller.results
        masks, heatmaps = self.parse_results(results)

        rospy.loginfo(f"[{rospy.get_name()}]:" +"Received {} masks from vlmaps".format(masks.shape[0]))

        # Postprocess the results
        outputs = postprocess_masks(masks,self.labels)
        for label in [obj1, obj2]:
            goal2D = list(self.get_2D_goal_rtabmap(outputs,label=label,safe= False))
            goals2D.append(goal2D)
            goals_vis.append(goal2D)
        
        goals2D = np.mean(goals2D, axis=0,keepdims=False)
        theta = goals2D[2]

        # Verify that the goal is safe
        ## For now, we are not checking if the goal is safe
        ## Costmap based functions are not stable. Need to fix them

        if (safe and (not self.costmap_processor.isSafe(goals2D[0],goals2D[1],0,ref_cost_min=10,ref_cost_max=50))):
            rospy.loginfo(f"[{rospy.get_name()}]:" +"Computed goal point is not safe, finding nearest safe point")
            x,y,z = self.costmap_processor.findNearestSafePoint(goals2D[0],goals2D[1],0,ref_cost_min=10,ref_cost_max=50)
            rospy.loginfo(f"[{rospy.get_name()}]:" +"Computed safe goal point: {}, {}, {}".format(x,y,z))
            safe_goal2D = (x,y,theta)
        else:
            rospy.loginfo(f"[{rospy.get_name()}]:" +"Computed goal mean point is safe")
            safe_goal2D = (goals2D[0],goals2D[1],theta)

        goals_vis.append(safe_goal2D)

        safe_movebase_goal = convertToMovebaseGoals(safe_goal2D)
        rospy.loginfo(f"[{rospy.get_name()}]:" +"Sending goal to movebase {}".format(safe_movebase_goal))

        ##### Publish to Rviz for debugging #####
        if(self.show_vis):
            publish_markers(goals_vis)
            # labels of interest
            USEFUL_LABELS = [obj1,obj2]
            publish_heatmap(heatmaps, self.labels, USEFUL_LABELS)
            return
        else:
            self.navigate_to_location(safe_movebase_goal)
        
    def move_object_closest_to(self, obj1, obj2):
        """ Navigates to the object of category 'obj1' 
        that is closest to the object of category 'obj2' """

        for obj in [obj1, obj2]:
            print("Object=",obj)
            if not obj in self.labels:
                rospy.logwarn(f"[{rospy.get_name()}]:" +"Object {} not in vlmaps labels".format(obj))
                rospy.loginfo(f"[{rospy.get_name()}]:" +"Adding {} to vlmaps labels".format(obj))
                self.labels.append(obj)

        # Call vlmaps
        self.vlmaps_caller.send_goal(self.labels)
        rospy.sleep(0.5)
        results = self.vlmaps_caller.results
        masks, heatmaps = self.parse_results(results)

        rospy.loginfo(f"[{rospy.get_name()}]:" +"Received {} masks from vlmaps".format(masks.shape[0]))

        # Postprocess the results
        outputs = postprocess_masks(masks,self.labels)
        safe_goal2D = self.find_closest_pair_2D_goal(outputs,obj1,obj2,safe=False)
        safe_movebase_goal = convertToMovebaseGoals(safe_goal2D)
        rospy.loginfo(f"[{rospy.get_name()}]:" +"Sending goal to movebase {}".format(safe_movebase_goal))

        # Send goal to movebase
        # self.navigate_to_location(safe_movebase_goal)
        
        if (self.show_vis):
            # labels of interest
            USEFUL_LABELS = [obj1,obj2]
            publish_heatmap(heatmaps, self.labels, USEFUL_LABELS)

    def find_closest_pair_2D_goal(self,results:dict,obj1:str,obj2:str,safe:False)-> tuple:
        """ Finds the object of category 'obj1' 
        that is closest to the object of category 'obj2'"""

        assert obj1 in results.keys(), "Object {} not found in the results {}".format(obj1, results.keys())
        assert obj2 in results.keys(), "Object {} not found in the results {}".format(obj2, results.keys())

        ### Get the largest bbox and find the centroid of the bbox
        mask1 = results[obj1]['mask'].astype(np.uint8)
        mask2 = results[obj2]['mask'].astype(np.uint8)
        res = self.get_locations_dict_rtabmap(results,obj1)
        goal2d_obj2 = self.get_2D_goal_rtabmap(results,obj2,safe=False)
        X2,Y2,theta2 = goal2d_obj2
        dist_min = 100000

        for i in range(len(res['x_locs'])):
            area = res['areas'][i]
            if(area < self.area_threshold_to_match):
                continue
            X1,Y1, Z1 = res['x_locs'][i],res['y_locs'][i], res['z_locs'][i]
            # find distance between x1,y1 and X2,Y2
            dist = np.sqrt((X1-X2)**2 + (Y1-Y2)**2)
            if(dist < dist_min):
                dist_min = dist
                X, Y, Z = X1,Y1,Z1
        
        # set 2d goal
        if(safe):
            X, Y, Z = self.costmap_processor.findNearestSafePoint(X,Y,Z,ref_cost_min=10, ref_cost_max=50)
            rospy.loginfo(f"[{rospy.get_name()}]:" +" Computed safe goal point: {}, {}, {}".format(X,Y,Z))

        theta_goal = self.find_theta(obj1)
        goal2D = (X,Y,theta_goal)

        # Publish markers
        if(self.show_vis):
            goals_vis = []
            goals_vis.append(goal2D)
            goals_vis.append(goal2d_obj2)
            publish_markers(goals_vis)
           
        return goal2D

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
    
    def transform_to_world_frame(self, pos_des_cam_frame: np.ndarray):
        """Transforms the desired position from camera frame to world frame"""

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

        # Position in world frame
        pos_des_world_frame = tf_world2rtabmap @ tf_rtabmap2base @ \
                                tf_base2cam @ tf_camera_optical2pcd @ pos_des_cam_frame
        assert pos_des_world_frame.shape == (4,1), "pos_des_world_frame should be of shape (4,1) but found {}".format(pos_des_world_frame.shape)

        return pos_des_world_frame

    def convert_vlmap_id_to_world_frame(self, x,y):
        
        # Vlmap to rtabmap world coordinate
        YY =0 # 2d navigation
        XX,ZZ = grid_id2pos(self.gs,self.cs,x,y)
        print("XX,YY,ZZ: ", XX,YY,ZZ)
        pos_des_cam_frame = np.array([XX,YY,ZZ,1]).reshape(-1,1)

        # Position in world frame
        pos_des_world_frame = self.transform_to_world_frame(pos_des_cam_frame)
        assert pos_des_world_frame.shape == (4,1), "pos_des_world_frame should be of shape (4,1) but found {}".format(pos_des_world_frame.shape)

        X, Y ,Z = pos_des_world_frame[:3,0]

        return X,Y,Z

    def get_2D_goal_rtabmap(self,results:dict,label:str, safe: bool = True)-> tuple:
        """Returns a single 2D goal from the vlmap in map frame"""

        ########### WARNING: This is a hacky way to get the initial pose of the robot. Get Tfs from VLMaps ros action server instead ###########
        #TODO: @ag6 - Get the initial pose of the robot and associated transforms from the vlmaps action server

        assert label in results.keys(), "Label {} not found in the results {}".format(label, results.keys())
        rospy.loginfo(f"[{rospy.get_name()}]:" +"Running test goal")

        ### Get the largest bbox and find the centroid of the bbox
        mask = results[label]['mask'].astype(np.uint8)
        x,y  = find_best_centroid(results[label]['bboxes'])
        X, Y ,Z = self.convert_vlmap_id_to_world_frame(x,y)

        if(safe):
            goal_des = (X,Y,Z)
            curr_pos = self.curr_pos
            home_pos = self.home_pos
            # X, Y, Z = self.costmap_processor.findNearestSafePoint(X,Y,Z,ref_cost_min=10, ref_cost_max=15)
            # X, Y, Z = self.costmap_processor.findStraightLineSafePoint(goal_des,curr_pos,num_points=100,ref_cost_min=10, ref_cost_max=20)
            X, Y, Z = self.findStraightLinePoint(goal_des,home_pos,num_points=100)
            rospy.loginfo(f"[{rospy.get_name()}]:" +" Computed safe goal point: {}, {}, {} for label {}".format(X,Y,Z,label))

        theta = self.find_theta(label)
        goal2D = (X,Y,theta)

        rospy.loginfo(f"[{rospy.get_name()}]:" +" Computed goal point: {}, {}, {} for label{}".format(X,Y,Z,label))

        return goal2D
    
    def findStraightLinePoint(self, goal_des, curr_pos, num_points=100,radius = 1):
        """Finds a point on the straight line connecting the goal and current position"""

        assert len(goal_des) == 3, "goal_des should be of length 3 but found {}".format(len(goal_des))
        assert len(curr_pos) == 3, "curr_pos should be of length 3 but found {}".format(len(curr_pos))

        # Find straight line
        x1, y1, z1 = goal_des
        x2, y2, z2 = curr_pos
        x = np.linspace(x1,x2,num_points)
        y = np.linspace(y1,y2,num_points)

        # Iterate
        for i in range(num_points):
            x_curr, y_curr = x[i], y[i]
            dist_to_goal = np.sqrt((x_curr-x1)**2 + (y_curr-y1)**2)

            if(dist_to_goal > radius):
                rospy.loginfo(f"[{rospy.get_name()}]:" +"Found safe point on straight line")
                return x_curr, y_curr, 0
            
        rospy.loginfo(f"[{rospy.get_name()}]:" +"Could not find safe point on straight line, returning current position")
        return curr_pos
            
    def get_locations_dict_rtabmap(self,results:dict,label:str)-> tuple:
        """Returns the 2D locations of all labels bboxes from the vlmap in 2D map frame"""
        res={}
        res['x_locs'] = []
        res['y_locs'] = []
        res['z_locs'] = []
        res['areas'] = []
        res['theta'] = []    
        assert label in results.keys(), "Label {} not found in the results {}".format(label, results.keys())

        ### Get the largest bbox and find the centroid of the bbox
        mask = results[label]['mask'].astype(np.uint8)
        centroids_dict  = find_centroids(results[label]['bboxes'])

        # Vlmap to rtabmap world coordinate
        for i in range(len(centroids_dict['x_locs'])):
            X,Y,Z = self.convert_vlmap_id_to_world_frame(centroids_dict['x_locs'][i],centroids_dict['y_locs'][i])
            theta=self.find_theta(label)
            res['x_locs'].append(X)
            res['y_locs'].append(Y)
            res['z_locs'].append(Z)
            res['theta'].append(theta)
            res['areas'].append(centroids_dict['areas'][i])
            rospy.loginfo(f"[{rospy.get_name()}]:" +" Area of mask: {}".format(centroids_dict['areas'][i]))

        return res
    
    def navigate_to_location(self, goal:MoveBaseGoal):
        """Navigates to the given location"""
        self.navigation_client.send_goal(goal, feedback_cb = self.navigation_feedback)
        wait = self.navigation_client.wait_for_result()

        if self.navigation_client.get_state() != actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo(f"[{rospy.get_name()}]:" +"Failed to reach goal at X = {}, Y = {}".format(goal.target_pose.pose.position.x, goal.target_pose.pose.position.y))
            # cancel navigation
            self.navigation_client.cancel_goal()
            self.publish_goal(False)
            return False
        
        rospy.loginfo(f"[{rospy.get_name()}]:" +"Reached goal at X = {}, Y = {}".format(goal.target_pose.pose.position.x, goal.target_pose.pose.position.y))
        self.publish_goal(True)
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

    def find_theta(self, label):
        """ Takes hardcoded orientations"""

        if (label in self.location_priors.keys()): # hardcoding the orientation for now
            theta = self.location_priors[label]['yaw']
            theta = np.deg2rad(theta)
        else:
            theta = 0

        return theta

    def start(self):
        """Starts the action server"""
        self.action_server.start()
        rospy.loginfo(f"[{rospy.get_name()}] " + " VLMaps Primitive Server Started")

## Utility functions

def find_best_centroid(bboxes: np.ndarray):
    """Finds the centroid of the mask"""

    bboxes = np.array(bboxes)
    bboxes = np.reshape(bboxes,(-1,4))
    assert bboxes.shape[1] == 4, "bboxes should be of shape (N,4) but found {}".format(bboxes.shape)

    # Find areas
    areas = (bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1])
    print("areas: ", areas)
    max_id = np.argmax(areas)
    max_bbox = bboxes[max_id]

    # Find centroid
    x,y  = (max_bbox[0] + max_bbox[2])/2, (max_bbox[1] + max_bbox[3])/2

    return x,y

def find_centroids(bboxes: np.ndarray):
    """Finds the centroids of the mask"""

    centroids_dict={}

    bboxes = np.array(bboxes)
    bboxes = np.reshape(bboxes,(-1,4))
    assert bboxes.shape[1] == 4, "bboxes should be of shape (N,4) but found {}".format(bboxes.shape)

    # Find areas
    areas = (bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1])

    # Find centroid
    x_locs,y_locs  = (bboxes[:,0] + bboxes[:,2])/2, (bboxes[:,1] + bboxes[:,3])/2

    assert x_locs.shape == (bboxes.shape[0],), "x_locs should be of shape (N,) but found {}".format(x_locs.shape)

    centroids_dict['areas'] = areas
    centroids_dict['x_locs'] = x_locs
    centroids_dict['y_locs'] = y_locs

    return centroids_dict

def convertToMovebaseGoals(goals2D: tuple)-> MoveBaseGoal:
    """Converts 2D goals to movebase goals"""

    assert isinstance(goals2D, tuple), "goals2D should be of type tuple but found {}".format(type(goals2D))
    assert len(goals2D) == 3, "goals2D should be of length 3 (x,y,theta) but found {}".format(len(goals2D))

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.pose.position.x = goals2D[0]
    goal.target_pose.pose.position.y = goals2D[1]
    theta = goals2D[2]
    quaternion = Quaternion(*tf.transformations.quaternion_from_euler(0.0, 0.0, math.radians(theta)))
    goal.target_pose.pose.orientation = quaternion

    return goal

def get_marker(pose: Pose, marker_id:int = 0):
    """Creates a marker message for visualization in rviz"""

    # Create a Marker message and set its properties
    marker = Marker()
    marker.id = marker_id
    marker.header.frame_id = "map"
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose = pose
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0
    marker.color.r = ((marker_id+1)%3)/2
    marker.color.g = ((marker_id+2)%3)/2
    marker.color.b = (((marker_id+3))%3)/2
    return marker

def publish_markers(goals2D: list):
    """Publishes markers array for visualization in rviz"""
    marker_array_pub = rospy.Publisher('/markersarray', MarkerArray, queue_size=10)
    marker_array_msg = MarkerArray()

    for idx,goal in enumerate(goals2D):

        x,y,yaw = goal
        z, roll, pitch = 0, 0,0
        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation = Quaternion(*quaternion)

        marker = get_marker(pose,marker_id=idx)
        marker_array_msg.markers.append(marker)

    # Publish 10 times for visualization
    for i in range(5):
        rospy.loginfo(f"[{rospy.get_name()}]:" +"Publishing marker for all goals {}".format(goals2D))
        marker_array_pub.publish(marker_array_msg)
        rospy.sleep(0.5)

def publish_heatmap(heatmaps, labels, useful_labels):

    height = heatmaps.shape[1]
    width = heatmaps.shape[2]
    heatmap_img = []
    for idx,label in enumerate(labels):
        if label in useful_labels:
            heatmap_img.append(heatmaps[idx])

    heatmap_img = np.mean(heatmap_img, axis=0,keepdims=False)
    heatmap_img = np.reshape(heatmap_img, (height,width,3))

    # Publish heatmap
    heatmap_img_msg = sensor_msgs.msg.Image()
    heatmap_img_msg.height = heatmap_img.shape[0]
    heatmap_img_msg.width = heatmap_img.shape[1]
    heatmap_img_msg.encoding = "bgr8"
    heatmap_img_msg.is_bigendian = 0
    heatmap_img_msg.data = heatmap_img.tostring()
    heatmap_img_msg.step = len(heatmap_img_msg.data) // heatmap_img_msg.height

    rospy.loginfo(f"[{rospy.get_name()}]:" +"Publishing heatmap")
    heatmap_pub = rospy.Publisher('/heatmap', Image, queue_size=10)
    heatmap_pub.publish(heatmap_img_msg)

    return

def read_home_pos(data_config_dir:str):
    """ Reads home amcl location from json file"""

    # ensure that file exists
    initial_pose_path = os.path.join(data_config_dir, 'amcl_pose.json')
    assert os.path.exists(initial_pose_path), "Initial pose json file not found at {}".format(initial_pose_path)
    
    f = open(initial_pose_path)
    initial_pose_dict = json.load(f)
    pos = initial_pose_dict['position']
    home_pos = (pos['x'],pos['y'],pos['z'])

    return home_pos

if __name__ == "__main__":
    task_planner = TestTaskPlanner()

    obj_list = ["sofa", "potted plant", "sink", "refrigerator", "table"]

    # go to object
    # for obj in obj_list:
    #     rospy.loginfo(f"[{rospy.get_name()}]:" +"Going to object {}".format(obj))
    #     task_planner.go_to_object(obj, None)
    #     rospy.sleep(1)
    
    # task_planner.go_to_object("table")
    # task_planner.move_between_objects("sofa", "refrigerator")
    # task_planner.move_between_objects("sofa", "potted plant")
    # task_planner.move_between_objects("potted plant", "table")
    # task_planner.move_between_objects("table", "sink")
    # task_planner.move_object_closest_to("table", "sofa")

    # rospy.sleep(2)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    
    

