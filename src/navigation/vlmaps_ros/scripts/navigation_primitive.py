#Author: Abhinav Gupta
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
from vlmaps_ros.msg import VLMaps_primitiveAction, VLMaps_primitiveResult, VLMaps_primitiveFeedback

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

        # Defines navigation primitives
        self.primitives_mapping = {"go_to_object": self.go_to_object,
                                    "move_between_objects": self.move_between_objects,
                                    "move_object_closest_to": self.move_object_closest_to}

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Initializing costmap processor node...")
        self.costmap_processor = ProcessCostmap()

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


    def publish_goal(self, goal):

        msg = VLMaps_primitiveResult()
        msg.header = rospy.Header()
        msg.goal = goal
        msg.success = True
        self.action_server.set_succeeded(msg)
        rospy.logdebug(f"[{rospy.get_name()}] " + "Published goals for primitive successfully")
    

    def go_to_object(self, obj1, obj2):
        """ Inferences over vlmaps and navigates to object location """

        assert obj1 in self.labels, "Object not in vlmaps labels"

        # Call vlmaps
        self.vlmaps_caller.send_goal(self.labels)
        rospy.sleep(0.5)
        mask_list = self.vlmaps_caller.results
        masks = np.array(mask_list)

        rospy.loginfo(f"[{rospy.get_name()}]:" +"Received {} masks from vlmaps".format(masks.shape[0]))

        # Postprocess the results
        outputs = postprocess_masks(masks,self.labels)
        safe_goal2D = self.get_2D_goal_rtabmap(outputs,label=obj1)
        safe_movebase_goal = convertToMovebaseGoals(safe_goal2D)

        rospy.loginfo(f"[{rospy.get_name()}]:" +"Found goal to movebase {}".format(safe_movebase_goal))
        return safe_movebase_goal


    def move_between_objects(self, obj1, obj2):
        """ Inferences over vlmaps and navigates between the two objects in the scene."""

        assert obj1 in self.labels, "Object not in vlmaps labels"
        assert obj2 in self.labels, "Object not in vlmaps labels"
        goals2D = []

        # Call vlmaps
        self.vlmaps_caller.send_goal(self.labels)
        rospy.sleep(0.5)
        mask_list = self.vlmaps_caller.results
        masks = np.array(mask_list)

        rospy.loginfo(f"[{rospy.get_name()}]:" +"Received {} masks from vlmaps".format(masks.shape[0]))

        # Postprocess the results
        outputs = postprocess_masks(masks,self.labels)
        for label in [obj1, obj2]:
            goals2D.append(list(self.get_2D_goal_rtabmap(outputs,label=label,safe= False)))
        
        goals2D = np.mean(goals2D, axis=0,keepdims=False)
        theta = goals2D[2]

        # Verify that the goal is safe
        if (not self.costmap_processor.isSafe(goals2D[0],goals2D[1],0)):
            rospy.loginfo(f"[{rospy.get_name()}]:" +"Computed goal point is not safe, finding nearest safe point")
            x,y,z = self.costmap_processor.findNearestSafePoint(goals2D[0],goals2D[1],0)
            rospy.loginfo(f"[{rospy.get_name()}]:" +"Computed safe goal point: {}, {}, {}".format(x,y,z))
            safe_goal2D = (x,y,theta)
        else:
            safe_goal2D = (goals2D[0],goals2D[1],theta)

        safe_movebase_goal = convertToMovebaseGoals(safe_goal2D)

        rospy.loginfo(f"[{rospy.get_name()}]:" +"Found goal to movebase {}".format(safe_movebase_goal))
        return safe_movebase_goal   


    def move_object_closest_to(self, obj1, obj2):
        """ Navigates to the object of category 'obj1' 
        that is closest to the object of category 'obj2' """

        for obj in zip(obj1, obj2):
            if not obj in self.labels:
                rospy.logwarn(f"[{rospy.get_name()}]:" +"Object {} not in vlmaps labels".format(obj))
                rospy.loginfo(f"[{rospy.get_name()}]:" +"Adding {} to vlmaps labels".format(obj))
                self.labels.append(" " + obj)

        # Call vlmaps
        self.vlmaps_caller.send_goal(self.labels)
        rospy.sleep(0.5)
        mask_list = self.vlmaps_caller.results
        masks = np.array(mask_list)

        rospy.loginfo(f"[{rospy.get_name()}]:" +"Received {} masks from vlmaps".format(masks.shape[0]))

        # Postprocess the results
        outputs = postprocess_masks(masks,self.labels)
        safe_goal2D = self.find_closest_pair_2D_goal(outputs,obj1,obj2)
        safe_movebase_goal = convertToMovebaseGoals(safe_goal2D)

        rospy.loginfo(f"[{rospy.get_name()}]:" +"Found goal to movebase {}".format(safe_movebase_goal))
        return safe_movebase_goal


    def find_closest_pair_2D_goal(self,results:dict,obj1:str,obj2:str)-> tuple:
        """ Finds the object of category 'obj1' 
        that is closest to the object of category 'obj2'"""

        assert obj1 in results.keys(), "Object {} not found in the results {}".format(obj1, results.keys())
        assert obj2 in results.keys(), "Object {} not found in the results {}".format(obj2, results.keys())

        ### Get the largest bbox and find the centroid of the bbox
        mask1 = results[obj1]['mask'].astype(np.uint8)
        mask2 = results[obj2]['mask'].astype(np.uint8)
        res = self.get_locations_dict_rtabmap(results,obj1)
        goal2d_obj2 = self.get_2D_goal_rtabmap(results,obj2)
        X2,Y2,theta2 = goal2d_obj2
        dist_min = 100000

        for i in range(res['x_locs'].shape[0]):
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
        X, Y, Z = self.costmap_processor.findNearestSafePoint(X,Y,Z)
        rospy.loginfo(f"[{rospy.get_name()}]:" +" Computed safe goal point: {}, {}, {}".format(X,Y,Z))
        theta_goal =0
        goal2D = (X,Y,theta_goal)
        return goal2D


    def navigation_feedback(self, msg : MoveBaseFeedback):
        self.position = msg.base_position.pose.position
        self.orientation = msg.base_position.pose.orientation
        
        if time.time() - self.last_update_time > 1:
            self.last_update_time = time.time()
            rospy.logdebug(f"[{rospy.get_name()}]:" +"Current position: {}".format(self.position))
            rospy.logdebug(f"[{rospy.get_name()}]:" +"Current orientation: {}".format(self.orientation))


    def get_2D_goal_rtabmap(self,results:dict,label:str, safe: bool= True)-> tuple:
        """Returns a single 2D goal from the vlmap in map frame"""

        ########### WARNING: This is a hacky way to get the initial pose of the robot. Get Tfs from VLMaps ros action server instead ###########
        #TODO: @ag6 - Get the initial pose of the robot and associated transforms from the vlmaps action server

        assert label in results.keys(), "Label {} not found in the results {}".format(label, results.keys())

        ### Get the largest bbox and find the centroid of the bbox
        mask = results[label]['mask'].astype(np.uint8)
        x,y  = find_best_centroid(results[label]['bboxes'])
        X, Y ,Z = convert_vlmap_id_to_world_frame(self.data_config_dir,x,y,self.cs,self.gs)
        print("X,Y,Z (unfiltered): ", X,Y,Z)

        if(safe):
            X, Y, Z = self.costmap_processor.findNearestSafePoint(X,Y,Z)
            rospy.loginfo(f"[{rospy.get_name()}]:" +" Computed safe goal point: {}, {}, {}".format(X,Y,Z))

        # set 2d goal
        theta=0 # 2d navigation
        goal2D = (X,Y,theta)

        return goal2D


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
        for i in range(res['x_locs'].shape[0]):
            X,Y,Z = convert_vlmap_id_to_world_frame(self.data_config_dir,
                                                    centroids_dict['x_locs'][i],
                                                    centroids_dict['y_locs'][i],
                                                    self.cs, self.gs)
            theta=0
            res['x_locs'].append(X)
            res['y_locs'].append(Y)
            res['z_locs'].append(Z)
            res['theta'].append(theta)
            res['areas'].append(centroids_dict['areas'][i])

        return res


    def start(self):
        """Starts the action server"""
        self.action_server.start()
        rospy.loginfo(f"[{rospy.get_name()}] " + " VLMaps Server Started")

## Utility functions

def convert_vlmap_id_to_world_frame(data_config_dir:str,x,y,cs=0.01,gs=1000):
    
    # Vlmap to rtabmap world coordinate
    YY =0 # 2d navigation
    XX,ZZ = grid_id2pos(gs,cs,x,y)
    print("XX,YY,ZZ: ", XX,YY,ZZ)
    pos_des_cam_frame = np.array([XX,YY,ZZ,1]).reshape(-1,1)

    # Position in world frame
    pos_des_world_frame = transform_to_world_frame(data_config_dir, pos_des_cam_frame)
    assert pos_des_world_frame.shape == (4,1), "pos_des_world_frame should be of shape (4,1) but found {}".format(pos_des_world_frame.shape)

    X, Y ,Z = pos_des_world_frame[:3,0]

    return X,Y,Z
    
def transform_to_world_frame(data_config_dir: str, pos_des_cam_frame: np.ndarray):
    """Transforms the desired position from camera frame to world frame"""

    # ensure that file exists
    initial_pose_path = os.path.join(data_config_dir, 'amcl_pose.json')
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

    tf_rtabmap2base = get_initial_rtabmap_pose(data_config_dir)
    tf_camera_optical2pcd = get_tf_camera_optical2pcd()
    tf_base2cam = get_tf_base2cam(data_config_dir)

    # Position in world frame
    pos_des_world_frame = tf_world2rtabmap @ tf_rtabmap2base @ \
                            tf_base2cam @ tf_camera_optical2pcd @ pos_des_cam_frame
    assert pos_des_world_frame.shape == (4,1), "pos_des_world_frame should be of shape (4,1) but found {}".format(pos_des_world_frame.shape)

    return pos_des_world_frame

def get_initial_rtabmap_pose(data_config_dir: str)-> np.ndarray:
        """Returns the initial pose of the base in /rtabmap/odom frame """

        pose_dir = os.path.join(data_config_dir,"pose")
        assert os.path.exists(pose_dir), "Pose directory not found at {}".format(pose_dir)
        pose_list = sorted(os.listdir(pose_dir), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        pose_list = [os.path.join(pose_dir, x) for x in pose_list]

        pos, rot = load_pose(pose_list[0])
        
        base_pose = np.eye(4)
        base_pose[:3, :3] = rot
        base_pose[:3, 3] = pos.reshape(-1)

        return base_pose

def get_tf_base2cam(data_config_dir: str):
    """Returns the transformation matrix from base to camera frame"""

    add_params_dir = os.path.join(data_config_dir, "add_params")
    assert os.path.exists(add_params_dir), "Additional params directory not found at {}".format(add_params_dir)
    tf_base2cam_path = os.path.join(add_params_dir, "rosbag_data_0tf_base2cam.txt")
    tf_base2cam = load_tf(tf_base2cam_path)
    return tf_base2cam

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

if __name__ == "__main__":
    task_planner = TestTaskPlanner()

    # go to object
    task_planner.go_to_object(" sofa")

    # rospy.sleep(2)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    
    

