import numpy as np
import rospy
import tf
# from scene_parser import SceneParser
from scene_parser_refactor import SceneParser
from manip_basic_control import ManipulationMethods
import actionlib
from manipulation.msg import PlaceTriggerAction, PlaceTriggerFeedback, PlaceTriggerResult, PlaceTriggerGoal
from helpers import move_to_pose
import time
import actionlib

class PlaceManager():
    def __init__(self, scene_parser : SceneParser, trajectory_client, manipulation_methods : ManipulationMethods):
        self.node_name = 'pick_manager'
        self.scene_parser = scene_parser
        self.trajectory_client = trajectory_client
        self.manipulationMethods = manipulation_methods
        
        self.server = actionlib.SimpleActionServer('place_action', PlaceTriggerAction, execute_cb=self.execute_cb, auto_start=False)
        
        self.vs_range = rospy.get_param('/manipulation/vs_range') 
        self.vs_range[0] *= np.pi/180
        self.vs_range[1] *= np.pi/180
        
        
        self.offset_dict = rospy.get_param('/manipulation/offsets')
        self.isContactDict = rospy.get_param('/manipulation/contact')
        
        self.class_list = rospy.get_param('/object_detection/class_list')
        self.label2name = {i : self.class_list[i] for i in range(len(self.class_list))}
        self.server.start()
        
        rospy.loginfo("Loaded pick manager.")
    
    def place(self, fixed_place = False, location = None, is_rotate = True):
        
        self.heightOfObject = self.goal.heightOfObject
        
        if is_rotate:
            base_rotate = np.pi/2
        else:
            base_rotate = 0
            
        move_to_pose(
            self.trajectory_client,
            {
                'head_pan;to' : -np.pi/2,
                'head_tilt;to' : - 30 * np.pi/180,
                'base_rotate;by': base_rotate,
            }
        )
        if base_rotate != 0:
            rospy.sleep(5)

        if location is not None:
            rospy.loginfo("Placing at location: {}".format(location))
            placingLocation = location
        else:
            if not fixed_place:
                rospy.loginfo("Scanning for planes")
                self.scene_parser.set_point_cloud(publish = True) #converts depth image into point cloud
                plane = self.scene_parser.get_plane(publish = True)
                placingLocation = self.scene_parser.get_placing_location(plane, self.heightOfObject, publish = True)
                rospy.loginfo("Obtained placing location: {}".format(placingLocation))
            else:
                placingLocation = np.array([
                    0.0, 0.7, 0.9
                ])
                rospy.loginfo("Using fixed placing location: {}".format(placingLocation))
                
        success = self.manipulationMethods.place(placingLocation)
        return success
    
    def execute_cb(self, goal : PlaceTriggerGoal):
        starttime = time.time()
        self.goal = goal
        
        place_location = None        
        use_fixed_place = goal.fixed_place
        is_rotate = goal.is_rotate
        if goal.use_place_location:
            place_location = goal.place_location
            
        self.scene_parser.set_parse_mode("YOLO", None)
        success = self.place(
            fixed_place = use_fixed_place,
            location = place_location,
            is_rotate = goal.is_rotate
        )
        rospy.loginfo("Place action took " + str(time.time() - starttime) + " seconds.")
        result = PlaceTriggerResult()
        result.success = success
        if success:
            self.server.set_succeeded(result)
        else:
            self.server.set_aborted(result)