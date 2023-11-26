import numpy as np
import rospy
import tf
# from scene_parser import SceneParser
from scene_parser_refactor import SceneParser
from visual_servoing import AlignToObject
from manip_basic_control import ManipulationMethods
import actionlib
from manipulation.msg import DrawerTriggerAction, DrawerTriggerActionGoal, DrawerTriggerActionResult
from helpers import move_to_pose
import time
from enum import Enum
class States(Enum):
    IDLE = 0
    ALIGNING = 1
    MOVING_TO_OBJECT = 2
    REALIGNMENT = 3
    OPEN_DRAWER = 4
    CLOSE_DRAWER = 5
    COMPLETE = 6

class DrawerManager():
    def __init__(self, scene_parser : SceneParser, trajectory_client, manipulation_methods : ManipulationMethods):
        self.node_name = 'drawer_manager'
        self.scene_parser = scene_parser
        self.trajectory_client = trajectory_client
        self.manipulationMethods = manipulation_methods
        
        self.vs_range = rospy.get_param('/manipulation/vs_range') 
        self.vs_range[0] *= np.pi/180
        self.vs_range[1] *= np.pi/180
        
        self.head_tilt_angle_search = rospy.get_param('/manipulation/head_tilt_angle_search_drawer') 
        self.head_tilt_angle_grasp = rospy.get_param('/manipulation/head_tilt_angle_grasp')
        
        self.recoveries = rospy.get_param('/manipulation/recoveries')
        
        self.forward_vel = rospy.get_param('/manipulation/forward_velocity')
        self.interval_before_restart = rospy.get_param('/manipulation/forward_velocity')
        self.max_graspable_distance = rospy.get_param('/manipulation/max_graspable_distance')

        self.server = actionlib.SimpleActionServer('drawer_action', DrawerTriggerAction, execute_cb=self.execute_cb, auto_start=False)
        
        self.aruco_id = 0
        self.drawer_location = None

        rospy.loginfo("Drawer manager is ready...")
        self.server.start()
    
    def execute_cb(self, goal: DrawerTriggerActionGoal):
        self.scene_parser.set_parse_mode("ARUCO", self.aruco_id)

        if goal.to_open:
            location, success = self.open_drawer()
        else:
            success = self.close_drawer()
            self.drawer_location = None
            
        if success:
            self.drawer_location = location
            response = DrawerTriggerActionResult()
            response.success= True
            response.saved_position = location
            self.server.set_succeeded(response)
        else:
            response = DrawerTriggerActionResult()
            response.success= False
            self.server.set_aborted(response)
    
    def find_and_align_to_drawer(self):
        """
        as soon as the aruco is found, reorient and align to it instantly.
        """
        
        rospy.loginfo(f"[{self.node_name}]: Finding and orienting to Drawer")
        move_to_pose(self.trajectory_client, {
            'head_tilt;to' : - self.head_tilt_angle_search * np.pi/180,
            'head_pan;to' : self.vs_range[0],
        })
        rospy.sleep(2)
        nRotates = 0
        while not rospy.is_shutdown():
            for i, angle in enumerate(np.linspace(self.vs_range[0], self.vs_range[1], self.vs_range[2])):
                move_to_pose(self.trajectory_client, {
                    'head_pan;to' : angle,
                })
                rospy.sleep(self.vs_range[3]) #settling time.
                (x, y, z, r, p, angleToGo, confidence, pred_time), isLive = self.scene_parser.get_latest_observation()
                if isLive:
                    drawer_loc, success = self.realign_base_to_aruco(offset = -np.pi/2)
                    if success:
                        move_to_pose(self.trajectory_client, {
                            'head_pan;to' : 0,
                        })
                        return True

            rospy.loginfo("Object not found, rotating base.")
            move_to_pose(self.trajectory_client, {
                'base_rotate;by' : -np.pi/2,
                'head_pan;to' : self.vs_range[0],
            })
            rospy.sleep(5)
            self.scene_parser.clear_observations()
            
            nRotates += 1
            if nRotates >= self.recoveries['n_scan_attempts']:
                return False
        
        return False

    def move_to_drawer(self):
        rospy.loginfo(f"{self.node_name} Moving towards the object")
        startTime = time.time()

        vel = self.forward_vel
        intervalBeforeRestart = self.interval_before_restart

        move_to_pose(self.trajectory_client, {
            'head_tilt;to' : - self.head_tilt_angle_grasp * np.pi/180,
            'head_pan;to' : 0.0,
        })
        rospy.sleep(1)
        
        if not self.scene_parser.clearAndWaitForNewObject(3):
            return False

        maxGraspableDistance = self.max_graspable_distance

        object_properties, success = self.scene_parser.estimate_object_location()
        x_dist = object_properties['x']
        if success:
            distanceToMove = x_dist - maxGraspableDistance
        else:
            return False

        rospy.loginfo("distanceToMove = {}".format(distanceToMove))
        if distanceToMove < 0:
            rospy.loginfo("Object is close enough. Not moving.")
            return True
        

        distanceToMove = object_properties['x'] - maxGraspableDistance
        
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            motionTime = time.time() - startTime
            distanceMoved = motionTime * vel
            
            move_to_pose(self.trajectory_client, {
                'base_translate;vel' : vel,
            })

            rospy.loginfo("Distance to move = {}, Distance moved = {}".format(distanceToMove, distanceMoved))
            object_properties, isLive = self.scene_parser.get_latest_observation()
            lastDetectedTime = object_properties['time']

            if isLive:
                if object_properties[0] < maxGraspableDistance:
                    rospy.loginfo("Drawer is close enough. Stopping.")
                    rospy.loginfo("Drawer location = {}".format(object_properties['x']))
                    move_to_pose(self.trajectory_client, {
                        'base_translate;vel' : 0.0,
                    })          
                    return True
            elif time.time() - lastDetectedTime > intervalBeforeRestart:
                    rospy.loginfo("Lost track of the object. Going back to search again.")
                    move_to_pose(self.trajectory_client, {
                        'base_translate;vel' : 0.0,
                    })  
                    return False
                
            rate.sleep()
        return False

    def realign_base_to_aruco(self, offset = 0):
        """
            ensure that the aruco is visible in the camera frame
        """
        isLive = False
        starttime = time.time()
        while not isLive and not (time.time() - starttime > 5):
            object_properties, isLive = self.scene_parser.get_latest_observation()
            rospy.sleep(0.1)

        if not isLive:
            return None, False
        else:
            angleToGo = object_properties['angle']  - np.pi/2 + offset
            x = object_properties['x']
            y = object_properties['y']
            z = object_properties['z']
            
            self.manipulationMethods.reorient_base(self.trajectory_client, angleToGo)
            return (x, y, z, angleToGo), True
            
    def open_drawer(self):
        ntries = 0
        ntries_max = 4
        
        self.state = States.ALIGNING
        while not rospy.is_shutdown() and not (ntries > ntries_max):
            if self.state == States.ALIGNING:
                success = self.find_and_align_to_drawer()
                if not success:
                    ntries += 1
                    rospy.loginfo(f"[{self.node_name}]: Failed to find drawer. Retrying...")    
                else:
                    self.state = States.MOVING_TO_OBJECT
            elif self.state == States.MOVING_TO_OBJECT:
                success = self.move_to_drawer()
                if not success:
                    ntries += 1
                    rospy.loginfo(f"[{self.node_name}]: Failed to move to drawer. Retrying...")
                    self.state = States.ALIGNING
                else:
                    self.state = States.REALIGNMENT
                    
            elif self.state == States.REALIGNMENT:
                drawer_loc, success = self.realign_base_to_aruco(offset = np.pi/2)
                if not success:
                    ntries += 1
                    rospy.loginfo(f"[{self.node_name}]: Failed to realign base. Retrying...")
                else:
                    self.drawer_location = drawer_loc
                    self.state = States.OPEN_DRAWER
                    
            elif self.state == States.OPEN_DRAWER:
                success = self.open_drawer()
                if not success:
                    ntries += 1
                    rospy.loginfo(f"[{self.node_name}]: Failed to open drawer. Retrying...")
                else:
                    self.state = States.COMPLETE
                    break
                
        if self.state == States.COMPLETE:
            return True
    
        return False
    
    def close_drawer(self):
        if self.drawer_location:
            (x, y, z, angleToGo) = self.drawer_location
            self.manipulationMethods.close_drawer(self.trajectory_client, x, y, z)
            self.drawer_location = None
            return True
        else:
            return False
    