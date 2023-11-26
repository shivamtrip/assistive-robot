import numpy as np
import rospy
import tf
from scene_parser import SceneParser
from visual_servoing import AlignToObject
from manip_basic_control import ManipulationMethods
import actionlib
from manipulation.msg import DrawerTriggerAction, DrawerTriggerActionGoal, DrawerTriggerActionResult
from helpers import move_to_pose
import time
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
            drawer_location = goal.saved_position
            success = self.close_drawer(drawer_location)
            
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
        x_dist = object_properties[4]
        if success:
            distanceToMove = object_properties[4] - maxGraspableDistance
        else:
            return False

        rospy.loginfo("distanceToMove = {}".format(distanceToMove))
        if distanceToMove < 0:
            rospy.loginfo("Object is close enough. Not moving.")
            return True
        

        distanceToMove = object_properties[4] - maxGraspableDistance
        
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            motionTime = time.time() - startTime
            distanceMoved = motionTime * vel
            
            move_to_pose(self.trajectory_client, {
                'base_translate;vel' : vel,
            })

            rospy.loginfo("Distance to move = {}, Distance moved = {}".format(distanceToMove, distanceMoved))
            object_location, isLive = self.scene_parser.get_latest_observation()

            lastDetectedTime = object_location[-1]

            if isLive:
                if object_location[0] < maxGraspableDistance:
                    rospy.loginfo("Drawer is close enough. Stopping.")
                    rospy.loginfo("Drawer location = {}".format(self.scene_parser.objectLocArr[-1]))
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
            (x, y, z, r, p, angleToGo, confidence, pred_time), isLive = self.scene_parser.get_latest_observation()
            rospy.sleep(0.1)
            
        if not isLive:
            return (x, y, z, angleToGo), False
        else:
            self.manipulationMethods.reorient_base(self.trajectory_client, angleToGo + offset)
            return (x, y, z, angleToGo), True
            

    def open_drawer(self):
        
        self.find_and_align_to_drawer()

        drawer_loc, success = self.realign_base_to_aruco(offset = np.pi/2)
        if not success:
            return False
        
        x, y, z, angleToGo = drawer_loc
        
        success = self.manipulationMethods.open_drawer(self.trajectory_client, x, y, z)
        if not success:
            return False
        
        self.drawer_location = (x, y, z, angleToGo)
        return True
    
    def close_drawer(self):
        if self.drawer_location:
            (x, y, z, angleToGo) = self.drawer_location
            self.manipulationMethods.close_drawer(self.trajectory_client, x, y, z)
            self.drawer_location = None
            return True
    