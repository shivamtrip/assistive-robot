import numpy as np
import rospy
import tf
# from scene_parser import SceneParser
from scene_parser_refactor import SceneParser
from visual_servoing import AlignToObject
from manip_basic_control import ManipulationMethods
import actionlib
from manipulation.msg import DrawerTriggerAction, DrawerTriggerActionGoal, DrawerTriggerActionResult, DrawerTriggerResult
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
    ALIGN_HORIZONTAL = 7
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
        self.mean_err_horiz_alignment = rospy.get_param('/manipulation/mean_err_horiz_alignment')
        self.moving_avg_n = rospy.get_param('/manipulation/moving_average_n')

        
        self.pid_gains = rospy.get_param('/manipulation/pid_gains')
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
            success = self.open_drawer()
        else:
            success = self.close_drawer()
            self.drawer_location = None
            
        if success:
            result = DrawerTriggerResult(
                success = True,
                saved_position = self.drawer_location
            )
            self.server.set_succeeded(result)
        else:
            result = DrawerTriggerResult()
            result.success= False
            self.server.set_aborted(result)
    
    def find_and_align_to_drawer(self):
        """
        as soon as the aruco is found, reorient and align to it instantly.
        """
        
        rospy.loginfo(f"Finding and orienting to Drawer")
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
                rospy.sleep(1) #settling time.
                object_properties, isLive = self.scene_parser.get_latest_observation()
                if isLive:
                    drawer_loc, success = self.realign_base_to_aruco(offset = 0)
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
                if object_properties['x'] < maxGraspableDistance:
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

    def get_drawer_loc(self):
        isLive = False
        starttime = time.time()
        while not isLive and not (time.time() - starttime > 5):
            object_properties, isLive = self.scene_parser.get_latest_observation()
            rospy.sleep(0.1)

        return object_properties, isLive
        
        
    
    def realign_base_to_aruco(self, offset = 0):
        """
            ensure that the aruco is visible in the camera frame
        """
        
        object_properties, isLive = self.get_drawer_loc()
        if not isLive:
            return None, False
        else:
            angleToGo = object_properties['yaw']  - np.pi/2 + offset
            x = object_properties['x']
            y = object_properties['y']
            z = object_properties['z']
            
            self.manipulationMethods.reorient_base(angleToGo)
            return (x, y, z, angleToGo), True
    
    def alignObjectHorizontal(self, ee_pose_x = 0.0, debug_print = {}):
        """
        Visual servoing to align gripper with the object of interest.
        """
        
        self.requestClearObject = True
        if not self.scene_parser.clearAndWaitForNewObject(2):
            return False

        xerr = np.inf
        rospy.loginfo("Aligning object horizontally")
        
        prevxerr = np.inf
        moving_avg_n = self.moving_avg_n
        moving_avg_err = []
        mean_err = np.inf
        
        kp = self.pid_gains['kp']
        kd = self.pid_gains['kd']
        curvel = 0.0
        prevsettime = time.time()
        while len(moving_avg_err) <= moving_avg_n and  mean_err > self.mean_err_horiz_alignment:
            # [x, y, z, r, p, y, conf, pred_time], isLive = self.scene_parser.get_latest_observation()
            object_properties, isLive = self.scene_parser.get_latest_observation()
            x = object_properties['x']
            y = object_properties['y']
            z = object_properties['z']
            pred_time = object_properties['time']
            # if isLive:
            rospy.loginfo("Time since last detection = {}".format(time.time() - pred_time))
            xerr = (x - ee_pose_x)
            
                
            moving_avg_err.append(xerr)
            if len(moving_avg_err) > moving_avg_n:
                moving_avg_err.pop(0)
                
            dxerr = (xerr - prevxerr)
            vx = (kp * xerr + kd * dxerr)
            prevxerr = xerr
            
            mean_err = np.mean(np.abs(np.array(moving_avg_err)))
            debug_print["xerr"] = xerr
            debug_print["mean_err"] = mean_err
            debug_print["x, y, z"] = [x, y, z]
            
            print(debug_print)

            move_to_pose(self.trajectory_client, {
                    'base_translate;vel' : vx,
            }, asynchronous= True) 
            curvel = vx
            prevsettime = time.time()
            # else:
            #     rospy.logwarn("Object not detected. Using open loop motions")
            #     xerr = (x - curvel * (time.time() - prevsettime)) - ee_pose_x
            
            
            rospy.sleep(0.1)
        
        move_to_pose(self.trajectory_client, {
            'base_translate;vel' : 0.0,
        }, asynchronous= True) 
        
        return True
    
    
    def align_for_manipulation(self):
        """
        Rotates base to align manipulator for grasping
        """
        rospy.loginfo(f"Aligning object for manipulation")
        move_to_pose(self.trajectory_client, {
                'head_pan;to' : -np.pi/2,
                'base_rotate;by' : np.pi/2,
            }
        )
        rospy.sleep(4)
        return True
    
    def open_drawer(self):
        starttime = time.time()
        ntries = 0
        ntries_max = 4
        
        self.state = States.ALIGNING
        while not rospy.is_shutdown() and not (ntries > ntries_max):
            if self.state == States.ALIGNING:
                success = self.find_and_align_to_drawer()
                if not success:
                    ntries += 1
                    rospy.loginfo(f"Failed to find drawer. Retrying...")    
                else:
                    self.state = States.MOVING_TO_OBJECT
            elif self.state == States.MOVING_TO_OBJECT:
                success = self.move_to_drawer()
                if not success:
                    ntries += 1
                    rospy.loginfo(f"Failed to move to drawer. Retrying...")
                    self.state = States.ALIGNING
                else:
                    self.state = States.REALIGNMENT
                    
            elif self.state == States.REALIGNMENT:
                self.align_for_manipulation()
                drawer_loc, success = self.realign_base_to_aruco(offset = np.pi/2)
                if not success:
                    ntries += 1
                    rospy.loginfo(f"Failed to realign base. Retrying...")
                else:
                    self.drawer_location = drawer_loc
                    self.state = States.ALIGN_HORIZONTAL
                    
            elif self.state == States.ALIGN_HORIZONTAL:
                success = self.alignObjectHorizontal(ee_pose_x = 0.08)
                if not success:
                    ntries += 1
                    rospy.loginfo(f"Failed to align horizontally. Retrying...")
                    self.state = States.ALIGNING
                else:
                    self.state = States.OPEN_DRAWER
                    
            elif self.state == States.OPEN_DRAWER:
                object_properties, isLive = self.get_drawer_loc()
                self.drawer_location = object_properties['x'], object_properties['y'], 0.76, object_properties['yaw']
                
                
                success = self.manipulationMethods.open_drawer(
                    self.drawer_location[0],
                    self.drawer_location[1],
                    self.drawer_location[2],
                )
                if not success:
                    ntries += 1
                    rospy.loginfo(f"Failed to open drawer. Retrying...")
                else:
                    self.state = States.COMPLETE
                    break
        rospy.loginfo("Drawer operation time = {}".format(time.time() - starttime))
        if self.state == States.COMPLETE:
            return True
    
        return False
    
    def close_drawer(self):
        if self.drawer_location:
            (x, y, z, angleToGo) = self.drawer_location
            self.manipulationMethods.close_drawer(x)
            self.drawer_location = None
            return True
        else:
            return False
    