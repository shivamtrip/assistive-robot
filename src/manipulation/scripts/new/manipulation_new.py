import numpy as np
import rospy
from std_srvs.srv import Trigger
from helpers import *
from control_msgs.msg import FollowJointTrajectoryAction
from planner import Planner
import open3d as o3d

class Manipulation:
    def __init__(self):
        self.node_name = "manipulation"
        rospy.init_node(self.node_name, anonymous=True)
        self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
        self.stow_robot_service.wait_for_service()
        
        self.navigate_to_location_service = rospy.ServiceProxy('/navigate_to_location', Trigger)
        self.navigate_to_location_service.wait_for_service()
        
        self.trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.search_range = rospy.get_param('/manipulation/search_range')

        self.planner = Planner()
        
        self.capture_tsdf_snapshot = None
        self.start_tsdf_accumulation = None
        self.stop_tsdf_accumulation = None
        
        self.start_object_tracking = None
        self.stop_object_tracking = None
        self.capture_object_snapshot = None
    
        
    def pan_camera_region(self, 
                          range = [0, 2 * np.pi, 10],
                          pitch = -10 * np.pi/180,
                          object_of_interest = 0,
            ):
        
        rospy.loginfo(f"[{self.node_name}]: Panning area. pitch: {pitch}, range: {range}")


        self.start_tsdf_accumulation()
        self.start_object_tracking(object_of_interest)
        
        move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : pitch,
            'head_pan;to' : range[0],
        })
        
        for i, angle in enumerate(np.arange(range[0], range[1], range[2])):
            move_to_pose(self.trajectoryClient, {
                'head_pan;to' : angle,
            })
            self.capture_tsdf_snapshot()
            self.capture_object_snapshot()
            
        move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : -10 * np.pi/180,
            'head_pan;to' : 0,
        })
        
        object_properties = self.stop_object_tracking()
        self.stop_tsdf_accumulation()
        path_to_tsdf = "/home/hello-robot/alfred-autonomy/src/perception/uatu/outputs/tsdf.ply"
        return object_properties, path_to_tsdf
        
    def execute_plan(self, plan):
        pass
    
    def execute_base_plan(self, plan):
        pass
    
    def execute_manipulator_plan(self, plan):
        pass
    
    def main(self):
        
        object_properties, path_to_tsdf = self.pan_camera_region(
            range = self.search_range,
            pitch = -10 * np.pi/180,
        )
        point_cloud = o3d.io.read_point_cloud(path_to_tsdf)
        
        base_plan = self.planner.plan_base(point_cloud, object_properties)
        self.execute_base_plan(base_plan)
        
        
        object_properties, path_to_tsdf = self.pan_camera_region(
            range = self.search_range,
            pitch = -10 * np.pi/180,
        )
        
        point_cloud = o3d.io.read_point_cloud(path_to_tsdf)

        grasp_target = self.planner.plan_grasp_target(point_cloud, object_properties)
        manipulator_plan = self.planner.plan_manipulator(point_cloud, grasp_target)

        self.execute_manipulator_plan(manipulator_plan)
        