import numpy as np
import rospy
from std_srvs.srv import Trigger
from helpers_sim import *
from control_msgs.msg import FollowJointTrajectoryAction
from planner.planner import Planner
import open3d as o3d
from uatu.parse_scene import SceneParser

class Manipulation:
    def __init__(self):
        self.node_name = "manipulation"
        rospy.init_node(self.node_name, anonymous=True)
        
        
        rospy.loginfo(f"[{self.node_name}]: Waiting for services...")
        # self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
        # self.stow_robot_service.wait_for_service()
        
        # self.navigate_to_location_service = rospy.ServiceProxy('/navigate_to_location', Trigger)
        # self.navigate_to_location_service.wait_for_service()
        
        
        self.switch_to_manipulation_mode = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
        self.switch_to_manipulation_mode.wait_for_service()
        
        self.switch_to_navigation_mode = rospy.ServiceProxy('/switch_to_navigation_mode', Trigger)
        self.switch_to_navigation_mode.wait_for_service()
        
        
        self.trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        
        # self.search_range = rospy.get_param('/manipulation/search_range')
        self.search_range = [-np.pi/6, np.pi/6, 3]

        self.scene_parser = SceneParser()

        self.planner = Planner()
        
        self.capture_tsdf_snapshot = self.scene_parser.tsdf_snapshot
        self.start_tsdf_accumulation = self.scene_parser.start_accumulate_tsdf
        self.stop_tsdf_accumulation = self.scene_parser.stop_accumulate_tsdf
        
        self.start_object_tracking = self.scene_parser.start_tracking_object
        self.capture_object_snapshot = self.scene_parser.image_snapshot
        self.get_three_d_bbox_from_tracker = self.scene_parser.get_three_d_bbox_from_tracker
        
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
        
        for i, angle in enumerate(np.linspace(range[0], range[1], range[2])):
            move_to_pose(self.trajectoryClient, {
                'head_pan;to' : angle,
            })
            rospy.sleep(1)
            rospy.loginfo(f"[{self.node_name}]: Capturing snapshot {angle}...")
            self.capture_tsdf_snapshot()
            self.capture_object_snapshot()
            
        move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : pitch,
            'head_pan;to' : 0,
        })
        
        self.stop_tsdf_accumulation()
        object_bbox = self.get_three_d_bbox_from_tracker()
        
        path_to_tsdf = "/home/praveenvnktsh/alfred-autonomy/src/perception/uatu/outputs/tsdf.ply"
        return object_bbox, path_to_tsdf
        
    def execute_base_plan(self, plan):
        # write code that converts local plan into map coordinates and sends it to movebase
        
        self.switch_to_navigation_mode()


        self.switch_to_manipulation_mode()
            
    def execute_manipulator_plan(self, plan):
        for waypoint in plan:
            x, y, theta, z, phi, ext = waypoint
            move_to_pose(self.trajectoryClient, {
                'base_translate;by' : y, 
                'base_rotate;by' : theta,
                'lift;to' : z,
                'arm_phi;to' : phi,
                'arm_ext;to' : ext,
            })

    
    def main(self):  
        
        object_bbox, path_to_tsdf = self.pan_camera_region(
            range = self.search_range,
            pitch = 0 * np.pi/180,
            object_of_interest = 8
        )
        
        point_cloud = o3d.io.read_point_cloud(path_to_tsdf)
        
        object_cloud = point_cloud.crop(
            o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(object_bbox)
            )
        )
        
        object_points = np.asarray(object_cloud.points)
        medianz = np.median(object_points[:,2])
        x, y = np.mean(object_points[object_points[:,2] == medianz, :2], axis=0)
        
        base_plan, is_plan_reqd = self.planner.plan_base(point_cloud, np.array([x, y]))

        if is_plan_reqd:
            self.execute_base_plan(base_plan)
        
        
            object_bbox, path_to_tsdf = self.pan_camera_region(
                range = self.search_range,
                pitch = -30 * np.pi/180,
            )
            
            point_cloud = o3d.io.read_point_cloud(path_to_tsdf)

        grasp_target = self.planner.plan_grasp_target(point_cloud, object_bbox)
        
        # x, y, theta, z, phi, ext - this is the stowed state
        curstate = np.array(
            [
                0, 0, 0, 0.3, np.pi, 0
            ]
        )
        
        manipulator_plan = self.planner.plan_manipulator(point_cloud, curstate, grasp_target)
        self.execute_manipulator_plan(manipulator_plan)
        
if __name__ == "__main__":
    manipulation = Manipulation()
    manipulation.main()
    