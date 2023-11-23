#!/usr/bin/env python3

import numpy as np
import cv2
import open3d as o3d
import fusion
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import rospy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from control_msgs.msg import FollowJointTrajectoryAction
from nav_msgs.msg import OccupancyGrid
from rospy.numpy_msg import numpy_msg
import tf
from scipy.spatial.transform import Rotation as R
import actionlib
from helpers import move_to_pose

class ManipulationPerception:

    def __init__(self):
        rospy.init_node('voxel_analyse', anonymous=False)
        color_topic = '/camera/color/image_raw'
        depth_topic = '/camera/depth/image_raw'
        intrinsic_topic = '/camera/depth/camera_info'
        
        self.color_sub = message_filters.Subscriber(color_topic, Image)
        self.depth_sub = message_filters.Subscriber(depth_topic, Image)
        self.cameraInfoSub = rospy.Subscriber(intrinsic_topic, CameraInfo, self.getCamInfo)
        self.depth_img = None
        self.color_img = None

        # self.sub = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.getPointcloud, queue_size=1)
        # self.pub = rospy.Publisher('/points_from_stuff', PointCloud2, queue_size=1)
        self.trajectoryClient = None
        # self.trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        # self.trajectoryClient.wait_for_server()
        self.image_sub_timesynced = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1, allow_headerless=True)
        self.image_sub_timesynced.registerCallback(self.getImages)

        self.bridge = CvBridge()

        self.obtainedIntrinsics = False
        self.obtainedInitialPoints = False
        self.voxelResolution = 0.05

        self.listener = tf.TransformListener()

        while not self.obtainedInitialPoints and not self.obtainedIntrinsics :
            rospy.loginfo(f"[{rospy.get_name()}] " + " Waiting to start services")
            rospy.loginfo(f"{self.obtainedInitialPoints}, {self.obtainedIntrinsics}")
            # rospy.sleep(1)
        rospy.loginfo(f"[{rospy.get_name()}] " + "Node Ready")

        # range of occupancy grid is from:
        # x: -2 to 2
        # y: 0 to 2
        # z: 0 to 3
        self.xr = [-2, 2]
        self.yr = [-2, 2]
        self.zr = [0, 3]

        self.occupancyGrid = np.zeros((
            int((self.xr[1] - self.xr[0]) / self.voxelResolution), 
            int((self.yr[1] - self.yr[0]) / self.voxelResolution), 
            int((self.zr[1] - self.zr[0]) / self.voxelResolution)
        ))
        print("Voxel grid size is ", self.occupancyGrid.shape)

    def getImages(self, color, depth):
        self.depth = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        self.color = self.bridge.imgmsg_to_cv2(color, desired_encoding="bgr8")
        self.obtainedInitialPoints = True

    def getCamInfo(self,msg):
        self.intrinsics = np.array(msg.K).reshape(3,3)
        self.obtainedIntrinsics = True
        rospy.logdebug(f"[{rospy.get_name()}] " + "Obtained camera intrinsics")
        self.cameraInfoSub.unregister()

    def get_transform(self):
        while not rospy.is_shutdown():
            try:
                (trans,rot) = self.listener.lookupTransform('/base_link', '/camera_color_optical_frame', rospy.Time(0))

                trans_mat = np.array(trans).reshape(3, 1)
                rot_mat = R.from_quat(rot).as_matrix()
                transform_mat = np.hstack((rot_mat, trans_mat))
                print(rot_mat)
                print(trans_mat)
                return transform_mat
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        

    def getPointcloud(self):
        # pc = ros_numpy.numpify(data)
        # self.points=np.zeros((pc.shape[0],3))
        # self.points[:,0]=pc['x']
        # self.points[:,1]=pc['y']
        # self.points[:,2]=pc['z']
        # self.obtainedInitialPoints = True
        intrinsic = self.intrinsics
        camera = Camera(self.color.shape[1], self.color.shape[0], intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], 1000)

        self.points = create_point_cloud_from_depth_image(self.depth, camera, False)
    
    
    

    def aggregateCloudIntoOccupancy(self):
        transform_mat = self.get_transform()            
        
        points = self.points
        points = np.hstack((points,np.ones((points.shape[0],1)))) 


        base_points = np.matmul(transform_mat, points.T).T  
        indices = base_points[:, 2] < self.zr[1]
        base_points = base_points[indices]
        indices = base_points[:, 2] > self.zr[0]
        base_points = base_points[indices]
        indices = base_points[:, 1] < self.yr[1]
        base_points = base_points[indices]
        indices = base_points[:, 1] > self.yr[0]
        base_points = base_points[indices]
        indices = base_points[:, 0] < self.xr[1]
        base_points = base_points[indices]
        indices = base_points[:, 0] > self.xr[0]
        base_points = base_points[indices]
        

        base_points[:, 0] -= self.xr[0]
        base_points[:, 1] -= self.yr[0]
        base_points[:, 2] -= self.zr[0]

        voxels = np.floor(base_points / self.voxelResolution).astype(int)
        
        print(voxels.max(axis = 0))
        print(voxels.min(axis = 0))

        self.occupancyGrid[voxels[:, 0], voxels[:, 1], voxels[:, 2]] += 1
    
    
    def collectData(self, fuser ):
        # pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
       
        
        # self.getPointcloud()
        # self.aggregateCloudIntoOccupancy()
        # plot_plane(self.occupancyGrid, self.voxelResolution)
        # print("Starting")
        rospy.loginfo("Starting")
        move_to_pose(self.trajectoryClient, {
            'head_pan;to' : -np.pi/3,
            'head_tilt;to' : np.deg2rad(-10)
        })
        rospy.sleep(3)
        # time.sleep(3)
        # self.aggregateCloudIntoOccupancy()

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        for angle in np.arange(-np.pi/2, np.pi/2, np.deg2rad(10)):
            move_to_pose(self.trajectoryClient, {
                'head_pan;to' : angle,
            })
            rospy.sleep(0.3)
            color = o3d.geometry.Image(self.color.astype(np.uint8))
            depth = self.depth.astype(np.uint16)
            depth[depth < 150] = 0
            depth = o3d.geometry.Image(depth)
            
            extrinsics = self.get_transform()
            extrinsics = np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0)
            intrinsics = self.intrinsics
            weight = 1
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000, convert_rgb_to_intensity=False)
            cam = o3d.camera.PinholeCameraIntrinsic()
            cam.intrinsic_matrix = intrinsics
            cam.width = self.color.shape[1]
            cam.height = self.color.shape[0]
            volume.integrate(
                rgbd,
                cam,
                np.linalg.inv(extrinsics))
            # fuser.integrate(color, depth, intrinsics, extrinsics, weight)
            # self.getPointcloud()
            # self.aggregateCloudIntoOccupancy(
        # tsdf, col = fuser.get_volume()
        # np.save("tsdf.npy", tsdf)
        # np.save("col.npy", col)
        # point_cloud = fuser.get_point_cloud()
        # # fusion.meshwrite("mesh.ply", verts, faces, norms, colors)
        # fusion.pcwrite("pc.ply", point_cloud)
        # plot_plane(self.occupancyGrid, self.voxelResolution)
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh],
                                      )
        o3d.io.write_triangle_mesh("mesh.ply", mesh)
        # np.save("occupancyGrid_005.npy", self.occupancyGrid)  # 0.1m resolution and 0.05       


class Camera():
    """ Camera intrisics for point cloud creation. """
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale
def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert(depth.shape[1] == camera.width and depth.shape[0] == camera.height)
    # xmap = np.arange(-camera.width//2, camera.width//2)

    #camera width =1280
    #camera height = 720
    xmap = np.arange(0, camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud
def create_geometry_at_points(points):
    geometries = o3d.geometry.TriangleMesh()
    for point in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05) #create a small sphere to represent point
        sphere.translate(point) #translate this sphere to point
        geometries += sphere
    geometries.paint_uniform_color([1.0, 0.0, 0.0])
    return geometries

  
def plot_plane(occupancyGrid, resolution):
    pointsFromOccupancy =  np.argwhere(occupancyGrid > 0)
    colors = (occupancyGrid[pointsFromOccupancy[:, 0], pointsFromOccupancy[:, 1], pointsFromOccupancy[:, 2]] * 0.8/occupancyGrid.max())
    print(colors.shape)
    # reshape 
    colors = np.stack((colors, colors, colors), axis = 1)
    pointsFromOccupancy = pointsFromOccupancy.astype(float)
    pointsFromOccupancy *= resolution
    plane_cloud = o3d.geometry.PointCloud()                 # creating point cloud of plane
    plane_cloud.points = o3d.utility.Vector3dVector(pointsFromOccupancy)
    # plane_cloud.colors = o3d.utility.Vector3dVector(np.ones((pointsFromOccupancy.shape[0], 3)))
    # color point cloud based on occupancy grid values  
    plane_cloud.colors = o3d.utility.Vector3dVector(colors)



    # app = gui.Application.instance
    # app.initialize()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(plane_cloud)
    opt = vis.get_render_option()
    opt.point_size = 3/resolution
    vis.run()
    # o3d.visualization.draw_geometries([plane_cloud])

if __name__ == "__main__":
    objectLocalizer = ManipulationPerception()
    from helpers import *
    armclient.wait_for_server()
    headclient.wait_for_server()
    gripperclient.wait_for_server()
    print("Starting to collect data")
    from fusion import TSDFVolume
    vol_bnds = np.zeros((3,2))
    vol_bnds[:, 0] = np.array([-2, -2, 0])
    vol_bnds[:, 1] = np.array([2, 2, 3])
    fuser = TSDFVolume(vol_bnds, voxel_size=0.05)
    
    objectLocalizer.collectData(fuser)
    # fuser.integrate(objectLocalizer.points, np.eye(4))
    rospy.loginfo("Compelte :)")
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass