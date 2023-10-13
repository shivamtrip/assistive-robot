import cv2
import numpy as np
import rospy
from sklearn.linear_model import LinearRegression, RANSACRegressor
import open3d as o3d
from tf import TransformListener
import tf
from scipy.spatial.transform import Rotation as R


def get_transform(tf_listener: TransformListener, base_frame, camera_frame):

    while not rospy.is_shutdown():
        try:
            t = tf_listener.getLatestCommonTime(base_frame, camera_frame)
            (trans,rot) = tf_listener.lookupTransform(base_frame, camera_frame, t)
            

            trans_mat = np.array(trans).reshape(3, 1)
            rot_mat = R.from_quat(rot).as_matrix()
            transform_mat = np.hstack((rot_mat, trans_mat))

            return transform_mat
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("not found")
            print(tf_listener.allFramesAsString())
            continue
def fit_plane(depthimg, img):
    global tf_listener_
    # K: [908.4595336914062, 0.0, 648.8724365234375, 0.0, 907.7884521484375, 367.76092529296875, 0.0, 0.0, 1.0]
    # R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    # P: [908.4595336914062, 0.0, 648.8724365234375, 0.0, 0.0, 907.7884521484375, 367.76092529296875, 0.0, 0.0, 0.0, 1.0, 0.0]
    intr = o3d.open3d.camera.PinholeCameraIntrinsic(1280, 720, cx=648.87, cy=367.76, fx=908.459, fy=907.78)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(img), o3d.geometry.Image(depthimg), 
        depth_scale=1000, depth_trunc=1000.0, convert_rgb_to_intensity=False)
    pcd_from_depth = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        intr)

    # fit plane
    plane_model, inliers = pcd_from_depth.segment_plane(distance_threshold=0.01,
                                                        ransac_n=3,
                                                        num_iterations=1000)
    [a, b, c, d] = plane_model
    print("Plane equation: {}x + {}y + {}z + {} = 0".format(a, b, c, d))
    inlier_cloud = pcd_from_depth.select_by_index(inliers)
    outlier_cloud = pcd_from_depth.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([1.0, 0, 0])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    
    camera_to_base = get_transform(tf_listener_, '/base_link', '/camera_depth_optical_frame')    
    normal_vec = np.array([a, b, c]).reshape(3, 1)
    
    plane_vec = np.matmul(camera_to_base[:3, :3], normal_vec)
    
    plane_vec_norm = plane_vec / np.linalg.norm(plane_vec)
    plane_vec_yaw = np.arctan2(plane_vec_norm[1], plane_vec_norm[0])
    plane_vec_pitch = np.arctan2(plane_vec_norm[2], plane_vec_norm[0])
    plane_vec_roll = np.arctan2(plane_vec_norm[2], plane_vec_norm[1])
    
    move_to_pose(trajectoryClient, {
            'base_rotate;by' : plane_vec_yaw,
    }) 
    plane_vec_pitch *= 180 / np.pi
    plane_vec_yaw *= 180 / np.pi
    plane_vec_roll *= 180 / np.pi
    
    print("Plane rpy: {}, {}, {}".format(plane_vec_roll, plane_vec_pitch, plane_vec_yaw))
    

    

def depth_image_callback(img_msg):
    dimage = np.frombuffer(img_msg.data, dtype=np.uint16).reshape(img_msg.height, img_msg.width)
    dimage = dimage.astype(np.uint16)
    global depth_image
    depth_image = dimage

def rgb_image(img_msg):
    image = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
    image = image.astype(np.uint8)
    global rgbimg
    rgbimg = image

if __name__ == "__main__":
    from sensor_msgs.msg import Image
    import time
    import actionlib
    from control_msgs.msg import FollowJointTrajectoryAction
    from helpers import move_to_pose
    from std_srvs.srv import Trigger, TriggerResponse
    depth_image = None
    rgbimg = None
    trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    rospy.init_node("listener")
    sb = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth_image_callback)
    sb1 = rospy.Subscriber('/camera/color/image_raw', Image, rgb_image)
    tf_listener_ = TransformListener()
    switch_to_manipulation = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    switch_to_manipulation.wait_for_service()
    switch_to_manipulation()
        # rospy.spin()
    lasttime = time.time()
    while time.time() - lasttime <= 5:
        try:     
            if depth_image is not None and rgbimg is not None:   
                # detect_cliff(depth_image, 0.006, 2.0, [0,0], display_images=True)
                fit_plane(depth_image, rgbimg)
                rospy.sleep(10)
                break
        except KeyboardInterrupt:
            print('Shutting down')
            exit()
    # cv2.waitKey(0)