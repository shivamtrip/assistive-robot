import cv2
import numpy as np
import rospy
from sklearn.linear_model import LinearRegression, RANSACRegressor
import open3d as o3d
from tf import TransformListener
import tf
from scipy.spatial.transform import Rotation as R
from yolo.msg import Detections
from geometry_msgs.msg import Vector3, Vector3Stamped, PoseStamped, Point, Pose, PointStamped

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

def convertPointToBaseFrame(x_f, y_f, z):
    global intrinsics
    
    fx = intrinsics[0][0]
    fy = intrinsics[1][1] 
    cx = intrinsics[0][2] 
    cy = intrinsics[1][2]

    x_gb = (x_f - cx) * z/ fx
    y_gb = (y_f - cy) * z/ fy

    camera_point = PointStamped()
    camera_point.header.frame_id = '/camera_color_optical_frame'
    camera_point.point.x = x_gb
    camera_point.point.y = y_gb
    camera_point.point.z = z
    point = tf_listener_.transformPoint('base_link', camera_point).point
    return point
def parseScene(msg):
    global isDetected, objectLocArr, depth_image
    if depth_image is None:
        return
    
    dimage = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
    num_detections = (msg.nPredictions)
    msg = {
        "boxes" : np.array(msg.box_bounding_boxes).reshape(num_detections, 4),
        "box_classes" : np.array(msg.box_classes).reshape(num_detections),
        'confidences' : np.array(msg.confidences).reshape(num_detections),
    }

    loc = np.where(np.array(msg['box_classes']).astype(np.uint8) == 0)[0]
    upscale_fac = 1/rospy.get_param('/image_shrink/downscale_ratio')
    if len(loc) > 0:
        loc = loc[0]
        box = np.squeeze(msg['boxes'][loc])
        confidence = np.squeeze(msg['confidences'][loc])
        x1, y1, x2, y2 =  box
        x1 = int(x1 * upscale_fac)
        y1 = int(y1 * upscale_fac)
        x2 = int(x2 * upscale_fac)
        y2 = int(y2 * upscale_fac)
        # print(x1, y1, x2, y2)
        crop = dimage[y1 : y2, x1 : x2]
        temp = cv2.rotate(rgbimg, cv2.ROTATE_90_CLOCKWISE)
        # cv2.imwrite("crop.png", temp[y1 : y2, x1 : x2])
        # print(y1, y2, x1, x2)
        # cv2.imwrite("crop.png", crop.astype(np.uint8))
        z_f = np.median(crop[crop != np.nan])/1000.0
        h, w = dimage.shape[:2]
        y_new1 = w - x2 
        x_new1= y1 
        y_new2 = w - x1
        x_new2 = y2 

        x_f = (x_new1 + x_new2)/2
        y_f = (y_new1 + y_new2)/2
        # print("cam frame", x_f, y_f, z_f)
        point = convertPointToBaseFrame(x_f, y_f, z_f)

        x, y, z = point.x, point.y, point.z
        rospy.loginfo("Object detected at" + str((x, y, z)))

        radius = np.sqrt(x**2 + y**2)
        confidence /= radius
        isDetected = True
        objectLocArr.append((x, y, z, confidence))
            
    else:
        isDetected = False

objectLocArr = []
isDetected = False

def alignObjectHorizontal(offset = 0.05):
    xerr = np.inf
    rospy.loginfo("Aligning object horizontally")
    kp = {
        'velx' : 0.5,
    }
    kd = {
        'velx' : 0.0,
    }
    prevxerr = np.inf
    curvel = 0.0
    prevsettime = time.time()
    
    while abs(xerr) > 0.008:
        try:
            x, y, z, confidence = objectLocArr[-1]
            print(xerr)
            if isDetected:
                xerr = (x) + offset
            else:
                rospy.logwarn("Object not detected. Using open loop motions")
                xerr = (x - curvel * (time.time() - prevsettime)) + offset
            rospy.loginfo("Alignment Error = " + str(xerr))
            dxerr = (xerr - prevxerr)
            vx = (kp['velx'] * xerr + kd['velx'] * dxerr)
            prevxerr = xerr
            move_to_pose(trajectoryClient, {
                    'base_translate;vel' : vx,
            }) 
            curvel = vx
            prevsettime = time.time()
            
            rospy.sleep(0.1)
        except KeyboardInterrupt:
            exit()
    move_to_pose(trajectoryClient, {
        'base_translate;vel' : 0.0,
    }) 
    
    return True


def wrap_to_pi(angle):
    return np.mod((angle + 2 * np.pi),  (2 * np.pi))

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
    
    # check if the normal is vertically upwards:
    if abs(plane_vec_norm[2]) > 0.5:
        print("Plane is horizontal, redoing.")
        
        
        plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.01,
                                                        ransac_n=3,
                                                        num_iterations=1000)
        [a, b, c, d] = plane_model
        normal_vec = np.array([a, b, c]).reshape(3, 1)
        plane_vec = np.matmul(camera_to_base[:3, :3], normal_vec)
        plane_vec_norm = plane_vec / np.linalg.norm(plane_vec)
        
        # inlier_points = np.asarray(inlier_cloud.points)
        # # get dominant directions:
        # covmat = np.cov(inlier_points.T)
        # eigval, eigvec = np.linalg.eig(covmat)
        # dominant_vec = eigvec[:, np.argmax(eigval)]
        # weak_vec = eigvec[:, np.argmin(eigval)]
        # print("Dominant vec: {}".format(dominant_vec))
        # yaw = np.arctan2(dominant_vec[1], dominant_vec[0]) 
        # yaw = wrap_to_pi(yaw)
        # print("Yaw: {} degrees".format(yaw * 180 / np.pi))
        # return 
    # else: 
    plane_vec_yaw = np.arctan2(plane_vec_norm[1], plane_vec_norm[0])[0]
    plane_vec_pitch = np.arctan2(plane_vec_norm[2], plane_vec_norm[0])[0]
    plane_vec_roll = np.arctan2(plane_vec_norm[2], plane_vec_norm[1])[0]
    # yaw = wrap_to_pi(plane_vec_yaw)
    yaw = plane_vec_yaw + np.pi/2
    # print(yaw)
    # print(yaw * 180/np.pi)
    # if yaw > np.pi / 2:
    #     yaw -= np.pi
    
    # yaw = wrap_to_pi(plane_vec_yaw)
    plane_vec_pitch *= 180 / np.pi
    plane_vec_yaw *= 180 / np.pi
    plane_vec_roll *= 180 / np.pi
    print("Plane rpy: {}, {}, {}".format(plane_vec_roll, plane_vec_pitch, plane_vec_yaw))
        
    move_to_pose(trajectoryClient, {
        'base_rotate;by' : yaw,
    }) 
    print("Rotating base.")
    rospy.sleep(3)
        
    

    

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

def getCamInfo(msg):
    global intrinsics, cameraInfoSub
    intrinsics = np.array(msg.K).reshape(3,3)
    cameraInfoSub.unregister()

def open_drawer():
    x, y, z, _ = objectLocArr[-1]
    move_to_pose(trajectoryClient, {
        'wrist_yaw;to': 90
    }
    )
    rospy.sleep(2)
    move_to_pose(trajectoryClient, {
        'arm;to': 0.2
        }
    )
    rospy.sleep(2)
    move_to_pose(trajectoryClient, {
        'lift;to': z - 0.08
        }
    )
    rospy.sleep(3)
    move_to_pose(trajectoryClient, {
            'arm;to' : (1, 30)
        }, custom_contact_thresholds=True
    )
    rospy.sleep(5)
    # customcontactthresh
    move_to_pose(trajectoryClient, {
        'lift;to' : (z - 0.15, 30),
    }, custom_contact_thresholds=True)
    
    rospy.sleep(2)
    move_to_pose(trajectoryClient, {
            'arm;to' : (0.0, 30)
        }, custom_contact_thresholds=True
    )
    rospy.sleep(5)
    

if __name__ == "__main__":
    from sensor_msgs.msg import Image
    import time
    import actionlib
    from control_msgs.msg import FollowJointTrajectoryAction
    from helpers import move_to_pose
    from std_srvs.srv import Trigger, TriggerResponse
    from sensor_msgs.msg import JointState, CameraInfo, Image
    
    depth_image = None
    rgbimg = None
    intrinsics = None

    trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    rospy.init_node("listener")
    tf_listener_ = TransformListener()
    sb = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth_image_callback)
    sb1 = rospy.Subscriber('/camera/color/image_raw', Image, rgb_image)
    cameraInfoSub = rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, getCamInfo)
    while intrinsics is None:
        rospy.sleep(0.1)
    boundingBoxSub = rospy.Subscriber("/object_bounding_boxes", Detections, parseScene)
        
    switch_to_manipulation = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    switch_to_manipulation.wait_for_service()
    switch_to_manipulation()
    move_to_pose(trajectoryClient, {
        'head_pan;to' : -np.pi/2,
        # 'head_pan;to' : 0,
        # 'head_tilt;to' : 0,
    }) 
    move_to_pose(trajectoryClient, {
        # 'head_pan;to' : -np.pi/2,
        'head_tilt;to' : -np.pi/6,
        # 'head_tilt;to' : -np.pi/8,
    }) 
    rospy.sleep(5)
    #     # rospy.spin()
    lasttime = time.time()
    while time.time() - lasttime <= 20:
        try:     
            if depth_image is not None and rgbimg is not None:   
                # detect_cliff(depth_image, 0.006, 2.0, [0,0], display_images=True)
                fit_plane(depth_image, rgbimg)
                alignObjectHorizontal()
                open_drawer()
                # rospy.sleep(60)
                break
        except KeyboardInterrupt:
            print('Shutting down')
            exit()
    # cv2.waitKey(0)