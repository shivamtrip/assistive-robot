import rospy 
import tf
from geometry_msgs.msg import PointStamped
import numpy as np

def convertEEpointToBaseFrame(listener):
    base_point = PointStamped()
    base_point.header.frame_id = '/link_grasp_center'
    base_point.point.x = 0
    base_point.point.y = 0
    base_point.point.z = 0
    point = listener.transformPoint('base_link', base_point).point
    return [point.x, point.y, point.z]

def getEndEffectorPose(listener):
    from_frame_rel = 'base_link'
    to_frame_rel = 'link_grasp_center'
    while not rospy.is_shutdown():
        try:
            translation, rotation = listener.lookupTransform(to_frame_rel, from_frame_rel, rospy.Time(0))
            
            return translation
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        
    rospy.logerr(f"[{rospy.get_name()}] " + "Transform not found between {} and {}".format(to_frame_rel, from_frame_rel))
    return None


if __name__ == "__main__":
    
    rospy.init_node('ee_transform')
    listener = tf.TransformListener()
    rospy.sleep(1)
    while not rospy.is_shutdown():
        pt = convertEEpointToBaseFrame(listener)
        # transform = getEndEffectorPose(listener)
        # print(np.round(transform, 2))
        print(np.round(pt, 2))
    # extension, base_trans, height
    
    rospy.spin()