#!/usr/local/lib/obj_det_env/bin/python3


import rospy
import cv2
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import os


class FaceDetector:

    def __init__(self):

        rospy.init_node('face_detection', anonymous=False)

        self.rgb_image_subscriber = message_filters.Subscriber(rospy.get_param('/telepresence/subscribe_topic'), Image)
        self.rgb_image_subscriber.registerCallback(self.callback)
        self.cv_bridge = CvBridge()

        self.face_detection_enabled = False

        # Get the current directory of the script
        current_directory = os.path.dirname(os.path.abspath(__file__))

        # Specify the path to the XML file (example of a relative path)
        xml_file_path = os.path.join(current_directory, 'haarcascade_frontalface_default.xml')


        self.haar_cascade_detector = cv2.CascadeClassifier(xml_file_path)
        self.annotated_image_pub = rospy.Publisher(rospy.get_param('/telepresence/visualize_topic'), Image, queue_size=1)

        rospy.sleep(3)
        if self.haar_cascade_detector.empty():
            print("XML file not loaded properly or file path incorrect")
        else:
            self.face_detection_enabled = True
            print("XML file loaded successfully")




    def callback(self,ros_rgb_image):

        if self.face_detection_enabled:

            print("Face Detector will START INFERENCING")

            rgb_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image)
            rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)

            # gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            # faces = self.haar_cascade_detector.detectMultiScale(gray, 1.3, 5)

            # for (x,y,w,h) in faces:
            #     cv2.rectangle(rgb_image,(x,y),(x+w,y+h),(255,0,0),2)


            # Initializing the HOG person
            # detector
            # hog = cv2.HOGDescriptor()
            # hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Reading the Image
            # image = cv2.imread('img.png')
            
            # # Resizing the Image
            # image = imutils.resize(rgb_image,
            #                     width=min(400, rgb_image.shape[1]))
            
            # resized_image = cv2.resize(rgb_image, (min(400, image.shape[1]), int(image.shape[0] * min(400, image.shape[1]) / image.shape[1])))

        
            # Detecting all the regions in the 
            # Image that has a pedestrians inside it
            (regions, _) = hog.detectMultiScale(rgb_image, 
                                                winStride=(4, 4),
                                                padding=(4, 4),
                                                scale=1.05)
            
            # Drawing the regions in the Image
            for (x, y, w, h) in regions:
                cv2.rectangle(rgb_image, (x, y), 
                            (x + w, y + h), 
                            (0, 0, 255), 2)

            
            # annotated_img = cv2.resize(annotated_img, (0, 0), fx = 0.5, fy = 0.5)
            # annotated_img = cv2.rotate(annotated_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.annotated_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(rgb_image))




if __name__ == "__main__":
    
    node = FaceDetector()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

    