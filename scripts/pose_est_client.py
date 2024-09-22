#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import SetBool

import cv2
import numpy as np

from foundation_pose_ros.srv._EstPose import EstPose,EstPoseResponse,EstPoseRequest

class PoseEstClient:

    def __init__(self) -> None:

        self._bridge = CvBridge()

        rospy.wait_for_service("pose_est")
        print("[PoseEstClient]: Initialized")

        color_img_path = "/home/jau/Desktop/cleanup_tests/rgb_new.png"
        color = cv2.imread(color_img_path)
        depth_img_path = "/home/jau/Desktop/cleanup_tests/depth_new.png"
        depth = cv2.imread(depth_img_path)

        pose_request = EstPoseRequest(self.cv2_to_ros(color),self.cv2_to_ros(depth))
    
        pose_est_service_handle = rospy.ServiceProxy("pose_est", EstPose)
        print("[PoseEstClient]: Request sent")
        pose_response = pose_est_service_handle(pose_request)
        print("[PoseEstClient]: Got pose: ")
        pose_msg = pose_response.T_ce
        print(pose_msg)


    def ros_to_cv2(self, frame: Image, desired_encoding="bgr8"):
        return self._bridge.imgmsg_to_cv2(frame, desired_encoding=desired_encoding)

    def cv2_to_ros(self, frame: np.ndarray):
        return self._bridge.cv2_to_imgmsg(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), encoding="rgb8")

    
    


if __name__ == "__main__":
    rospy.init_node('pose_est_client')
    pose_client = PoseEstClient()
    #pose_detector.run()
    rospy.spin()