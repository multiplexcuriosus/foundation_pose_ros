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
        self.color = None
        self.depth = None
        self.color_msg = None
        self.depth_msg = None

        rospy.wait_for_service("pose_est")
        print("[PoseEstClient]: Initialized")

        color_topic = rospy.get_param("ros/color_image_topic")
        depth_topic = rospy.get_param("ros/depth_image_topic")
        self._color_sub = rospy.Subscriber(color_topic, Image, self._color_callback)
        self._depth_sub = rospy.Subscriber(depth_topic, Image, self._depth_callback)

        print("[PoseEstClient]: Waiting for color and depth imgs...")
        while self.color is None or self.depth is None:
            pass
        print("[PoseEstClient]: Color and depth img received!")

        pose_request = EstPoseRequest(self.color_msg,self.depth_msg)
    
        pose_est_service_handle = rospy.ServiceProxy("pose_est", EstPose)
        print("[PoseEstClient]: Request sent")
        pose_response = pose_est_service_handle(pose_request)
        print("[PoseEstClient]: Got pose: ")
        pose_msg = pose_response.T_ce
        print(pose_msg)


    def _color_callback(self, data: Image):
        self.color = self.ros_to_cv2(data, desired_encoding="bgr8")
        self.color_msg = data


    def _depth_callback(self, data: Image):
        self.depth_msg = data
        self.depth = self.ros_to_cv2(data, desired_encoding="passthrough").astype(np.uint16).copy() / 1000.0
        self.depth[(self.depth < 0.1)] = 0


    def ros_to_cv2(self, frame: Image, desired_encoding="bgr8"):
        return self._bridge.imgmsg_to_cv2(frame, desired_encoding=desired_encoding)

    def cv2_to_ros(self, frame: np.ndarray):
        return self._bridge.cv2_to_imgmsg(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), encoding="rgb8")

    
    


if __name__ == "__main__":
    rospy.init_node('pose_est_client')
    pose_client = PoseEstClient()
    #pose_detector.run()
    rospy.spin()