#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import SetBool
#
# from foundation_pose_ros.srv import CreateMask

from foundation_pose_ros.srv._CreateMask import CreateMask,CreateMaskRequest
from foundation_pose_ros.srv._ShelfPose import ShelfPose,ShelfPoseResponse

import cv2
import copy
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from threading import Lock
from scipy.spatial.transform import Rotation as R

from foundation_pose.estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from foundation_pose.utils import set_logging_format, set_seed, draw_posed_3d_box, draw_xyz_axis

from coordinate_frame_converter import CoordinateFrameConverter

class CreateMaskClient:

    def __init__(self) -> None:

        self._bridge = CvBridge()

        rospy.wait_for_service("create_mask_service")
        print("[MaskCreatorClient]: Initialized")

        color_img_path = "/home/jau/Desktop/cleanup_tests/rgb_new.png"
        color = cv2.imread(color_img_path)

        mask_request = CreateMaskRequest()
        mask_request.data = self.cv2_to_ros(color)

        create_mask_service_handle = rospy.ServiceProxy("create_mask_service", CreateMask)
        print("[MaskCreatorClient]: Request sent")
        mask_response = create_mask_service_handle(mask_request)
        print("[MaskCreatorClient]: Got mask: ")
        mask_msg = mask_response.mask
        mask = self.ros_to_cv2(mask_msg, desired_encoding="passthrough").astype(np.uint8)   
        mask_path = "/home/jau/Desktop/cleanup_tests/resulting_mask.png"
        cv2.imwrite(mask_path,mask)


   
    def _color_callback(self, data: Image):
        self.last_color = self.ros_to_cv2(data, desired_encoding="bgr8")


    def ros_to_cv2(self, frame: Image, desired_encoding="bgr8"):
        return self._bridge.imgmsg_to_cv2(frame, desired_encoding=desired_encoding)

    def cv2_to_ros(self, frame: np.ndarray):
        return self._bridge.cv2_to_imgmsg(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), encoding="rgb8")

    
    


if __name__ == "__main__":
    rospy.init_node('pose_est_client')
    create_mask_client = CreateMaskClient()
    rospy.spin()