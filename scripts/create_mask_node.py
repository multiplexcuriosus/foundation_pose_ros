#!/usr/bin/env python3

import rospy
import rospkg
import os
import torch
import gc
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from foundation_pose_ros.srv import CreateMask, CreateMaskResponse
from plotting_utils import get_seg_points, show_masks, select_mask

import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor


class SAMMaskPredictor:

    def __init__(self):
        checkpoint = rospy.get_param("sam/checkpoint")
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(rospkg.RosPack().get_path('foundation_pose_ros'), checkpoint)
        self._checkpoint = checkpoint
        self._model_type = rospy.get_param("sam/model_type")
        self._device = rospy.get_param("sam/device")

        self._bridge = CvBridge()
        self.service = rospy.Service("create_marker", CreateMask, self._handle_create_mask)

    def ros_to_cv2(self, frame: Image, desired_encoding="bgr8"):
        return self._bridge.imgmsg_to_cv2(frame, desired_encoding=desired_encoding)

    def cv2_to_ros(self, frame: np.ndarray):
        return self._bridge.cv2_to_imgmsg(frame, encoding="mono8")

    @torch.no_grad()
    def _predict(self, image, input_points, input_labels):
        sam = sam_model_registry[self._model_type](checkpoint=self._checkpoint)
        sam.to(device=self._device)
        predictor = SamPredictor(sam)
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # Free up memory
        del sam
        gc.collect()
        torch.cuda.empty_cache()
        return masks, scores, logits

    def _handle_create_mask(self, req):
        image = req.data
        image = self.ros_to_cv2(image, desired_encoding="rgb8")
        input_points, input_labels = get_seg_points(image)
        masks, scores, logits = self._predict(image, input_points, input_labels)
        fig, axs = plt.subplots(len(masks), figsize=(6, 12))
        show_masks(image, input_points, input_labels, masks, scores, axs)
        best_mask_idx = select_mask(fig, axs)
        plt.close(fig)
        if best_mask_idx is None:
            return None
        best_mask = masks[best_mask_idx].astype(np.uint8) * 255

 
        img_msg = self.cv2_to_ros(best_mask)

        return CreateMaskResponse(mask=img_msg)


if __name__ == "__main__":

    rospy.init_node("create_mask_node", anonymous=True)
    predictor = SAMMaskPredictor()
    rospy.spin()
