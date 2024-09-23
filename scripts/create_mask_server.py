#!/usr/bin/env python3

import rospy
import rospkg
import os
import torch
import gc
import cv2
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from foundation_pose_ros.srv import CreateMask, CreateMaskResponse
from plotting_utils import get_seg_points, show_masks, select_mask
from std_srvs.srv import SetBool
from std_msgs.msg import Bool
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor,SamAutomaticMaskGenerator,SamPredictor


class SAMMaskPredictor:

    def __init__(self):
        """
        The SamMaskPredictor provides the create_mask service (CreateMask.srv).
        It receives an RGB image as input, uses the SamAutomaticMaskGenerator to infer all masks, 
        selects the best fitting mask and returns it in the servcice response.
        """

        # Get approach param (True: automatically choose largest found mask, False: Choose largest with 5 holes, largest if none has five holes)
        self.choose_largest = rospy.get_param("sam/choose_largest_mask") 

        # Init model params + CVBridge
        checkpoint = rospy.get_param("sam/checkpoint")
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(rospkg.RosPack().get_path('foundation_pose_ros'), checkpoint)
        self._checkpoint = checkpoint
        self._model_type = rospy.get_param("sam/model_type")
        self._device = rospy.get_param("sam/device")
        self._bridge = CvBridge()

        # Start create_mask service
        self.service = rospy.Service("create_mask_service", CreateMask, self._handle_create_mask)

        # Start shutdown-service
        self._shutdown_srv = rospy.Subscriber("/shutdown_spice_up", Bool, self._shutdown_cb)

        print("[CreateMaskServer] :  Initialized")

    def _handle_create_mask(self, req):
        print("[CreateMaskServer] :  Received request")
        
        t0 = time.perf_counter() # Time measurement

        target_color_img = self._ros_to_cv2(req.data, desired_encoding="rgb8")
        all_masks = self._get_all_masks(target_color_img)

        # Select best mask
        if self.choose_largest:
             # Assume largest mask is correct one and use it
            best_mask = all_masks[0].astype(np.uint8) * 255
            has_five_contours = self._count_contours(best_mask) == 5
        else:
            # Choose largest mask with 5 contours, or the largest mask if no mask has 5 contours
            best_mask,has_five_contours = self._get_best_mask(all_masks)

        if not has_five_contours:
            print("[CreateMaskServer] :  WARNING: Mask does not have 5 contours")

        
        best_mask_msg = self._cv2_to_ros(best_mask)
        print("[CreateMaskServer] :  Response sent")

        print("[CreateMaskServer] :  Total duration: "+ "%0.2f" % (time.perf_counter() - t0)+"s")

        return CreateMaskResponse(mask=best_mask_msg,has_five_contours=has_five_contours)

    def _get_all_masks(self,image):
      
        # Mask inferece with SAM 
        # reference: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-anything-with-sam.ipynb
        sam = sam_model_registry[self._model_type](checkpoint=self._checkpoint)
        sam.to(device=self._device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        sam_result = mask_generator.generate(image)
        masks = [mask['segmentation'] for mask in sorted(sam_result, key=lambda x: x['area'],reverse=True)]

        # Free up memory
        del sam,mask_generator,sam_result
        gc.collect()
        torch.cuda.empty_cache()
        
        return masks

    def _get_best_mask(self,masks):

        inspection = rospy.get_param("sam/inspect_masks") # Loop through found masks and display them

        first_mask = masks[0].astype(np.uint8) * 255
        N_conts = None
        for mask in masks:
                mask = mask.astype(np.uint8) * 255
                N_conts = self._count_contours(mask)     
                if inspection: 
                    print("N contours: "+str(N_conts))
                    cv2.imshow("",mask)
                    cv2.waitKey(0)
                if N_conts == 5 and not inspection:
                    return mask,True 
        if inspection:
            cv2.destroyAllWindows() 
        return first_mask,N_conts == 5

    def _count_contours(self,mask):
        mask = self._clean_mask(mask,5)
        contours, hier = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)

    # Utils -------------------------------------------------------

    def _clean_mask(self,mask,kernel_size):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
        morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return morph

    def _ros_to_cv2(self, frame: Image, desired_encoding="bgr8"):
        return self._bridge.imgmsg_to_cv2(frame, desired_encoding=desired_encoding)

    def _cv2_to_ros(self, frame: np.ndarray):
        return self._bridge.cv2_to_imgmsg(frame, encoding="mono8")

    def _shutdown_cb(self, signal):
        if signal.data:
            print("[CreateMaskServer] :  Shutting down")
            rospy.signal_shutdown("Job done")
        return True
    
    # ---------------------------------------------------------------

if __name__ == "__main__":

    rospy.init_node("create_mask_server", anonymous=True)
    predictor = SAMMaskPredictor()
    rospy.spin()
