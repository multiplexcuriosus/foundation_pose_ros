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
from std_srvs.srv import SetBool
from std_msgs.msg import Bool
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor,SamAutomaticMaskGenerator,SamPredictor


class SAMMaskPredictor:

    def __init__(self):
        
        self.done = False

        checkpoint = rospy.get_param("sam/checkpoint")
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(rospkg.RosPack().get_path('foundation_pose_ros'), checkpoint)
        self._checkpoint = checkpoint
        self._model_type = rospy.get_param("sam/model_type")
        self._device = rospy.get_param("sam/device")

        self._bridge = CvBridge()
        self.service = rospy.Service("create_marker", CreateMask, self._handle_create_mask)

        #self._shutdown_srv = rospy.Service("shutdown_spice_up", SetBool, self._shutdown_cb)
        self._shutdown_srv = rospy.Subscriber("/shutdown_spice_up", Bool, self._shutdown_cb)
        print("[CreateMaskServer] :  Initialized")


    def _shutdown_cb(self, signal):
        if signal.data:
            print("[CreateMaskServer] :  Shutting down")
            rospy.signal_shutdown("Job done")
        return True

    def ros_to_cv2(self, frame: Image, desired_encoding="bgr8"):
        return self._bridge.imgmsg_to_cv2(frame, desired_encoding=desired_encoding)

    def cv2_to_ros(self, frame: np.ndarray):
        return self._bridge.cv2_to_imgmsg(frame, encoding="mono8")


    def _predict(self,image):
      
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
    
    def _handle_create_mask(self, req):
        print("[CreateMaskServer] :  Received request")
        image = req.data
        image = self.ros_to_cv2(image, desired_encoding="rgb8")
        #input_points, input_labels = get_seg_points(image)
        #masks, scores, logits = self._predict(image, input_points, input_labels)
        masks = self._predict(image)

        best_mask,has_5_contours = self.get_best_mask(masks)
        if not has_5_contours:
            print("[CreateMaskServer] :  WARNING: USING MASK WITH N CONTOURS != 5")
        
        ''' 
        cv2.imshow("",best_mask)
        cv2.waitKey(0)
        
        best_mask_clean = self.clean_mask(best_mask,10)
        cv2.imshow("",best_mask_clean)
        cv2.waitKey(0)
        '''
        best_mask = masks[0].astype(np.uint8) * 255
        img_msg = self.cv2_to_ros(best_mask)
        print("[CreateMaskServer] :  Response sent")

        return CreateMaskResponse(mask=img_msg)
    
    def clean_mask(self,mask,k):
        # Apply structuring element
        #k = 10
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k,k))
        morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return morph

    def get_best_mask(self,masks):
        first_mask = masks[0].astype(np.uint8) * 255
        for mask in masks:
                mask = mask.astype(np.uint8) * 255
                mask = self.clean_mask(mask,5)
                #mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                contours, hier = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                N_conts = len(contours)
                print("N contours: "+str(N_conts))
                if N_conts == 5:
                    return mask,True       
                #cv2.imshow("",mask)
                #cv2.waitKey(0)
        #cv2.destroyAllWindows() 
        return first_mask,False


if __name__ == "__main__":

    rospy.init_node("create_mask_node", anonymous=True)
    predictor = SAMMaskPredictor()
    rospy.spin()