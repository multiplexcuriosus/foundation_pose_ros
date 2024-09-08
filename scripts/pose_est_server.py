#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import SetBool,SetBoolResponse
from std_msgs.msg import Bool
from foundation_pose_ros.srv._CreateMask import CreateMask,CreateMaskRequest
from foundation_pose_ros.srv._ShelfPose import ShelfPose,ShelfPoseResponse


from foundation_pose.utils import set_logging_format, set_seed, draw_posed_3d_box, draw_xyz_axis, \
                                  project_3d_to_2d,draw_line3d,get_box_corners,draw_point3d,get_T_ce


import cv2
import copy
import numpy as np
import torch
import gc
import trimesh
import nvdiffrast.torch as dr
from threading import Lock
from scipy.spatial.transform import Rotation as R

from foundation_pose.estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from foundation_pose.utils import set_logging_format, set_seed, draw_posed_3d_box, draw_xyz_axis

from coordinate_frame_converter import CoordinateFrameConverter

class PoseDetector:
    """
    Subscribe to the image topic and publish the pose of the tracked object.
    """
    def __init__(self):

        seed = rospy.get_param("pose_detector/seed", 0)
        set_seed(seed)
        # set_logging_format()
        self._debug = True

        mesh_file = rospy.get_param("pose_detector/mesh_file")
        self._mesh, self._mesh_props = self._load_mesh(mesh_file)

        self._color_lock = Lock()
        self._depth_lock = Lock()
        self._initialized = False
        self._running = False
        self._has_color = False
        self._has_depth = False
        self._rate = rospy.Rate(rospy.get_param("pose_detector/refresh_rate"))

        self._est_refine_iter = rospy.get_param("pose_detector/estimator_refine_iters")
        self._track_refine_iter = rospy.get_param("pose_detector/tracker_refine_iters")

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self._estimator = FoundationPose(
            model_pts=self._mesh.vertices,
            model_normals=self._mesh.vertex_normals,
            mesh=self._mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            glctx=self.glctx,
        )


        self.final_mask = None
        self.service = rospy.Service("pose_est_server", ShelfPose, self._pose_est_server_cb)

        self._init_ros()
        print("[PoseEstServer]: Initialized")
    
    def create_pose_msg(self,T):
            pose_mat = T.reshape(4, 4)
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            #pose_msg.header.frame_id = self.color_frame_id
            pose_msg.pose.position.x = pose_mat[0, 3]
            pose_msg.pose.position.y = pose_mat[1, 3]
            pose_msg.pose.position.z = pose_mat[2, 3]
            quat = R.from_matrix(pose_mat[:3, :3]).as_quat()
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]
            return pose_msg


    def _pose_est_server_cb(self,request):
        print("[PoseEstServer]: Computing pose")

        # for real:
        color = self.ros_to_cv2(request.color_frame)
        depth = self.ros_to_cv2(request.depth_frame, desired_encoding="passthrough").astype(np.uint16) / 1000.0
        #depth = self.ros_to_cv2(request.depth_frame,desired_encoding="passthrough")
        #depth = depth.astype(np.uint8)

        #print("color.shape: "+str(color.shape))

        T_cs = self._detect_pose(color, depth)
        print("T_cs: "+str(T_cs))

        frame_converter = CoordinateFrameConverter(T_cs,self._mesh_props,self._K)
        T_ce = frame_converter.T_ce

        # Create response
        response = ShelfPoseResponse()
        T_ce_msg = self.create_pose_msg(T_ce)
        print("T_ce_msg: "+str(T_ce_msg))
        response.T_ce = T_ce_msg
        response.mask = self.cv2_to_ros(self.final_mask)
        response.has_five_contours = self.final_mask_has_five_contours

                
        if self._debug:
            # Draw bbox and T_CA coord axes
            pose_visualized = draw_posed_3d_box(self._K, img=color, ob_in_cam=T_cs, bbox=self._mesh_props["bbox"]) # 
            pose_visualized = draw_xyz_axis(color,ob_in_cam=T_cs,scale=0.1,K=self._K,thickness=3, transparency=0,is_input_rgb=True) 
            
            # Draw grasp poses 
            pose_visualized = draw_xyz_axis(pose_visualized,ob_in_cam=T_cs,scale=0.05,K=self._K,thickness=2,transparency=0,is_input_rgb=True)

            pose_visualized_msg = self.cv2_to_ros(pose_visualized)
            self._pose_debug_pub.publish(pose_visualized_msg)

        #self.customShutdown()
        return response
    
    def customShutdown(self):
        del self.scorer
        del self.refiner
        del self. glctx 
        del self._estimator

        gc.collect()
        torch.cuda.empty_cache()
        
    def _init_ros(self):
        pose_debug_image_topic = rospy.get_param("ros/pose_debug_image_topic")
        mask_debug_image_topic = rospy.get_param("ros/mask_debug_image_topic")
        pose_topic = rospy.get_param("ros/pose_topic", "/pose_detector/pose")

        self._bridge = CvBridge()

        self._pose_pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)
        self._pose_debug_pub = rospy.Publisher(pose_debug_image_topic, Image, queue_size=1)
        self._mask_debug_pub = rospy.Publisher(mask_debug_image_topic, Image, queue_size=1)
        self._debug_srv = rospy.Service("~debug_pose", SetBool, self._debug_callback)
        #self._shutdown_srv = rospy.Service("/shutdown_spice_up", SetBool, self._shutdown_cb)
        self._shutdown_srv = rospy.Subscriber("/shutdown_spice_up", Bool, self._shutdown_cb)

    def _load_mesh(self, mesh_file):
        mesh = trimesh.load(mesh_file, force="mesh")
        rospy.loginfo("[PoseDetectorNode]: Loaded mesh from %s", mesh_file)
        mesh_props = dict()
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        mesh_props["to_origin"] = to_origin
        mesh_props["bbox"] = bbox
        mesh_props["extents"] = extents
        return mesh, mesh_props

    def _shutdown_cb(self, signal):
        if signal.data:
         self.customShutdown()
         rospy.signal_shutdown("Job done")

    def _debug_callback(self, req):
        self._debug = req.data
        return True, "Debug mode set to {}".format(self._debug)

    def ros_to_cv2(self, frame: Image, desired_encoding="bgr8"):
        return self._bridge.imgmsg_to_cv2(frame, desired_encoding=desired_encoding)

    def cv2_to_ros(self, frame: np.ndarray):
        return self._bridge.cv2_to_imgmsg(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), encoding="rgb8")

    def _get_mask(self, color):
        try:
            rospy.wait_for_service("create_marker", timeout=10)
            mask_request = CreateMaskRequest()
            mask_request.data = self.cv2_to_ros(color)
            service_proxy = rospy.ServiceProxy("create_marker", CreateMask)
            mask_response = service_proxy(mask_request)
            mask_msg = mask_response.mask
            self._mask_debug_pub.publish(mask_msg)

            mask = self.ros_to_cv2(mask_msg, desired_encoding="passthrough").astype(np.uint8).astype(bool)
            
            self.final_mask = mask.astype(np.uint8) * 255
            self.final_mask_has_five_contours = mask_response.has_five_contours
            print("gotten mask shape: "+str(mask.shape))


            return mask
        except rospy.ROSException:
            rospy.logerr("[PoseDetectorNode]: Could not find service 'create_marker', Exiting!")
            rospy.signal_shutdown("Could not find service 'create_marker'")
            exit(1)

    def _get_intrinsics(self):
        intrinsics_topic = rospy.get_param("camera_info_topic", "/camera/color/camera_info")
        try:
            data = rospy.wait_for_message(intrinsics_topic, CameraInfo, timeout=10.0)
            K = np.array(data.K).reshape(3, 3).astype(np.float64)
            return K
        except rospy.ROSException:
            rospy.logwarn(f"[PoseDetectorNode]: Failed to get intrinsics from topic '{intrinsics_topic}', retrying...")
            return self._get_intrinsics()

    @torch.no_grad()
    def _detect_pose(self, color, depth):

        self._K = self._get_intrinsics()
        mask = self._get_mask(color)
        T_ca = self._estimator.register(K=self._K,rgb=color,depth=depth,ob_mask=mask,iteration=self._est_refine_iter)
        self._initialized = True

        # Tramsform from "anchor" A frame to shelf frame
        T_sa = self._mesh_props["to_origin"]
        
        T_cs = T_ca @ np.linalg.inv(T_sa)

        return T_cs

    def _run_detector(self):

        if self._running or not self._has_color or not self._has_depth:
            return

        self._color_lock.acquire()
        color = self.color.copy()
        self._has_color = False
        self._color_lock.release()

        self._depth_lock.acquire()
        depth = self.depth.copy()
        self._has_depth = False
        self._depth_lock.release()

        self._detect_pose(color, depth)

    def run(self):
        while not rospy.is_shutdown():
            self._run_detector()
            self._rate.sleep()


if __name__ == "__main__":
    rospy.init_node('pose_est_server', anonymous=True)
    pose_detector = PoseDetector()
    #pose_detector.run()
    rospy.spin()