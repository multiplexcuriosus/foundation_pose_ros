#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import SetBool
from foundation_pose_ros.srv import CreateMask

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


class PoseDetector:
    """
    Subscribe to the image topic and publish the pose of the tracked object.
    """
    def __init__(self):

        seed = rospy.get_param("pose_detector/seed", 0)
        set_seed(seed)
        # set_logging_format()
        self._debug = True

        self.done = False
       

        # Resize
        shorter_side = 480

        self.H,self.W = (480,640) # HARDCODED!!!

        if shorter_side is not None:
            self.downscale = shorter_side/min(self.H, self.W)

        self.H = int(self.H*self.downscale)
        self.W = int(self.W*self.downscale)


        mesh_file = rospy.get_param("pose_detector/mesh_file")
        self._mesh, self._mesh_props = self._load_mesh(mesh_file)

        self._color_lock = Lock()
        self._depth_lock = Lock()
        self._initialized = False
        #self._running = False
        self._has_color = False
        self._has_depth = False
        self._rate = rospy.Rate(rospy.get_param("pose_detector/refresh_rate"))

        self._est_refine_iter = rospy.get_param("pose_detector/estimator_refine_iters")
        self._track_refine_iter = rospy.get_param("pose_detector/tracker_refine_iters")

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self._estimator = FoundationPose(
            model_pts=self._mesh.vertices,
            model_normals=self._mesh.vertex_normals,
            mesh=self._mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
        )

        self._init_ros()
        rospy.loginfo("[PoseDetectorNode]: Initialized FoundationPose")

    def _init_ros(self):
        color_topic = rospy.get_param("ros/color_image_topic")
        depth_topic = rospy.get_param("ros/depth_image_topic")
        debug_topic = rospy.get_param("ros/debug_image_topic", "/pose_detector/debug/image")
        pose_topic = rospy.get_param("ros/pose_topic", "/pose_detector/pose")

        self._bridge = CvBridge()
        self._img_sub = rospy.Subscriber(color_topic, Image, self._color_callback)
        self._depth_sub = rospy.Subscriber(depth_topic, Image, self._depth_callback)
        self._pose_pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)
        self._debug_pub = rospy.Publisher(debug_topic, Image, queue_size=1)
        self._debug_srv = rospy.Service("~debug_pose", SetBool, self._debug_callback)

    def _load_mesh(self, mesh_file):
        mesh = trimesh.load(mesh_file, force="mesh")
        rospy.loginfo("[PoseDetectorNode]: Loaded mesh from %s", mesh_file)
        mesh_props = dict()
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        mesh_props["to_origin"] = to_origin
        mesh_props["bbox"] = bbox
        return mesh, mesh_props

    def _debug_callback(self, req):
        self._debug = req.data
        return True, "Debug mode set to {}".format(self._debug)

    def ros_to_cv2(self, frame: Image, desired_encoding="bgr8"):
        return self._bridge.imgmsg_to_cv2(frame, desired_encoding=desired_encoding)

    def cv2_to_ros(self, frame: np.ndarray):
        return self._bridge.cv2_to_imgmsg(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), encoding="rgb8")

    def _color_callback(self, data: Image):
        self._color_lock.acquire()
        self.color_frame_id = data.header.frame_id
        self.color = self.ros_to_cv2(data, desired_encoding="bgr8")
        self.color = cv2.resize(self.color, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
        self._has_color = True
        self._color_lock.release()

    def _depth_callback(self, data: Image):
        self._depth_lock.acquire()
        self.depth = self.ros_to_cv2(data, desired_encoding="passthrough").astype(np.uint16) / 1000.0
        self.depth = cv2.resize(self.depth, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
        self.depth[(self.depth < 0.1)] = 0
        self._has_depth = True
        self._depth_lock.release()

    def _get_mask(self, color):
        try:
            rospy.wait_for_service("create_marker", timeout=10)
            service_proxy = rospy.ServiceProxy("create_marker", CreateMask)
            data = service_proxy(self.cv2_to_ros(color))
            mask = self.ros_to_cv2(data.mask, desired_encoding="passthrough").astype(np.uint8).astype(bool)
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
        self._K[:2] *= self.downscale # RESIZING
        mask = self._get_mask(color)
        rospy.loginfo("[PoseDetectorNode]: Computing Pose")
        pose = self._estimator.register(
            K=self._K,
            rgb=color,
            depth=depth,
            ob_mask=mask,
            iteration=self._est_refine_iter,
        )

        pose_mat = pose.reshape(4, 4)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = self.color_frame_id
        pose_msg.pose.position.x = pose_mat[0, 3]
        pose_msg.pose.position.y = pose_mat[1, 3]
        pose_msg.pose.position.z = pose_mat[2, 3]
        quat = R.from_matrix(pose_mat[:3, :3]).as_quat()
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        self._pose_pub.publish(pose_msg)
        rospy.loginfo("[PoseDetectorNode]: Pose message published")



        if self._debug:
            center_pose = pose @ np.linalg.inv(self._mesh_props["to_origin"])
            pose_visualized = draw_posed_3d_box(self._K, img=color, ob_in_cam=center_pose, bbox=bounding_box)
            pose_visualized = draw_xyz_axis(
                color,
                ob_in_cam=center_pose,
                scale=0.1,
                K=self._K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            pose_visualized_msg = self.cv2_to_ros(pose_visualized)
            pose_visualized_msg.header.stamp = pose_msg.header.stamp
            self._debug_pub.publish(pose_visualized_msg)
            rospy.loginfo("[PoseDetectorNode]:Visualization image published")
       
        self.done = True

    def _run_detector(self):

        if self.done or not self._has_color or not self._has_depth:
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

            if self.done:
                rospy.signal_shutdown("[PoseDetectorNode]: Done -> Shutting down")



if __name__ == "__main__":
    rospy.init_node('foundation_pose_detector', anonymous=True)
    pose_detector = PoseDetector()
    pose_detector.run()
