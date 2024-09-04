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
from foundation_pose.utils import set_logging_format, set_seed, draw_posed_3d_box, draw_xyz_axis, \
                                  project_3d_to_2d,draw_line3d,get_box_corners,draw_point3d,get_T_ce


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
        self._mesh, self._mesh_props,self._extents = self._load_mesh(mesh_file)
        

        self._color_lock = Lock()
        self._depth_lock = Lock()
        self._initialized = False
        self._running = False
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
        self._mask_pub = rospy.Publisher("fp_mask", Image, queue_size=1)
        self._debug_pub = rospy.Publisher(debug_topic, Image, queue_size=1)
        self._debug_srv = rospy.Service("~debug_pose", SetBool, self._debug_callback)

    def _load_mesh(self, mesh_file):
        mesh = trimesh.load(mesh_file, force="mesh")
        rospy.loginfo("[PoseDetectorNode]: Loaded mesh from %s", mesh_file)
        mesh_props = dict()
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        print("extents: "+str(extents))
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        #print("bbox: "+str(bbox))
        mesh_props["to_origin"] = to_origin
        #print("to origin: "+str(to_origin))
        mesh_props["bbox"] = bbox
        return mesh, mesh_props,extents

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
        self._has_color = True
        self._color_lock.release()

    def _depth_callback(self, data: Image):
        self._depth_lock.acquire()
        self.depth = self.ros_to_cv2(data, desired_encoding="passthrough").astype(np.uint16) / 1000.0
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

    def create_pose_msg(self,T):
        pose_mat = T.reshape(4, 4)
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
        return pose_msg

    def compute_grasp_poses(self,T_eg,T_cg,T_bc,T_ec):
        T_cb = np.linalg.inv(T_bc)
        T_eb = T_ec @ T_cb
        T_bg = T_bc @ T_cg

        G_E = T_eg[:,3]
        B_E = T_eb[:,3]

        gamma_E = np.array([B_E[0],B_E[1],G_E[2],1.0]) # point on height of grasp positions below frame (used to make grasp tf orthonormal)
        gammaG_r_E = G_E - gamma_E
        gammaG_r_E_norm = gammaG_r_E / np.linalg.norm(gammaG_r_E)
        normal_to_gammaG_in_gx_gy_plane = np.array([-gammaG_r_E_norm[1],gammaG_r_E_norm[0],0])
        
        BG_r_B = T_bg[:,3]
        BG_r_C = T_cb @ BG_r_B
        BG_r_C_norm = BG_r_C[0:3] / np.linalg.norm(BG_r_C[0:3])

        grasp_x = normal_to_gammaG_in_gx_gy_plane
        grasp_y = BG_r_C_norm
        grasp_z = np.cross(grasp_x,grasp_y)
        grasp_rot = np.array([grasp_x,grasp_y,grasp_z])

        grasp_tf_C = np.zeros((4,4))
        grasp_tf_C[0:3,0:3] = grasp_rot
        grasp_tf_C[:,3] = BG_r_C
        grasp_tf_C[3,3] = 1

        return grasp_tf_C

    def get_grasp_and_drop_off_poses(self,T_cs):

        # Get shelf corners in C frame
        bounding_box =self._mesh_props["bbox"]
        corners_S = get_box_corners(bounding_box)
        corners_C = []
        for p_S in corners_S:
            p_C =(T_cs @ p_S)[0:3]
            corners_C.append(p_C)
        
        # Get transform from C-frame to E-frame
        T_ce = get_T_ce(corners_C)
        T_ec = np.linalg.inv(T_ce)

        # CONSTRUCT GRASPING POSES in C-frame
        EP0_r_E = np.array([0.040,0.285,0.392,1.0])
        EP1_r_E = np.array([0.040,0.125,0.392,1.0])
        EP2_r_E = np.array([0.035,0.280,0.042,1.0])
        EP3_r_E = np.array([0.035,0.130,0.042,1.0])
        z_off = np.array([0,0,0.03,0.0])
        EG0_r_E = EP0_r_E + z_off
        EG1_r_E = EP1_r_E + z_off
        EG2_r_E = EP2_r_E + z_off
        EG3_r_E = EP3_r_E + z_off

        T_cb = np.array([[1,0,0,0], # fake base transform --> TODO Get from tf pub?
                        [0,1,0,0.6],
                        [0,0,1,0.6],
                        [0,0,0,1]])
        T_bc = np.linalg.inv(T_cb)
        
        # Construct transforms from E t0 g0,g1,g2,g3 (grasp poses)
        T_eg0 = np.zeros((4,4))
        T_eg0[0:3,0:3] = np.identity(3)
        T_eg0[0:4,3] = EG0_r_E
        T_eg0[3,3] = 1
        
        T_eg1 = np.zeros((4,4))
        T_eg1[0:3,0:3] = np.identity(3)
        T_eg1[0:4,3] = EG1_r_E
        T_eg1[3,3] = 1

        T_eg2 = np.zeros((4,4))
        T_eg2[0:3,0:3] = np.identity(3)
        T_eg2[0:4,3] = EG2_r_E
        T_eg2[3,3] = 1

        T_eg3 = np.zeros((4,4))
        T_eg3[0:3,0:3] = np.identity(3)
        T_eg3[0:4,3] = EG3_r_E
        T_eg3[3,3] = 1

        # Construct transforms from C to GRASP_i
        T_cg0 = T_ce @ T_eg0
        T_cg1 = T_ce @ T_eg1
        T_cg2 = T_ce @ T_eg2
        T_cg3 = T_ce @ T_eg3
        
        # Construct pick up grasp poses
        T_cg0 = self.compute_grasp_poses(T_eg0,T_cg0,T_bc,T_ec)
        T_cg1 = self.compute_grasp_poses(T_eg1,T_cg1,T_bc,T_ec)
        T_cg2 = self.compute_grasp_poses(T_eg2,T_cg2,T_bc,T_ec)
        T_cg3 = self.compute_grasp_poses(T_eg3,T_cg3,T_bc,T_ec)

        # Create drop off pose DO0,DO1 ->TODO: drop off position bottom part of bottle or com?
        shelf_depth = self._extents[0]
        shelf_height = self._extents[1] # TODO: CHECK IF STILL TRUE FOR SMALLER KALLAX !! 
        shelf_width = self._extents[2] # TODO: CHECK IF STILL TRUE FOR SMALLER KALLAX !! 
        
        DO0_E = np.array([shelf_depth/2,shelf_width*0.75,shelf_height,1.0])
        EDO0_rot = np.identity(3)
        CDO0_rot = T_ce[0:3,0:3] @ EDO0_rot
        T_cdo0 = np.zeros((4,4))
        T_cdo0[0:3,0:3] = CDO0_rot
        T_cdo0[:,3] = T_ce @ DO0_E
        T_cdo0[3,3] = 1

        DO1_E = np.array([shelf_depth/2,shelf_width*0.25,shelf_height,1.0])
        EDO1_rot = np.identity(3)
        CDO1_rot = T_ce[0:3,0:3] @ EDO1_rot
        T_cdo1 = np.zeros((4,4))
        T_cdo1[0:3,0:3] = CDO1_rot
        T_cdo1[:,3] = T_ce @ DO1_E
        T_cdo1[3,3] = 1

        return T_cg0,T_cg1,T_cg2,T_cg3,T_cdo0,T_cdo1


    def _detect_pose(self, color, depth):
        self._running = True
        if not self._initialized:
            self._K = self._get_intrinsics()
            mask = self._get_mask(color)
            T_ca = self._estimator.register(K=self._K,rgb=color,depth=depth,ob_mask=mask,iteration=self._est_refine_iter)
            self._initialized = True
        else:
            T_ca = self._estimator.track_one(rgb=color,depth=depth,K=self._K,iteration=self._track_refine_iter)
        
        # Create pose msg for T_ca
        T_ca_msg = self.create_pose_msg(T_ca)

        # Compute other important transforms
        T_sa = self._mesh_props["to_origin"]
        T_cs = T_ca @ np.linalg.inv(T_sa)

        # Get all poses
        T_cg0,T_cg1,T_cg2,T_cg3,T_cdo0,T_cdo1 = self.get_grasp_and_drop_off_poses(T_cs)

        # Create pose msgs
        T_cg0_msg = self.create_pose_msg(T_cg0)
        T_cg1_msg = self.create_pose_msg(T_cg1)
        T_cg2_msg = self.create_pose_msg(T_cg2)
        T_cg3_msg = self.create_pose_msg(T_cg3)
        T_cdo0_msg = self.create_pose_msg(T_cdo0)
        T_cdo1_msg = self.create_pose_msg(T_cdo1)

        # Publish pose msgs
        self._pose_pub.publish(T_ca_msg)


        if self._debug:
            # Draw bbox and T_CA coord axes
            pose_visualized = draw_posed_3d_box(self._K, img=color, ob_in_cam=T_cs, bbox=self._mesh_props["bbox"]) # 
            pose_visualized = draw_xyz_axis(color,ob_in_cam=T_cs,scale=0.1,K=self._K,thickness=3, transparency=0,is_input_rgb=True) 
            
            # Draw grasp poses 
            pose_visualized = draw_xyz_axis(pose_visualized,ob_in_cam=T_cg0,scale=0.05,K=self._K,thickness=2,transparency=0,is_input_rgb=True)
            pose_visualized = draw_xyz_axis(pose_visualized,ob_in_cam=T_cg1,scale=0.05,K=self._K,thickness=2,transparency=0,is_input_rgb=True)
            pose_visualized = draw_xyz_axis(pose_visualized,ob_in_cam=T_cg2,scale=0.05,K=self._K,thickness=2,transparency=0,is_input_rgb=True)
            pose_visualized = draw_xyz_axis(pose_visualized,ob_in_cam=T_cg3,scale=0.05,K=self._K,thickness=2,transparency=0,is_input_rgb=True)

            # Draw drop off poses
            pose_visualized = draw_xyz_axis(pose_visualized,ob_in_cam=T_cdo0,scale=0.05,K=self._K,thickness=2,transparency=0,is_input_rgb=True)
            pose_visualized = draw_xyz_axis(pose_visualized,ob_in_cam=T_cdo1,scale=0.05,K=self._K,thickness=2,transparency=0,is_input_rgb=True)

            pose_visualized_msg = self.cv2_to_ros(pose_visualized)
            pose_visualized_msg.header.stamp = T_ca_msg.header.stamp
            self._debug_pub.publish(pose_visualized_msg)

        self._running = False

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
    rospy.init_node('foundation_pose_detector', anonymous=True)
    pose_detector = PoseDetector()
    pose_detector.run()