# Foundation Pose ROS

This package provides a ROS node for detecting the pose of a tracked object using the Foundation Pose. It subscribes to an image topic and publishes the pose of the object in the form of a `PoseStamped` message. Additionally, it also provides a service to initialize the object mask using Segment Anything.

## Installation

1. Follow the instructions [here](https://github.com/leggedrobotics/FoundationPose/tree/feature/realsense?tab=readme-ov-file#env-setup-option-2-conda-experimental) to setup a conda environment for FoundationPose.

2. Clone the internal leggedrobotics FoundationPose repository. Switch to the branch `feature/realsense` and install the python module in your conda environment.

    ```
    git clone git@github.com:leggedrobotics/FoundationPose.git
    cd FoundationPose && git checkout `feature/realsense`
    pip install -e .
    ```
3. Install Segment Anything.

    ```
    pip install git+https://github.com/facebookresearch/segment-anything.git
    ```

4. Download the [model checkpoints](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) for SAM in the `foundation_pose_ros/models/sam` directory.

5. Install the python module for ROS:
    ```
    pip install rospkg
    ```

## Usage

1. Launch the ROS node:

    ```
    roslaunch foundation_pose_ros pose_detector.launch
    ```

2. The node will subscribe to the specified color and depth image topics and publish the pose of the tracked object on the specified pose topic.

3. You can enable or disable debug mode by calling the `~debug_pose` service with the desired boolean value.

## Configuration

The behavior of the pose detector can be configured by modifying the parameters in the `pose_detector_node.py` file or by setting the corresponding ROS parameters.

- `pose_detector/seed`: Random seed for reproducibility.
- `pose_detector/mesh_file`: Path to the mesh file representing the tracked object.
- `pose_detector/refresh_rate`: Refresh rate of the pose detector.
- `pose_detector/estimator_refine_iters`: Number of iterations for refining the pose estimation.
- `pose_detector/tracker_refine_iters`: Number of iterations for refining the pose tracking.
- `ros/color_image_topic`: ROS topic for the color image.
- `ros/depth_image_topic`: ROS topic for the depth image.
- `ros/debug_image_topic`: ROS topic for publishing debug images.
- `ros/pose_topic`: ROS topic for publishing the pose.
