# The foundation_pose_ros node
This ReadMe is structured into:
* **Overview** 
* **Installation** (foundation_pose_ros+spice_selection_gui)
* **Configuration** (foundation_pose_ros+spice_selection_gui)
* **Launch** (foundation_pose_ros+spice_selection_gui)


## Overview
![sa_slide_extraction-7](https://github.com/user-attachments/assets/c232ff90-586b-401d-b2fb-4ca687f82a6d)
## Installation
### foundation_pose_ros  

1. Choose an appropriate location to store the leggedrobotics foundationpose fork, clone it and cd into it:
```
git clone https://github.com/leggedrobotics/FoundationPose.git`
cd Foundationpose
```
2. Install conda or mamba  
3. Setup the conda environment (based on instructions [here](https://github.com/leggedrobotics/foundation_pose_ros))
```
# create conda environment
conda create -n foundationpose python=3.9 

# activate conda environment
conda activate foundationpose

# Install requirements
python -m pip install -r requirements.txt


#Install cuda toolkit & runtime: do not install any package with cuda in the name into the conda venv that has a version number not equal to 11.8 !
conda install nvidia/label/cuda-11.8.0::cuda-toolkit -c nvidia/label/cuda-11.8.0
conda install nvidia/label/cuda-11.8.0::cuda-runtime -c nvidia/label/cuda-11.8.0

# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"

# Install NVDiffRast
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
 ```

Executing the last line will most likely cause errors. To remedy them do the following:  

Error: `cannot find -lcudart: No such file or directory`  
Solution:  
```
mkdir /home/<username>/miniforge3/envs/foundationpose/lib64
cp /home/<username>/miniforge3/envs/foundationpose/lib/libcudart.* /home/<username>/miniforge3/envs/foundationpose/lib64 
```

Error:`RuntimeError: Error compiling objects for extension` or other `eigen3` related problems  
Solution: 
In `/FoundationPose/foundation_pose/bundlesdf/mycuda/setup.py` edit the `include_dirs` as follows:
```
    include_dirs=[
        #"/usr/local/include/eigen3",
        #"/usr/include/eigen3",
        "/home/<username>/miniforge3/envs/foundationpose/include/eigen3"

    ],
```
The `CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh` command was succesfull if it results in 
"Successfully installed common" being displayed in the terminal.

4. In the Foundationpose directory, switch to the branch feature/realsense  with:
```
git checkout feature/realsense
```
This will only work if you commit the changes made to setup.py.  

5. In the Foundationpose directory, install the module into the venv with: 
```
pip install -e .
```
6. Download the foundationpose model weigths from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i) and put them into `Foundationpose/foundation_pose/weights`.
7. Clone Jaú's fork of the foundationpose wrapper into `catkin_ws/src` with:
```
git clone https://github.com/multiplexcuriosus/foundation_pose_ros.git`
```
8. Follow steps 3-5 [here](https://github.com/leggedrobotics/foundation_pose_ros). For step 4: Put the SAM weights into `foundation_pose_ros/models/sam`
9. Clone the rqt plugin into `catkin_ws/src` with:
```
git clone https://github.com/multiplexcuriosus/spice_selection_gui.git
```
10. Build the `foundation_pose_ros` & the `spice_selection_gui` packages with:
```
catkin build foundation_pose_ros
catkin build spice_selection_gui
```  
### Troubleshooting 
Error: `Unable to find either executable 'empy' or Python module 'em'...  try installing the package 'python3-empy'`   
Solution:  
```
catkin build foundation_pose_ros -DPYTHON_EXECUTABLE=/usr/bin/python3
catkin build spice_selection_gui -DPYTHON_EXECUTABLE=/usr/bin/python3
```  
Error: `PermissionError: [Errno 13] Permission denied: '/tmp/material_0.png'`  
Solution:  
```
sudo rm /tmp/material_0.png
sudo rm /tmp/material.mtl
```

## Configuration
### create_mask_server (foundation_pose_ros/config/create_mask.yaml) ###  
Possible configurations:  
**I**     
`sam/choose_largest_mask:  True`: The create_mask_server node will automatically choose the largest mask among the masks it got from SAM. 'choose_largest_mask' has no effect  
**II**     
`sam/choose_largest_mask: False` && `sam/inspect_masks: False`: The create_mask_server node will loop through all masks and return the largest one with five contours. If no mask has five contours the largest one is returned.  
**III**   
`sam/choose_largest_mask: False` && `sam/inspect_masks: True`: Same as II + ALL found masks are displayed (with cv2.waitKey(0)). The largest one is returned.
### pose_est_server (foundation_pose_ros/config/pose_detector.yaml)
* Set `pose_detector/mesh_file` to path with valid mesh file
* (optional): Tune `pose_detector/estimator_refine_iters` & `pose_detector/tracker_refine_iters`
* When testing on real robot: replace `/camera` in all relevant `ros: ...` yaml-parameters, e.g `/camera/color/camera_info`

## Launch
### foundation_pose_ros
There is only one relevant launch file: `only_estimate.launch`. It launches:
* the parameter files:  
  * `pose_detector.yaml`
  * `create_mask.yaml`  
* the `foundation_pose_ros` node
* the services:  
  * `pose_est_server` 
  * `create_mask_server`   

Launch with:
```
roslaunch foundation_pose_ros all.launch
```
### spice_up_plugin
The spice_selection_gui node is launched by selecting the `SpiceSelectionPlugin` in the rqt-plugin-dropdown. The spice_selection_gui will then provide the `spice_name_server` service.

## Testing
### create_mask_service
The `create_mask_service` can be tested by running:  
```
conda activate foundationpose # Terminal 1
rosrun foundation_pose_ros create_mask_server.py # Terminal 1
rosrun foundation_pose_ros create_mask_client.py # Terminal 2
```
In the file `create_mask_server.py`, the following variables need to be adjusted:  
* `color_img_path`
* `mask_path`

### estimate_pose_service

The `estimate_pose_service` service can be tested by running:  
```
conda activate foundationpose # Terminal 1
rosrun foundation_pose_ros pose_est_server.py # Terminal 1
rosrun foundation_pose_ros pose_est_client.py # Terminal 2
```
Color-imgs, depth-imgs and intrinsics-info must be present in the ros network.




