cmake_minimum_required(VERSION 3.0.2)
project(foundation_pose_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  std_srvs
  sensor_msgs
  geometry_msgs
  message_generation
)

catkin_python_setup()


## Generate services in the 'srv' folder
add_service_files(
  FILES
  CreateMask.srv
  EstPose.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs roscpp

)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

