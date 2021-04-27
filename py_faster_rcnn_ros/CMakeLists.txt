cmake_minimum_required(VERSION 2.8.3)
project(py_faster_rcnn_ros)

find_package(catkin REQUIRED)

catkin_package(
  #  INCLUDE_DIRS include
  #  LIBRARIES py-faster-rcnn
  #  CATKIN_DEPENDS other_catkin_pkg
  #  DEPENDS system_lib
  )

include_directories(
  # include
  # ${catkin_INCLUDE_DIRS}
  )
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )

install(PROGRAMS
  src/detector_manager.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

install(DIRECTORY models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )
