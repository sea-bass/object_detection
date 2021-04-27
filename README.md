# py_faster_rcnn_ros
This package provides a ros warpper for py-faster-rcnn [here](https://github.com/rbgirshick/py-faster-rcnn).

## Usage

### Live:
```bash
roslaunch py_faster_rcnn_ros detector.launch
rosservice call /start_detection <confidence_threshold>
```

### Bag test:
```bash
roslaunch py_faster_rcnn_ros bag_test.launch \
           filename:=[full path to bag file] \
           topic:=[topic in bag to run detections on] \
           <compressed:=true|false>
```

## Building
```bash
rosdep install --from-paths . --ignore-src --rosdistro=$ROS_DISTRO -y
catkin build
```


You need to have nvidia-docker2 installed
Follow the instructions here.
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide



Start the docker image
```
docker run -rm -v ~/curiosity/catkin_ws_docker:/curiosity/catkin_ws_docker -it --network host --gpus all -w /curiosity/catkin_ws_docker 88da16acc73c  bash
```


Inside the docker image, run
```
roslaunch py_faster_rcnn_ros detector.launch
```