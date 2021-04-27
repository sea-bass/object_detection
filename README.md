# py_faster_rcnn_ros
This package provides a ros warpper for py-faster-rcnn [here](https://github.com/rbgirshick/py-faster-rcnn).

## Usage

You should make a new catkin workspace for the code in this repo.  This is all of the code that will need to run in the docker image.


You need to have nvidia-docker2 installed
Follow the instructions here.
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide

Download the docker image from dockerhub
```
docker pull kschmeckpeper/py_faster_rcnn_ros
```


Start the docker image
```
docker run -rm -v ~/PATH/TO/catkin_ws_docker:/curiosity/catkin_ws_docker -it --network host --gpus all -w /curiosity/catkin_ws_docker kschmeckpeper/py_faster_rcnn_ros  bash
```

The first time you start the docker image, you will need to build the code in the catkin workspace.  


On the host machine, start ros.

Inside the docker image, run
```
roslaunch py_faster_rcnn_ros detector.launch
```

This should connect to the ros network on the host machine.