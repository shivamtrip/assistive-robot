#!/usr/bin/bash
nvidia-smi | grep 'python' | awk '{ print $5 }' | sudo xargs -n1 kill -9
sleep 3
source /home/praveen/alfred-autonomy/devel/setup.bash
# export ROS_MASTER_URI=http://172.26.19.239:11311/
export ROS_IP=172.26.55.74
export ROS_MASTER_URI=http://172.26.25.59:11311/
roslaunch yolo object_detector.launch