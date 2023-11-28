#!/usr/bin/bash
nvidia-smi | grep 'python' | awk '{ print $5 }' | sudo xargs -n1 kill -9
sleep 3
source /home/praveen/alfred-autonomy/devel/setup.bash
export ROS_MASTER_URI=http://stretch-re1-1056:11311/
export ROS_IP=192.168.0.138
roslaunch alfred_core perception_brain.launch