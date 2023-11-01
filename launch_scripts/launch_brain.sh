#!/usr/bin/bash
nvidia-smi | grep 'python' | awk '{ print $5 }' | sudo xargs -n1 kill -9
sleep 3
source /home/praveen/ws1/devel/setup.bash
export ROS_MASTER_URI=http://stretch-re1-1061:11311/
roslaunch alfred_core perception_brain.launch