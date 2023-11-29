#!/usr/bin/bash
nvidia-smi | grep 'python' | awk '{ print $5 }' | sudo xargs -n1 kill -9
sleep 3
source /home/abhinav/FVD/alfred-autonomy/devel/setup.bash
export ROS_MASTER_URI=http://192.168.0.232:11311/ # 1056
export ROS_IP=192.168.0.138
roslaunch vlmaps_ros vlmap_server.launch