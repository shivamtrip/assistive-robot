#!/usr/bin/bash

./art.sh
echo "----------DOWNLOADING ALFRED's BRAIN POWER ONTO YOUR PC----------"

export WORKING_DIR=$(pwd)

echo "----------Setting up Object Detector----------"
sudo python3 -m venv /usr/local/lib/obj_det_env
sudo /usr/local/lib/obj_det_env/bin/pip3 install wheel ultralytics
sudo /usr/local/lib/obj_det_env/bin/pip3 install -r ../src/perception/object_detection/config/requirements.txt
sudo /usr/local/lib/obj_det_env/bin/pip3 install rospkg

echo "----------Setting up Grasp Detector----------"
sudo python3 -m venv /usr/local/lib/graspnet_env
sudo /usr/local/lib/graspnet_env/bin/pip3 install wheel
sudo /usr/local/lib/graspnet_env/bin/pip3 install -r ../src/perception/grasp_detector/config/requirements.txt
sudo /usr/local/lib/graspnet_env/bin/pip3 install rospkg

echo "----------Installing Dependencies for GraspNet----------"
cd ../src/perception/grasp_detector/scripts/pointnet2
sudo /usr/local/lib/graspnet_env/bin/python3 setup.py install
cd ../knn
sudo /usr/local/lib/graspnet_env/bin/python3 setup.py install
cd ../graspnetAPI
sudo /usr/local/lib/graspnet_env/bin/pip3 install .
rosdep install --from-paths ../src --ignore-src -r -y

cd ~/ws/
source /opt/ros/noetic/setup.bash

catkin build
source ./devel/setup.bash