#!/usr/bin/bash
export WORKING_DIR=$(pwd)
./art.sh

echo "----------Installing Dependencies for Robot----------"

sudo apt install python3.8-venv
sudo apt-get install ros-noetic-catkin python3-catkin-tools
sudo apt install python3-rosdep
rosdep install --from-paths ~/ws/src --ignore-src -r -y

sudo python3 -m venv /usr/local/lib/robot_env
<<<<<<< HEAD
sudo /usr/local/lib/robot_env/bin/pip3 install wheel firebase pyrebase
sudo /usr/local/lib/robot_env/bin/pip3 install -r ~/ws/src/interface/alfred_hri/config/requirements.txt
=======
sudo /usr/local/lib/robot_env/bin/pip3 install wheel firebase pyrebase pvporcupine pvrecorder google-cloud-texttospeech openai PyAudio
sudo /usr/local/lib/robot_env/bin/pip3 install -r ~/ws/src/interface/interface_meta/config/requirements.txt
>>>>>>> fc19e240d6c9410d1a83240f9fd9c411f4e5929c
sudo /usr/local/lib/robot_env/bin/pip3 install rospkg
