<h1 align="center">
  <br>
  <img src="images/Auxilio_Logo.jpg" alt="Auxilio" width="197">
  <br>
  <p>Alfred</p>
</h1>

# Repository organisation

The code is organised in several top level packages/directories. The top level should adhere to the following subdivision of functionality (a more detailed description can be found in the folders themselves):

**common/** - top-level launchfiles, msgs, and other files used by many packages

**perception/** - folder for perception packages

**navigation/** - folder for navigation packages

**manipulation/** - folder for manipulation packages

**control/** - folder for control packages

**interface/** - folder for interface packages

**simulation/** - folder for simulation packages

# Placement of ROS packages
ROS Packages should be added in one of the top level work-package folders. The top level work-package folders themselves should not be used to store ros package information. 

The directory tree should look like:

```
~/ws
  |__ src
      |__ common
      |   |__ alfred_msgs
      |   |__ ...
      |
      |__ perception
      |   |__ ...
      |
      |__ control
      |   |__ ...
      |
      |__ manipulation
      |   |__ ...
      |
      |__ navigation
      |   |__ ...
      |
      |__ interface
      |   |__ speech_recognition
      |   |__ ...
      |
      |__ simulation
          |__ ...
```

# Setting up the workspace

```
cd ~
git clone --recursive git@github.com:Auxilio-Robotics/alfred.git ws
```

# Install dependencies

```
cd ws
source /opt/ros/noetic/setup.bash
sudo apt install python3-rosdep
rosdep install --from-paths src --ignore-src -r -y
```

# Building the workspace

```
cd ~/ws
source /opt/ros/noetic/setup.bash # if not done already
catkin init # to check if config is valid
catkin build
```

# Testing the workspace
```
cd ~/ws
catkin build
catkin test
```

# Run Nursing Home Simulation

```
cd ~/ws
source devel/setup.bash
roslaunch alfred_gazebo simulation.launch
```