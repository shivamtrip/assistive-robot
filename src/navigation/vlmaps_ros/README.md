# Description

VLMaps fuse pre- trained visual-language features into a geometrical reconstruction of the scene. It’ll help us to execute navigation tasks directly by inferencing object ’labels’ over the map. Hence, it makes open-vocabulary object goal navigation possible. For more information on VLMaps, please refer to [VLMaps paper](https://arxiv.org/abs/2103.16857).

# Working with VLMaps (standalone)

Broadly, there are just three steps to use VLMaps for navigation:

1. **Save Rosbag data**: This step ensures that you have the data in the right format to use VLMaps. You can skip this step if you already have the data in the right format. However, if you haven't made data directories yet, you can run the following commands to store the data in the right format:

    ```bash
    # Run rosbag in terminal 1
    rosbag play -l <path_to_rosbag> 

    # Run the save_data.py script in terminal 2 inside the 
    # `vlmaps` conda environment 
    cd ~/FVD/alfred-autonomy/src/navigation/vlmaps_ros/scripts/
    python save_data.py --fps <desired_fps> --img_save_dir <path_to_directory> --color_topic <color_topic_name> --depth_topic <depth_topic_name> --pose_topic <pose_topic_name> --intrinsic_topic <intrinsic_topic_name> --is_rtabmap <bool value depending on mapping method>

    ## Example command
    python save_data.py --fps 3 --img_save_dir ~/FVD/alfred-autonomy/src/navigation/vlmaps_ros/data/rosbag_data_nov11
    ```

2. **Create VLMaps**: This step creates the VLMaps from the data saved in the previous step. You can run the following command to create the VLMaps:

    ```bash
    # Run the main.py script in terminal 3 inside the 
    # `vlmaps` conda environment 

    cd ~/FVD/alfred-autonomy/src/navigation/vlmaps_ros/scripts/

    python main.py --root_dir <path-to-root-dir> --data_dir <path-to imgsavedir-in-last-step> --cs 0.01 --gs 1000 --depth_filter <max-depth-used> --depth_sample_rate 30 

    ## It is important to note that some arguments above are optional
    ## Example command
    python main.py --data_dir '/home/abhinav/FVD/alfred-autonomy/src/navigation/vlmaps_ros/data/rosbag_data_nov11' --cs 0.01 --gs 1000 --depth_filter 5 --depth_sample_rate 30
    ```

3. **Run VLMaps**: In this step, we can run inferences on VLMaps and get segmentation masks for a default `LABELS` input. You can run the following command to run VLMaps:

    ```bash
    # Run the main.py script in terminal 4 inside the 
    # `vlmaps` conda environment 

    cd ~/FVD/alfred-autonomy/src/navigation/vlmaps_ros/scripts/

    python main.py --data_dir <path-to imgsavedir-in-last-step> --cs 0.01 --gs 1000 --depth_filter <max-depth-used> --depth_sample_rate 30 --inference True

    ## It is important to note that some arguments above are optional
    ## Example command
    python main.py --data_dir '/home/abhinav/FVD/alfred-autonomy/src/navigation/vlmaps_ros/data/rosbag_data_nov11' --cs 0.01 --gs 1000 --depth_filter 5 --depth_sample_rate 30 --inference True --show_vis True
    ```

# Working with VLMaps (alfred-autonomy ROS)

Previously we looked at how to get VLMaps inferences on standalone data. However, we actually need to run it on the robot live. The following steps will help you do that:

1. **Sanity Checks**: a) Ensure that you are using Stretch-re-1056 since so far VLMap has been tested on this robot only,b) Ensure that the VLMap has been successfully created and saved in the `vlmaps_ros/data` directory, c) Ensure that the `vlmaps_ros` package is built and sourced on both the `brain` and `robot`, d) Ensure that the network connectivity between `brain` and `robot` is established and is good (>10fps), e) Ensure all file paths in config files are correct, f) Ensure that the robot is in the same position as the one in which the VLMap was created i.e home position.

2. **Run starter scripts**: This step runs the starter scripts on robot that are necessary for navigation. You can run the following commands on separate terminals to run the starter scripts:

    ```bash
    roslaunch alfred_core driver.launch
    roslaunch alfred_navigation navigation_no_driver.launch
    roslaunch alfred_core perception_robot_tuned.launch
    ```

3. **Run the VLMaps Server**: This step runs the VLMaps server on the `brain` machine. You can run the following command to run the VLMaps server:

    ```bash
    # Run server
    roslaunch vlmaps_ros vlmaps_server.launch
    ```
    Wait for the server to start. You should see confirmation messages on the terminal.

4. **Run the VLMaps Client**: This step runs the VLMaps client on the `robot` machine. You can run the following command to run the VLMaps client:

    ```bash
    # Run client
    roslaunch vlmaps_ros vlmaps_robot.launch
    ```

    The default script will command it with the 'go_to('sofa')' primitive. You can change the `LABELS` input as desired in the test_navigation_rtabmap file.


    

