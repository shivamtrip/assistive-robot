from landmark_index import *
from lseg_map import *
from vis_tools import *

import argparse
from collections import OrderedDict 
import os
import rospy
import cv2
import json
from utils.clip_mapping_utils import *
from scipy.spatial.transform import Rotation as R

class VLMaps():

    def __init__(self, params: dict):

        self.root_dir = params["root_dir"]
        self.data_dir = params["data_dir"]
        self.cs = params["cs"]
        self.gs = params["gs"]
        self.camera_height = params["camera_height"]
        self.max_depth_value = params["depth_filter"]
        self.depth_sample_rate = params["depth_sample_rate"]
        self.use_self_built_map = params["use_self_built_map"]
        self.map_save_dir = params["map_save_dir"]
        self.save_video = params["save_video"]
        self.show_vis = params["show_vis"]
        self.map_save_dir = os.path.join(self.data_dir, self.map_save_dir)
        self.data_config_dir = os.path.join(self.root_dir, "config/data_configs")
        self.device = params["device"] if torch.cuda.is_available() else "cpu"

        # check if checkpoints exists
        checkpoints_dir = os.path.join(self.root_dir, "scripts/lseg", "checkpoints")
        if not os.path.exists(checkpoints_dir):
            raise Exception("Checkpoints not found in {}. Please download them and place them in the checkpoints directory".format(checkpoints_dir))

        map_save_dir = os.path.join(self.data_dir, "map_correct")
        if self.use_self_built_map:
            map_save_dir = os.path.join(self.data_dir, "map")
        os.makedirs(map_save_dir, exist_ok=True)

        # Set the save paths
        self.color_top_down_save_path = os.path.join(map_save_dir, f"color_top_down_1.npy")
        self.grid_save_path = os.path.join(map_save_dir, f"grid_lseg_1.npy")
        self.obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")

        if(params["inference"]):
            print("Loading CLIP")
            device = self.device if torch.cuda.is_available() else "cpu"
            clip_version = "ViT-B/32"
            self.clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                            'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
            self.clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
            self.clip_model.to(device).eval()

        if(self.save_video):
            print("Saving video")
            create_video(data_dir=self.data_dir, output_dir=self.root_dir, fps= 30)

    def generate_vlmap(self,):
        """Generates the vlmap and saves it in the data_dir"""

        create_lseg_map_batch(
            img_save_dir=self.data_dir,
            use_self_built_map=self.use_self_built_map,
            camera_height=self.camera_height,
            cs=self.cs,
            gs=self.gs,
            depth_sample_rate=self.depth_sample_rate,
            max_depth_value=self.max_depth_value,
        )

    def get_initial_rtabmap_pose(self,):
        """Returns the initial pose of the base in /rtabmap/odom frame """

        pose_dir = os.path.join(self.data_config_dir,"pose")
        assert os.path.exists(pose_dir), "Pose directory not found at {}".format(pose_dir)
        pose_list = sorted(os.listdir(pose_dir), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        pose_list = [os.path.join(pose_dir, x) for x in pose_list]

        pos, rot = load_pose(pose_list[0])
        
        base_pose = np.eye(4)
        base_pose[:3, :3] = rot
        base_pose[:3, 3] = pos.reshape(-1)

        return base_pose
    
    def get_tf_base2cam(self,):
        """Returns the transformation matrix from base to camera frame"""

        add_params_dir = os.path.join(self.data_config_dir, "add_params")
        assert os.path.exists(add_params_dir), "Additional params directory not found at {}".format(add_params_dir)
        tf_base2cam_path = os.path.join(add_params_dir, "rosbag_data_0tf_base2cam.txt")
        tf_base2cam = load_tf(tf_base2cam_path)
        return tf_base2cam


    def get_movebase_goals(self,results:dict,label:str):
        """Returns the movebase goals from the vlmap"""

        # ensure that file exists
        initial_pose_path = os.path.join(self.data_config_dir, 'amcl_pose.json')
        assert os.path.exists(initial_pose_path), "Initial pose json file not found at {}".format(initial_pose_path)
        
        f = open(initial_pose_path)
        initial_pose_dict = json.load(f)
        pos = initial_pose_dict['position']
        pos = np.array([pos['x'],pos['y'],pos['z']])
        rot = initial_pose_dict['orientation']
        rot = np.array([rot['x'],rot['y'],rot['z'],rot['w']])

        tf_world2rtabmap = np.eye(4)
        tf_world2rtabmap[:3,:3] = R.from_quat(rot).as_matrix()
        tf_world2rtabmap[:3,3] = np.array(pos)

        tf_rtabmap2base = self.get_initial_rtabmap_pose()
        tf_camera_optical2pcd = get_tf_camera_optical2pcd()
        tf_base2cam = self.get_tf_base2cam()

        assert label in results.keys(), "Label {} not found in the results {}".format(label, results.keys())
        mask = results[label]['mask'].astype(np.uint8)

        bboxes = np.array(results[label]['bboxes'])
        bboxes = np.reshape(bboxes,(-1,4))

        print("No. of bboxes: ", bboxes.shape[0])
        assert bboxes.shape[1] == 4, "bboxes should be of shape (N,4) but found {}".format(bboxes.shape)

        # find areas of all bboxes
        areas = (bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1])
        max_id = np.argmax(areas)
        # find the largest bbox
        max_bbox = bboxes[max_id]

        # find centroid of the bbox
        x,y  = (max_bbox[0] + max_bbox[2])/2, (max_bbox[1] + max_bbox[3])/2

        print("x,y: ", x,y)

        # Vlmap to rtabmap world coordinate
        YY =0 # 2d navigation
        XX,ZZ = grid_id2pos(self.gs,self.cs,x,y)
        print("XX,YY,ZZ: ", XX,YY,ZZ)
        pos_des_cam_frame = np.array([XX,YY,ZZ,1]).reshape(-1,1)

        # find euler angles from tf_base2cam
        euler_angles = R.from_matrix(tf_base2cam[:3,:3]).as_euler('xyz', degrees=True)
        
        # Print all tfs
        print("tf_world2rtabmap: ", tf_world2rtabmap)
        print("tf_rtabmap2base: ", tf_rtabmap2base)
        print("tf_base2cam: ", tf_base2cam)
        print("tf_camera_optical2pcd: ", tf_camera_optical2pcd)
        print("pos_des_cam_frame: ", pos_des_cam_frame)
        # Position in world frame
        pos_des_world_frame = tf_world2rtabmap @ tf_rtabmap2base @ \
                                tf_base2cam @ tf_camera_optical2pcd @ \
                                pos_des_cam_frame
        
        assert pos_des_world_frame.shape == (4,1), "pos_des_world_frame should be of shape (4,1) but found {}".format(pos_des_world_frame.shape)

        X, Y ,Z = pos_des_world_frame[:3,0]

        print("X,Y,Z: ", X,Y,Z)

        # Make Z =0 and yaw =0 for 2d navigation
        Z = 0; theta =0 
        print("X,Y,Z,theta: ", X,Y,Z,theta)

        return [X,Y,Z,theta]

    def inference_labels(self,labels:list):

        """ Performs natural languuage indexing on VLMap for the given labels"""

        mask_list = []
        crop_res = False # Set to true for visualizing zoomed in results
        obstacles = load_map(self.obstacles_save_path)
        xmin = 0; xmax = obstacles.shape[0]-1; ymin = 0; ymax = obstacles.shape[1]-1
        x_indices, y_indices = np.where(obstacles == 0)

        if not x_indices.shape[0] or not y_indices.shape[0]:
            rospy.logwarn("No obstacles found in the map. Please run the generate_vlmap function first.")
            x_indices,y_indices = np.where(obstacles)

        if(crop_res):
            xmin = np.min(x_indices)
            xmax = np.max(x_indices)
            ymin = np.min(y_indices)
            ymax = np.max(y_indices)
  
        # Free space mask
        no_map_mask = obstacles[xmin:xmax+1, ymin:ymax+1] > 0
        predictions = get_seg_mask(labels,self.clip_model,self.clip_feat_dim,
                                    self.grid_save_path, obstacles,
                                    xmin, xmax, ymin, ymax,device=self.device)

        for i in range(len(labels)):
            label_mask = np.array(predictions == i)
            label_mask[no_map_mask] = 0
            mask_list.append(label_mask)

        outputs = postprocess_masks(mask_list,labels)
        # movebase_goal = self.get_movebase_goals(outputs,label=' '+'sofa')
        # print("movebase_goal: ", movebase_goal)

        if(self.show_vis):
            # Visualize label masks
            # show_label_masks(mask_list,labels)

            # Visualize all bounding boxes over masks
            # show_bounding_boxes(outputs,labels)
            
            seg, patches = get_colored_seg_mask(labels,self.clip_model,self.clip_feat_dim,
                                    self.grid_save_path, obstacles,
                                    xmin, xmax, ymin, ymax) 
            
            print("patches: ", patches)
            print("seg shape: ", seg.shape)
            print("Length mask list", len(mask_list))
            print("mask list shape: ", mask_list[0].shape)

            # More visualizations
            color_top_down = load_map(self.color_top_down_save_path)
            print("color_top_down shape: ", color_top_down.shape)
            show_color_top_down(color_top_down,obstacles=obstacles)
            show_seg_mask(seg, patches)
            # show_obstacle_map(obstacles)
            show_heatmap(cell_size=self.cs,color_top_down=color_top_down,obstacles=obstacles, \
                          mask_list=mask_list, labels=labels,outputs=outputs)

if __name__ == "__main__":

    # reads arugments
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--cs", type=float, default=0.05)
    parser.add_argument("--gs", type=int, default=1000)
    parser.add_argument("--camera_height", type=float, default=1.3)
    parser.add_argument("--depth_filter", type=int, default=5, 
                        help="Filter out depth values greater than this value")
    parser.add_argument("--depth_sample_rate", type=int, default=100,
                        help="Determine the step size while sampling points. Lower-> More dense sampling")
    parser.add_argument("--use_self_built_map", type=bool, default=True)
    parser.add_argument("--map_save_dir", type=str, default="map_correct")
    parser.add_argument("--save_video", type=bool, default=False)
    parser.add_argument("--inference", type=bool, default=False)
    parser.add_argument("--show_vis", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda:3")

    args = parser.parse_args()
    params = vars(args)
    print(params)

    if(params["root_dir"] is None):
        # Find parent directory of current file directory
        params["root_dir"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if(params["data_dir"] is None):
        data_dir = os.path.join(params["root_dir"], "data", "rosbag_data/")
        params["data_dir"] = data_dir
    
    else:
        # Ensure path exists
        assert os.path.exists(params["data_dir"]), "Data directory not found at {}".format(params["data_dir"])

    vlmaps = VLMaps(params=params)

    if not params["inference"]:
        params['camera_height'] = 1.2 # stretch cam height roughly
        vlmaps.generate_vlmap()

    else:
        # LABELS = "big flat counter, sofa, floor, chair, wash basin, other" # @param {type: "string"}
        # LABELS = "table, chair, floor, sofa, bed, other"
        # LABELS = "table,kitchen,floor,sofa,sink,refrigerator,potted plant,other"
        LABELS = "table,kitchen,floor,sofa,sink,refrigerator,potted plant,laptop,other"
        # LABELS = "television, chair, floor, sofa, refrigerator, other"
        LABELS = LABELS.split(",")
        vlmaps.inference_labels(labels=LABELS)


    

