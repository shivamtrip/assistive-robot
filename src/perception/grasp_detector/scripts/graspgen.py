#!/usr/local/lib/graspnet_env/bin/python3

import open3d as o3d
import torch
import cv2
import numpy as np 
from models.graspnet import GraspNet, pred_decode
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup

class GraspGenerator:

    def __init__(self, 
                checkpoint_path, 
                num_point, 
                num_view,
                collision_thresh, 
                voxel_size, 
                max_depth, 
                num_angle,
                num_depth,
                cylinder_radius,
                input_feature_dim,
                hmin,
                hmax_list,
                device_name
    ):
        #initialize grasp arguments
        self.checkpoint_path = checkpoint_path
        self.num_point = num_point
        self.num_view = num_view
        self.collision_thresh=collision_thresh
        self.voxel_size = voxel_size
        self.max_depth = max_depth
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.cylinder_radius = cylinder_radius
        self.input_feature_dim = input_feature_dim
        self.hmin = hmin
        self.hmax_list = hmax_list
        self.device_name = device_name
        self.net = self.get_net()

    def get_net(self):

        # Init the model
        net = GraspNet(
            input_feature_dim=self.input_feature_dim, 
            num_view = self.num_view, 
            num_angle = self.num_angle, 
            num_depth = self.num_depth,
            cylinder_radius = self.cylinder_radius, 
            hmin = self.hmin, 
            hmax_list = self.hmax_list, 
            is_training= False
        )
        device = torch.device(self.device_name)
        net.to(device)

        checkpoint = torch.load(self.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        return net
    
    def get_and_process_data(self, color, depth, bbox, intrinsic):

        self.color = color.astype(np.float64)/255.0
        inflate=0
        bbox = bbox.astype(int)
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        self.workspace_mask = np.zeros((color.shape[0],color.shape[1]),dtype=np.uint8)
        self.workspace_mask[
            x2-inflate:x1+inflate,
            y1-inflate:y2 - inflate
        ] = 255

        factor_depth = 1000
        
        camera = CameraInfo(self.color.shape[1], self.color.shape[0], intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True) #shape=(480,640,3)
        mask = (self.workspace_mask & (depth > 0)) #shape =(720,1280)
      
        cloud_masked = cloud[mask>0,:]
        color_masked = color[mask>0,:]

       
        if len(cloud_masked) >= 20000:
            idxs = np.random.choice(len(cloud_masked), 20000, replace=False)
        else:
            print("Cloud_masked shape",)
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), 20000-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
            
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud

    def get_grasps(self,net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg
    
    def get_centroid(self,end_points:dict):

        pts = end_points["point_clouds"].to("cpu").detach().numpy()
        pts=pts.squeeze(axis=0)
        print("Mean=",np.mean(pts,axis=0))
        print("Median=",np.median(pts,axis=0))
        return np.median(pts,axis=0)


    def collision_detection(self,gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=0.01)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
        gg = gg[~collision_mask]
        return gg
    
    def sort_grasps(self,gg):
        gg.nms()
        gg.sort_by_score()
        print("Sorting grasps - -- - - - - - - -")
        

    def vis_grasps(self,gg, cloud):
        # gg = self.sort_grasps(gg)
        gg.nms()
        gg.sort_by_score()
        gg = gg[:10]
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])

    def vis_pcd(self, cloud):
        o3d.visualization.draw_geometries([cloud])

    def centroid_grasp(self, color, depth, bbox, intrinsic):

        end_points, cloud = self.get_and_process_data(color, depth, bbox, intrinsic)
        # self.vis_pcd()
        centroid = self.get_centroid(end_points)
        return centroid