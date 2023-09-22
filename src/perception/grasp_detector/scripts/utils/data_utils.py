""" Tools for data processing.
    Author: chenxi-wang
"""
import numpy as np
import scipy.io as scio
import cv2
from graspnetAPI import GraspGroup
from scipy.spatial.transform import Rotation as R
import math

class CameraInfo():
    """ Camera intrisics for point cloud creation. """
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


class GraspGroup_(GraspGroup):

    def __init__(self,centroid: np.ndarray,tf_mat_base2cam: np.ndarray):
        super(GraspGroup, self).__init__()
        self.pcd_centroid = centroid
        self.rank_grasps(tf_mat_base2cam)


    def rank_grasps(self,tf_mat_base2cam: np.ndarray):

        #rank grasps after calculating all losses and GM
        self.centroid_loss = self.calc_centroid_loss()
        self.orientation_loss = self.calc_orientation_loss(tf_mat_base2cam)

        assert self.centroid_loss.shape == self.orientation_loss.shape
        self.grasp_group_array[:,0] = np.sqrt(self.grasp_group_array[:,0]*(self.centroid_loss+self.orientation_loss)/2.0)

    def calc_centroid_loss(self,index: int):
        
        #calculate distance from the centroid
        dist = np.linalg.norm(self.grasp_group_array[index, 13:16] - self.pcd_centroid)
        loss = 1 - dist/dist.sum()
        return loss

    
    def calc_orientation_loss(self,tf_mat_base2cam):

        # Calculates orientation error

        # angles = r.as_euler('zyx', degrees=True)
        # yaw,pitch,roll = angles[0],angles[1],angles[2]

        #Talk to Prof. Kroemer about this!!

        rot_mat= self.rotation_matrices.astype(np.float32)
        t = self.translations.astype(np.float32)

        r=[]
        for i in range(0,360,4):
            r.append(R.from_euler('z', 90, degrees=True))

        loss=[]
        err_vec=[]

        for i in range(len(self.grasp_group_array)):
            tf_mat_cam2obj = np.hstack((rot_mat[i],t[i].reshape((3,1))))
            tf_mat_cam2obj = np.vstack((tf_mat_cam2obj,np.array([[0,0,0,1]])))
            tf_mat_base2obj = np.matmul(tf_mat_base2cam,tf_mat_cam2obj)
            r_grasp = R.from_matrix(tf_mat_base2obj[:3,:3])
            
            _, err = self.R2axisang(np.matmul(r_grasp.as_matrix().T,r[i].as_matrix()))

            err_vec.append(abs(np.mod(err,np.pi)))
        
        loss[i] = np.min(np.array(err_vec))

        return np.array(loss)

    def R2axisang(self,R):
        ang = math.acos((R[0,0] + R[1,1] + R[2,2] - 1)/2)
        Z = np.linalg.norm([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
        if Z==0:
            return[1,0,0], 0.
        x = (R[2,1] - R[1,2])/Z
        y = (R[0,2] - R[2,0])/Z
        z = (R[1,0] - R[0,1])/Z 	
        
        return[x, y, z], ang

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert(depth.shape[1] == camera.width and depth.shape[0] == camera.height)
    # xmap = np.arange(-camera.width//2, camera.width//2)

    #camera width =1280
    #camera height = 720
    xmap = np.arange(0, camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                points in original coordinates
            transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [np.ndarray, (N,3), np.float32]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed

def compute_point_dists(A, B):
    """ Compute pair-wise point distances in two matrices.

        Input:
            A: [np.ndarray, (N,3), np.float32]
                point cloud A
            B: [np.ndarray, (M,3), np.float32]
                point cloud B

        Output:
            dists: [np.ndarray, (N,M), np.float32]
                distance matrix
    """
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A-B, axis=-1)
    return dists

def remove_invisible_grasp_points(cloud, grasp_points, pose, th=0.01):
    """ Remove invisible part of object model according to scene point cloud.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                scene point cloud
            grasp_points: [np.ndarray, (M,3), np.float32]
                grasp point label in object coordinates
            pose: [np.ndarray, (4,4), np.float32]
                transformation matrix from object coordinates to world coordinates
            th: [float]
                if the minimum distance between a grasp point and the scene points is greater than outlier, the point will be removed

        Output:
            visible_mask: [np.ndarray, (M,), np.bool]
                mask to show the visible part of grasp points
    """
    grasp_points_trans = transform_point_cloud(grasp_points, pose)
    dists = compute_point_dists(grasp_points_trans, cloud)
    min_dists = dists.min(axis=1)
    visible_mask = (min_dists < th)
    return visible_mask

def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    """ Keep points in workspace as input.

        Input:
            cloud: [np.ndarray, (H,W,3), np.float32]
                scene point cloud
            seg: [np.ndarray, (H,W,), np.uint8]
                segmantation label of scene points
            trans: [np.ndarray, (4,4), np.float32]
                transformation matrix for scene points, default: None.
            organized: [bool]
                whether to keep the cloud in image shape (H,W,3)
            outlier: [float]
                if the distance between a point and workspace is greater than outlier, the point will be removed
                
        Output:
            workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
                mask to indicate whether scene points are in workspace
    """
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h*w, 3])
        seg = seg.reshape(h*w)
    if trans is not None:
        cloud = transform_point_cloud(cloud, trans)
    foreground = cloud[seg>0]
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = ((cloud[:,0] > xmin-outlier) & (cloud[:,0] < xmax+outlier))
    mask_y = ((cloud[:,1] > ymin-outlier) & (cloud[:,1] < ymax+outlier))
    mask_z = ((cloud[:,2] > zmin-outlier) & (cloud[:,2] < zmax+outlier))
    workspace_mask = (mask_x & mask_y & mask_z)
    if organized:
        workspace_mask = workspace_mask.reshape([h, w])

    return workspace_mask

