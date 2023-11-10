
import numpy as np
from typing import List
import open3d as o3d
mahalanobis = lambda pos, cov, mean : np.sqrt(np.dot(np.dot(pos - mean, np.linalg.inv(cov)), pos - mean))



class Node:
    
    def __init__(self, cls_type, intrinsics, max_samples=500):
        self.samples = []
        self.cls_type = cls_type
        self.max_samples = max_samples
        self.intrinsics = intrinsics
        
        
    def add_sample(self, colorimg, depthimg, extrinsics_image_to_base, detections, extrinsics_image_to_map):
        
        if len(detections) == 0:
            return False
        # instance disambiguation
        if len(detections) > 1:
            mean, cov = self.estimate_location('base')
            detection = None
            for d in detections:
                meandepth = np.mean(depthimg[d[1]:d[3], d[0]:d[2]])
                if mahalanobis(meandepth, cov, mean) < 0.5:
                    detection = d
                    break
                
            if detection == None:
                return False
        else:
            detection = detections[0]
            
        
        colormask = np.zeros_like(colorimg)
        colormask[detection[1]:detection[3], detection[0]:detection[2]] = 1
        depthmask = np.zeros_like(depthimg)
        depthmask[detection[1]:detection[3], detection[0]:detection[2]] = 1
        
        depthimg = depthimg * depthmask
        colorimg = colorimg * colormask
        
        self.samples.append([colorimg, depthimg, detection, extrinsics_image_to_base, extrinsics_image_to_map])
        return True
        
    def estimate_location(self, to_frame = 'map'):
        locs = []
        for sample in self.samples:
            colorimg, depthimg, detection, extrinsics_image_to_base, extrinsics_image_to_map = sample
            if to_frame == 'map':
                extrinsics = extrinsics_image_to_map
            else:
                extrinsics = extrinsics_image_to_base
            
            x1, y1, x2, y2 = detection
            
            midpoint = np.array([(x1 + x2)/2, (y1 + y2)/2, 1])
            midpoint[2] = np.median(depthimg[int(y1):int(y2), int(x1):int(x2)])
            midpoint = np.dot(np.linalg.inv(self.intrinsics), midpoint)
            midpoint = np.dot(extrinsics[:3, :3], midpoint)
            midpoint += extrinsics[:3, 3]
            locs.append(midpoint)
            
        mean = np.mean(locs, axis=0)
        cov = np.cov(locs, rowvar=False)
        return mean, None, cov
    
    
    def reconstruct_object(self):
        tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        for sample in self.samples:
            colorimg, depthimg, detection, extrinsics_image_to_base, extrinsics_image_to_map = sample
            depth = o3d.geometry.Image(depthimg)
            color = o3d.geometry.Image(colorimg.astype(np.uint8))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000, convert_rgb_to_intensity=False
            )
            cam = o3d.camera.PinholeCameraIntrinsic()
            cam.intrinsic_matrix = self.intrinsics
            cam.width = colorimg.shape[1]
            cam.height = colorimg.shape[0]
            tsdf_volume.integrate(
                rgbd,
                cam,
                extrinsics_image_to_base)
            
        pcd = tsdf_volume.extract_point_cloud()
        return pcd
            
            
            


class Graph:
    
    def __init__(self):
        self.nodes : List[Node] = []
        self.classes = []
    
    def get_instance(self, class_id):
        indices = np.where(np.array(self.classes) == class_id)[0]
        if len(indices) == 0:
            return None
        else:
            return self.nodes[indices[0]]
    
    
    def add_observation(self, color, depth, class_type, confidence, timestamp):
        embedding = None # CLIP embedding
        if class_type in self.classes: # if it has been observed before
            indices = np.where(np.array(self.classes) == class_type)[0]
            mindist, minidx = np.inf, None
            for index in indices:
                node : Node = self.nodes[index]
                pos = np.array([x, y, z])
                
                # get mahalanobis distance of all nodes in the class

                mean = node.get_mean()
                cov = node.get_cov()                    
                mahalanobis = np.sqrt(np.dot(np.dot(pos - mean, np.linalg.inv(cov)), pos - mean))
                if mahalanobis < mindist:
                    mindist = mahalanobis
                    minidx = index
            if mindist < 0.5:
                self.nodes[minidx].add_sample(x, y, z, embedding)
            else:
                self.classes.append(class_type)
                node = Node(class_type)
                node.add_sample(x, y, z, embedding, timestamp, confidence)
                self.nodes.append(node)

        
    def clear(self):
        self.nodes = []
        self.classes = []