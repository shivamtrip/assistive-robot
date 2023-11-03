
import numpy as np
from typing import List

class Node:
    
    def __init__(self, cls_type, max_samples=500):
        self.samples = []
        self.cls_type = cls_type
        self.max_samples = max_samples
        
    def add_sample(self, x, y, z, embedding, timestamp, confidence):
        self.samples.append([x, y, z, embedding, timestamp, confidence])
        
        if len(self.samples) > self.max_samples:
            self.max_samples.pop(0)

        
    def get_mean(self):
        
        mean = np.mean(np.array(self.samples)[:, :3], axis=0)
        return mean
    
    def get_cov(self):
        return np.cov(self.samples[:, :3], rowvar=False)
        
    
    def get_weighted_mean(self):
        timestamps = np.array([sample[4] for sample in self.samples])
        timestamps -= np.min(timestamps)
        timestamps /= np.max(timestamps)
        
        confidences = np.array([sample[5] for sample in self.samples])
        
        weights = timestamps * confidences
        weights /= np.sum(weights)
            
        weighted_mean = np.sum(np.array(self.samples)[:, :3] * weights.reshape(-1, 1), axis=0)
        return weighted_mean
    
    def clear_samples(self):
        self.samples = []
        
    


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
    
    
    def add_observation(self, x, y, z, class_type, confidence, timestamp):
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