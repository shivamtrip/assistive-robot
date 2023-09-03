""" Tools for data processing.
    Author: Abhinav Gupta
"""

import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import cv2

import torch
from graspnetAPI import GraspGroup


class GraspGroup_(GraspGroup):

    def rank_grasps(self,):
        #rank grasps after calculating all losses and GM

    def calc_centroid_loss(self,):
        #calculate distance from the centroid
    
    def calc_orientation_loss(self,):
        #calculates orientation error

