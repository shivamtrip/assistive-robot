import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.clip_mapping_utils import load_map, get_new_pallete, get_new_mask_pallete
from utils.clip_utils import get_text_feats
from utils.mp3dcat import mp3dcat
import clip
import cv2
from collections import OrderedDict

def get_colored_seg_mask(lang:list,clip_model,
                 clip_feat_dim, grid_save_path, 
                 obstacles, xmin, xmax, ymin, ymax,cropped=True):

    grid = load_map(grid_save_path)
    grid = grid[xmin:xmax+1, ymin:ymax+1]
    no_map_mask = obstacles[xmin:xmax+1, ymin:ymax+1] > 0
    obstacles_rgb = np.repeat(obstacles[xmin:xmax+1, ymin:ymax+1,
                                         None],3, axis=2)
    text_feats = get_text_feats(lang, clip_model, clip_feat_dim)

    map_feats = grid.reshape((-1, grid.shape[-1]))
    scores_list = map_feats @ text_feats.T

    predicts = np.argmax(scores_list, axis=1)
    predicts = predicts.reshape((xmax - xmin + 1, ymax - ymin + 1))
    floor_mask = predicts == 2

    new_pallete = get_new_pallete(len(lang))
    mask, patches = get_new_mask_pallete(predicts, new_pallete, 
                                         out_label_flag=True, labels=lang)
    seg = mask.convert("RGBA")
    seg = np.array(seg)
    seg[no_map_mask] = [225, 225, 225, 255]
    seg[floor_mask] = [225, 225, 225, 255]

    return seg, patches

def get_seg_mask(lang:list,clip_model,
                 clip_feat_dim, grid_save_path, 
                 obstacles, xmin, xmax, ymin, ymax):

    grid = load_map(grid_save_path)
    grid = grid[xmin:xmax+1, ymin:ymax+1]

    no_map_mask = obstacles[xmin:xmax+1, ymin:ymax+1] > 0
    obstacles_rgb = np.repeat(obstacles[xmin:xmax+1, ymin:ymax+1,
                                         None],3, axis=2)
    print(no_map_mask.shape)
    text_feats = get_text_feats(lang, clip_model, clip_feat_dim)

    map_feats = grid.reshape((-1, grid.shape[-1]))
    scores_list = map_feats @ text_feats.T

    predicts = np.argmax(scores_list, axis=1)
    predicts = predicts.reshape((xmax - xmin + 1, ymax - ymin + 1))
    print("predicts shape: ", predicts.shape)

    return predicts

def postprocess_masks(masks:np.ndarray,labels_list:list):
    """ Post process the masks to remove noise and yield bounding boxes on clusters"""

    # Initialize parameters for morphological operations
    size_o = 3
    size_c = 5
    k_opening = np.ones((size_o,size_o),np.uint8)
    k_closing = np.ones((size_c,size_c),np.uint8)
    iterations = 1
    min_area = 100
    results = OrderedDict()

    # Perform morphological operations on each mask
    for mask,label in zip(masks,labels_list):

        results[label] = OrderedDict()
        results[label]['bboxes'] = []
        results[label]['centroids'] = []

        img = mask.copy().astype(np.uint8)*255
        assert img.dtype == np.uint8

        # Gaussian blurr
        # img = cv2.GaussianBlur(img,(size,size),0)

        # Opening
        img = cv2.erode(img,k_opening,iterations = iterations)
        img = cv2.dilate(img,k_opening,iterations = iterations)

        # Closing
        img = cv2.dilate(img,k_closing,iterations = iterations)
        img = cv2.erode(img,k_closing,iterations = iterations)

        # get connected components with stats
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=4)

        # find bounding boxes
        bboxes = []
        centroids = []
        for i in range(1,n_labels):
            x,y,w,h,area = stats[i]
            if(area<min_area):
                continue
            bboxes.append([int(x),int(y),int(x+w),int(y+h)])
            centroids.append([int(x+w//2),int(y+h//2)])
        
        results[label]['bboxes'] = bboxes
        results[label]['centroids'] = centroids
        results[label]['mask'] = img

    return results



