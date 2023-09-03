#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np
import open3d as o3d



def ransac_error(pcd: o3d.geometry.PointCloud,
                 distances: np.ndarray,
                 threshold: float) -> Tuple[float, np.ndarray]:
    """ Calculate the RANSAC error which is the number of outliers.

    The RANSAC error is defined as the number of outliers given a specific model.
    A outlier is defined as a point which distance to the plane is larger (or equal ">=") than a threshold.

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param distances: The distances of each point to the proposed plane
    :type distances: np.ndarray with shape (num_of_points,)

    :param threshold: The distance of a point to the plane below which point is considered an inlier (in meters)
    :type threshold: float

    :return: (error, inliers)
        error: The calculated error
        inliers: Boolean mask of inliers (shape: (num_of_points,))
    :rtype: (float, np.ndarray)
    """
    ######################################################

    inliers = distances < threshold
    error = np.sum(~inliers)

    ######################################################
    return error, inliers