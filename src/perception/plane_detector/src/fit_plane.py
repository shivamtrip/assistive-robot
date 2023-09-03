#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple, Callable

import copy

import numpy as np
import open3d as o3d


def fit_plane(pcd,
              confidence: float,
              inlier_threshold: float,
              min_sample_distance: float,
              error_func: Callable) -> Tuple[np.ndarray, np.ndarray, int]:
    """ Find dominant plane in pointcloud with sample consensus.

    Detect a plane in the input pointcloud using sample consensus. The number of iterations is chosen adaptively.
    The concrete error function is given as an parameter.

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. distance of a point from the plane to be considered an inlier (in meters)
    :type inlier_threshold: float

    :param min_sample_distance: Minimum distance of all sampled points to each other (in meters). For robustness.
    :type min_sample_distance: float

    :param error_func: Function to use when computing inlier error defined in error_funcs.py
    :type error_func: Callable

    :return: (best_plane, best_inliers, num_iterations)
        best_plane: Array with the coefficients of the plane equation ax+by+cz+d=0 (shape = (4,))
        best_inliers: Boolean array with the same length as pcd.points. Is True if the point at the index is an inlier
        num_iterations: The number of iterations that were needed for the sample consensus
    :rtype: tuple (np.ndarray[a,b,c,d], np.ndarray, int)
    """
    ######################################################

    points = pcd #np.asarray(pcd.points)
    N = len(points)
    m = 3
    eta_0 = 1-confidence
    k, eps, error_star = 0, m/N, np.inf
    I = 0
    best_inliers = np.full(shape=(N,),fill_value=0.)
    best_plane = np.full(shape=(4,), fill_value=-1.)
    while pow((1-pow(eps, m)),k) >= eta_0:
        p1, p2, p3 = points[np.random.randint(N)], points[np.random.randint(N)], points[np.random.randint(N)]
        if np.linalg.norm(p1-p2) < min_sample_distance or np.linalg.norm(p2-p3) < min_sample_distance or np.linalg.norm(p1-p3) < min_sample_distance:
            continue
        n = np.cross(p2-p1,p3-p1)               # get normal vector to the plane

        # n[0] = np.random.rand(1) * 2 - 1
        # n[1] = np.random.rand(1) * 2 - 1
        # n[2] = 1.0
        # print(n.shape)
        n = n/np.linalg.norm(n)                 # normalizing the normal vector (magnitude = 1)
        if n[2] < 0:                            # positive z direction
            n = -n
        d = -np.dot(n, p1)                       # parameter d of line eqn (d = -(ax + by + cz))
        
        distances = np.abs(np.dot(points, n)+d)         # distance of each point from the current plane

        error, inliers = error_func(pcd, distances, inlier_threshold)          # number of outliers and inliers
        if error < error_star:
            I = np.sum(inliers)
            eps = I/N
            best_inliers = inliers
            error_star = error
        if k == 250:                    # Exiting ransac loop if number of iterations in 250
            break
        k = k + 1

    A = points[best_inliers]
    y = np.full(shape=(len(A),),fill_value=1.)
    best_plane[0:3] = np.linalg.lstsq(A, y, rcond=-1)[0]
    if best_plane[2] < 0:  ### positive z direction
        best_plane = -best_plane
    return best_plane, best_inliers, k
