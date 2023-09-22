#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Helper functions to plot the results
Author: Matthias Hirschmanner
"""
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def create_geometry_at_points(points):
    geometries = o3d.geometry.TriangleMesh()
    for point in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05) #create a small sphere to represent point
        sphere.translate(point) #translate this sphere to point
        geometries += sphere
    geometries.paint_uniform_color([1.0, 0.0, 0.0])
    return geometries



def plot_plane(plane_points, place_point):

    plane_cloud = o3d.geometry.PointCloud()                 # creating point cloud of plane
    plane_cloud.points = o3d.utility.Vector3dVector(plane_points)

    place_cloud = create_geometry_at_points(place_point.reshape(1, 3))
    o3d.visualization.draw_geometries([plane_cloud, place_cloud])