#!/usr/bin/env python3

""" Test if an image is blurred """

import numpy as np
import cv2
import os

def is_blurred(img_dir_path, threshold=300):
    """
    Check if an image is blurred using the Laplacian method.
    :param image: image to check
    :param threshold: threshold to use
    :return: True if the image is blurred, False otherwise
    """

    # Read dir like a list
    img_list = sorted(os.listdir(img_dir_path))
    indices_list = []

    laplacian_list = []

    for idx, img_name in enumerate(img_list):
        img_path = os.path.join(img_dir_path, img_name)
        image = cv2.imread(img_path)
        val = cv2.Laplacian(image, cv2.CV_64F).var()
        laplacian_list.append(val)

        if val > threshold:
            indices_list.append(idx)

    # plot laplacian_list
    import matplotlib.pyplot as plt
    plt.plot(laplacian_list)
    plt.show()

    return indices_list



if __name__ == '__main__':

    img_dir_path = '/home/abhinav/alfred-autonomy/src/navigation/vlmaps_ros/data/rosbag_data/rgb/'

    idx_list  = is_blurred(img_dir_path)

    print(idx_list)