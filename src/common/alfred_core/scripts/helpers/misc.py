#!/usr/bin/env python

from __future__ import print_function

import time
import os
import glob
import math

import rospy
import tf2_ros
import ros_numpy


def create_time_string():
    t = time.localtime()
    time_string = str(t.tm_year) + str(t.tm_mon).zfill(2) + str(t.tm_mday).zfill(2) + str(t.tm_hour).zfill(2) + str(t.tm_min).zfill(2) + str(t.tm_sec).zfill(2)
    return time_string


def get_recent_filenames(filename_without_time_suffix, filename_extension, remove_extension=False): 
    filenames = glob.glob(filename_without_time_suffix + '_*[0-9]' + '.' + filename_extension)
    filenames.sort()
    if remove_extension:
        return [os.path.splitext(f)[0] for f in filenames]
    return filenames


def get_most_recent_filename(filename_without_time_suffix, filename_extension, remove_extension=False):
    filenames = get_recent_filenames(filename_without_time_suffix, filename_extension, remove_extension=remove_extension) 
    most_recent_filename = filenames[-1]
    return most_recent_filename


def angle_diff_deg(target_deg, current_deg):
    # I've written this type of function many times before, and it's
    # always been annoying and tricky. This time, I looked on the web:
    # https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    diff_deg = target_deg - current_deg
    diff_deg = ((diff_deg + 180.0) % 360.0) - 180.0
    return diff_deg


def angle_diff_rad(target_rad, current_rad):
    # I've written this type of function many times before, and it's
    # always been annoying and tricky. This time, I looked on the web:
    # https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    diff_rad = target_rad - current_rad
    diff_rad = ((diff_rad + math.pi) % (2.0 * math.pi)) - math.pi
    return diff_rad


def get_p1_to_p2_matrix(p1_frame_id, p2_frame_id, tf2_buffer, lookup_time=None, timeout_s=None):
    # If the necessary TF2 transform is successfully looked up, this
    # returns a 4x4 affine transformation matrix that transforms
    # points in the p1_frame_id frame to points in the p2_frame_id.
    try:
        if lookup_time is None:
            lookup_time = rospy.Time(0) # return most recent transform
        if timeout_s is None:
            timeout_ros = rospy.Duration(0.1)
        else:
            timeout_ros = rospy.Duration(timeout_s)
        stamped_transform =  tf2_buffer.lookup_transform(p2_frame_id, p1_frame_id, lookup_time, timeout_ros)

        # http://docs.ros.org/melodic/api/geometry_msgs/html/msg/TransformStamped.html

        p1_to_p2_mat = ros_numpy.numpify(stamped_transform.transform)
        return p1_to_p2_mat, stamped_transform.header.stamp
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        print('WARNING: get_p1_to_p2_matrix failed to lookup transform from p1_frame_id =', p1_frame_id, ' to p2_frame_id =', p2_frame_id)
        print('         exception =', e)
        return None, None

def bound_ros_command(bounds, ros_pos, fail_out_of_range_goal, clip_ros_tolerance=1e-3):
    """Clip the command with clip_ros_tolerance, instead of
    invalidating it, if it is close enough to the valid ranges.
    """
    if ros_pos < bounds[0]:
        if fail_out_of_range_goal:
            return bounds[0] if (bounds[0] - ros_pos) < clip_ros_tolerance else None
        else:
            return bounds[0]

    if ros_pos > bounds[1]:
        if fail_out_of_range_goal:
            return bounds[1] if (ros_pos - bounds[1]) < clip_ros_tolerance else None
        else:
            return bounds[1]

    return ros_pos
