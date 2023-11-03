#!/usr/bin/env python3

#
# Copyright (C) 2023 Auxilio Robotics
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
 
from geometry_msgs.msg import Quaternion
from tf import transformations

import math

def get_quaternion(theta):
    """
    A function to build Quaternians from Euler angles. Since the Stretch only
    rotates around z, we can zero out the other angles.
    :param theta: The angle (degrees) the robot makes with the x-axis.
    """
    return Quaternion(*transformations.quaternion_from_euler(0.0, 0.0, math.radians(theta)))