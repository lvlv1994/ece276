#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 01:59:25 2017

@author: chunyilyu
"""
import numpy as np
from quaternion import Quaternion
def q_average(quaternions, weights=None):
    if weights is None:
        weights = np.ones((len(quaternions), 1))/len(quaternions)
    else:
        weights = np.array(weights).reshape((len(quaternions),1))
    q_avg = quaternions[0]
    while True:
        error_quaternions = [quaternion*q_avg.inv() for quaternion in quaternions]
        rotation_vectors = [2*error_quaternion.log().v for error_quaternion in error_quaternions]
        new_rotation_vectors = (-np.pi + np.mod(np.linalg.norm(rotation_vectors, axis=1)+np.pi, 2*np.pi)).reshape((len(quaternions), 1)) * \
            np.nan_to_num(np.array(rotation_vectors)/np.linalg.norm(rotation_vectors, axis=1).reshape((len(quaternions), 1)))
        weighted_sum = np.sum(weights*new_rotation_vectors, axis=0)
        q_avg = Quaternion(0, weighted_sum/2).exp()*q_avg
        if np.linalg.norm(weighted_sum) < .001:
            return q_avg.unit()

