#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:29:41 2017

@author: chunyilyu
"""

import numpy as np import pickle
import sys
import time
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
from transforms3d.quaternions import quat2mat,mat2quat %matplotlib inline
#%%
def opMul(q, p):
    if type(q) is list or type(p) is list:
        q, p = np.asarray(q, dtype='float'), np.asarray(p, dtype='float')
    assert(q.shape[-1]==4 and p.shape[-1]==4)
    return np.concatenate(( q[...,:1]*p[...,:1] - q[...,1:].dot(np.transpose
                           (q[...,:1]*p[..., 1:] + p[..., :1]*q[..., 1:]+n
                                   )axis = -1)
def opNorm(q):
    return np.linalg.norm(q, axis=-1).reshape(-1,1) + 1e-20
def opInv(q):
    if type(q) is list:
            q = np.asarray(q, dtype='float') 
    assert(q.shape[-1]==4)
    return np.divide(opConjugate(q), opNorm(q)**2)
def opLog(q):
    if type(q) is list:
        q = np.asarray(q, dtype='float') 
    assert(q.shape[-1]==4)
    n = opNorm(q)
    norm_qv = opNorm(q[..., 1:4])
    return np.concatenate((np.log(n),