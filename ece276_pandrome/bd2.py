#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:33:41 2017

@author: chunyilyu
"""
import numpy as np

imud = imu_data_raw
imud['vals'] = np.asarray(imud['vals'], dtype='float64') #calculating bias
scaler_w = 3300./1023*np.pi/180/3.33
scaler_v = 11./1023
bias = np.zeros(6)
s = np.sum(imud['vals'][:,:100],axis=-1)/100.
bias = np.concatenate((s[:3]-np.asarray([0,0,1])/scaler_v, s[3:])) 
def convertData(X):
    assert(X.ndim==1)
# -Ax, -Ay, -Az, Wz, Wx, Wy
    a = (X-bias)*np.concatenate(([scaler_v]*3, [scaler_w]*3))
    return (X-bias)*np.concatenate(([scaler_v]*3, [scaler_w]*3))
print bias
val = np.transpose(imud['vals'])
w = convertData(np.asarray(val[1],dtype='float64'))