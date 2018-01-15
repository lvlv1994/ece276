#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 23:58:48 2017

@author: chunyilyu
"""
#%%
from quaternion import Quaternion 
import numpy as np
import time
import os
import pickle

import sys
import cv2
import pylab as pl
from transforms3d import quaternions as q3d
from transforms3d import euler as e3d
from transforms3d.quaternions import quat2mat,mat2quat

#%%
def read_dataset(folder_name, data_type, number):
    if data_type == 'cam':
        filename = folder_name + "/cam/cam" + number + ".p"
    elif data_type == 'imu':
        filename = folder_name + "/imu/imuRaw" + number + ".p"
    elif data_type == 'vicon':
        filename = folder_name + "/vicon/viconRot" + number + ".p"
    else:
        filename = None
        assert False and "type doesnt exist"

    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset
#%%

def calibrate_imu_data(imu_data_raw,debug=True):

    gyro = 3300/1023*np.pi/180/3.33
    accel = 3300/(1023.*300)

    scale_factors = np.hstack(([np.ones((3,))*accel], [np.ones((3,))*gyro])).T
    num_cal = 100

    imu_data_cut = np.sum(imu_data_raw['vals'][:, :num_cal],axis=-1)/num_cal

    biases = np.concatenate((imu_data_cut[:3]-np.asarray([0,0,1])/accel, imu_data_cut[3:]))
    
    imu_data = (imu_data_raw['vals']-np.tile(biases,(imu_data_raw['vals'].shape[1],1)).T)*scale_factors

    imu_data_calibrated = imu_data_raw
    imu_data_calibrated['vals'] = imu_data

    return imu_data_calibrated

#%%
imu_data_raw = read_dataset('trainset', 'imu', '1')

#%%
imu_data = calibrate_imu_data(imu_data_raw)
#%%
vicon_data = read_dataset('trainset', 'vicon', '1')
#%%
def integrate(imu_data,train=True):
    dt = [(imu_data['ts'][0,i+1] - imu_data['ts'][0,i]) for i in range(imu_data['ts'].shape[1]-1)]
    angular_velocities = imu_data['vals'][3:, :]
    quaternion_estimates = [Quaternion(1, [0, 0, 0])]
    for t in range(angular_velocities.shape[1]-1):     
        w = angular_velocities[:,t][[1,2,0]]
        
        #quaternion_estimates.append(quaternion_estimates[-1].multiply(Quaternion(0, (0.5*w*dt[t])).exp().unit()))
        quaternion_estimates.append((quaternion_estimates[t].multiply(Quaternion(0, 0.5*w*dt[t]).exp())).unit())
    euler_estimates = []
    for quaternion_estimate in quaternion_estimates:
        euler_estimates.append(e3d.quat2euler(quaternion_estimate.to_numpy()))

    vicon_data = read_dataset('trainset', 'vicon', '1')
    vicon_euler = []
    for i in range(vicon_data['rots'].shape[2]):
        vicon_euler.append(e3d.mat2euler(vicon_data['rots'][:, :, i]))
    if train:
        for i in range(3):
            pl.plot(imu_data['ts'][0],np.array(euler_estimates)[:, i])
            pl.plot(vicon_data['ts'][0],np.array(vicon_euler)[:, i])
            #pl.plot(vicon_data['ts'][0], np.vstack((np.array(euler_estimates)[:len(vicon_euler), i], np.array(vicon_euler)[:, i])).T)
            pl.show()
    else:
        for i in range(3):
            pl.plot(imu_data['ts'][0],np.array(euler_estimates)[:, i])
            #pl.plot(vicon_data['ts'][0],np.array(vicon_euler)[:, i])
            #pl.plot(vicon_data['ts'][0], np.vstack((np.array(euler_estimates)[:len(vicon_euler), i], np.array(vicon_euler)[:, i])).T)
            pl.show()        

#%%
integrate(imu_data)
#%%
imu_data_test_raw = read_dataset('testset', 'imu', '10')
#%%
imu_data_test = imu_data = calibrate_imu_data(imu_data_test_raw)
#%%
integrate(imu_data_test,False)
#%%
import scipy.linalg
import numpy as np
def process_noise():
    q = np.eye(3)
    q[0][0] = 0.001
    q[1][1] = 0.001
    q[2][2] = 0.001
    return q
#%%
q = process_noise()    
#%%
def get_sigmas(n_dim,mu,p,q):
    n_sig = 2 * n_dim + 1   
    ret = np.zeros((n_sig, n_dim))
    tmp_mat = (n_dim)* (p+q) 
    print(tmp_mat)
    spr_mat = scipy.linalg.cholesky(tmp_mat)
    ret[0] = mu
    for i in range(n_dim):
        ret[i+1] = mu + np.sqrt(n_dim)*spr_mat[i]
        ret[i+1+n_dim] = mu - np.sqrt(n_dim)*spr_mat[i]
    return ret
#%%
def average(quaternions, weights=None):
    if weights is None:
        weights = np.ones((len(quaternions), 1))/len(quaternions)
    else:
        weights = np.array(weights).reshape((len(quaternions),1))
    q_avg = quaternions[0]
    while True:
        errors = [quaternion*q_avg.inv() for quaternion in quaternions]
        rotation = [2*error.log().v for error in errors]
        new_rotation = (-np.pi + np.mod(np.linalg.norm(rotation, axis=1)+np.pi, 2*np.pi)).reshape((len(quaternions), 1)) * \
            np.nan_to_num(np.array(rotation)/np.linalg.norm(rotation, axis=1).reshape((len(quaternions), 1)))
        weighted_sum = np.sum(weights*new_rotation, axis=0)
        q_avg = Quaternion(0, weighted_sum/2).exp()*q_avg
        if np.linalg.norm(weighted_sum) < .001:
            return q_avg.unit()
 
#%% 
def predict(timestep,sigmas,q_t_t,w,n_dim):
    
    n_sig = 2 * n_dim + 1
    
    covar_weights = np.zeros(n_sig)
    mean_weights = np.zeros(n_sig)
    #print(sigmas)
    
    covar_weights[0] = 2
    for i in range(1,n_sig):
        covar_weights[i] = 1. / (n_sig-1)
        mean_weights[i] = 1. / (n_sig-1)
    covar_weights = covar_weights.reshape(-1,1)
    q_tt_t = []
    
    for i in range(sigmas.shape[0]):
        
        a = Quaternion(0,1/2.*sigmas[i]).exp()       
        b = Quaternion(0,1/2.*w*timestep).exp()
        q_trans = a.multiply(b)
        q_tt_t.append(q_t_t.multiply(q_trans))
    
    avg_q = average(q_tt_t, mean_weights)
    sigmas_out = []
    tmp = []
    for i in range(len(q_tt_t)):
        e = 2 * (avg_q.inv().multiply(q_tt_t[i])).log()
        tmp.append(e.v)
    tmp = np.asarray(tmp)
    sigmas_out = np.transpose(tmp*covar_weights).dot(tmp)

    return avg_q, sigmas_out, q_tt_t


#%%
q_0 = Quaternion(1,[0,0,0])
sigma_0_0 = np.eye(3)*0.0001
q = [q_0]
sigma_tt_t = sigma_0_0 
noise = process_noise()
n_dim = 6
angular_velocities = imu_data['vals']
quaternion_estimates = [Quaternion(1, [0, 0, 0])]
dt = [(imu_data['ts'][0,i+1] - imu_data['ts'][0,i]) for i in range(imu_data['ts'].shape[1]-1)]
train = imu_data['ts']
for t in range(train.shape[1]-1):
    w = angular_velocities[:,t][[4,5,3]]
    sigmas = get_sigmas(3,np.asarray([0,0,0]),sigma_tt_t,noise)
    ave_q,sigma_tt_t,_ = predict(dt[t],sigmas, q[-1], w,3)
    q.append(ave_q)
    print(ave_q)
    
#%%
euler_estimates = []
for quaternion_estimate in q:
    euler_estimates.append(e3d.quat2euler(quaternion_estimate.to_numpy()))

vicon_data = read_dataset('trainset', 'vicon', '1')
vicon_euler = []
ground_truth = []
for i in range(vicon_data['rots'].shape[2]):
    vicon_euler.append(e3d.mat2euler(vicon_data['rots'][:, :, i]))

for i in range(3):
    pl.plot(imu_data['ts'][0],np.array(euler_estimates)[:, i])
    pl.plot(vicon_data['ts'][0],np.array(vicon_euler)[:, i])
    #pl.plot(vicon_data['ts'][0], np.vstack((np.array(euler_estimates)[:len(vicon_euler), i], np.array(vicon_euler)[:, i])).T)
    pl.show()


#%%
q_0 = Quaternion(1,[0,0,0])
sigma_0_0 = np.eye(3)*0.0001
q = [q_0]
sigma_tt_t = sigma_0_0 
noise = process_noise()
n_dim = 6
angular_velocities = imu_data_test['vals']
quaternion_estimates = [Quaternion(1, [0, 0, 0])]
dt = [(imu_data_test['ts'][0,i+1] - imu_data_test['ts'][0,i]) for i in range(imu_data_test['ts'].shape[1]-1)]
train = imu_data_test['ts']
for t in range(train.shape[1]-1):
    w = angular_velocities[:,t][[4,5,3]]
    sigmas = get_sigmas(3,np.asarray([0,0,0]),sigma_tt_t,noise)
    ave_q,sigma_tt_t,_ = predict(dt[t],sigmas, q[-1], w,3)
    q.append(ave_q)
    print(ave_q)
    
#%%
euler_estimates = []
for quaternion_estimate in q:
    euler_estimates.append(e3d.quat2euler(quaternion_estimate.to_numpy()))

vicon_data = read_dataset('trainset', 'vicon', '1')
vicon_euler = []
ground_truth = []
for i in range(vicon_data['rots'].shape[2]):
    vicon_euler.append(e3d.mat2euler(vicon_data['rots'][:, :, i]))

for i in range(3):
    pl.plot(imu_data_test['ts'][0],np.array(euler_estimates)[:, i])
    #pl.plot(vicon_data['ts'][0],np.array(vicon_euler)[:, i])
    #pl.plot(vicon_data['ts'][0], np.vstack((np.array(euler_estimates)[:len(vicon_euler), i], np.array(vicon_euler)[:, i])).T)
    pl.show()

 
#%%
def measure_model(q_trans,g):
    #g = np.asarray(g,dtype='float64')
    Z = []
    for i in range(len(q_trans)):
        Z.append(((q_trans[i].multiply(g)).multiply(q_trans[i].inv())))
    return np.asarray(Z)



#%%

def update(q_tt_t,ave_q_tt_t,Cov_tt_t,z_acc,g=Quaternion(0,[0,0,1]),noise = np.eye(3)*0.001):
    n = len(q_tt_t)
    #g = np.asarray(g)
    tmp = []
    for i in range(len(q_tt_t)):
        e = 2 * ((ave_q_tt_t.inv()).multiply(q_tt_t[i])).log()
        tmp.append(e.v)
    e = np.asarray(tmp)

    Z_tt  = measure_model(q_tt_t,g)
    Z_tt = [x.v for x in Z_tt]
    Z_tt = np.asarray(Z_tt)
    print(Z_tt,'ztt')
    weight_c = np.asarray([2.]+[1./(n-1)]*(n-1)).reshape(-1,1)
    Ave_Z_tt = np.mean(Z_tt[:,:],axis = 0)
    print(Ave_Z_tt)
    Pzz = np.transpose((Z_tt-Ave_Z_tt)*weight_c).dot(Z_tt-Ave_Z_tt) + noise
    Pxz = np.transpose(e*weight_c).dot(-Z_tt+Ave_Z_tt)
    K_tt = Pxz.dot(np.linalg.inv(Pzz)) 
    q_tt_tt = ave_q_tt_t.multiply(Quaternion(0,K_tt.dot(z_acc-Ave_Z_tt)*(1/2)).exp())
    Cov_tt_tt = Cov_tt_t - K_tt.dot(Pzz).dot(np.transpose(K_tt))
    return q_tt_tt,Cov_tt_tt


#%%
'''
def update(q_tt_t, ave_q_tt_t, Cov_tt_t, z_acc, g=Quaternion(0,[0,0,1]), noise = np.eye(3)*0.001):
    
        n = len(q_tt_t)
        tmp = []
        for i in range(len(q_tt_t)):
            e = 2 * ((ave_q_tt_t.inv()).multiply(q_tt_t[i])).log()
            tmp.append(e.v)
        e = np.asarray(tmp)

        Z_tt = measure_model(q_tt_t, g)
        #print Z_tt,z_acc
        Z_tt = [x.v for x in Z_tt]
        Z_tt = np.asarray(Z_tt)
        assert(Z_tt.shape[-1]==3)
        weight_c = np.asarray([2.]+[1./(n-1)]*(n-1)).reshape(-1,1)
        Ave_Z_tt = np.mean(Z_tt[:,:], axis=0)
        Cov_Z_tt = np.transpose((Z_tt-Ave_Z_tt)*weight_c).dot(Z_tt-Ave_Z_tt) + noise
        Cov_xz_tt = np.transpose(weight_c*e).dot(Z_tt-Ave_Z_tt)
        K_tt = Cov_xz_tt.dot(np.linalg.inv(Cov_Z_tt))
        q_tt_tt = ave_q_tt_t.multiply(Quaternion(0,K_tt.dot(z_acc-Ave_Z_tt)*(1/2)).exp())
        Cov_tt_tt = Cov_tt_t - K_tt.dot(Cov_Z_tt).dot(np.transpose(K_tt)) 
        return q_tt_tt, Cov_tt_tt

'''
'''
#%%
from copy import deepcopy
def update(states,data,r_matrix,sigmas,n_dim,x,n_sig,p):

    x = x.v
    num_states = len(states)
    print(sigmas)
    sigmas = np.asarray(sigmas).T
    # create y, sigmas of just the states that are being updated
    #sigmas_split = np.split(sigmas, n_dim)
    #y = np.concatenate([sigmas_split[i] for i in states])
    y = sigmas
    # create y_mean, the mean of just the states that are being updated
    #x_split = np.split(x, n_dim)
    #y_mean = np.concatenate([x_split[i] for i in states])
    y_mean = x
    # differences in y from y mean
    y_diff = deepcopy(y)
    x_diff = deepcopy(sigmas)

    for i in range(n_sig):
        for j in range(num_states):
            y_diff[j][i] -= y_mean[j]
        for j in range(n_dim):
            x_diff[j][i] -= x[j]

    # covariance of measurement
    covar_weights = np.asarray([2.]+[1./(n_sig-1)]*(n_sig-1)).reshape(-1,1)
    p_yy = np.zeros((num_states, num_states))
    for i, val in enumerate(np.array_split(y_diff, n_sig, 1)):
        p_yy += covar_weights[i] * val.dot(val.T)

    # add measurement noise
    p_yy += r_matrix

    # covariance of measurement with states
    p_xy = np.zeros((n_dim, num_states))
    for i, val in enumerate(zip(np.array_split(y_diff, n_sig, 1), np.array_split(x_diff, n_sig, 1))):
        p_xy += covar_weights[i] * val[1].dot(val[0].T)
    k = np.dot(p_xy, np.linalg.inv(p_yy))

    y_actual = data
    x += np.dot(k, (y_actual - y_mean))
    p -= np.dot(k, np.dot(p_yy, k.T))
    #sigmas = get_sigmas(3,np.asarray([0,0,0]), Cov_tt_tt,Motion_noise) 

    return x,p
'''
'''
#%%
def predict(timestep,sigma,n_dim = 3,n_sig = 7):
    """
    performs a prediction step
    :param timestep: float, amount of time since last prediction
    """

    sigma = np.asarray(sigma).T
    x_out = np.zeros(n_dim)
    covar_weights = np.zeros(n_sig)
    mean_weights = np.zeros(n_sig)
    #print(sigmas)
    
    covar_weights[0] = 2.
    for i in range(1,n_sig):
        covar_weights[i] = 1. / (n_sig-1)
        mean_weights[i] = 1. / (n_sig-1)
    covar_weights = covar_weights.reshape(-1,1)
    print(mean_weights)
    print(sigma)
    print('sigma')
    # for each variable in X
    for i in range(n_dim):
        # the mean of that variable is the sum of
        # the weighted values of that variable for each iterated sigma point\
        x_out[i] = sum((mean_weights[j] * sigma[i][j] for j in range(n_sig)))

    p_out = np.zeros((n_dim, n_dim))
    # for each sigma point
    for i in range(n_sig):
        # take the distance from the mean
        # make it a covariance by multiplying by the transpose
        # weight it using the calculated weighting factor
        # and sum
        diff = sigma.T[i] - x_out
        diff = np.atleast_2d(diff)
        p_out += covar_weights[i] * np.dot(diff.T, diff)

    # add process noise
    #p_out += timestep * q
    return sigma,x_out,p_out
'''
#%%

q_tt_tt = Quaternion(1,[0.,0.,0.])
#q_tt_tt = [0.,0.,0.]
q = [q_tt_tt]
Cov_tt_tt = np.eye(3)*0.01 
Oberservation_noise = process_noise() 
angular_velocities = imu_data['vals']
dt = [(imu_data['ts'][0,i+1] - imu_data['ts'][0,i]) for i in range(imu_data['ts'].shape[1]-1)]
Motion_noise = process_noise()
train = imu_data['ts'][:1000]
for t in range(train.shape[1]-1):
        w = angular_velocities[:,t][[4,5,3]]
        z_acc = angular_velocities[:,t+1][:3]
        z_acc[:2] = -z_acc[:2]
        z_acc /= np.linalg.norm(z_acc)
       
        #print (np.linalg.norm(z_acc),z_acc,t)
        sigmas = get_sigmas(3,np.asarray([0.,0.,0.]), Cov_tt_tt,Motion_noise)
        #ave_q_tt_t, Cov_tt_t,q_tt_t = predict(dt[t],sigmas)
        ave_q_tt_t, Cov_tt_t,q_tt_t = predict(dt[t],sigmas, q_tt_tt, w,3) 
        #print(q_tt_t)
        q_tt_tt, Cov_tt_tt = update(q_tt_t, ave_q_tt_t, Cov_tt_t, z_acc) 
        #q_tt_tt, Cov_tt_tt = update([0,1,2],z_acc, np.eye(3)*0.001,sigmas,3,q_tt_tt,7,Cov_tt_tt) 
        q.append(ave_q_tt_t)
        print(ave_q_tt_t)

#%%
euler_estimates = []
for quaternion_estimate in q:
    euler_estimates.append(e3d.quat2euler(quaternion_estimate.to_numpy()))

vicon_data = read_dataset('trainset', 'vicon', '1')
vicon_euler = []
ground_truth = []
for i in range(vicon_data['rots'].shape[2]):
    vicon_euler.append(e3d.mat2euler(vicon_data['rots'][:, :, i]))

for i in range(3):
    pl.plot(imu_data['ts'][0],np.array(euler_estimates)[:, i])
    #pl.plot(vicon_data['ts'][0],np.array(vicon_euler)[:, i])
    #pl.plot(vicon_data['ts'][0], np.vstack((np.array(euler_estimates)[:len(vicon_euler), i], np.array(vicon_euler)[:, i])).T)
    pl.show()
#%%
imu_data_test_raw = read_dataset('testset', 'imu', '11')
#%%
imu_data_test = imu_data = calibrate_imu_data(imu_data_test_raw)
 
#%%
cam_data = read_dataset('trainset', 'cam', '8')
vicon_data = read_dataset('trainset', 'vicon', '8')
#%%
def sep2world():
    sep = [((119.5-row)/240.*45/180*np.pi,(159.5-column)/320./3*np.pi) for row in range(0,240) for column in range(0,320)]
    
    x = [np.cos(beta,dtype='float64')*np.cos(alpha,dtype='float64') for (alpha,beta) in sep]
    y = [np.sin(beta,dtype='float64')*np.cos(alpha,dtype='float64') for (alpha,beta) in sep]
    z = [np.sin(alpha,dtype='float64') for (alpha,beta) in sep]
    return x,y,z

#%%

def car2Cyl(points,r):
    points = np.asarray(points, dtype='float64')
    alpha = -np.arctan(1.0*points[..., 1]/points[..., 0]) 
    alpha[points[:,0] > 0] += np.pi
    alpha[alpha < 0] += 2*np.pi
    beta = -np.arctan(1.0*np.asarray(np.sqrt(points[:,1]**2+points[:,0]**2))/(points[:,2]))
    #print(np.round(1.0*r*(points[:,2])/np.asarray(np.sqrt(points[:,1]**2+points[:,0]**2))))
    #beta = 1.*points[:,2]/np.linalg.norm(points[:,2])
    #beta[points[:,0] > 0] += np.pi
    beta[beta < 0] += np.pi
    return np.vstack((np.round(alpha*r*1.0),np.round(1.0*r*beta)))
    

#%%
def align(t1, t2):
    return [[i, np.argmin(np.abs(t2-t1[i]))] for i in range(len(t1))]


#%%
import matplotlib.pyplot as plt
pic_3d = sep2world()
time_pair = align(cam_data['ts'][0], vicon_data['ts'][0])
r = 500
count = 0
pic = np.zeros((2500,int(np.floor(2*np.pi*r))+10, 3), dtype='uint8')
for item in time_pair[::]:
    img = cam_data['cam'][..., item[0]].reshape(-1,3)
    q = vicon_data['rots'][...,item[1]]
    #q1 = q[item[1]]
    #print(item[1])
    rotate = np.transpose(q.dot(pic_3d))
    #rotate = np.transpose(q.dot(pic_3d))
    X_2d = np.transpose(car2Cyl(rotate,r))
    #print(X_2d)
    X_2d = np.asarray(X_2d, dtype='int')
    X_2d = (X_2d + np.asarray([[0,800]]))
    if np.max(X_2d[:,1]) > 2500 or np.min(X_2d[:,1])<=0:
        count += 1
        continue
    pic[X_2d[:,1], X_2d[:,0], :] = img
plt.figure(figsize=(18,5))
plt.imshow(pic[::-1,:,:])