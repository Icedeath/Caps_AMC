#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:25:51 2018

@author: icedeath
"""
import scipy.io as sio
import numpy as np

data = sio.loadmat('final_output.mat', appendmat=False)  #matlab 非-v7.3
for i in data:
    locals()[i] = data[i]
del data
del i
num_classes = 10

y_pred = (np.sign(y_pred1-0.62)+1)/2
idx_yt = np.sum(y_train, axis = 1)
idx_yp = np.sum(y_pred, axis = 1)
idx_cm = np.zeros([num_classes + 1, num_classes+1])
idx = np.arange(0, 10)
for i in xrange(y_pred.shape[0]):
    if np.mod(i,200000)==0:
        print(i)
    y_p = y_pred[i,:]
    y_t = y_train[i,:]
    y_ref = y_p + y_t
    
    idx1 = idx[y_ref==2]
    if idx1.shape[0]!=0:
        y_p[idx1] = 0
        y_t[idx1] = 0
        y_ref[idx1] = 0
        idx_cm[idx1, idx1] += 1
    if np.sum(y_ref)!=0:
        idx2_p = idx[y_p==1]
        idx2_t = idx[y_t==1]    
        max_tar = np.max([idx2_p.shape[0],idx2_t.shape[0]])
        re_p = np.ones(max_tar - idx2_p.shape[0],dtype = int)*num_classes
        re_t = np.ones(max_tar - idx2_t.shape[0],dtype = int)*num_classes
        
        idx2_p = np.concatenate([idx2_p, re_p])
        idx2_t = np.concatenate([idx2_t, re_t])
        
        idx_cm[idx2_p, idx2_t] += 1

acc = get_accuracy(idx_cm) 
pm = np.sum(idx_cm[num_classes,:])/np.sum(idx_cm)  # Missing Alarm
pf = np.sum(idx_cm[:, num_classes])/np.sum(idx_cm)  #False Alarm
print('-' * 30 + 'End  : test' + '-' * 30)   

print(pf)
print(pm)
print(np.mean(acc))