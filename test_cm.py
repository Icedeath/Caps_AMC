#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:10:47 2018

@author: icedeath
"""
import numpy as np
idx_cm = np.zeros([6, 6])
y_t = np.array([0,1,0,0,1])
y_p = np.array([1,0,0,1,0])


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
    re_p = np.ones(max_tar - idx2_p.shape[0],dtype = int)*5
    re_t = np.ones(max_tar - idx2_t.shape[0],dtype = int)*5
      
    idx2_p = np.concatenate([idx2_p, re_p])
    idx2_t = np.concatenate([idx2_t, re_t])
        
    idx_cm[idx2_p, idx2_t] += 1
    
print(idx_cm)