#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:25:51 2018

@author: icedeath
"""

print('-'*30 + 'Begin: test' + '-'*30)
y_pred_tr = model.predict(x_train, batch_size=args.batch_size,verbose=1)
_, y_pred1_tr = tf.nn.top_k(y_pred_tr, 2)
_, y_1 = tf.nn.top_k(y_train, 2)
y_pred1_tr = K.eval(y_pred1_tr)
y_pred1_tr.sort(axis = 1)
y_1 = K.eval(y_1)
y_1.sort(axis = 1)
y_pred1_tr = np.reshape(y_pred1_tr, np.prod(y_pred1_tr.shape))
y_1 = np.reshape(y_1, np.prod(y_1.shape))
print('Train acc:', np.sum(y_pred1_tr == y_train1)/np.float(y_train1.shape[0]))
print('-' * 30 + 'End: test' + '-' * 30)   
