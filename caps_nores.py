#coding=utf

from keras.utils import multi_gpu_model
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.layers import Lambda
#import matplotlib.pyplot as plt
import tensorflow as tf
from capsulelayers2 import CapsuleLayer, PrimaryCap, Length, Mask
from keras import callbacks
from keras.layers.normalization import BatchNormalization as BN
import argparse
import scipy.io as sio
import h5py
from keras.layers.advanced_activations import ELU
'''
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
'''
K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=64, kernel_size=(1,3), strides=(1,2), padding='same')(x)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=64, kernel_size=(1,3), strides=(1,1), padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.MaxPooling2D((1, 2), strides=(1, 2))(conv1)
    
    conv1 = layers.Conv2D(filters=96, kernel_size=(1,3), strides=1, padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=96, kernel_size=(1,3), strides=1, padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.MaxPooling2D((1, 2), strides=(1, 2))(conv1)
    
    conv1 = layers.Conv2D(filters=128, kernel_size=(1,3), strides=1, padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=128, kernel_size=(1,3), strides=1, padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.MaxPooling2D((1, 2), strides=(1, 2))(conv1)

    conv1 = layers.Conv2D(filters=192, kernel_size=(1,3), strides=1, padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=192, kernel_size=(1,3), strides=1, padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=192, kernel_size=(1,3), strides=1, padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.MaxPooling2D((1, 2), strides=(1, 2))(conv1)

    conv1 = layers.Conv2D(filters=256, kernel_size=(1,3), strides=1, padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=256, kernel_size=(1,3), strides=1, padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    conv1 = layers.Conv2D(filters=256, kernel_size=(1,3), strides=1, padding='same')(conv1)
    conv1 = ELU(alpha=0.5)(conv1)
    conv1 = BN()(conv1)
    
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=(1,3),
                             strides=1, padding='same')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=args.dim_capsule, routings=routings,
                             name='digitcaps')(primarycaps)
    out_caps = Length(name='capsnet')(digitcaps)
    
    

    model = models.Model(x, out_caps)
    return model


def margin_loss(y_true, y_pred, margin = 0.4, threshold = 0.1, diver = 0.1):
    y_pred = y_pred - 0.5
    positive_cost = y_true * K.cast(
                    K.less(y_pred, margin), 'float32') * K.pow((y_pred - margin), 2)
    negative_cost = (1 - y_true) * K.cast(
                    K.greater(y_pred, -margin), 'float32') * K.pow((y_pred + margin), 2)
    positive_threshold_cost = y_true * K.cast(
                    K.less(y_pred, threshold + diver), 'float32') * K.pow((y_pred - threshold - diver), 2)
    negative_threshold_cost = (1 - y_true) * K.cast(
                    K.greater(y_pred, -threshold - diver), 'float32') * K.pow((y_pred + threshold + diver), 2)
    return 0.5 * positive_cost + 0.5 * negative_cost + 2*positive_threshold_cost + 2*negative_threshold_cost


def margin_loss1(y_true, y_pred, margin = 0.4):
    y_pred = y_pred - 0.5
    positive_cost = y_true * K.cast(
                    K.less(y_pred, margin), 'float32') * K.pow((y_pred - margin), 2)
    negative_cost = (1 - y_true) * K.cast(
                    K.greater(y_pred, -margin), 'float32') * K.pow((y_pred + margin), 2)
    return 0.5 * positive_cost + 0.5 * negative_cost


def train(model, data, args):
    (x_train, y_train) = data

    checkpoint = callbacks.ModelCheckpoint(args.save_file, monitor='val_loss', verbose=1, save_best_only=True, 
                                  save_weights_only=True, mode='auto', period=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    model = multi_gpu_model(model, gpus=2)
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss= margin_loss,
                  metrics={})
    if args.load == 1:
        model.load_weights(args.save_file)
        print('Loading %s' %args.save_file)
    hist = model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
                     validation_split = 0.1, callbacks=[checkpoint, lr_decay])
    return hist.history

def get_accuracy(cm):
    return [float(cm[i,i]/np.sum(cm[0:args.num_classes,:])) for i in xrange(args.num_classes)]



    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=0.002, type=float,
                        help="初始学习率")
    parser.add_argument('--lr_decay', default=0.92, type=float,
                        help="学习率衰减")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="routing迭代次数")
    parser.add_argument('-sf', '--save_file', default='3500_015_92.h5',
                        help="权重文件名称")
    parser.add_argument('-t', '--test', default=1,type=int,
                        help="测试模式，设为非0值激活，跳过训练")
    parser.add_argument('-l', '--load', default=1,type=int,
                        help="是否载入模型，设为1激活")
    parser.add_argument('-p', '--plot', default=0,type=int,
                        help="训练结束后画出loss变化曲线，设为1激活")
    parser.add_argument('-d', '--dataset', default='./samples/dataset_MAMC_10.mat',
                        help="需要载入的数据文件，MATLAB -v7.3格式")
    parser.add_argument('-n', '--num_classes', default=8,
                        help="类别数")
    parser.add_argument('-dc', '--dim_capsule', default=16)
    parser.add_argument('-tm', '--target_max', default=2, type=int)
    args = parser.parse_args()
    print(args)
    
    K.set_image_data_format('channels_last')
    ''' 
    data = sio.loadmat(args.dataset, appendmat=False)  #matlab 非-v7.3
    for i in data:
        locals()[i] = data[i]
    del data
    del i
    '''
    '''
    with np.load(args.dataset) as data: #npz格式
        x_train = data['x_train']
    with np.load(args.dataset) as data:
        y_train = data['y_train']
    '''
    with h5py.File(args.dataset, 'r') as data:
        for i in data:
            locals()[i] = data[i].value
    
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], 1)
    

    model = CapsNet(input_shape=x_train.shape[1:], n_class=args.num_classes, routings=args.routings)
        

    if args.test == 0:    
        history = train(model=model, data=((x_train, y_train)), args=args)
        if args.plot == 1:    
            train_loss = np.array(history['loss'])
            val_loss = np.array(history['val_loss'])
            plt.plot(np.arange(0, args.epochs, 1),train_loss,label="train_loss",color="red",linewidth=1.5)
            plt.plot(np.arange(0, args.epochs, 1),val_loss,label="val_loss",color="blue",linewidth=1.5)
            plt.legend()
            plt.show()
            plt.savefig('loss.png')
    else:
        model.load_weights(args.save_file)
        print('Loading %s' %args.save_file)
      
    print('-'*30 + 'Begin: test' + '-'*30)
    
    y_pred1 = model.predict(x_train, batch_size=args.batch_size,verbose=1)
    sio.savemat('final_output.mat', {'y_pred1':y_pred1, 'y_train':y_train})
    y_pred = (np.sign(y_pred1-0.62)+1)/2
    idx_yt = np.sum(y_train, axis = 1)
    idx_yp = np.sum(y_pred, axis = 1)
    idx_cm = np.zeros([args.num_classes + 1, args.num_classes+1])
    idx = np.arange(0, args.num_classes)
    for i in xrange(y_pred.shape[0]):
        if np.mod(i,20000)==0:
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
            re_p = np.ones(max_tar - idx2_p.shape[0],dtype = int)*args.num_classes
            re_t = np.ones(max_tar - idx2_t.shape[0],dtype = int)*args.num_classes
        
            idx2_p = np.concatenate([idx2_p, re_p])
            idx2_t = np.concatenate([idx2_t, re_t])
        
            idx_cm[idx2_p, idx2_t] += 1

    acc = get_accuracy(idx_cm) 
    pm = np.sum(idx_cm[args.num_classes,:])/(np.sum(
            idx_cm[0:args.num_classes,0:args.num_classes])+np.sum(idx_cm[args.num_classes,:]))  # Missing Alarm
    pf = np.sum(idx_cm[:, args.num_classes])/(np.sum(
            idx_cm[0:args.num_classes,0:args.num_classes])+np.sum(idx_cm[:,args.num_classes]))  #False Alarm
    print('-' * 30 + 'End  : test' + '-' * 30)   

    #sio.savemat('final_output.mat', {'y_pred1':y_pred1, 'y_train':y_train})
    
'''
    from keras.utils import plot_model
    plot_model(model, to_file='model.png',show_shapes = True)
'''