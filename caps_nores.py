#coding=utf

from keras.utils import multi_gpu_model
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.layers import Lambda
import matplotlib.pyplot as plt
import tensorflow as tf
from capsulelayers2 import CapsuleLayer, PrimaryCap, Length, Mask
from keras import callbacks
from keras.layers.normalization import BatchNormalization as BN
import argparse
#import scipy.io as sio
import h5py
from keras.layers.advanced_activations import ELU

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


def margin_loss(y_true, y_pred, margin = 0.4, downweight = 0.5):
    y_pred = y_pred - 0.5
    positive_cost = y_true * K.cast(
                    K.less(y_pred, margin), 'float32') * K.pow((y_pred - margin), 2)
    negative_cost = (1 - y_true) * K.cast(
                    K.greater(y_pred, -margin), 'float32') * K.pow((y_pred + margin), 2)
    return 0.5 * positive_cost + downweight * 0.5 * negative_cost


def train(model, data, args):
    (x_train, y_train) = data

    checkpoint = callbacks.ModelCheckpoint(args.save_file, monitor='val_loss', verbose=1, save_best_only=True, 
                                  save_weights_only=True, mode='auto', period=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    #model = multi_gpu_model(model, gpus=2)
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss= margin_loss,
                  metrics={})
    if args.load == 1:
        model.load_weights(args.save_file)
        print('Loading %s' %args.save_file)
    hist = model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
                     validation_split = 0.1, callbacks=[checkpoint, lr_decay])
    return hist.history


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--lr', default=0.002, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.92, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('-sf', '--save_file', default='0_15_3500_16.h5',
                        help="Name of saved weight file")
    parser.add_argument('-t', '--test', default=0,type=int,
                        help="Test only model")
    parser.add_argument('-l', '--load', default=0,type=int,
                        help="load weight file or not")
    parser.add_argument('-p', '--plot', default=0,type=int,
                        help="plot training loss after finished if plot==1")
    parser.add_argument('-d', '--dataset', default='./samples/dataset_MAMC_10.mat',
                        help="name of dataset that needs loading")
    parser.add_argument('-n', '--num_classes', default=10)
    parser.add_argument('-dc', '--dim_capsule', default=16)
    args = parser.parse_args()
    print(args)
    
    K.set_image_data_format('channels_last')
    ''' 
    data = sio.loadmat(args.dataset, appendmat=False)
    for i in data:
        locals()[i] = data[i]
    del data
    del i
    '''
    '''
    with np.load(args.dataset) as data:
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
    y_pred_tr = model.predict(x_train, batch_size=args.batch_size,verbose=1)
    _, y_pred1_tr = tf.nn.top_k(y_pred_tr, 2)
    _, y_1 = tf.nn.top_k(y_train, 2)
    y_pred1_tr = K.eval(y_pred1_tr)
    y_pred1_tr.sort(axis = 1)
    y_1 = K.eval(y_1)
    y_1.sort(axis = 1)
    y_pred1_tr = np.reshape(y_pred1_tr, np.prod(y_pred1_tr.shape))
    y_1 = np.reshape(y_1, np.prod(y_1.shape))
    print('Train acc:', np.sum(y_pred1_tr == y_1)/np.float(y_1.shape[0]))
    print('-' * 30 + 'End: test' + '-' * 30)   

    
    
'''
    from keras.utils import plot_model
    plot_model(model, to_file='model.png',show_shapes = True)
'''