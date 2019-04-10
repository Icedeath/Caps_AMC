#coding=utf
from keras import utils
import numpy as np
from keras import backend as K
import argparse
import scipy.io as sio
import h5py

def load_data(args):
    with h5py.File(args.data, 'r') as f:
        x_train = np.transpose(np.float64(f['train_x'].value))
        
    with h5py.File(args.data, 'r') as f:
        y_train = np.transpose(np.float64(f['train_y'].value-1))
    
    y_train = utils.to_categorical(y_train, args.num_classes)

    print('Data loading finished...')
    print('Start shifting train set')
    idx_tr = range(len(x_train))
    np.random.shuffle(idx_tr)
    X_train = np.zeros([x_train.shape[0], args.length])
    for i in xrange(x_train.shape[0]):
        shifti = np.random.randint(-args.max_shift, args.max_shift, size=2)
        aci = np.random.rand(2)*(args.ac_max-args.ac_min)+args.ac_min
        x_train_0 = shift(x_train[i], shifti[0], args)
        x_train_1 = shift(x_train[idx_tr[i]], shifti[1], args)
        X_train[i] = np.add(aci[0]*x_train_0, aci[1]*x_train_1)
        
    Y_train1 = np.vstack([y_train.argmax(1), y_train[idx_tr].argmax(1)]).T
    del x_train
    print('Eliminating repeat categoteries..')
    X_train = X_train[Y_train1[:,0] != Y_train1[:,1]]
    Y_train1 = Y_train1[Y_train1[:,0] != Y_train1[:,1]]
    Y_train = K.eval(K.one_hot(Y_train1, args.num_classes))
    
    with h5py.File(args.data, 'r') as f:
        x_test = np.transpose(np.float64(f['test_x'].value))
    
    with h5py.File(args.data, 'r') as f:
        y_test = np.transpose(np.float64(f['test_y'].value-1))
    
    print('Start shifting test set')
    y_test = utils.to_categorical(y_test, args.num_classes)
    idx_te = range(len(x_test))
    np.random.shuffle(idx_te)
    X_test = np.zeros([x_test.shape[0], args.length])
    for i in xrange(x_test.shape[0]):
        shifti = np.random.randint(-args.max_shift, args.max_shift, size=2)
        aci = np.random.rand(2)*(args.ac_max-args.ac_min)+args.ac_min
        x_test_0 = shift(x_test[i], shifti[0], args)
        x_test_1 = shift(x_test[idx_te[i]], shifti[1], args)
        X_test[i] = np.add(aci[0]*x_test_0, aci[1]*x_test_1)
    Y_test1 = np.vstack([y_test.argmax(1), y_test[idx_te].argmax(1)]).T
    del y_train
    print('Eliminating repeat categoteries..')
    X_test = X_test[Y_test1[:,0] != Y_test1[:,1]] 
    Y_test1 = Y_test1[Y_test1[:,0] != Y_test1[:,1]]
    Y_test = K.eval(K.one_hot(Y_test1, args.num_classes))
    

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], 1)
    return (X_train, Y_train), (X_test, Y_test), (Y_train1, Y_test1)

def shift(signal, shift, args):
    max_shift = args.max_shift
    max_shift += 1
    padded = np.pad(signal, max_shift, 'constant')
    rolled = np.roll(padded, shift)
    shifted_signal = rolled[max_shift:args.length+max_shift]
    return shifted_signal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-class signal generation.")
    parser.add_argument('-sd', '--save_data', default='test_shifted0_10.npz',
                        help="Name of saved data file")
    parser.add_argument('-m', '--max_shift', default=500, type=int,
                        help="maximum shift of mnist images before adding them together")
    parser.add_argument('-l', '--length', default=3500, type=int,
                        help="length of signal")
    parser.add_argument('-ds', '--data', default='./samples/test0_10.mat',
                        help="maximum shift of mnist images before adding them together")
    parser.add_argument('-max', '--ac_max', default=1.25)
    parser.add_argument('-min', '--ac_min', default=0.75)
    parser.add_argument('-n', '--num_classes', default=9)
    args = parser.parse_args()
    print(args)
    
    
    (x_train, y_train), (x_test, y_test), (y_train1, y_test1) = load_data(args=args)
    y_test = K.eval(K.sum(y_test, -2))
    y_train = K.eval(K.sum(y_train, -2))
    
    
    print('Saving data:%s' %args.save_data)
    '''
    sio.savemat(args.save_data, {'x_train':x_train,'y_train':y_train,
                                 'x_test':x_test,'y_test':y_test,
                                 'x_train0':x_train0,'x_train1':x_train1,
                                 'x_test0':x_test0,'x_test1':x_test1,
                                 'y_train1':y_train1, 'y_test1':y_test1})
    '''
    np.savez(args.save_data, x_train = x_train, y_train = y_train,x_test = x_test, 
                             y_test = y_test, y_train1 = y_train1, y_test1 = y_test1)
    
    