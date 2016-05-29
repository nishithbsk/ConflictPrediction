import os
import sys
import numpy as np
import scipy.io as sio

class DataSet(object):
    def __init__(self, X, y, mask):
        self._data = X
        self._labels = y
        self._mask = mask
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = self._data.shape[0]
    
    @property
    def data(self):
        return self._data
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples: # if we have reached end of epoch
            self._epochs_completed += 1

            # shuffle data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._labels = self._labels[perm]

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._data[start:end][0], self._labels[start:end][0], self._mask

def read_datasets(data_path='', dataset_type='train'):
    print('loading data...')
    
    # load data
    X_train, X_test, y_train, y_test, mask = np.load(data_path)

    if dataset_type == 'train':
        print('Training data shape:', X_train.shape)
        return DataSet(X_train, y_train, mask)
    else:
        print('Test data shape:', X_test.shape)
        return DataSet(X_test, y_test, mask)
