import tensorflow as tf
import matplotlib.pyplot as plt
from config import image_config, training_config, dataset_config
import pickle
import random
import numpy as np
from config import dataset_config
import os

class ProcessImages(object):
    def __init__(self):
        self.train_ds = self.load_fingerprint(os.path.join(dataset_config['save_dataset_path'], 'train'))
        self.validation_ds = self.load_fingerprint(os.path.join(dataset_config['save_dataset_path'], 'val'))
        self.test_ds = self.load_fingerprint(os.path.join(dataset_config['save_dataset_path'], 'test'))

        

    @staticmethod
    def load_fingerprint(path):
        with open(path + '/element_spec', 'rb') as in_:
            es = pickle.load(in_)

        loaded = tf.data.experimental.load(
            path, es, compression='GZIP'
        )
        print('loaded#####################################################')
        return loaded

