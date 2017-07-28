import pickle
import os
import sys
import tarfile
from six.moves import urllib
import numpy as np

def maybe_download_and_extract(data_dir, url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    if not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(data_dir)

def unpickle(file):
    fo = open(file, 'rb')
    d = pickle.load(fo, encoding="bytes")
    fo.close()
    return {'x': np.cast[np.float32]((-127.5 + np.transpose(d[b'data'].reshape((10000,3,32,32)), (0,2,3,1)))/128.), 'y': np.array(d[b'labels']).astype(np.uint8)}

#load cifar
def load(data_dir, subset='train'):
    maybe_download_and_extract(data_dir)
    if subset=='train':
        train_data = [unpickle(os.path.join(data_dir,'cifar-10-batches-py/data_batch_' + str(i))) for i in range(1,6)]
        trainx = np.concatenate([d['x'] for d in train_data],axis=0)
        trainy = np.concatenate([d['y'] for d in train_data],axis=0)
        return trainx, trainy
    elif subset=='test':
        test_data = unpickle(os.path.join(data_dir,'cifar-10-batches-py/test_batch'))
        testx = test_data['x']
        testy = test_data['y']
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')


# load cars
def load_cars(data_dir, subset='train'):
    split = 1500 # number of test set images
    if subset=='train':
        trainx = np.load(os.path.join(data_dir, 'images.npy'), mmap_mode='r')[:-split]
        trainy = np.load(os.path.join(data_dir, 'labels.npy'))[:-split]
        return trainx, trainy
    if subset=='test':
        testx = np.load(os.path.join(data_dir, 'images.npy'), mmap_mode='r')[-split:]
        testy = np.load(os.path.join(data_dir, 'labels.npy'))[-split:]
        return testx, testy

