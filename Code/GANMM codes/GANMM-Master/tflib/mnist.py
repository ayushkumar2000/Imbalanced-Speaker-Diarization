import numpy

import os
import urllib.request, urllib.parse, urllib.error
import gzip
import pickle as pickle
import scipy.io as sio

def mnist_generator(data, batch_size, n_labelled, limit=None):
    images, targets = data

    rng_state = numpy.random.get_state()
    numpy.random.shuffle(images)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(targets)
    if limit is not None:
        print("WARNING ONLY FIRST {} MNIST DIGITS".format(limit))
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        if n_labelled is not None:
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(labelled)

        image_batches = images.reshape(-1, batch_size, 784)
        target_batches = targets.reshape(-1, batch_size)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in range(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]), numpy.copy(labelled))

        else:

            for i in range(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]))

    return get_epoch

def load(batch_size, test_batch_size, scale=1.0, n_labelled=None):
    filepath = "/content/gdrive/MyDrive/ML Datasets/GANMM-master/Data/mnist.pkl.gz"
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print("Couldn't find MNIST dataset in /tmp, downloading...")
        urllib.request.urlretrieve(url, filepath)

    with gzip.open('/content/gdrive/MyDrive/ML Datasets/GANMM-master/Data/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f,encoding = 'latin1')
        trn_ft,trn_lbl = train_data
        dev_ft,dev_lbl = dev_data
        tst_ft,tst_lbl = test_data
        print(trn_ft.shape,dev_ft.shape,tst_ft.shape)
        print(trn_lbl.shape, dev_lbl.shape, tst_lbl.shape)
        full_ft = numpy.vstack((trn_ft,tst_ft))
        full_lbl = numpy.hstack((trn_lbl,tst_lbl))

        rng_state = numpy.random.get_state()
        numpy.random.shuffle(full_ft)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(full_lbl)

        full_ft = full_ft[:int(full_ft.shape[0] * scale), :]
        full_lbl = full_lbl[:int(full_lbl.shape[0] * scale)]

        print(full_ft.shape, full_lbl.shape)
        # sio.savemat('/tmp/mnist.mat',{'trn_ft':trn_ft, 'dev_ft':dev_ft, 'tst_ft':tst_ft, 'trn_lbl':trn_lbl, 'dev_lbl':dev_lbl, 'tst_lbl':tst_lbl})
        # print 'save mat done'

    return mnist_generator((full_ft,full_lbl), batch_size, n_labelled)

def getTrainData():
    filepath = '/content/gdrive/MyDrive/ML Datasets/GANMM-master/Data/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print("Couldn't find MNIST dataset in /tmp, downloading...")
        urllib.request.urlretrieve(url, filepath)

    with gzip.open('/content/gdrive/MyDrive/ML Datasets/GANMM-master/Data/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f,encoding = 'latin1')
        trn_ft, trn_lbl = train_data
        dev_ft, dev_lbl = dev_data
        tst_ft, tst_lbl = test_data
    full_ft = numpy.vstack((trn_ft, tst_ft))
    full_lbl = numpy.hstack((trn_lbl, tst_lbl))
    return full_ft,full_lbl

def splitGenerator(batch_size, scale=1.0):
    with gzip.open('/content/gdrive/MyDrive/ML Datasets/GANMM-master/Data/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f,encoding = 'latin1')
        trn_ft, trn_lbl = train_data
        dev_ft, dev_lbl = dev_data
        tst_ft, tst_lbl = test_data
    full_ft = numpy.vstack((trn_ft, tst_ft))
    full_lbl = numpy.hstack((trn_lbl, tst_lbl))

    rng_state = numpy.random.get_state()
    numpy.random.shuffle(full_ft)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(full_lbl)

    full_ft = full_ft[:int(full_ft.shape[0] * scale), :]
    full_lbl = full_lbl[:int(full_lbl.shape[0] * scale)]

    images, labels = full_ft, full_lbl

    rng_state = numpy.random.get_state()
    numpy.random.shuffle(images)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(labels)
    data_size = len(labels)

    def getEpoch(num):
        length = int(data_size/(batch_size*10))
        i=num*length
        while True:
            yield numpy.copy(images[i * batch_size:(i + 1) * batch_size]), numpy.copy(labels[i * batch_size:(i + 1) * batch_size])
            i=i+1
            if i==(num+1)*length:
                i=num*length
    return [getEpoch(0),getEpoch(1),getEpoch(2),getEpoch(3),getEpoch(4),getEpoch(5),getEpoch(6),getEpoch(7),getEpoch(8),getEpoch(9)]

def getVisData():
    vis_data = numpy.load('/content/gdrive/MyDrive/ML Datasets/GANMM-master/Data/mnist_vis_data.npy')
    return vis_data
