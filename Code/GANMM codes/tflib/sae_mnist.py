import numpy

import os
import pickle
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

        image_batches = images.reshape(-1, batch_size, 256)
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


    with open('/content/gdrive/MyDrive/ML Datasets/GANMM-master/Data/sae_mnist.pkl', 'rb') as f:
        full_ft = (numpy.load('/content/gdrive/MyDrive/ML Datasets/test_1.npy'))
        full_lbl = (numpy.load('/content/gdrive/MyDrive/ML Datasets/test_1_y.npy'))[:len(full_ft)]-1
        full_ft = full_ft/(full_ft.max(axis=0)-full_ft.min(axis=0))

        rng_state = numpy.random.get_state()
        numpy.random.shuffle(full_ft)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(full_lbl)

        full_ft = full_ft[:int(full_ft.shape[0]*scale),:]
        full_lbl = full_lbl[:int(full_lbl.shape[0]*scale)]
        print(full_ft.shape, full_lbl.shape)
        # sio.savemat('/tmp/mnist.mat',{'trn_ft':trn_ft, 'dev_ft':dev_ft, 'tst_ft':tst_ft, 'trn_lbl':trn_lbl, 'dev_lbl':dev_lbl, 'tst_lbl':tst_lbl})
        # print 'save mat done'

    return mnist_generator((full_ft,full_lbl), batch_size, n_labelled)

def getTrainData():
    with open('/content/gdrive/MyDrive/ML Datasets/GANMM-master/Data/sae_mnist.pkl', 'rb') as f:
        # full_ft,full_lbl = pickle.load(f)
        full_ft = (numpy.load('/content/gdrive/MyDrive/ML Datasets/test_1.npy'))
        full_lbl = (numpy.load('/content/gdrive/MyDrive/ML Datasets/test_1_y.npy'))[:len(full_ft)]-1
    full_ft = full_ft/(full_ft.max(axis=0)-full_ft.min(axis=0))
    return full_ft,full_lbl

def splitGenerator(batch_size, scale=1.0):
    with open('/content/gdrive/MyDrive/ML Datasets/GANMM-master/Data/sae_mnist.pkl', 'rb') as f:
        # full_ft,full_lbl = pickle.load(f)
        full_ft = (numpy.load('/content/gdrive/MyDrive/ML Datasets/test_1.npy'))
        full_lbl = (numpy.load('/content/gdrive/MyDrive/ML Datasets/test_1_y.npy'))[:len(full_ft)]-1
    full_ft = full_ft/(full_ft.max(axis=0)-full_ft.min(axis=0))

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
        length = int(data_size/(batch_size*4))
        i=num*length
        while True:
            yield numpy.copy(images[i * batch_size:(i + 1) * batch_size]), numpy.copy(labels[i * batch_size:(i + 1) * batch_size])
            i=i+1
            if i==(num+1)*length:
                i=num*length
    ans = []
    for i in range(0,4):
      ans.append(getEpoch(i))
    return ans

