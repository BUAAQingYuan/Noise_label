__author__ = 'PC-LiNing'

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def get_sample_target(phi, target, sample):
    sum = 0.0
    for id_, v in enumerate(phi[target]):
        sum += v
        if sum > sample:
            return id_


def transfer_label(target, num_class=10):
    vec = np.zeros(shape=(num_class,), dtype=np.float32)
    vec[target] = 1.0
    return vec


# train_labels = [60000,10]
def add_uniform_noise(train_labels, q=0.2, num_class=10):
    e = np.identity(num_class, dtype=np.float32)
    ones = np.ones(shape=(num_class, num_class), dtype=np.float32)
    phi = (1-q)*e + (q/num_class)*ones
    all_size = train_labels.shape[0]
    noise_labels = np.zeros(shape=(all_size, num_class), dtype=np.float32)
    correct = 0
    for id_ in range(all_size):
        sample = np.random.random_sample()
        target = np.argmax(train_labels[id_])
        sample_target = get_sample_target(phi, target=target, sample=sample)
        if target == sample_target:
            correct += 1
        noise_labels[id_] = transfer_label(sample_target, num_class)

    print("generate noise label, " + str(correct/all_size*100) + "% is correct.")
    return noise_labels, phi


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    len(data) is not divisible by batch_size .
    """
    data = np.array(data)
    data_size = len(data)
    if len(data) % batch_size == 0:
        num_batches_per_epoch = int(len(data)/batch_size)
    else:
        num_batches_per_epoch = int(len(data)/batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            # shuffle_indices = train_shuffle[epoch]
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
