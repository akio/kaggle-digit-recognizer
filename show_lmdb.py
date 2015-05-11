#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import lmdb
import caffe

filename = sys.argv[1]

env = lmdb.open(filename, map_size=8*28*28*50000)

labels = []
values = []
with env.begin() as txn:
    with txn.cursor() as cursor:
        for key, value in cursor:
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
            data = caffe.io.datum_to_array(datum)
            labels.append(datum.label)
            values.append(data[0])

ps = np.random.random_integers(0, len(values), 25)

for i, p in enumerate(np.nditer(ps)):
    print p, len(values), len(labels)
    data = values[p]
    label = labels[p]
    plt.subplot(5, 5, i + 1)
    plt.axis('off')
    plt.imshow(data.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('{0}'.format(label))
plt.show()
