#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import caffe

filename = sys.argv[1]
modelfile = sys.argv[2]
pretrained = sys.argv[3]

net = caffe.Classifier(modelfile, pretrained, raw_scale=255, image_dims=(28, 28))

dataset = []
with open(filename) as f:
    reader = csv.reader(f)
    reader.next()
    for row in reader:
        data = bytearray(int(x) for x in row)
        len(data)
        dataset.append(np.array(data, dtype=np.float32))

ps = np.random.random_integers(0, len(dataset), 25)

for i, p in enumerate(ps):
    data = dataset[p]
    plt.subplot(5, 5, i + 1)
    plt.axis('off')
    plt.imshow(data.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
    predictions = net.predict([data.reshape(28, 28, 1)])
    label = predictions[0].argmax()
    label = predictions[0]
    plt.title('{0} ({1})'.format(label, ))
plt.show()
