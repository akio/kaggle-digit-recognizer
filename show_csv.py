#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

dataset = []
with open(sys.argv[1]) as f:
    reader = csv.reader(f)
    reader.next()
    for row in reader:
        data = bytearray(int(x) for x in row[1:])
        dataset.append([int(row[0]), np.array(data, dtype=np.float32)])

ps = np.random.random_integers(0, len(dataset), 25)

for i, p in enumerate(ps):
    label, data = dataset[p]
    plt.subplot(5, 5, i + 1)
    plt.axis('off')
    plt.imshow(data.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('{0}'.format(label))
plt.show()
