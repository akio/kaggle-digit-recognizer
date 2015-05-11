#!/usr/bin/env python

import sys
import os
import csv
import lmdb
import caffe


filename = sys.argv[1]
dirname = os.path.dirname(filename)
basename, ext = os.path.splitext(filename)

env = lmdb.open(basename, map_size=8*28*28*50000)

with open(filename, 'rb') as f:
    reader = csv.reader(f)
    reader.next()  # skip header
    for i, row in enumerate(reader):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.height = 28
        datum.width = 28
        datum.channels = 1
        datum.label = int(row[0])
        data = bytearray([chr(int(x)) for x in row[1:]])
        datum.data = bytes(data)
        str_id = '{:08}'.format(i)
        with env.begin(write=True) as txn:
            txn.put(str_id, datum.SerializeToString())

