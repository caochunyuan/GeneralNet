import os
import sys
import numpy as np
import json
import caffe_pb2

class CaffeDataReader(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.json_dict = {}
        self.offset = 0
        self.load_using_pb()

    def load_using_pb(self):
        data = caffe_pb2.NetParameter()
        print("Loading the caffemodel. This takes a couple of minutes.")
        data.MergeFromString(open(self.data_path, 'rb').read())
        print("Done reading")
        pair = lambda layer: (layer.name, self.read_size(layer))
        layers = data.layers or data.layer
        map(pair, [layer for layer in layers if layer.blobs])
        with open('offset_'+self.data_path[:-11]+'.json', 'wb') as f:
            f.write(json.dumps(self.json_dict))
            print("Done writing.")

    def read_size(self, layer):
        print("Reading layer %s" % layer.name)
        key = ["weight", "bias"]
        offset_dict = {}
        for idx, blob in enumerate(layer.blobs):
            c_o  = blob.num
            c_i  = blob.channels
            h    = blob.height
            w    = blob.width
            offset_dict[key[idx]] = self.offset
            self.offset += c_o * c_i * h *w
        self.json_dict[layer.name] = offset_dict
        self.json_dict['file_size'] = self.offset * 4


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("usage: %s path.caffemodel" % os.path.basename(__file__))
        exit(-1)
    data_path = args[0]
    CaffeDataReader(data_path)

if __name__ == '__main__':
    main()
