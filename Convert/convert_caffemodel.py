# The VGGNet Caffe model stores the weights for each layer in this shape:
#    (outputChannels, inputChannels, kernelHeight, kernelWidth)
#
# The Metal API expects weights in the following shape:
#    (outputChannels, kernelHeight, kernelWidth, inputChannels)
#

import os
import sys
import numpy as np

class CaffeDataReader(object):
    def __init__(self, def_path, data_path):
        self.def_path = def_path
        self.data_path = data_path
        self.load_using_pb()

    def load_using_pb(self):
        import caffe_pb2
        data = caffe_pb2.NetParameter()
        print("Loading the caffemodel. This takes a couple of minutes.")
        data.MergeFromString(open(self.data_path, 'rb').read())
        print("Done reading")
        pair = lambda layer: (layer.name, self.transform_data(layer))
        layers = data.layers or data.layer
        self.parameters = [pair(layer) for layer in layers if layer.blobs]
        print("Done transforming")

    def transform_data(self, layer):
        print("Transforming layer %s" % layer.name)
        transformed = []
        for idx, blob in enumerate(layer.blobs):
            c_o  = blob.num
            c_i  = blob.channels
            h    = blob.height
            w    = blob.width
            print("  %d: %d x %d x %d x %d" % (idx, c_o, c_i, h, w))

            arr = np.array(blob.data, dtype=np.float32)
            transformed.append(arr.reshape(c_o, c_i, h, w))

        print()
        return tuple(transformed)

    def dump(self):
        params = []
        def convert(data):
            if data.ndim == 4:
                # (c_o, c_i, h, w) -> (c_o, h, w, c_i)
                data = data.transpose((0, 2, 3, 1))
            else:
                print("Unsupported layer:", data.shape)
            return data

        offset = 0
        s = ""
        all = np.array([], dtype=np.float32)
        for key, data_pair in self.parameters:
            print(key)
            ext = ["weights", "bias"]
            for i, data in enumerate(map(convert, data_pair)):
                s += ("- (float *) %s_%s {\n    return _basePtr + %s;\n}\n\n" % (ext[i], key, offset))
                print("  ", data.shape)
                offset += data.size
                all = np.append(all, data.ravel())

        f = open(os.getcwd() + "/params.dat", "wb")        
        all.tofile(f)
        f.close()

        print("\nCopy this code into Net.m:")
        print(s[:-1])
        print("static const uint fileSize = " + str(all.shape[0] * 4) + ";\n")
        print("Done!")

def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print("usage: %s path.prototxt path.caffemodel" % os.path.basename(__file__))
        exit(-1)
    def_path, data_path = args
    CaffeDataReader(def_path, data_path).dump()

if __name__ == '__main__':
    main()
    