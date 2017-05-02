def generate_param_txt(path):
    import caffe_pb2
    from google.protobuf.text_format import Merge
    from enum import Enum
    import math

    net = caffe_pb2.NetParameter()
    Merge((open(path, 'r').read()), net)

    class LayerType(Enum):
        none = 0
        input = 1
        conv = 2
        fc = 3
        lrn = 4
        pmax = 5
        pave = 6
        sft = 7
        concat = 8

    class GeneralLayer(object):
        type = LayerType.none
        in_channels = 0
        out_channels = 0
        out_width = 0
        out_height = 0
        name = ''
        bottom = ''
        top = ''
        layer_param = caffe_pb2.LayerParameter()
        param_txt = ''
        # encoded = False

    # relu_list[string] stores names of layers who use ReLU as its neuron
    relu_list = []
    # layers_list[GeneralLayer] stores all layers that will be used in the App
    # layers like "Dropout" should be excluded
    layers_list = []
    # concat_name_dict{string(name of the concat layer) :
    #  list[string](a list of names of bottom layers of the concat layer)}
    concat_name_dict = {}
    # concat_layer_dict{string(name of the concat layer) :
    #  list[GeneralLayer](a list of bottom layers of the concat layer)}
    concat_layer_dict = {}
    # concat_offset_dict{string(name of the concat layer) :
    #  list[int](a list of destinationFeatureChannelOffset that will be used by
    #  the bottom layers of the concat layer; first element is 0, last element
    #  is equal to the output channels number of the concat layer, while the
    #  number of elements equals the number of bottom layers plus 1 )}
    concat_offset_dict = {}
    # concat_parent_dict{string(name of a layer that serves as the bottom of
    #  a concat layer) : string(name of its top layer, of type concat)}
    concat_parent_dict = {}

# load params that cannot be found in [layer_type]_param
    for layer in net.layer:
        new_layer = GeneralLayer()

        if layer.type == 'Input':
            new_layer.type = LayerType.input
            new_layer.in_channels = layer.input_param.shape[0].dim[1]
            new_layer.out_channels = layer.input_param.shape[0].dim[1]
            new_layer.out_width = layer.input_param.shape[0].dim[2]
            new_layer.out_height = layer.input_param.shape[0].dim[3]

        elif layer.type == 'Convolution':
            new_layer.type = LayerType.conv
            new_layer.out_channels = layer.convolution_param.num_output

        elif layer.type == 'InnerProduct':
            new_layer.type = LayerType.fc
            new_layer.out_channels = layer.inner_product_param.num_output
            new_layer.out_width = 1
            new_layer.out_height = 1

        elif layer.type == 'LRN':
            new_layer.type = LayerType.lrn

        elif layer.type == 'Pooling' and layer.pooling_param.pool == 0:
            new_layer.type = LayerType.pmax

        elif layer.type == 'Pooling' and layer.pooling_param.pool == 1:
            new_layer.type = LayerType.pave

        elif layer.type == 'Softmax':
            new_layer.type = LayerType.sft
            new_layer.out_width = 1
            new_layer.out_height = 1

        elif layer.type == 'Concat':
            new_layer.type = LayerType.concat
            concat_name_dict[layer.name] = layer.bottom
            concat_layer_dict[layer.name] = []
            concat_offset_dict[layer.name] = [0]

        elif layer.type == 'ReLU':
            relu_list.append(layer.bottom[0])

        elif layer.type == 'Dropout':
            pass

        else:
            print 'Unsupported layer: ' + layer.name + ', of type: ' + layer.type

        if new_layer.type != LayerType.none:
            new_layer.name = layer.name
            new_layer.top = layer.top[0]
            new_layer.layer_param = layer

            if new_layer.type != LayerType.input:
                new_layer.bottom = layer.bottom[0]

            layers_list.append(new_layer)

# find bottom layers of concat layers
    for layer0 in layers_list:
        if layer0.type == LayerType.concat:
            for name in concat_name_dict[layer0.name]:
                for layer1 in layers_list:
                    if layer1.name == name:
                        concat_layer_dict[layer0.name].append(layer1)
                        concat_parent_dict[layer1.name] = layer0.name
                        break

# complete params by reading the context of the layer
    should_iterate = True
    while should_iterate:
        should_iterate = False

        for layer0 in layers_list:
            if layer0.in_channels == 0:
                should_iterate = True

                if layer0.type == LayerType.concat:
                    in_channels = 0
                    for layer1 in concat_layer_dict[layer0.name]:
                        if layer1.out_channels == 0:
                            in_channels = 0
                            break
                        else:
                            in_channels += layer1.out_channels
                    layer0.in_channels = in_channels
                    layer0.out_width = (in_channels != 0 and [concat_layer_dict[layer0.name][0].out_width] or [0])[0]
                    layer0.out_height = (in_channels != 0 and [concat_layer_dict[layer0.name][0].out_height] or [0])[0]
                else:
                    for layer1 in layers_list:
                        if layer1.name == layer0.bottom:
                            layer0.in_channels = layer1.out_channels

                            if layer0.type == LayerType.conv:
                                param = layer0.layer_param.convolution_param
                                stride = (param.stride and [param.stride[0]] or [1])[0]
                                pad = (param.pad and [param.pad[0]] or [0])[0]
                                layer0.out_width = \
                                    int(math.floor(float(layer1.out_width - param.kernel_size[0] + 2 * pad)
                                                   / stride)) + 1
                                layer0.out_height = \
                                    int(math.floor(float(layer1.out_height - param.kernel_size[0] + 2 * pad)
                                                   / stride)) + 1

                            elif layer0.type == LayerType.lrn:
                                layer0.out_width = layer1.out_width
                                layer0.out_height = layer1.out_height

                            elif layer0.type == LayerType.pmax:
                                param = layer0.layer_param.pooling_param
                                stride = param.stride or 1
                                pad = param.pad or 0

                                layer0.out_width = \
                                    int(math.ceil(float(layer1.out_width - param.kernel_size + 2 * pad) / stride)) + 1
                                layer0.out_height = \
                                    int(math.ceil(float(layer1.out_height - param.kernel_size + 2 * pad) / stride)) + 1

                            elif layer0.type == LayerType.pave:
                                param = layer0.layer_param.pooling_param
                                stride = param.stride or 1
                                pad = param.pad or 0

                                if param.global_pooling:
                                    layer0.out_width = 1
                                    layer0.out_height = 1
                                else:
                                    layer0.out_width = \
                                        int(math.ceil(float(layer1.out_width - param.kernel_size + 2 * pad)
                                                      / stride)) + 1
                                    layer0.out_height = \
                                        int(math.ceil(float(layer1.out_height - param.kernel_size + 2 * pad)
                                                      / stride)) + 1

                            break

                if layer0.type == LayerType.pmax or layer0.type == LayerType.lrn or layer0.type == LayerType.sft\
                        or layer0.type == LayerType.pave or layer0.type == LayerType.concat:
                    layer0.out_channels = layer0.in_channels

    # find offsets of concat layers
    for layer0 in layers_list:
        if layer0.type == LayerType.concat:
            for layer1 in concat_layer_dict[layer0.name]:
                concat_offset_dict[layer0.name].append(layer1.out_channels + concat_offset_dict[layer0.name][-1])

# figure out the sequence of encoding
    encode_list = []

    for layer in layers_list:
        if layer.type != LayerType.none and layer.type != LayerType.input:
            encode_list.append(layer)

# need this when the prototxt is not sequenced according to the encoding sequence
    # waiting_list = []
    #
    # for layer in layers_list:
    #     if layer.type == LayerType.sft:
    #         layer.encoded = True
    #         encode_list.append(layer)
    #         waiting_list.append(layer)
    #
    # while len(encode_list) != len(layers_list):
    #     new_waiting_list = []
    #
    #     for layer0 in waiting_list:
    #         flag = 0
    #
    #         for layer1 in layers_list:
    #             if layer1 != layer0 and layer1.bottom == layer0.bottom and ~layer1.encoded:
    #                 new_waiting_list.append(layer0)
    #                 flag = -1
    #                 break
    #
    #         if flag == 0:
    #             for layer2 in layers_list:
    #                 if layer2.name == layer0.bottom:
    #                     encode_list.insert(0, layer2)
    #                     layer2.encoded = True
    #                     new_waiting_list.append(layer2)
    #                     break
    #
    #     waiting_list = new_waiting_list
    #
    # del encode_list[0]  # don't need data layer

# generate OC code
    concat_txt = ''
    add_layer_txt = ''
    encode_txt = ''

    print '- (void)initLayers {'

    if len(relu_list) != 0:
        print 'MPSCNNNeuronReLU *relu = [[MPSCNNNeuronReLU alloc] initWithDevice:self.device a:0];' + '\n'

    for layer in encode_list:
        show_name = modify_name(layer.name)

        if layer.name not in concat_parent_dict.keys():
            layer.param_txt += 'MPSImageDescriptor *' + show_name + \
                               '_id = [MPSImageDescriptor imageDescriptorWithChannelFormat:self.textureFormat\n' + \
                               'width:' + str(layer.out_width) + '\n' + 'height:' + str(layer.out_height) + '\n' + \
                               'featureChannels:' + str(layer.out_channels) + '];\n\n'

        # used to set the readCount of the
        as_bottom = 0
        for top_layer in encode_list:
            if top_layer.type == LayerType.concat and layer.name in concat_name_dict[top_layer.name]:
                as_bottom += 1
            elif top_layer.bottom == layer.name:
                as_bottom += 1

        offset = 0
        if layer.name in concat_parent_dict.keys():
            idx = 0
            for name in concat_name_dict[concat_parent_dict[layer.name]]:
                if name == layer.name:
                    break
                idx += 1
            offset = concat_offset_dict[concat_parent_dict[layer.name]][idx]

        if layer.type == LayerType.conv:
            param = layer.layer_param.convolution_param

            layer.param_txt += 'SlimMPSCNNConvolution *' + show_name + '_kernel' + \
                               '  = [[SlimMPSCNNConvolution alloc] ' + \
                               'initWithKernelWidth:' + str(param.kernel_size[0]) + '\n' + \
                               'kernelHeight:' + str(param.kernel_size[0]) + '\n' + \
                               'inputFeatureChannels:' + str(layer.in_channels) + '\n' + \
                               'outputFeatureChannels:' + str(layer.out_channels) + '\n' + \
                               'neuron:' + (layer.name in relu_list and 'relu' or 'nil') + '\n' + \
                               'device:self.device' + '\n' + \
                               'weights:[self weights_' + show_name + ']' + '\n' + \
                               'bias:[self bias_' + show_name + ']' + '\n' + \
                               'willPad:' + (param.pad and 'YES' or 'NO') + '\n' + \
                               'strideX:' + (param.stride and [str(param.stride[0])] or ['1'])[0] + '\n' + \
                               'strideY:' + (param.stride and [str(param.stride[0])] or ['1'])[0] + '\n' + \
                               'destinationFeatureChannelOffset:' + str(offset) + '\n' + \
                               'group:' + str(param.group) + '];' + '\n\n'

        elif layer.type == LayerType.fc:
            kernel_width = 0
            kernel_height = 0
            for previous_layer in layers_list:
                if previous_layer.name == layer.bottom:
                    kernel_width = previous_layer.out_width
                    kernel_height = previous_layer.out_height
            layer.param_txt += 'SlimMPSCNNFullyConnected *' + show_name + '_kernel' + \
                               ' = [[SlimMPSCNNFullyConnected alloc] ' + \
                               'initWithKernelWidth:' + str(kernel_width) + '\n' + \
                               'kernelHeight:' + str(kernel_height) + '\n' + \
                               'inputFeatureChannels:' + str(layer.in_channels) + '\n' + \
                               'outputFeatureChannels:' + str(layer.out_channels) + '\n' + \
                               'neuron:' + (layer.name in relu_list and 'relu' or 'nil') + '\n' + \
                               'device:self.device' + '\n' + \
                               'weights:[self weights_' + show_name + ']' + '\n' + \
                               'bias:[self bias_' + show_name + ']' + '\n' + \
                               'destinationFeatureChannelOffset:' + str(offset) + '];' + '\n\n'

        elif layer.type == LayerType.pmax:
            param = layer.layer_param.pooling_param
            layer.param_txt += 'SlimMPSCNNPoolingMax *' + show_name + '_kernel' + \
                               ' = [[SlimMPSCNNPoolingMax alloc] initWithDevice:self.device' + '\n' + \
                               'kernelWidth:' + str(param.kernel_size) + '\n' + \
                               'kernelHeight:' + str(param.kernel_size) + '\n' + \
                               'strideInPixelsX:' + str(param.stride) + '\n' + \
                               'strideInPixelsY:' + str(param.stride) + '\n' + \
                               'willPad:' + (param.pad and 'YES' or 'NO') + '];' + '\n\n'

        elif layer.type == LayerType.pave:
            param = layer.layer_param.pooling_param

            if param.global_pooling:
                in_width = 0
                in_height = 0
                for bottom_layer in encode_list:
                    if bottom_layer.name == layer.bottom:
                        in_width = bottom_layer.out_width
                        in_height = bottom_layer.out_height
                        break

                layer.param_txt += 'SlimMPSCNNPoolingGlobalAverage *' + show_name + '_kernel' + \
                                   ' = [[SlimMPSCNNPoolingGlobalAverage alloc] initWithDevice:self.device' + '\n' + \
                                   'kernelWidth:' + (in_width % 2 and str(in_width) or str(in_width+1)) + '\n' + \
                                   'kernelHeight:' + (in_height % 2 and str(in_height) or str(in_height+1)) + '];\n\n'

            else:
                layer.param_txt += 'MPSCNNPoolingAverage *' + show_name + '_kernel' + \
                                   ' = [[MPSCNNPoolingAverage alloc] initWithDevice:self.device' + '\n' + \
                                   'kernelWidth:' + str(param.kernel_size) + '\n' + \
                                   'kernelHeight:' + str(param.kernel_size) + '\n' + \
                                   'strideInPixelsX:' + str(param.stride) + '\n' + \
                                   'strideInPixelsY:' + str(param.stride) + '];' + '\n\n'

        elif layer.type == LayerType.lrn:
            param = layer.layer_param.lrn_param
            layer.param_txt += 'SlimMPSCNNLocalResponseNormalization *' + show_name + '_kernel' + \
                               ' = [[SlimMPSCNNLocalResponseNormalization alloc] initWithDevice:self.device' + '\n' + \
                               'localSize:' + str(param.local_size) + '\n' + \
                               'alpha:' + str(param.alpha) + '\n' + \
                               'beta:' + str(param.beta) + '];' + '\n\n'

        elif layer.type == LayerType.sft:
            layer.param_txt += 'MPSImage *' + show_name + '_image' + \
                               ' = [[MPSImage alloc] initWithDevice:self.device ' + '\n' + \
                               'imageDescriptor:' + show_name + '_id];' + '\n\n' + \
                               'MPSCNNSoftMax *' + show_name + '_kernel' + \
                               ' = [[MPSCNNSoftMax alloc] initWithDevice:self.device];' + '\n\n'

        if layer.name in concat_parent_dict.keys():
            layer.param_txt += 'GeneralLayer *' + show_name + \
                               '_layer = [[GeneralLayer alloc] initWithImageDescriptor:nil' + '\n' + \
                               'readCount:0' + '\n' + \
                               'outputImage:nil' + '\n' + \
                               'kernel:' + show_name + '_kernel' + '];\n'
        elif layer.type == LayerType.sft:
            layer.param_txt += 'GeneralLayer *' + show_name + \
                               '_layer = [[GeneralLayer alloc] initWithImageDescriptor:nil' + '\n' + \
                               'readCount:0' + '\n' + \
                               'outputImage:' + show_name + '_image' + '\n' + \
                               'kernel:' + show_name + '_kernel' + '];\n'
        elif layer.type == LayerType.concat:
            as_bottom = 0
            for top_layer in encode_list:
                if top_layer.type == LayerType.concat and layer.name in concat_name_dict[top_layer.name]:
                    as_bottom += 1
                elif top_layer.bottom == layer.name:
                    as_bottom += 1

            layer.param_txt += 'GeneralLayer *' + show_name + \
                               '_layer = [[GeneralLayer alloc] initWithImageDescriptor:' + show_name + '_id' + '\n' + \
                               'readCount:' + str(as_bottom) + '\n' + \
                               'outputImage:nil' + '\n' + \
                               'kernel:nil];' + '\n'
        else:
            layer.param_txt += 'GeneralLayer *' + show_name + \
                               '_layer = [[GeneralLayer alloc] initWithImageDescriptor:' + show_name + '_id' + '\n' + \
                               'readCount:' + str(as_bottom) + '\n' + \
                               'outputImage:nil' + '\n' + \
                               'kernel:' + show_name + '_kernel' + '];\n'

        add_layer_txt += '[self addLayer:' + show_name + '_layer];' + '\n'

        if layer.type == LayerType.concat:
            for name in concat_name_dict[layer.name]:
                encode_txt += '[self.encodeList addObject:@[' + modify_name(name) + '_layer, ' + \
                              modify_name(layer.name) + '_layer]];' + '\n'
        elif layer.bottom != 'data':
            encode_txt += '[self.encodeList addObject:@[' + modify_name(layer.bottom) + '_layer, ' + \
                               modify_name(layer.name) + '_layer]];' + '\n'

    for layer in encode_list:
        print layer.param_txt
        if layer.name in concat_parent_dict.keys():
            concat_txt += modify_name(layer.name) + '_layer.concatLayer = ' + \
                          modify_name(concat_parent_dict[layer.name]) + '_layer;' + '\n'

    print concat_txt
    print add_layer_txt
    print encode_txt[:-1]

    print '}'
    
    
def generate_mid_result():
    import numpy as np
    import os, sys
    import h5py
    
    sys.path.append('C:/Users/wei/Downloads/caffe/python/')
    import caffe
    
    if __name__ == '__main__':
    
    	caffe.set_mode_cpu()
    
    	proto = os.getcwd() + '/alexnet.prototxt'
    	model = os.getcwd() + '/alexnet.caffemodel'
    	image = os.getcwd() + '/test.jpg'
    
    	net = caffe.Net(proto, model, caffe.TEST)
    
    	print '\n\nLoaded network {:s}'.format(model)
    
    	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    	transformer.set_transpose('data', (2,0,1))
    	transformer.set_mean('data', np.array([120,120,120]))
    	transformer.set_raw_scale('data', 255)
    	transformer.set_channel_swap('data', (2,1,0))
    	
    	img = caffe.io.load_image(image)
    	net.blobs['data'].data[...] = transformer.preprocess('data',img)
    	net.forward()
    
    	print 'Feature map name wrote to feature_map_name.txt.'
    
    	with open('feature_map_name.txt','wb') as f:
    		blob_keys = net.blobs.keys()
    		for i in range(len(blob_keys)):
    		    f.write(str(i) + ' ' + blob_keys[i] + '\r\n')
    
    	print 'Feature map matrix wrote to final_feature.h5.'
    
    	with h5py.File('final_feature.h5') as f:
    		for i,blob_name in enumerate(net.blobs.keys()):
    			print i,blob_name,net.blobs[blob_name].data.dtype
    			f.create_dataset(str(i),data = net.blobs[blob_name].data)


def get_mid_result():
    import h5py
    import numpy as np

    with h5py.File('final_feature.h5') as f:
        print f.keys()
        result = f[u'54']  # [0].transpose((1, 2, 0))
        print np.sum(np.abs(result))
        print np.sum(np.square(result))
        i = 0
        for item in result[0][:1000]:
            print i, item
            i += 1


def get_labels(path):
    f = open(path, 'r')
    words = 'static const NSString *labels[] = {\n'

    while True:
        line = f.readline()
        if not line:
            break
        words += '@"' + line[:-1] + '",\n'

    words = words[:-2]
    words += '};'
    print words


def test_layer(path):
    import caffe_pb2
    from google.protobuf.text_format import Merge

    net = caffe_pb2.NetParameter()
    Merge((open(path, 'r').read()), net)

    concat_dict = {}

    for layer0 in net.layer:
        if layer0.type == 'Concat':
            concat_dict[layer0.name] = []

            for layer1 in net.layer:
                if layer1.name in layer0.bottom:
                    concat_dict[layer0.name].append(layer1)

    print concat_dict


def modify_name(old):
    new = []
    for char in old:
        if char != '/':
            new.append(char)
        else:
            new.append('_')
    return ''.join(new)


if __name__ == '__main__':
    generate_param_txt('squeezenet.prototxt')
#     generate_mid_result()
#     get_mid_result()
#     get_labels('synset_words.txt')
#     test_layer('googlenet.prototxt')
