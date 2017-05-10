def generate_param_txt(prototxt_path, labels_path, json_file_name):
    import caffe_pb2
    from google.protobuf.text_format import Merge
    from enum import Enum
    import math
    import json

    net = caffe_pb2.NetParameter()
    Merge((open(prototxt_path, 'r').read()), net)

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
        in_channel = 0
        out_channel = 0
        out_size = 0
        name = ''
        bottom = ''
        top = ''
        layer_param = caffe_pb2.LayerParameter()
        param_dict = {}
        param_txt = ''

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
            new_layer.in_channel = layer.input_param.shape[0].dim[1]
            new_layer.out_channel = layer.input_param.shape[0].dim[1]
            new_layer.out_size = layer.input_param.shape[0].dim[2]

        elif layer.type == 'Convolution':
            new_layer.type = LayerType.conv
            new_layer.out_channel = layer.convolution_param.num_output

        elif layer.type == 'InnerProduct':
            new_layer.type = LayerType.fc
            new_layer.out_channel = layer.inner_product_param.num_output
            new_layer.out_size = 1

        elif layer.type == 'LRN':
            new_layer.type = LayerType.lrn

        elif layer.type == 'Pooling' and layer.pooling_param.pool == 0:
            new_layer.type = LayerType.pmax

        elif layer.type == 'Pooling' and layer.pooling_param.pool == 1:
            new_layer.type = LayerType.pave

        elif layer.type == 'Softmax':
            new_layer.type = LayerType.sft
            new_layer.out_size = 1

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
            exit(-1)

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
            if layer0.in_channel == 0:
                should_iterate = True

                if layer0.type == LayerType.concat:
                    in_channel = 0
                    for layer1 in concat_layer_dict[layer0.name]:
                        if layer1.out_channel == 0:
                            in_channel = 0
                            break
                        else:
                            in_channel += layer1.out_channel
                    layer0.in_channel = in_channel
                    layer0.out_size = (in_channel != 0 and [concat_layer_dict[layer0.name][0].out_size] or [0])[0]

                else:
                    for layer1 in layers_list:
                        if layer1.name == layer0.bottom:
                            layer0.in_channel = layer1.out_channel

                            if layer0.type == LayerType.conv:
                                param = layer0.layer_param.convolution_param
                                stride = (param.stride and [param.stride[0]] or [1])[0]
                                pad = (param.pad and [param.pad[0]] or [0])[0]
                                layer0.out_size = \
                                    int(math.floor(float(layer1.out_size - param.kernel_size[0] + 2 * pad)
                                                   / stride)) + 1

                            elif layer0.type == LayerType.lrn:
                                layer0.out_size = layer1.out_size

                            elif layer0.type == LayerType.pmax:
                                param = layer0.layer_param.pooling_param
                                stride = param.stride or 1
                                pad = param.pad or 0

                                layer0.out_size = \
                                    int(math.ceil(float(layer1.out_size - param.kernel_size + 2 * pad) / stride)) + 1

                            elif layer0.type == LayerType.pave:
                                param = layer0.layer_param.pooling_param
                                stride = param.stride or 1
                                pad = param.pad or 0

                                if param.global_pooling:
                                    layer0.out_size = 1
                                else:
                                    layer0.out_size = \
                                        int(math.ceil(float(layer1.out_size - param.kernel_size + 2 * pad)
                                                      / stride)) + 1

                            break

                if layer0.type == LayerType.pmax or layer0.type == LayerType.lrn or layer0.type == LayerType.sft\
                        or layer0.type == LayerType.pave or layer0.type == LayerType.concat:
                    layer0.out_channel = layer0.in_channel

    # find offsets of concat layers
    for layer0 in layers_list:
        if layer0.type == LayerType.concat:
            for layer1 in concat_layer_dict[layer0.name]:
                concat_offset_dict[layer0.name].append(layer1.out_channel + concat_offset_dict[layer0.name][-1])

# generate JSON
# data layer will be excluded
    json_dict = {}

    data_file_offset = 0
    layer_info = []
    encode_seq = []

    for layer in layers_list[1:]:

        # used to set the readCount
        as_bottom = 0
        for top_layer in layers_list[1:]:
            if top_layer.type == LayerType.concat and layer.name in concat_name_dict[top_layer.name]:
                as_bottom += 1
            elif top_layer.bottom == layer.name:
                as_bottom += 1

        # used to set the destinationFeatureChannelOffset
        offset = 0
        if layer.name in concat_parent_dict.keys():
            idx = 0
            for name in concat_name_dict[concat_parent_dict[layer.name]]:
                if name == layer.name:
                    break
                idx += 1
            offset = concat_offset_dict[concat_parent_dict[layer.name]][idx]

        # layer-specific parameters
        if layer.type == LayerType.conv:
            param = layer.layer_param.convolution_param

            weight_size = layer.in_channel * layer.out_channel * \
                          param.kernel_size[0] * param.kernel_size[0] / param.group
            bias_size = layer.out_channel

            layer.param_dict = {'type': 'Convolution',
                                'kernel_size': param.kernel_size[0],
                                'weight_offset': data_file_offset,
                                'bias_offset': data_file_offset + weight_size,
                                'input_channel': layer.in_channel,
                                'stride': (param.stride and param.stride[0] or 1),
                                'destination_channel_offset': offset,
                                'group': param.group,
                                'activation': (layer.name in relu_list and 'ReLU' or 'Identity'),
                                'pad': (param.pad and param.pad[0] or 0)
                                }
            data_file_offset += (weight_size + bias_size)

        elif layer.type == LayerType.fc:
            kernel_size = 0
            for previous_layer in layers_list:
                if previous_layer.name == layer.bottom:
                    kernel_size = previous_layer.out_size

            weight_size = layer.in_channel * layer.out_channel * kernel_size * kernel_size
            bias_size = layer.out_channel

            layer.param_dict = {'type': 'FullyConnected',
                                'kernel_size': kernel_size,
                                'weight_offset': data_file_offset,
                                'bias_offset': data_file_offset + weight_size,
                                'input_channel': layer.in_channel,
                                'destination_channel_offset': offset,
                                'activation': (layer.name in relu_list and 'ReLU' or 'Identity'),
                                }
            data_file_offset += (weight_size + bias_size)

        elif layer.type == LayerType.pmax:
            param = layer.layer_param.pooling_param

            layer.param_dict = {'type': 'PoolingMax',
                                'kernel_size': param.kernel_size,
                                'stride': param.stride,
                                'pad': (param.pad or 0)
                                }

        elif layer.type == LayerType.pave:
            param = layer.layer_param.pooling_param

            if param.global_pooling:
                in_size = 0
                for bottom_layer in layers_list:
                    if bottom_layer.name == layer.bottom:
                        in_size = bottom_layer.out_size
                        break

                layer.param_dict = {'type': 'PoolingAverage',
                                    'global': True,
                                    'kernel_size': in_size
                                    }

            else:
                layer.param_dict = {'type': 'PoolingAverage',
                                    'global': False,
                                    'kernel_size': param.kernel_size,
                                    'stride': param.stride
                                    }

        elif layer.type == LayerType.lrn:
            param = layer.layer_param.lrn_param

            layer.param_dict = {'type': 'LocalResponseNormalization',
                                'local_size': param.local_size,
                                'alpha': param.alpha,
                                'beta': param.beta
                                }

        elif layer.type == LayerType.sft:
            layer.param_dict = {'type': 'SoftMax',
                                }

        elif layer.type == LayerType.concat:
            layer.param_dict = {'type': 'Concat',
                                'bottom_layer': [name for name in concat_name_dict[layer.name]]
                                }

        # shared parameters
        layer.param_dict['name'] = layer.name
        layer.param_dict['output_channel'] = layer.out_channel
        layer.param_dict['output_size'] = layer.out_size

        # for creating MPSImage
        if layer.name in concat_parent_dict.keys():
            layer.param_dict['image_type'] = 'None'
            layer.param_dict['concat_layer_name'] = concat_parent_dict[layer.name]
        elif layer.type == LayerType.sft:
            layer.param_dict['image_type'] = 'Permanent'
        else:
            layer.param_dict['image_type'] = 'Temporary'
            layer.param_dict['read_count'] = as_bottom

        layer_info.append(layer.param_dict)

        # [kernel, src, dst]
        if layer.type != LayerType.concat:
            if layer.name in concat_parent_dict.keys():
                encode_seq.append([layer.name, layer.bottom, concat_parent_dict[layer.name]])
            else:
                encode_seq.append([layer.name, layer.bottom, layer.name])

    json_dict['layer_info'] = layer_info
    json_dict['encode_seq'] = encode_seq[1:]    # encoding from outer source should be handled elsewhere

    # get labels
    with open(labels_path, 'r') as f:
        labels = []

        while True:
            line = f.readline()
            if not line:
                break
            labels.append(line[:-1])

        json_dict['labels'] = labels

    # get input & output info
    inout_info = {'input_size': layers_list[0].out_size,
                  'input_channel': layers_list[0].out_channel,
                  'output_channel': layers_list[-1].out_channel,
                  'file_size': data_file_offset * 4,
                  'first_layer': layers_list[1].name,
                  'last_layer': layers_list[-1].name
                  }
    json_dict['inout_info'] = inout_info

    # dump into JSON file
    with open(json_file_name+'.json', 'wb') as s:
        s.write(json.dumps(json_dict))

    print 'Finished.'


if __name__ == '__main__':
    import sys

    arg = sys.argv[1:]
    if len(arg) != 3:
        print 'usage: prototxt_path labels_path json_filename'
        exit(-1)
    generate_param_txt(arg[0], arg[1], arg[2])
