def convert(src_path):
	import caffe_pb2
	from google.protobuf.text_format import Merge
	from enum import Enum
	import math
	
	net = caffe_pb2.NetParameter()
	Merge((open(src_path, 'r').read()), net)
	
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
	itr = 0
	while should_iterate:
	    if itr >= 10000:
	        print 'too many iterations'
	        exit(-1)
	    itr += 1
		
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
	
	# generate OC code
	concat_txt = ''
	add_layer_txt = ''
	encode_txt = ''
	
	print '- (void)initLayers {'
	
	if len(relu_list) != 0:
	    print 'MPSCNNNeuronReLU *relu = [[MPSCNNNeuronReLU alloc] initWithDevice:self.device a:0];\n'
	
	for layer in encode_list:
	    show_name = modify_name(layer.name)
	
	    if layer.name not in concat_parent_dict.keys():
	        layer.param_txt += '\n'.join(['MPSImageDescriptor *%s_id = [MPSImageDescriptor \
imageDescriptorWithChannelFormat:self.textureFormat' % show_name,
	                                      'width:%d' % layer.out_width,
	                                      'height:%d' % layer.out_height,
	                                      'featureChannels:%d];\n\n' % layer.out_channels])
	
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
	
	        layer.param_txt += '\n'.join(['SlimMPSCNNConvolution *%s_kernel = [[SlimMPSCNNConvolution alloc] \
initWithKernelWidth:%d' % (show_name, param.kernel_size[0]),
	                                      'kernelHeight:%d' % param.kernel_size[0],
	                                      'inputFeatureChannels:%d' % layer.in_channels,
	                                      'outputFeatureChannels:%d' % layer.out_channels,
	                                      'neuron:%s' % (layer.name in relu_list and 'relu' or 'nil'),
	                                      'device:self.device',
	                                      'weights:[self weights_%s]' % show_name,
	                                      'bias:[self bias_%s]' % show_name,
	                                      'willPad:%s' % (param.pad and 'YES' or 'NO'),
	                                      'strideX:%d\nstrideY:%d'
	                                      % (param.stride and (param.stride[0],) * 2 or (1, 1)),
	                                      'destinationFeatureChannelOffset:%d' % offset,
	                                      'group:%d];\n\n' % param.group])
	
	    elif layer.type == LayerType.fc:
	        kernel_width = 0
	        kernel_height = 0
	        for previous_layer in layers_list:
	            if previous_layer.name == layer.bottom:
	                kernel_width = previous_layer.out_width
	                kernel_height = previous_layer.out_height
	
	        layer.param_txt += '\n'.join(['SlimMPSCNNFullyConnected *%s_kernel = [[SlimMPSCNNFullyConnected alloc] \
initWithKernelWidth:%d' % (show_name, kernel_width),
	                                      'kernelHeight:%d' % kernel_height,
	                                      'inputFeatureChannels:%d' % layer.in_channels,
	                                      'outputFeatureChannels:%d' % layer.out_channels,
	                                      'neuron:%s' % (layer.name in relu_list and 'relu' or 'nil'),
	                                      'device:self.device',
	                                      'weights:[self weights_%s]' % show_name,
	                                      'bias:[self bias_%s]' % show_name,
	                                      'destinationFeatureChannelOffset:%d];\n\n' % offset])
	
	    elif layer.type == LayerType.pmax:
	        param = layer.layer_param.pooling_param
	
	        layer.param_txt += '\n'.join(['SlimMPSCNNPoolingMax *%s_kernel = \
[[SlimMPSCNNPoolingMax alloc] initWithDevice:self.device' % show_name,
	                                      'kernelWidth:%d\nkernelHeight:%d'
	                                      % ((param.kernel_size,) * 2),
	                                      'strideInPixelsX:%d\nstrideInPixelsY:%d'
	                                      % ((param.stride,) * 2),
	                                      'willPad:%s];\n\n' % (param.pad and 'YES' or 'NO')])
	
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
	
	            layer.param_txt += '\n'.join(['SlimMPSCNNPoolingGlobalAverage *%s_kernel = \
[[SlimMPSCNNPoolingGlobalAverage alloc] initWithDevice:self.device' % show_name,
	                                          'kernelWidth:%d' % (in_width % 2 and in_width or in_width+1),
	                                          'kernelHeight:%d];\n\n' % (in_height % 2 and in_height or in_height + 1)])
	
	        else:
	            layer.param_txt += '\n'.join(['MPSCNNPoolingAverage *%s_kernel = \
[[MPSCNNPoolingAverage alloc] initWithDevice:self.device' % show_name,
	                                          'kernelWidth:%d\nkernelHeight:%d' % ((param.kernel_size,) * 2),
	                                          'strideInPixelsX:%d\nstrideInPixelsY:%d];\n\n'
	                                          % ((param.stride,) * 2)])
	
	    elif layer.type == LayerType.lrn:
	        param = layer.layer_param.lrn_param
	
	        layer.param_txt += '\n'.join(['SlimMPSCNNLocalResponseNormalization *%s_kernel = \
[[SlimMPSCNNLocalResponseNormalization alloc] initWithDevice:self.device' % show_name,
	                                      'localSize:%d' % param.local_size,
	                                      'alpha:%f' % param.alpha,
	                                      'beta:%f];\n\n' % param.beta])
	
	    elif layer.type == LayerType.sft:
	        layer.param_txt += '\n'.join(['MPSImage *%s_image = [[MPSImage alloc] initWithDevice:self.device'
	                                      % show_name,
	                                     'imageDescriptor:%s_id];\n' % show_name,
	                                     'MPSCNNSoftMax *%s_kernel = [[MPSCNNSoftMax alloc] \
initWithDevice:self.device];\n\n' % show_name])
	
	    if layer.name in concat_parent_dict.keys():
	        layer.param_txt += '\n'.join(['GeneralLayer *%s_layer = [[GeneralLayer alloc] initWithImageDescriptor:nil'
	                                      % show_name,
	                                      'readCount:0',
	                                      'outputImage:nil',
	                                      'kernel:%s_kernel];\n\n' % show_name])
	
	    elif layer.type == LayerType.sft:
	        layer.param_txt += '\n'.join(['GeneralLayer *%s_layer = [[GeneralLayer alloc] initWithImageDescriptor:nil'
	                                      % show_name,
	                                      'readCount:0',
	                                      'outputImage:%s_image' % show_name,
	                                      'kernel:%s_kernel];\n\n' % show_name])
	
	    elif layer.type == LayerType.concat:
	        as_bottom = 0
	        for top_layer in encode_list:
	            if top_layer.type == LayerType.concat and layer.name in concat_name_dict[top_layer.name]:
	                as_bottom += 1
	            elif top_layer.bottom == layer.name:
	                as_bottom += 1
	
	        layer.param_txt += '\n'.join(['GeneralLayer *%s_layer = [[GeneralLayer alloc] \
initWithImageDescriptor:%s_id' % ((show_name,) * 2),
	                                      'readCount:%d' % as_bottom,
	                                      'outputImage:nil',
	                                      'kernel:nil];\n\n'])
	
	    else:
	        layer.param_txt += '\n'.join(['GeneralLayer *%s_layer = [[GeneralLayer alloc] \
initWithImageDescriptor:%s_id' % ((show_name,) * 2),
	                                      'readCount:%d' % as_bottom,
	                                      'outputImage:nil',
	                                      'kernel:%s_kernel];\n\n' % show_name])
	
	    add_layer_txt += '[self addLayer:%s_layer];\n' % show_name
	
	    if layer.type == LayerType.concat:
	        for name in concat_name_dict[layer.name]:
	            encode_txt += '[self.encodeList addObject:@[%s_layer, %s_layer]];\n' % (modify_name(name), show_name)
	
	    elif layer.bottom != 'data':
	        encode_txt += '[self.encodeList addObject:@[%s_layer, %s_layer]];\n' % (modify_name(layer.bottom), show_name)
	
	for layer in encode_list:
	    print layer.param_txt[:-1]
	    if layer.name in concat_parent_dict.keys():
	        concat_txt += '%s_layer.concatLayer = %s_layer;\n' \
	                      % (modify_name(layer.name), modify_name(concat_parent_dict[layer.name]))
	
	if concat_txt:print concat_txt
	if add_layer_txt:print add_layer_txt
	if encode_txt:print encode_txt[:-1]
	print '}'
	
	
def modify_name(old):
    new = []
    for char in old:
        if char != '/':
            new.append(char)
        else:
            new.append('_')
    return ''.join(new)
	

if __name__ == '__main__':
	import sys
	
	arg = sys.argv[1:]
	if len(arg) != 1:
		print 'too more arguments'
		exit(-1)
	convert(arg[0])
