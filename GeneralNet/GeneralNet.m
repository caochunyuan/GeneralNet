//
//  GeneralNet.m
//  GeneralNet
//
//  Created by Lun on 2017/5/9.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import "GeneralNet.h"

#pragma mark - if use Metal
#if USE_METAL

@implementation GeneralLayer

- (instancetype)initWithName:(NSString *)name
             ImageDescriptor:(MPSImageDescriptor *)imageDescritor
                   readCount:(NSUInteger)readCount
                 outputImage:(MPSImage *)outputImage
                      kernel:(MPSCNNKernel *)kernel {
    if (self = [super init]) {
        _name = name;
        _imageDescriptor = imageDescritor;
        _outputImage = outputImage;
        _readCount = readCount;
        _kernel = kernel;
    }
    
    return self;
}

@end

@implementation GeneralNet

static const uint textureFormat = MPSImageFeatureChannelFormatFloat16;

- (instancetype)initWithDescriptionFile:(NSString *)descriptionFile
                               dataFile:(NSString *)dataFile {
    if (self = [super init]) {
        // preparation for Metal
        _device = MTLCreateSystemDefaultDevice();
        
        NSAssert(MPSSupportsMTLDevice(_device), @"Metal Performance Shaders not supported on current device");
        
        _commandQueue = [_device newCommandQueue];
        _textureLoader = [[MTKTextureLoader alloc] initWithDevice:_device];
        _lanczos = [[MPSImageLanczosScale alloc] initWithDevice:_device];
        
        id <MTLLibrary> library = [_device newDefaultLibrary];
        id <MTLFunction> adjust_mean_rgb = [library newFunctionWithName:@"adjust_mean_rgb"];
        _pipelineRGB = [_device newComputePipelineStateWithFunction:adjust_mean_rgb error:nil];
        
        // read JSON file
        NSData *jsonData = [NSData dataWithContentsOfFile:descriptionFile];
        NSDictionary *jsonDict = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:NULL];
        
        NSDictionary *inoutInfo = jsonDict[@"inout_info"];
        NSArray *layerInfo = jsonDict[@"layer_info"];
        NSArray *encodeSeq = jsonDict[@"encode_seq"];
        _labels = jsonDict[@"labels"];
        _layersDict = [[NSMutableDictionary alloc] init];
        _encodeSequence = [[NSMutableArray alloc] init];
        _prefetchList = [[NSMutableArray alloc] init];
        _tempImageList = [[NSMutableArray alloc] init];
        
        // create input id and output image
        _input_id = [MPSImageDescriptor imageDescriptorWithChannelFormat:textureFormat
                                                                   width:[(NSNumber *)inoutInfo[@"input_size"] intValue]
                                                                  height:[(NSNumber *)inoutInfo[@"input_size"] intValue]
                                                         featureChannels:[(NSNumber *)inoutInfo[@"input_channel"] intValue]];
        [_prefetchList addObject:_input_id];    // for srcImage
        [_prefetchList addObject:_input_id];    // for preImage
        
        _dstImage = [[MPSImage alloc] initWithDevice:_device
                                     imageDescriptor:[MPSImageDescriptor
                                                      imageDescriptorWithChannelFormat:textureFormat
                                                      width:1
                                                      height:1
                                                      featureChannels:[(NSNumber *)inoutInfo[@"output_channel"] intValue]]];
        
        // read parameters
        _fd = open([dataFile UTF8String], O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
        NSAssert(_fd != -1, @"Error: failed to open params file with errno = %d", errno);
        
        _basePtr = mmap(nil, [(NSNumber *)inoutInfo[@"file_size"] unsignedIntegerValue], PROT_READ, MAP_FILE | MAP_SHARED, _fd, 0);
        NSAssert(_basePtr, @"Error: mmap failed with errno = %d", errno);
        
        // construct layers and encode sequence
        _lastLayerName = inoutInfo[@"last_layer"];
        [self constructLayersFromInfo:layerInfo];
        _firstLayer = _layersDict[inoutInfo[@"first_layer"]];
        for (NSArray *triplet in encodeSeq) {
            [_encodeSequence addObject:@[_layersDict[triplet[0]], _layersDict[triplet[1]], _layersDict[triplet[2]]]];
        }
        
        // close file after initialization
        NSAssert(munmap(_basePtr, [(NSNumber *)inoutInfo[@"file_size"] unsignedIntegerValue]) == 0, @"Error: munmap failed with errno = %d", errno);
        close(_fd);
    }
    
    return self;
}

- (void)constructLayersFromInfo:(NSArray *)layers {
    for (NSDictionary *layer in layers) {
        NSString *layerName = layer[@"name"];
        NSString *layerType = layer[@"layer_type"];
        NSString *imageType = layer[@"image_type"];
        
        // construct kernel
        MPSCNNKernel *kernel;
        MPSCNNNeuronReLU *relu = [[MPSCNNNeuronReLU alloc] initWithDevice:_device a:0];
        
        if ([layerType isEqualToString:@"Convolution"]) {
            kernel = [[SlimMPSCNNConvolution alloc] initWithKernelSize:[(NSNumber *)layer[@"kernel_size"] intValue]
                                                  inputFeatureChannels:[(NSNumber *)layer[@"input_channel"] intValue]
                                                 outputFeatureChannels:[(NSNumber *)layer[@"output_channel"] intValue]
                                                                neuron:[(NSString *)layer[@"activation"] isEqualToString:@"ReLU"]? relu : nil
                                                                device:_device
                                                               weights:_basePtr + [(NSNumber *)layer[@"weight_offset"] intValue]
                                                                  bias:_basePtr + [(NSNumber *)layer[@"bias_offset"] intValue]
                                                               willPad:[(NSNumber *)layer[@"pad"] intValue] != 0? YES : NO
                                                                stride:[(NSNumber *)layer[@"stride"] intValue]
                                       destinationFeatureChannelOffset:[(NSNumber *)layer[@"destination_channel_offset"] intValue]
                                                                 group:[(NSNumber *)layer[@"group"] intValue]];
        } else if ([layerType isEqualToString:@"FullyConnected"]) {
            kernel = [[SlimMPSCNNFullyConnected alloc] initWithKernelSize:[(NSNumber *)layer[@"kernel_size"] intValue]
                                                     inputFeatureChannels:[(NSNumber *)layer[@"input_channel"] intValue]
                                                    outputFeatureChannels:[(NSNumber *)layer[@"output_channel"] intValue]
                                                                   neuron:[(NSString *)layer[@"activation"] isEqualToString:@"ReLU"]? relu : nil
                                                                   device:_device
                                                                  weights:_basePtr + [(NSNumber *)layer[@"weight_offset"] intValue]
                                                                     bias:_basePtr + [(NSNumber *)layer[@"bias_offset"] intValue]
                                          destinationFeatureChannelOffset:[(NSNumber *)layer[@"destination_channel_offset"] intValue]];
        } else if ([layerType isEqualToString:@"PoolingMax"]) {
            kernel = [[SlimMPSCNNPoolingMax alloc] initWithDevice:_device
                                                       kernelSize:[(NSNumber *)layer[@"kernel_size"] intValue]
                                                           stride:[(NSNumber *)layer[@"stride"] intValue]
                                                          willPad:[(NSNumber *)layer[@"pad"] intValue] != 0? YES : NO];
        } else if ([layerType isEqualToString:@"PoolingAverage"]) {
            if ((BOOL)layer[@"global"]) {
                kernel = [[SlimMPSCNNPoolingGlobalAverage alloc] initWithDevice:_device
                                                                      inputSize:[(NSNumber *)layer[@"input_size"] intValue]];
            } else {
                kernel = [[MPSCNNPoolingAverage alloc] initWithDevice:_device
                                                          kernelWidth:[(NSNumber *)layer[@"kernel_size"] intValue]
                                                         kernelHeight:[(NSNumber *)layer[@"kernel_size"] intValue]
                                                      strideInPixelsX:[(NSNumber *)layer[@"stride"] intValue]
                                                      strideInPixelsY:[(NSNumber *)layer[@"stride"] intValue]];
            }
        } else if ([layerType isEqualToString:@"LocalResponseNormalization"]) {
            kernel = [[SlimMPSCNNLocalResponseNormalization alloc] initWithDevice:_device
                                                                        localSize:[(NSNumber *)layer[@"local_size"] intValue]
                                                                            alpha:[(NSNumber *)layer[@"alpha"] floatValue]
                                                                             beta:[(NSNumber *)layer[@"beta"] floatValue]];
        } else if ([layerType isEqualToString:@"SoftMax"]) {
            kernel = [[MPSCNNSoftMax alloc] initWithDevice:_device];
        } else if ([layerType isEqualToString:@"Concat"]) {
            // does not need a kernel
        } else {
            assert("Unsupported layer!");
        }
        
        // construct output image
        MPSImageDescriptor *imageDescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:textureFormat
                                                                                             width:[(NSNumber *)layer[@"output_size"] intValue]
                                                                                            height:[(NSNumber *)layer[@"output_size"] intValue]
                                                                                   featureChannels:[(NSNumber *)layer[@"output_channel"] intValue]];
        MPSImage *outputImage;
        
        if ([layerName isEqualToString:_lastLayerName]) {
            outputImage = _dstImage;
        } else {
            if ([imageType isEqualToString:@"Permanent"]) {
                outputImage = [[MPSImage alloc] initWithDevice:_device imageDescriptor:imageDescriptor];
            } else if ([imageType isEqualToString:@"Temporary"]) {
                [_prefetchList addObject:imageDescriptor];
            }
        }
        
        // construct layer
        GeneralLayer *newLayer = [[GeneralLayer alloc] initWithName:layerName
                                                    ImageDescriptor:imageDescriptor
                                                          readCount:[imageType isEqualToString:@"Temporary"]? [(NSNumber *)layer[@"read_count"] intValue] : 0
                                                        outputImage:outputImage
                                                             kernel:kernel];
        if ([imageType isEqualToString:@"Temporary"]) [_tempImageList addObject:newLayer];
        [_layersDict setObject:newLayer forKey:layerName];
    }
}

- (NSString *)forwardWithImage:(UIImage *)image {
    NSError *error = NULL;
    _sourceTexture = [_textureLoader newTextureWithCGImage:image.CGImage options:nil error:&error];
    NSAssert(!error, error.localizedDescription);
    
    @autoreleasepool {
        id <MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        
#if ALLOW_PRINT
        [MPSTemporaryImage prefetchStorageWithCommandBuffer:commandBuffer imageDescriptorList:@[_input_id, _input_id]];
#else
        [MPSTemporaryImage prefetchStorageWithCommandBuffer:commandBuffer imageDescriptorList:_prefetchList];
#endif
        
        MPSTemporaryImage *srcImage = [MPSTemporaryImage temporaryImageWithCommandBuffer:commandBuffer imageDescriptor:_input_id];
        MPSTemporaryImage *preImage = [MPSTemporaryImage temporaryImageWithCommandBuffer:commandBuffer imageDescriptor:_input_id];
        
        // create MPSTemporaryImage for inside layers
        for (GeneralLayer *layer in _tempImageList) {
#if ALLOW_PRINT
            if (!layer.outputImage) {
                layer.outputImage = [[MPSImage alloc] initWithDevice:_device
                                                     imageDescriptor:layer.imageDescriptor];
            }
#else
            MPSTemporaryImage *tempImage = [MPSTemporaryImage temporaryImageWithCommandBuffer:commandBuffer imageDescriptor:layer.imageDescriptor];
            tempImage.readCount = layer.readCount;
            layer.outputImage = tempImage;
#endif
        }
        
        // scale input image to 227x227
        [_lanczos encodeToCommandBuffer:commandBuffer
                          sourceTexture:_sourceTexture
                     destinationTexture:srcImage.texture];
        
        // subtract mean RGB, and convert to GBR
        id <MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:_pipelineRGB];
        [encoder setTexture:srcImage.texture atIndex:0];
        [encoder setTexture:preImage.texture atIndex:1];
        MTLSize threadsPerGroups = MTLSizeMake(8, 8, 1);
        MTLSize threadGroups = MTLSizeMake(preImage.texture.width / threadsPerGroups.width,
                                           preImage.texture.height / threadsPerGroups.height, 1);
        [encoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadsPerGroups];
        [encoder endEncoding];
        srcImage.readCount -= 1;
        
        // run following layers
        [_firstLayer.kernel encodeToCommandBuffer:commandBuffer
                                      sourceImage:preImage
                                 destinationImage:_firstLayer.outputImage];
        
        for (NSArray *triplet in _encodeSequence) {
            [((GeneralLayer *)triplet[0]).kernel encodeToCommandBuffer:commandBuffer
                                                           sourceImage:((GeneralLayer *)triplet[1]).outputImage
                                                      destinationImage:((GeneralLayer *)triplet[2]).outputImage];
        }
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
#if ALLOW_PRINT
        for (NSArray *triplet in _encodeSequence) {
            if (((GeneralLayer *)triplet[2]).outputImage) {
                [self printImage:((GeneralLayer *)triplet[2]).outputImage
                         ofLayer:((GeneralLayer *)triplet[2]).name];
            }
        }
#endif
        
        return [self getTopProbs];
    }
}

- (NSString *)getTopProbs {
    // gather measurements of MPSImage to use to get out probabilities
    NSUInteger width = _dstImage.width;
    NSUInteger height = _dstImage.height;
    NSUInteger numSlices = (_dstImage.featureChannels + 3) / 4;
    NSUInteger count = _dstImage.texture.width * _dstImage.texture.height * _dstImage.featureChannels;
    NSUInteger channelsPerSlice = 4;    // textures are in RGBA format
    
    uint16_t *output = calloc(count, sizeof(uint16_t));
    float *outputF = calloc(count, sizeof(float));
    
    // get probabilities of each label in UIn16 array we use this to contain float16s
    for (int i = 0; i < numSlices; i++) {
        [_dstImage.texture getBytes:&output[height * width * channelsPerSlice * i]
                        bytesPerRow:sizeof(uint16_t) * width * channelsPerSlice
                      bytesPerImage:0
                         fromRegion:MTLRegionMake3D(0, 0, 0, width, height, 1)
                        mipmapLevel:0
                              slice:i];
    }
    
    // use VImage to convert Float16 to Float32 so we can use them
    vImage_Buffer fullResultVImagebuf = {outputF, 1, count, count * 4};
    vImage_Buffer halfResultVImagebuf = {output, 1, count, count * 2};
    
    if(vImageConvert_Planar16FtoPlanarF(&halfResultVImagebuf, &fullResultVImagebuf, 0) != kvImageNoError){
        NSLog(@"Error in vImage");
    }
    
    // copy output probabilities into an array of touples of (probability, index)
    NSMutableArray *indexedProbabilities = [[NSMutableArray alloc] initWithCapacity:count];
    for (int i = 0; i < count; i++) {
        [indexedProbabilities addObject:@[@(outputF[i]), @(i)]];
    }
    
    // sort the touple array to have top5 guesses in the front
    NSArray *sortedIndexedProbabilities = [indexedProbabilities sortedArrayUsingComparator:^NSComparisonResult(id a, id b) {
        NSNumber *first = [(NSArray *)a objectAtIndex:0];
        NSNumber *second = [(NSArray *)b objectAtIndex:0];
        return [second compare:first];
    }];
    
    // get top 5 valid guesses and add them to return string with top 5 guesses
    NSString *returnString = @"";
    for (int i = 0; i < 5; i++) {
        NSArray* probAndIndex = sortedIndexedProbabilities[i];
        returnString = [NSString stringWithFormat:@"%@%3.2f%%: %@\n", returnString, [(NSNumber *)probAndIndex[0] floatValue] * 100, _labels[[(NSNumber *)probAndIndex[1] intValue]]];
    }
    
    free(output);
    free(outputF);
    
    return returnString;
}

#if ALLOW_PRINT
- (void)printImage:(MPSImage *)image ofLayer:(NSString *)layer {
    NSLog(@"Now comes %@",layer);
    
    NSUInteger width = image.width;
    NSUInteger height = image.height;
    NSUInteger numSlices = (image.featureChannels + 3) / 4;
    NSUInteger count = image.texture.width * image.texture.height * image.featureChannels;
    NSUInteger channelsPerSlice = 4;    // textures are in RGBA format
    
    uint16_t *output = calloc(count, sizeof(uint16_t));
    float *outputF = calloc(count, sizeof(float));
    
    // get probabilities of each label in UIn16 array we use this to contain float16s
    for (int i = 0; i < numSlices; i++) {
        [image.texture getBytes:&output[height * width * channelsPerSlice * i]
                    bytesPerRow:sizeof(uint16_t) * width * channelsPerSlice
                  bytesPerImage:0
                     fromRegion:MTLRegionMake3D(0, 0, 0, width, height, 1)
                    mipmapLevel:0
                          slice:i];
    }
    
    // use VImage to convert Float16 to Float32 so we can use them
    vImage_Buffer fullResultVImagebuf = {outputF, 1, count, count * 4};
    vImage_Buffer halfResultVImagebuf = {output, 1, count, count * 2};
    
    if(vImageConvert_Planar16FtoPlanarF(&halfResultVImagebuf, &fullResultVImagebuf, 0) != kvImageNoError){
        NSLog(@"Error in vImage");
    }
    
    for (int i = 0; i < 8; i++) {
        printf("%d: %f\n", i, outputF[i]);
    }
    
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        sum += fabsf(outputF[i]);
    }
    printf("sum: %f\n",sum);
    
    float sqr = 0.0f;
    for (int i = 0; i < count; i++) {
        sqr += powf(outputF[i], 2);
    }
    printf("square: %f\n",sqr);
    
    free(output);
    free(outputF);
}
#endif

@end

#pragma mark - if not use Metal
#else

@implementation GeneralNet

- (instancetype)initWithDescriptionFile:(NSString *)descriptionFile
                               dataFile:(NSString *)dataFile {
    if (self = [super init]) {
        
        // read JSON file
        NSData *jsonData = [NSData dataWithContentsOfFile:descriptionFile];
        NSDictionary *jsonDict = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:NULL];
        
        NSDictionary *inoutInfo = jsonDict[@"inout_info"];
        NSArray *layerInfo = jsonDict[@"layer_info"];
        NSArray *encodeSeq = jsonDict[@"encode_seq"];
        
        // read parameters
        _fd = open([dataFile UTF8String], O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
        NSAssert(_fd != -1, @"Error: failed to open params file with errno = %d", errno);
        
        _basePtr = mmap(nil, [(NSNumber *)inoutInfo[@"file_size"] unsignedIntegerValue], PROT_READ, MAP_FILE | MAP_SHARED, _fd, 0);
        NSAssert(_basePtr, @"Error: mmap failed with errno = %d", errno);
        
        // construct layers and encode sequence
        _lastLayerName = inoutInfo[@"last_layer"];
        [self constructLayersFromInfo:layerInfo];
        _firstLayer = _layersDict[inoutInfo[@"first_layer"]];
        for (NSArray *triplet in encodeSeq) {
            [_encodeSequence addObject:_layersDict[triplet[0]]];
        }
    }
    
    return self;
}

- (void)constructLayersFromInfo:(NSArray *)layers {
    for (NSDictionary *layer in layers) {
        NSString *layerName = layer[@"name"];
        NSString *layerType = layer[@"layer_type"];
        NSString *imageType = layer[@"image_type"];
        
        CPULayer *cpuLayer;
        
        // construct forward method
        if ([layerType isEqualToString:@"Convolution"]) {
            cpuLayer = [[CPUConvolutionLayer alloc] initWithName:layerName
                                                          weight:_basePtr + [(NSNumber *)layer[@"weight_offset"] intValue]
                                                            bias:_basePtr + [(NSNumber *)layer[@"bias_offset"] intValue]
                                                           group:[(NSNumber *)layer[@"group"] intValue]
                                                    inputChannel:[(NSNumber *)layer[@"input_channel"] intValue]
                                                   outputChannel:[(NSNumber *)layer[@"output_channel"] intValue]
                                                       inputSize:[(NSNumber *)layer[@"input_size"] intValue]
                                                      outputSize:[(NSNumber *)layer[@"output_size"] intValue]
                                                      kernelSize:[(NSNumber *)layer[@"kernel_size"] intValue]
                                                             pad:[(NSNumber *)layer[@"pad"] intValue]
                                                          stride:[(NSNumber *)layer[@"stride"] intValue]
                                                          doReLU:[(NSString *)layer[@"activation"] isEqualToString:@"ReLU"]? YES : NO];
        } else if ([layerType isEqualToString:@"FullyConnected"]) {
            cpuLayer = [[CPUFullyConnectedLayer alloc] initWithName:layerName
                                                             weight:_basePtr + [(NSNumber *)layer[@"weight_offset"] intValue]
                                                               bias:_basePtr + [(NSNumber *)layer[@"bias_offset"] intValue]
                                                       inputChannel:[(NSNumber *)layer[@"input_channel"] intValue]
                                                      outputChannel:[(NSNumber *)layer[@"output_channel"] intValue]
                                                          inputSize:[(NSNumber *)layer[@"input_size"] intValue]
                                                             doReLU:[(NSString *)layer[@"activation"] isEqualToString:@"ReLU"]? YES : NO];
        }
    }
}

- (NSString *)forwardWithImage:(UIImage *)image {
    for (CPULayer *layer in _encodeSequence) {
        [layer forward];
    }
    
    return [self getTopProbs];
}

- (NSString *)getTopProbs {
    
    // copy output probabilities into an array of touples of (probability, index)
    NSMutableArray *indexedProbabilities = [[NSMutableArray alloc] initWithCapacity:_labels.count];
    for (int i = 0; i < _labels.count; i++) {
        [indexedProbabilities addObject:@[@(_dstPtr[i]), @(i)]];
    }
    
    // sort the touple array to have top5 guesses in the front
    NSArray *sortedIndexedProbabilities = [indexedProbabilities sortedArrayUsingComparator:^NSComparisonResult(id a, id b) {
        NSNumber *first = [(NSArray *)a objectAtIndex:0];
        NSNumber *second = [(NSArray *)b objectAtIndex:0];
        return [second compare:first];
    }];
    
    // get top 5 valid guesses and add them to return string with top 5 guesses
    NSString *returnString = @"";
    for (int i = 0; i < 5; i++) {
        NSArray* probAndIndex = sortedIndexedProbabilities[i];
        returnString = [NSString stringWithFormat:@"%@%3.2f%%: %@\n", returnString, [(NSNumber *)probAndIndex[0] floatValue] * 100, _labels[[(NSNumber *)probAndIndex[1] intValue]]];
    }
    
    return returnString;
}

- (void)dealloc {
    
    // close file
    NSAssert(munmap(_basePtr, [(NSNumber *)inoutInfo[@"file_size"] unsignedIntegerValue]) == 0, @"Error: munmap failed with errno = %d", errno);
    close(_fd);
}

@end

#endif
