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

#import <Accelerate/Accelerate.h>
#import "SlimMPSCNN.h"

static const uint kTextureFormat = MPSImageFeatureChannelFormatFloat16;

@implementation GeneralNet

- (instancetype)initWithDescriptionFile:(NSString *)descriptionFile
                               dataFile:(NSString *)dataFile {
    if (self = [super init]) {
        // preparation for Metal
         m_Device = MTLCreateSystemDefaultDevice();
        
        NSAssert(MPSSupportsMTLDevice(m_Device), @"Metal Performance Shaders not supported on current device");
        
        m_CommandQueue = [ m_Device newCommandQueue];
        m_TextureLoader = [[MTKTextureLoader alloc] initWithDevice: m_Device];
        m_Lanczos = [[MPSImageLanczosScale alloc] initWithDevice: m_Device];
        
        id <MTLLibrary> library = [ m_Device newDefaultLibrary];
        id <MTLFunction> adjust_mean_rgb = [library newFunctionWithName:@"adjust_mean_rgb"];
        m_PipelineRGB = [ m_Device newComputePipelineStateWithFunction:adjust_mean_rgb error:nil];
        
        // read JSON file
        NSData *jsonData = [NSData dataWithContentsOfFile:descriptionFile];
        NSDictionary *jsonDict = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:NULL];
        
        NSDictionary *inoutInfo = jsonDict[@"inout_info"];
        NSArray *layersInfo = jsonDict[@"layer_info"];
        NSArray *encodeSeq = jsonDict[@"encode_seq"];
        NSMutableDictionary *layersDict = [[NSMutableDictionary alloc] init];
        NSMutableArray *encodeSequence = [[NSMutableArray alloc] init];
        NSMutableArray *prefetchList = [[NSMutableArray alloc] init];
        NSMutableArray *tempImageList = [[NSMutableArray alloc] init];
        
        // create input id and output image
        m_InputImageDescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:kTextureFormat
                                                                    width:[(NSNumber *)inoutInfo[@"input_size"] unsignedIntegerValue]
                                                                   height:[(NSNumber *)inoutInfo[@"input_size"] unsignedIntegerValue]
                                                          featureChannels:[(NSNumber *)inoutInfo[@"input_channel"] unsignedIntegerValue]];
        [prefetchList addObject:m_InputImageDescriptor];    // for srcImage
        [prefetchList addObject:m_InputImageDescriptor];    // for preImage
        
        // read parameters
        m_Fd = open([dataFile UTF8String], O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
        NSAssert(m_Fd != -1, @"Error: failed to open params file with errno = %d", errno);
        
        m_BasePtr = mmap(nil, [(NSNumber *)inoutInfo[@"file_size"] unsignedIntegerValue], PROT_READ, MAP_FILE | MAP_SHARED, m_Fd, 0);
        NSAssert(m_BasePtr, @"Error: mmap failed with errno = %d", errno);
        
        // construct layers and encode sequence
        [self constructLayersWithInfo:layersInfo
                           layersDict:layersDict
                         prefetchList:prefetchList
                        tempImageList:tempImageList];
        for (NSArray *triplet in encodeSeq) {
            [encodeSequence addObject:@[layersDict[triplet[0]], layersDict[triplet[1]], layersDict[triplet[2]]]];
        }
        
        // they should not be changed after initialization
        m_FirstLayer = layersDict[inoutInfo[@"first_layer"]];
        m_LastLayer = layersDict[inoutInfo[@"last_layer"]];
        m_LayersDict = [layersDict copy];
        m_EncodeSequence = [encodeSequence copy];
        m_PrefetchList = [prefetchList copy];
        m_TempImageList = [tempImageList copy];
        m_Labels = jsonDict[@"labels"];
        
        // close file after initialization
        NSAssert(munmap(m_BasePtr, [(NSNumber *)inoutInfo[@"file_size"] unsignedIntegerValue]) == 0, @"Error: munmap failed with errno = %d", errno);
        close(m_Fd);
    }
    
    return self;
}

- (void)constructLayersWithInfo:(NSArray *)layersInfo
                     layersDict:(NSMutableDictionary *)layersDict
                   prefetchList:(NSMutableArray *)prefetchList
                  tempImageList:(NSMutableArray *)tempImageList {
    for (NSDictionary *layerInfo in layersInfo) {
        NSString *layerName = layerInfo[@"name"];
        NSString *layerType = layerInfo[@"layer_type"];
        NSString *imageType = layerInfo[@"image_type"];
        
        // construct kernel
        MPSCNNKernel *kernel;
        MPSCNNNeuronReLU *relu = [[MPSCNNNeuronReLU alloc] initWithDevice: m_Device a:0];
        
        if ([layerType isEqualToString:@"Convolution"]) {
            kernel = [[SlimMPSCNNConvolution alloc] initWithKernelSize:[(NSNumber *)layerInfo[@"kernel_size"] unsignedIntegerValue]
                                                  inputFeatureChannels:[(NSNumber *)layerInfo[@"input_channel"] unsignedIntegerValue]
                                                 outputFeatureChannels:[(NSNumber *)layerInfo[@"output_channel"] unsignedIntegerValue]
                                                                neuron:[(NSString *)layerInfo[@"activation"] isEqualToString:@"ReLU"]? relu : nil
                                                                device: m_Device
                                                               weights:m_BasePtr + [(NSNumber *)layerInfo[@"weight_offset"] unsignedIntegerValue]
                                                                  bias:m_BasePtr + [(NSNumber *)layerInfo[@"bias_offset"] unsignedIntegerValue]
                                                               willPad:[(NSNumber *)layerInfo[@"pad"] unsignedIntegerValue] != 0? YES : NO
                                                                stride:[(NSNumber *)layerInfo[@"stride"] unsignedIntegerValue]
                                       destinationFeatureChannelOffset:[(NSNumber *)layerInfo[@"destination_channel_offset"] unsignedIntegerValue]
                                                                 group:[(NSNumber *)layerInfo[@"group"] unsignedIntegerValue]];
        } else if ([layerType isEqualToString:@"FullyConnected"]) {
            kernel = [[SlimMPSCNNFullyConnected alloc] initWithKernelSize:[(NSNumber *)layerInfo[@"kernel_size"] unsignedIntegerValue]
                                                     inputFeatureChannels:[(NSNumber *)layerInfo[@"input_channel"] unsignedIntegerValue]
                                                    outputFeatureChannels:[(NSNumber *)layerInfo[@"output_channel"] unsignedIntegerValue]
                                                                   neuron:[(NSString *)layerInfo[@"activation"] isEqualToString:@"ReLU"]? relu : nil
                                                                   device: m_Device
                                                                  weights:m_BasePtr + [(NSNumber *)layerInfo[@"weight_offset"] unsignedIntegerValue]
                                                                     bias:m_BasePtr + [(NSNumber *)layerInfo[@"bias_offset"] unsignedIntegerValue]
                                          destinationFeatureChannelOffset:[(NSNumber *)layerInfo[@"destination_channel_offset"] unsignedIntegerValue]];
        } else if ([layerType isEqualToString:@"PoolingMax"]) {
            kernel = [[SlimMPSCNNPoolingMax alloc] initWithDevice: m_Device
                                                       kernelSize:[(NSNumber *)layerInfo[@"kernel_size"] unsignedIntegerValue]
                                                           stride:[(NSNumber *)layerInfo[@"stride"] unsignedIntegerValue]
                                                          willPad:[(NSNumber *)layerInfo[@"pad"] unsignedIntegerValue]? YES : NO];
        } else if ([layerType isEqualToString:@"PoolingAverage"]) {
            if ((BOOL)layerInfo[@"global"]) {
                kernel = [[SlimMPSCNNPoolingGlobalAverage alloc] initWithDevice: m_Device
                                                                      inputSize:[(NSNumber *)layerInfo[@"input_size"] unsignedIntegerValue]];
            } else {
                kernel = [[MPSCNNPoolingAverage alloc] initWithDevice: m_Device
                                                          kernelWidth:[(NSNumber *)layerInfo[@"kernel_size"] unsignedIntegerValue]
                                                         kernelHeight:[(NSNumber *)layerInfo[@"kernel_size"] unsignedIntegerValue]
                                                      strideInPixelsX:[(NSNumber *)layerInfo[@"stride"] unsignedIntegerValue]
                                                      strideInPixelsY:[(NSNumber *)layerInfo[@"stride"] unsignedIntegerValue]];
            }
        } else if ([layerType isEqualToString:@"LocalResponseNormalization"]) {
            kernel = [[SlimMPSCNNLocalResponseNormalization alloc] initWithDevice: m_Device
                                                                        localSize:[(NSNumber *)layerInfo[@"local_size"] unsignedIntegerValue]
                                                                            alpha:[(NSNumber *)layerInfo[@"alpha"] floatValue]
                                                                             beta:[(NSNumber *)layerInfo[@"beta"] floatValue]];
        } else if ([layerType isEqualToString:@"SoftMax"]) {
            kernel = [[MPSCNNSoftMax alloc] initWithDevice: m_Device];
        } else if ([layerType isEqualToString:@"Concat"]) {
            // does not need a kernel
        } else {
            assert("Unsupported layer!");
        }
        
        // construct output image
        MPSImageDescriptor *imageDescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:kTextureFormat
                                                                                             width:[(NSNumber *)layerInfo[@"output_size"] unsignedIntegerValue]
                                                                                            height:[(NSNumber *)layerInfo[@"output_size"] unsignedIntegerValue]
                                                                                   featureChannels:[(NSNumber *)layerInfo[@"output_channel"] unsignedIntegerValue]];
        
        MPSImage *outputImage;
        if ([imageType isEqualToString:@"Permanent"]) {
            outputImage = [[MPSImage alloc] initWithDevice: m_Device imageDescriptor:imageDescriptor];
        } else if ([imageType isEqualToString:@"Temporary"]) {
            [prefetchList addObject:imageDescriptor];
        }
        
        // construct layer
        MPSLayer *newLayer = [[MPSLayer alloc] initWithName:layerName
                                                     kernel:kernel
                                            ImageDescriptor:imageDescriptor
                                                  readCount:[imageType isEqualToString:@"Temporary"]? [(NSNumber *)layerInfo[@"read_count"] unsignedIntegerValue] : 0
                                                outputImage:outputImage];
        if ([imageType isEqualToString:@"Temporary"]) [tempImageList addObject:newLayer];
        [layersDict setObject:newLayer forKey:layerName];
    }
}

- (void)forwardWithImage:(UIImage *)image
              completion:(void (^)())completion {
    NSError *error = NULL;
    m_SourceTexture = [m_TextureLoader newTextureWithCGImage:image.CGImage options:nil error:&error];
    NSAssert(!error, error.localizedDescription);
    
    @autoreleasepool {
        id <MTLCommandBuffer> commandBuffer = [m_CommandQueue commandBuffer];
        
#if ALLOW_PRINT
        [MPSTemporaryImage prefetchStorageWithCommandBuffer:commandBuffer imageDescriptorList:@[m_InputImageDescriptor, m_InputImageDescriptor]];
#else
        [MPSTemporaryImage prefetchStorageWithCommandBuffer:commandBuffer imageDescriptorList:m_PrefetchList];
#endif
        
        MPSTemporaryImage *srcImage = [MPSTemporaryImage temporaryImageWithCommandBuffer:commandBuffer imageDescriptor:m_InputImageDescriptor];
        MPSTemporaryImage *preImage = [MPSTemporaryImage temporaryImageWithCommandBuffer:commandBuffer imageDescriptor:m_InputImageDescriptor];
        
        // create MPSTemporaryImage for inside layers
        for (MPSLayer *layer in m_TempImageList) {
#if ALLOW_PRINT
            if (!layer.outputImage) {
                layer.outputImage = [[MPSImage alloc] initWithDevice: m_Device
                                                     imageDescriptor:layer.imageDescriptor];
            }
#else
            MPSTemporaryImage *tempImg = [MPSTemporaryImage temporaryImageWithCommandBuffer:commandBuffer imageDescriptor:layer.imageDescriptor];
            tempImg.readCount = layer.readCount;
            layer.outputImage = tempImg;
#endif
        }
        
        // scale input image to 227x227
        [m_Lanczos encodeToCommandBuffer:commandBuffer
                          sourceTexture:m_SourceTexture
                     destinationTexture:srcImage.texture];
        
        // subtract mean RGB, and convert to GBR
        id <MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:m_PipelineRGB];
        [encoder setTexture:srcImage.texture atIndex:0];
        [encoder setTexture:preImage.texture atIndex:1];
        MTLSize threadsPerGroups = MTLSizeMake(8, 8, 1);
        MTLSize threadGroups = MTLSizeMake(preImage.texture.width / threadsPerGroups.width,
                                           preImage.texture.height / threadsPerGroups.height, 1);
        [encoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadsPerGroups];
        [encoder endEncoding];
        srcImage.readCount -= 1;
        
        // run following layers
        [m_FirstLayer.kernel encodeToCommandBuffer:commandBuffer
                                       sourceImage:preImage
                                  destinationImage:m_FirstLayer.outputImage];
        
        for (NSArray<MPSLayer *> *triplet in m_EncodeSequence) {
            [triplet[0].kernel encodeToCommandBuffer:commandBuffer
                                         sourceImage:(triplet[1]).outputImage
                                    destinationImage:(triplet[2]).outputImage];
        }
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        completion();
        
#if ALLOW_PRINT
        [self printImage:m_FirstLayer.outputImage
                 ofLayer:m_FirstLayer.name];
        for (NSArray<MPSLayer *> *triplet in m_EncodeSequence) {
            if (triplet[2].outputImage) {
                [self printImage:triplet[2].outputImage
                         ofLayer:triplet[2].name];
            }
        }
#endif
    }
}

- (NSString *)labelsOfTopProbs {
    
    // gather measurements of MPSImage to use to get out probabilities
    MPSImage *outputImage = m_LastLayer.outputImage;
    NSUInteger width = outputImage.width;
    NSUInteger height = outputImage.height;
    NSUInteger numSlices = (outputImage.featureChannels + 3) / 4;
    NSUInteger count = outputImage.texture.width * outputImage.texture.height * outputImage.featureChannels;
    NSUInteger channelsPerSlice = 4;    // textures are in RGBA format
    
    uint16_t *output = calloc(count, sizeof(uint16_t));
    float *outputF = calloc(count, sizeof(float));
    
    // get probabilities of each label in UIn16 array we use this to contain float16s
    for (int i = 0; i < numSlices; i++) {
        [outputImage.texture getBytes:&output[height * width * channelsPerSlice * i]
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
        NSArray *probAndIndex = sortedIndexedProbabilities[i];
        returnString = [NSString stringWithFormat:@"%@%3.2f%%: %@\n", returnString, [(NSNumber *)probAndIndex[0] floatValue] * 100, m_Labels[[(NSNumber *)probAndIndex[1] unsignedIntegerValue]]];
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

#import "CPULayer.h"

@implementation GeneralNet

- (instancetype)initWithDescriptionFile:(NSString *)descriptionFile
                               dataFile:(NSString *)dataFile {
    if (self = [super init]) {
        
        // read JSON file
        NSData *jsonData = [NSData dataWithContentsOfFile:descriptionFile];
        NSDictionary *jsonDict = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:NULL];
        NSDictionary *inoutInfo = jsonDict[@"inout_info"];
        NSArray *layersInfo = jsonDict[@"layer_info"];
        NSArray *encodeSeq = jsonDict[@"encode_seq"];
        NSMutableDictionary *layersDict = [[NSMutableDictionary alloc] init];
        NSMutableArray *encodeSequence = [[NSMutableArray alloc] init];
        
        m_FileSize = [(NSNumber *)inoutInfo[@"file_size"] unsignedIntegerValue];
        m_InputSize = [(NSNumber *)inoutInfo[@"input_size"] intValue];
        m_ImageRawData = (unsigned char *)calloc(m_InputSize * m_InputSize * 4, sizeof(unsigned char));
        m_ImageData = malloc(sizeof(float) * m_InputSize * m_InputSize * 3);
        
        // read parameters
        m_Fd = open([dataFile UTF8String], O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
        NSAssert(m_Fd != -1, @"Error: failed to open params file with errno = %d", errno);
        
        m_BasePtr = mmap(nil, m_FileSize, PROT_READ, MAP_FILE | MAP_SHARED, m_Fd, 0);
        NSAssert(m_BasePtr, @"Error: mmap failed with errno = %d", errno);
        
        // construct layers and encode sequence
        [self constructLayersWithInfo:layersInfo layersDict:layersDict];
        for (NSArray *triplet in encodeSeq) {
            [encodeSequence addObject:@[layersDict[triplet[0]], layersDict[triplet[1]], layersDict[triplet[2]]]];
        }
        
        // they should not be changed after initialization
        m_FirstLayer = layersDict[inoutInfo[@"first_layer"]];
        m_LastLayer = layersDict[inoutInfo[@"last_layer"]];
        m_LayersDict = [layersDict copy];
        m_EncodeSequence = [encodeSequence copy];
        m_Labels = jsonDict[@"labels"];
    }
    
    return self;
}

- (void)constructLayersWithInfo:(NSArray *)layersInfo
                     layersDict:(NSMutableDictionary *)layersDict {
    
    // find out the maximum of size of col_data
    // col_data will only be created once, and then shared by all convolution layers
    size_t maxColDataSize = 0;
    for (NSDictionary *layerInfo in layersInfo) {
        if ([layerInfo[@"layer_type"] isEqualToString:@"Convolution"]) {
            size_t colDataSize = [(NSNumber *)layerInfo[@"output_size"] intValue] * [(NSNumber *)layerInfo[@"output_size"] intValue] *
            [(NSNumber *)layerInfo[@"input_channel"] intValue] * [(NSNumber *)layerInfo[@"kernel_size"] intValue] *
            [(NSNumber *)layerInfo[@"kernel_size"] intValue];
            if (colDataSize > maxColDataSize) maxColDataSize = colDataSize;
        }
    }
    m_ColData = malloc(maxColDataSize * sizeof(float));
    
    for (NSDictionary *layerInfo in layersInfo) {
        NSString *layerName = layerInfo[@"name"];
        NSString *layerType = layerInfo[@"layer_type"];
        NSString *imageType = layerInfo[@"image_type"];
        
        CPULayer *newLayer;
        
        // construct forward method
        if ([layerType isEqualToString:@"Convolution"]) {
            newLayer = [[CPUConvolutionLayer alloc] initWithName:layerName
                                                          weight:m_BasePtr + [(NSNumber *)layerInfo[@"weight_offset"] intValue]
                                                            bias:m_BasePtr + [(NSNumber *)layerInfo[@"bias_offset"] intValue]
                                                           group:[(NSNumber *)layerInfo[@"group"] intValue]
                                                    inputChannel:[(NSNumber *)layerInfo[@"input_channel"] intValue]
                                                   outputChannel:[(NSNumber *)layerInfo[@"output_channel"] intValue]
                                                       inputSize:[(NSNumber *)layerInfo[@"input_size"] intValue]
                                                      outputSize:[(NSNumber *)layerInfo[@"output_size"] intValue]
                                                      kernelSize:[(NSNumber *)layerInfo[@"kernel_size"] intValue]
                                                             pad:[(NSNumber *)layerInfo[@"pad"] intValue]
                                                          stride:[(NSNumber *)layerInfo[@"stride"] intValue]
                                                          doReLU:[(NSString *)layerInfo[@"activation"] isEqualToString:@"ReLU"]? YES : NO
                                                         colData:m_ColData];
        } else if ([layerType isEqualToString:@"FullyConnected"]) {
            newLayer = [[CPUFullyConnectedLayer alloc] initWithName:layerName
                                                             weight:m_BasePtr + [(NSNumber *)layerInfo[@"weight_offset"] intValue]
                                                               bias:m_BasePtr + [(NSNumber *)layerInfo[@"bias_offset"] intValue]
                                                       inputChannel:[(NSNumber *)layerInfo[@"input_channel"] intValue]
                                                      outputChannel:[(NSNumber *)layerInfo[@"output_channel"] intValue]
                                                          inputSize:[(NSNumber *)layerInfo[@"input_size"] intValue]
                                                             doReLU:[(NSString *)layerInfo[@"activation"] isEqualToString:@"ReLU"]? YES : NO];
        } else if ([layerType isEqualToString:@"PoolingMax"]) {
            newLayer = [[CPUPoolingLayer alloc]initWithName:layerName
                                                poolingType:ePoolingMax
                                               inputChannel:[(NSNumber *)layerInfo[@"input_channel"] intValue]
                                              outputChannel:[(NSNumber *)layerInfo[@"output_channel"] intValue]
                                                  inputSize:[(NSNumber *)layerInfo[@"input_size"] intValue]
                                                 outputSize:[(NSNumber *)layerInfo[@"output_size"] intValue]
                                                 kernelSize:[(NSNumber *)layerInfo[@"kernel_size"] intValue]
                                                        pad:[(NSNumber *)layerInfo[@"pad"] intValue]
                                                     stride:[(NSNumber *)layerInfo[@"stride"] intValue]];
        } else if ([layerType isEqualToString:@"PoolingAverage"]) {
            if ((BOOL)layerInfo[@"global"]) {
                newLayer = [[CPUPoolingLayer alloc]initWithName:layerName
                                                    poolingType:ePoolingGlobalAverage
                                                   inputChannel:[(NSNumber *)layerInfo[@"input_channel"] intValue]
                                                  outputChannel:[(NSNumber *)layerInfo[@"output_channel"] intValue]
                                                      inputSize:[(NSNumber *)layerInfo[@"input_size"] intValue]
                                                     outputSize:[(NSNumber *)layerInfo[@"output_size"] intValue]
                                                     kernelSize:[(NSNumber *)layerInfo[@"input_size"] intValue]
                                                            pad:0
                                                         stride:[(NSNumber *)layerInfo[@"input_size"] intValue]];
            } else {
                newLayer = [[CPUPoolingLayer alloc]initWithName:layerName
                                                    poolingType:ePoolingAverage
                                                   inputChannel:[(NSNumber *)layerInfo[@"input_channel"] intValue]
                                                  outputChannel:[(NSNumber *)layerInfo[@"output_channel"] intValue]
                                                      inputSize:[(NSNumber *)layerInfo[@"input_size"] intValue]
                                                     outputSize:[(NSNumber *)layerInfo[@"output_size"] intValue]
                                                     kernelSize:[(NSNumber *)layerInfo[@"kernel_size"] intValue]
                                                            pad:0
                                                         stride:[(NSNumber *)layerInfo[@"stride"] intValue]];
            }
        } else if ([layerType isEqualToString:@"LocalResponseNormalization"]) {     // only support within-channel normalization for now
            newLayer = [[CPULocalResponseNormalizationLayer alloc] initWithName:layerName
                                                                   inputChannel:[(NSNumber *)layerInfo[@"input_channel"] intValue]
                                                                      inputSize:[(NSNumber *)layerInfo[@"input_size"] intValue]
                                                                          alpha:[(NSNumber *)layerInfo[@"alpha"] floatValue]
                                                                           beta:[(NSNumber *)layerInfo[@"beta"] floatValue]
                                                                          delta:1.0f
                                                                      localSize:[(NSNumber *)layerInfo[@"local_size"] intValue]];
        } else if ([layerType isEqualToString:@"SoftMax"]) {
            newLayer = [[CPUSoftMaxLayer alloc] initWithName:layerName
                                                inputChannel:[(NSNumber *)layerInfo[@"input_channel"] intValue]];
        } else if ([layerType isEqualToString:@"Concat"]) {
            newLayer = [[CPULayer alloc] initWithName:layerName];
        } else {
            assert("Unsupported layer!");
        }
        
        if (![imageType isEqualToString:@"None"]) {
            newLayer.outputNum = [(NSNumber *)layerInfo[@"output_size"] intValue] * [(NSNumber *)layerInfo[@"output_size"] intValue] *
                                 [(NSNumber *)layerInfo[@"output_channel"] intValue];
            newLayer.output = malloc(newLayer.outputNum * sizeof(float));
        }
        
        if ([layerInfo objectForKey:@"destination_channel_offset"]) {
            newLayer.destinationOffset = [((NSNumber *)layerInfo[@"destination_channel_offset"]) intValue] *
                                         [(NSNumber *)layerInfo[@"output_size"] intValue] * [(NSNumber *)layerInfo[@"output_size"] intValue];
        }
        
        [layersDict setObject:newLayer forKey:layerName];
    }
}

- (void)forwardWithImage:(UIImage *)image
              completion:(void (^)())completion {
    
    // scale the input image
    UIGraphicsBeginImageContext(CGSizeMake(m_InputSize, m_InputSize));
    [image drawInRect:CGRectMake(0, 0, m_InputSize, m_InputSize)];
    UIImage *scaledImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    // get the image into data buffer
    CGImageRef imageRef = [scaledImage CGImage];
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    NSUInteger bytesPerPixel = 4;
    NSUInteger bytesPerRow = bytesPerPixel * m_InputSize;
    NSUInteger bitsPerComponent = 8;
    CGContextRef context = CGBitmapContextCreate(m_ImageRawData, m_InputSize, m_InputSize, bitsPerComponent, bytesPerRow, colorSpace,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(colorSpace);
    
    CGContextDrawImage(context, CGRectMake(0, 0, m_InputSize, m_InputSize), imageRef);
    CGContextRelease(context);
    
    // imageRawData contains the image data in the RGBA8888 pixel format
    // substract mean RGB and flip to GBR
    for (int i = 0 ; i < m_InputSize * m_InputSize; i++) {
        m_ImageData[i+m_InputSize*m_InputSize*0] = (float)m_ImageRawData[i*4+2] - 120.0f;
        m_ImageData[i+m_InputSize*m_InputSize*1] = (float)m_ImageRawData[i*4+1] - 120.0f;
        m_ImageData[i+m_InputSize*m_InputSize*2] = (float)m_ImageRawData[i*4+0] - 120.0f;
    }
    
    [m_FirstLayer forwardWithInput:m_ImageData
                            output:m_FirstLayer.output + m_FirstLayer.destinationOffset];
    
    for (NSArray<CPULayer *> *triplet in m_EncodeSequence) {
        [triplet[0] forwardWithInput:triplet[1].output
                              output:triplet[2].output + triplet[0].destinationOffset];
    }
    
    completion();
    
#if ALLOW_PRINT
    [self printOutput:m_FirstLayer.output
              ofLayer:m_FirstLayer.name
               length:m_FirstLayer.outputNum];
    for (NSArray<CPULayer *> *triplet in m_EncodeSequence) {
        if (triplet[2].output) {
            [self printOutput:triplet[2].output
                      ofLayer:triplet[2].name
                       length:triplet[2].outputNum];
        }
    }
#endif
}

- (NSString *)labelsOfTopProbs {
    
    // copy output probabilities into an array of touples of (probability, index)
    NSMutableArray *indexedProbabilities = [[NSMutableArray alloc] initWithCapacity:m_Labels.count];
    for (int i = 0; i < m_Labels.count; i++) {
        [indexedProbabilities addObject:@[@(m_LastLayer.output[i]), @(i)]];
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
        NSArray *probAndIndex = sortedIndexedProbabilities[i];
        returnString = [NSString stringWithFormat:@"%@%3.2f%%: %@\n", returnString, [(NSNumber *)probAndIndex[0] floatValue] * 100, m_Labels[[(NSNumber *)probAndIndex[1] intValue]]];
    }
    
    return returnString;
}

#if ALLOW_PRINT
- (void)printOutput:(float *)output
            ofLayer:(NSString *)layer
             length:(size_t)length {
    NSLog(@"Now comes %@",layer);
    
    for (int i = 0; i < 8; i++) {
        printf("%d: %f\n", i, output[i]);
    }
    
    float sum = 0.0f, sqr = 0.0f;
    for (int i = 0; i < length; i++) {
        sum += fabsf(output[i]);
        sqr += powf(output[i], 2);
    }
    printf("sum: %f\nsquare: %f\n", sum, sqr);
}
#endif

- (void)dealloc {
    
    // close file
    NSAssert(munmap(m_BasePtr, m_FileSize) == 0, @"Error: munmap failed with errno = %d", errno);
    close(m_Fd);
    
    // release pointers
    if (m_ImageRawData) free(m_ImageRawData);
    if (m_ImageData)    free(m_ImageData);
    if (m_ColData)      free(m_ColData);
}

@end

#endif
