//
//  GeneralNet.m
//  GeneralNet
//
//  Created by Lun on 2017/4/18.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import "GeneralNet.h"

@implementation GeneralLayer

- (instancetype)initWithImageDescriptor:(MPSImageDescriptor *)imageDescritor
                              readCount:(NSUInteger)readCount
                            outputImage:(MPSImage *)outputImage
                                 kernel:(MPSCNNKernel *)kernel {
    if (self = [super init]) {
        _imageDescriptor = imageDescritor;
        _outputImage = outputImage;
        _readCount = readCount;
        _kernel = kernel;
    }
    
    return self;
}

@end

@implementation GeneralNet

- (instancetype)initWithCommandQueue:(id <MTLCommandQueue>)commandQueue {
    if (self = [super init]) {
        _device = commandQueue.device;
        _commandQueue = commandQueue;
        _layers = [[NSMutableArray alloc] init];
        _descriptors = [[NSMutableArray alloc] init];
        _encodeList = [[NSMutableArray alloc] init];
        _textureFormat = MPSImageFeatureChannelFormatFloat16;
    }
    
    return self;
}

- (void)addLayer:(GeneralLayer *)layer {
    [_layers addObject:layer];
    
    if (layer.imageDescriptor) {
        [_descriptors addObject:layer.imageDescriptor];
    }
}

- (MPSImage *)getOutputImage {
    return [(GeneralLayer *)_layers.lastObject outputImage];
}

- (void)forwardWithCommandBuffer:(id <MTLCommandBuffer>)commandBuffer
                     sourceImage:(MPSImage *)sourceImage {
    for (GeneralLayer *layer in _layers) {
        NSAssert(layer.outputImage || (layer.imageDescriptor && layer.readCount) || layer.concatLayer, @"No image!");
        
        if (layer.imageDescriptor) {
            layer.outputImage = [MPSTemporaryImage temporaryImageWithCommandBuffer:commandBuffer
                                                                   imageDescriptor:layer.imageDescriptor];
            ((MPSTemporaryImage *)layer.outputImage).readCount = layer.readCount;
        }
        
//        if (layer.imageDescriptor && !layer.outputImage && !layer.concatLayer) {
//            layer.outputImage = [[MPSImage alloc] initWithDevice:commandBuffer.device
//                                                 imageDescriptor:layer.imageDescriptor];
//        }
    }
    
    GeneralLayer *layer = (GeneralLayer *)_layers[0];
    [layer.kernel encodeToCommandBuffer:commandBuffer
                            sourceImage:sourceImage
                       destinationImage:layer.outputImage? layer.outputImage : layer.concatLayer.outputImage];
    
    for (NSArray *pair in _encodeList) {
        GeneralLayer *bottomLayer = pair[0];
        GeneralLayer *topLayer = pair[1];
        
        [topLayer.kernel encodeToCommandBuffer:commandBuffer
                                   sourceImage:bottomLayer.outputImage? bottomLayer.outputImage : bottomLayer.concatLayer.outputImage
                              destinationImage:topLayer.outputImage? topLayer.outputImage : topLayer.concatLayer.outputImage];
    }
    
//    [commandBuffer commit];
//    [commandBuffer waitUntilCompleted];
//    
//    [_layers enumerateObjectsUsingBlock:^(id  _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
//        GeneralLayer *layer = (GeneralLayer *)obj;
//        [self printImage:layer.outputImage? layer.outputImage : layer.concatLayer.outputImage
//                 ofLayer:layer != _layers.lastObject? [NSString stringWithFormat:@"%lu", idx+1] : @"prob"];
//        printf("\n");
//    }];
}

- (void)printImage:(MPSImage *)image ofLayer:(NSString *)layer {
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
    
    NSLog(@"Now comes %@",layer);
    
    for (int i = 0; i < 8; i++) {
        printf("%f  ",outputF[i]);
    }
    printf("\n");
    
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
    
    if ([layer isEqualToString:@"prob"]) {
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
        NSLog(@"Prob Top 5:");
        for (int i = 0; i < 5; i++) {
            NSArray* probAndIndex = sortedIndexedProbabilities[i];
            NSLog(@"%f",[(NSNumber *)probAndIndex[0] floatValue] * 100);
        }
    }
}

@end
