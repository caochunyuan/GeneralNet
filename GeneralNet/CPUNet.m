//
//  CPUNet.m
//  GeneralNet
//
//  Created by Lun on 2017/8/19.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import <sys/mman.h>
#import "CPUNet.h"
#import "CPULayer.h"

@implementation CPUNet

+ (id <GeneralNetProtocol>)netWithDescriptionFilename:(NSString *)descriptionFilename
                                         dataFilename:(NSString *)dataFilename {
    return [[CPUNet alloc] initWithDescriptionFile:[[NSBundle mainBundle] pathForResource:descriptionFilename ofType:@"json"]
                                          dataFile:[[NSBundle mainBundle] pathForResource:dataFilename ofType:@"dat"]];
}

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
