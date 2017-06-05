//
//  SlimMPSCNN.h
//  GeneralNet
//
//  Created by Lun on 2017/3/28.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface MPSLayer : NSObject

@property (strong, nonatomic) NSString *name;
@property (strong, nonatomic) MPSImageDescriptor *imageDescriptor;
@property (strong, nonatomic) MPSImage *outputImage;
@property (assign, nonatomic) NSUInteger readCount;
@property (strong, nonatomic) MPSCNNKernel *kernel;

- (instancetype)initWithName:(NSString *)name
             ImageDescriptor:(MPSImageDescriptor *)imageDescritor
                   readCount:(NSUInteger)readCount
                 outputImage:(MPSImage *)outputImage
                      kernel:(MPSCNNKernel *)kernel;

@end

@interface SlimMPSCNNConvolution : MPSCNNConvolution

@property (nonatomic) BOOL padding;

- (SlimMPSCNNConvolution *) initWithKernelSize:(NSUInteger)kernelSize
                          inputFeatureChannels:(NSUInteger)inChannels
                         outputFeatureChannels:(NSUInteger)outChannels
                                        neuron:(MPSCNNNeuron *)neuron
                                        device:(id <MTLDevice>)device
                                       weights:(const float *)weights
                                          bias:(const float *)bias
                                       willPad:(BOOL)willPad
                                        stride:(NSUInteger)stride
               destinationFeatureChannelOffset:(NSUInteger)offset
                                         group:(NSUInteger)group;

@end

@interface SlimMPSCNNFullyConnected : MPSCNNFullyConnected

- (SlimMPSCNNFullyConnected *) initWithKernelSize:(NSUInteger)kernelSize
                             inputFeatureChannels:(NSUInteger)inChannels
                            outputFeatureChannels:(NSUInteger)outChannels
                                           neuron:(MPSCNNNeuron *)neuron
                                           device:(id <MTLDevice>)device
                                          weights:(const float *)weights
                                             bias:(const float *)bias
                  destinationFeatureChannelOffset:(NSUInteger)offset;

@end

@interface SlimMPSCNNPoolingMax : MPSCNNPoolingMax

@property (nonatomic) BOOL padding;

- (SlimMPSCNNPoolingMax *) initWithDevice:(id <MTLDevice>)device
                               kernelSize:(NSUInteger)kernelSize
                                   stride:(NSUInteger)stride
                                  willPad:(BOOL)willPad;

@end

@interface SlimMPSCNNPoolingGlobalAverage : MPSCNNPoolingAverage

- (SlimMPSCNNPoolingGlobalAverage *) initWithDevice:(id <MTLDevice>)device
                                          inputSize:(NSUInteger)inputSize;

@end

@interface SlimMPSCNNLocalResponseNormalization : MPSCNNCrossChannelNormalization

- (SlimMPSCNNLocalResponseNormalization *) initWithDevice:(id <MTLDevice>)device
                                                localSize:(NSUInteger)localSize
                                                    alpha:(float)alpha
                                                     beta:(float)beta;

@end
