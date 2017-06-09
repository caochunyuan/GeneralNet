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

@property (readonly, nonatomic) NSString *name;
@property (readonly, nonatomic) MPSCNNKernel *kernel;
@property (readonly, nonatomic) MPSImageDescriptor *imageDescriptor;
@property (readonly, nonatomic) NSUInteger readCount;
@property (nonatomic) MPSImage *outputImage;

- (instancetype)initWithName:(NSString *)name
                      kernel:(MPSCNNKernel *)kernel
             ImageDescriptor:(MPSImageDescriptor *)imageDescritor
                   readCount:(NSUInteger)readCount
                 outputImage:(MPSImage *)outputImage;

@end

@interface SlimMPSCNNConvolution : MPSCNNConvolution

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
