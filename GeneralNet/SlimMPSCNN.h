//
//  SlimMPSCNN.h
//  GeneralNet
//
//  Created by Lun on 2017/3/28.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface SlimMPSCNNConvolution : MPSCNNConvolution

@property (nonatomic) BOOL padding;

- (SlimMPSCNNConvolution *) initWithKernelWidth:(uint)width
                                   kernelHeight:(uint)height
                           inputFeatureChannels:(uint)inChannels
                          outputFeatureChannels:(uint)outChannels
                                         neuron:(MPSCNNNeuron *)neuron
                                         device:(id <MTLDevice>)device
                                        weights:(const float *)weights
                                           bias:(const float *)bias
                                        willPad:(BOOL)willPad
                                        strideX:(uint)strideX
                                        strideY:(uint)strideY
                destinationFeatureChannelOffset:(uint)offset
                                          group:(uint)group;

@end

@interface SlimMPSCNNFullyConnected : MPSCNNFullyConnected

- (SlimMPSCNNFullyConnected *) initWithKernelWidth:(uint)width
                                      kernelHeight:(uint)height
                              inputFeatureChannels:(uint)inChannels
                             outputFeatureChannels:(uint)outChannels
                                            neuron:(MPSCNNNeuron *)neuron
                                            device:(id <MTLDevice>)device
                                           weights:(const float *)weights
                                              bias:(const float *)bias
                   destinationFeatureChannelOffset:(uint)offset;

@end

@interface SlimMPSCNNPoolingMax : MPSCNNPoolingMax

@property (nonatomic) BOOL padding;

- (SlimMPSCNNPoolingMax *) initWithDevice:(id <MTLDevice>)device
                              kernelWidth:(NSUInteger)kernelWidth
                             kernelHeight:(NSUInteger)kernelHeight
                          strideInPixelsX:(NSUInteger)strideInPixelsX
                          strideInPixelsY:(NSUInteger)strideInPixelsY
                                  willPad:(BOOL)willPad;

@end

@interface SlimMPSCNNPoolingGlobalAverage : MPSCNNPoolingAverage

- (SlimMPSCNNPoolingGlobalAverage *) initWithDevice:(id <MTLDevice>)device
                                        kernelWidth:(NSUInteger)kernelWidth
                                       kernelHeight:(NSUInteger)kernelHeight;

@end

@interface SlimMPSCNNLocalResponseNormalization : MPSCNNCrossChannelNormalization

- (SlimMPSCNNLocalResponseNormalization *) initWithDevice:(id <MTLDevice>)device
                                                localSize:(uint)localSize
                                                    alpha:(float)alpha
                                                     beta:(float)beta;

@end
