//
//  SlimMPSCNN.m
//  GeneralNet
//
//  Created by Lun on 2017/3/28.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import "SlimMPSCNN.h"

@implementation SlimMPSCNNConvolution {
@private
    BOOL _padding;
}

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
                                         group:(NSUInteger)group {
    
    MPSCNNConvolutionDescriptor *convDesc;
    convDesc = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:kernelSize
                                                                       kernelHeight:kernelSize
                                                               inputFeatureChannels:inChannels
                                                              outputFeatureChannels:outChannels
                                                                       neuronFilter:neuron];
    convDesc.strideInPixelsX = stride;
    convDesc.strideInPixelsY = stride;
    
    NSAssert(group > 0, @"Group size can't be less than 1");
    convDesc.groups = group;
    
    if (self = [super initWithDevice:device
               convolutionDescriptor:convDesc
                       kernelWeights:weights
                           biasTerms:bias
                               flags:MPSCNNConvolutionFlagsNone]) {
        self.destinationFeatureChannelOffset = offset;
        self.padding = willPad;
    }
    
    return self;
}

- (void) encodeToCommandBuffer:(nonnull id <MTLCommandBuffer>) commandBuffer
                   sourceImage:(MPSImage * __nonnull) sourceImage
              destinationImage:(MPSImage * __nonnull) destinationImage
MPS_SWIFT_NAME(encode(commandBuffer:sourceImage:destinationImage:)) {
    
    if (_padding) {
        long int pad_along_height = ((destinationImage.height - 1) * self.strideInPixelsY + self.kernelHeight - sourceImage.height);
        long int pad_along_width = ((destinationImage.width - 1) * self.strideInPixelsX + self.kernelWidth - sourceImage.width);
        long int pad_top = pad_along_height / 2;
        long int pad_left = pad_along_width / 2;
        
        MPSOffset offset;
        offset.x = self.kernelWidth / 2 - pad_left;
        offset.y = self.kernelHeight / 2 - pad_top;
        offset.z = 0;
        self.offset = offset;
    } else {
        MPSOffset offset;
        offset.x = self.kernelWidth / 2;
        offset.y = self.kernelHeight / 2;
        offset.z = 0;
        self.offset = offset;
    }

    [super encodeToCommandBuffer:commandBuffer
                     sourceImage:sourceImage
                destinationImage:destinationImage];
}

@end

@implementation SlimMPSCNNFullyConnected

- (SlimMPSCNNFullyConnected *) initWithKernelSize:(NSUInteger)kernelSize
                             inputFeatureChannels:(NSUInteger)inChannels
                            outputFeatureChannels:(NSUInteger)outChannels
                                           neuron:(MPSCNNNeuron *)neuron
                                           device:(id <MTLDevice>)device
                                          weights:(const float *)weights
                                             bias:(const float *)bias
                  destinationFeatureChannelOffset:(NSUInteger)offset{
    
    MPSCNNConvolutionDescriptor *convDesc;
    convDesc = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:kernelSize
                                                                       kernelHeight:kernelSize
                                                               inputFeatureChannels:inChannels
                                                              outputFeatureChannels:outChannels
                                                                       neuronFilter:neuron];
    
    self = [super initWithDevice:device
           convolutionDescriptor:convDesc
                   kernelWeights:weights
                       biasTerms:bias
                           flags:MPSCNNConvolutionFlagsNone];
    self.destinationFeatureChannelOffset = offset;
    
    return self;
}

@end

@implementation SlimMPSCNNPoolingMax {
@private
    BOOL _padding;
}

- (SlimMPSCNNPoolingMax *) initWithDevice:(id <MTLDevice>)device
                               kernelSize:(NSUInteger)kernelSize
                                   stride:(NSUInteger)stride
                                  willPad:(BOOL)willPad {
    if (self = [super initWithDevice:device
                         kernelWidth:kernelSize
                        kernelHeight:kernelSize
                     strideInPixelsX:stride
                     strideInPixelsY:stride]) {
        _padding = willPad;
    }
    
    return self;
}

- (void) encodeToCommandBuffer:(nonnull id <MTLCommandBuffer>) commandBuffer
                   sourceImage:(MPSImage * __nonnull) sourceImage
              destinationImage:(MPSImage * __nonnull) destinationImage
MPS_SWIFT_NAME(encode(commandBuffer:sourceImage:destinationImage:)) {
    
    if (_padding) {
        long int pad_along_height = ((destinationImage.height - 1) * self.strideInPixelsY + self.kernelHeight - sourceImage.height);
        long int pad_along_width = ((destinationImage.width - 1) * self.strideInPixelsX + self.kernelWidth - sourceImage.width);
        long int pad_top = pad_along_height / 2;
        long int pad_left = pad_along_width / 2;
        
        MPSOffset offset;
        offset.x = self.kernelWidth / 2 - pad_left;
        offset.y = self.kernelHeight / 2 - pad_top;
        offset.z = 0;
        self.offset = offset;
    } else {
        MPSOffset offset;
        offset.x = self.kernelWidth / 2;
        offset.y = self.kernelHeight / 2;
        offset.z = 0;
        self.offset = offset;
    }
    
    [super encodeToCommandBuffer:commandBuffer
                     sourceImage:sourceImage
                destinationImage:destinationImage];
}

@end

@implementation SlimMPSCNNPoolingGlobalAverage

- (SlimMPSCNNPoolingGlobalAverage *) initWithDevice:(id <MTLDevice>)device
                                         kernelSize:(NSUInteger)kernelSize{
    if (self = [super initWithDevice:device
                         kernelWidth:kernelSize
                        kernelHeight:kernelSize
                     strideInPixelsX:0
                     strideInPixelsY:0]) {
        MPSOffset offset;
        offset.x = self.kernelWidth / 2;
        offset.y = self.kernelHeight / 2;
        offset.z = 0;
        self.offset = offset;
        
        self.edgeMode = MPSImageEdgeModeClamp;
    }
    
    return self;
}

@end

@implementation SlimMPSCNNLocalResponseNormalization

- (SlimMPSCNNLocalResponseNormalization *) initWithDevice:(id<MTLDevice>)device
                                                localSize:(NSUInteger)localSize
                                                    alpha:(float)alpha
                                                     beta:(float)beta {
    if (self = [super initWithDevice:device kernelSize:localSize]) {
        self.alpha = alpha;
        self.beta = beta;
    }
    
    return self;
}

@end

