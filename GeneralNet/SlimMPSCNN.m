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
    /**
     A property to keep info from init time whether we will pad input image or not for use during encode call
     */
    BOOL _padding;
}

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
                                          group:(uint)group {
    
    // create appropriate convolution descriptor with appropriate stride
    MPSCNNConvolutionDescriptor *convDesc;
    convDesc = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:width
                                                                       kernelHeight:height
                                                               inputFeatureChannels:inChannels
                                                              outputFeatureChannels:outChannels
                                                                       neuronFilter:neuron];
    convDesc.strideInPixelsX = strideX;
    convDesc.strideInPixelsY = strideY;
    
    NSAssert(group > 0, @"Group size can't be less than 1");
    convDesc.groups = group;
    
    // initialize the convolution layer by calling the parent's (MPSCNNConvlution's) initializer
    if (self = [super initWithDevice:device
               convolutionDescriptor:convDesc
                       kernelWeights:weights
                           biasTerms:bias
                               flags:MPSCNNConvolutionFlagsNone]) {
        self.destinationFeatureChannelOffset = offset;
        
        // set padding for calculation of offset during encode call
        self.padding = willPad;
    }
    
    return self;
}

- (SlimMPSCNNConvolution *) initWithKernelSize:(uint)kernelSize
                          inputFeatureChannels:(uint)inChannels
                         outputFeatureChannels:(uint)outChannels
                                        neuron:(MPSCNNNeuron *)neuron
                                        device:(id <MTLDevice>)device
                                       weights:(const float *)weights
                                          bias:(const float *)bias
                                       willPad:(BOOL)willPad
                                        stride:(uint)stride
               destinationFeatureChannelOffset:(uint)offset
                                         group:(uint)group {
    
    // create appropriate convolution descriptor with appropriate stride
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
    
    // initialize the convolution layer by calling the parent's (MPSCNNConvlution's) initializer
    if (self = [super initWithDevice:device
               convolutionDescriptor:convDesc
                       kernelWeights:weights
                           biasTerms:bias
                               flags:MPSCNNConvolutionFlagsNone]) {
        self.destinationFeatureChannelOffset = offset;
        
        // set padding for calculation of offset during encode call
        self.padding = willPad;
    }
    
    return self;
}

- (void) encodeToCommandBuffer:(nonnull id <MTLCommandBuffer>) commandBuffer
                   sourceImage:(MPSImage * __nonnull) sourceImage
              destinationImage:(MPSImage * __nonnull) destinationImage
MPS_SWIFT_NAME(encode(commandBuffer:sourceImage:destinationImage:)) {
                    
    // select offset according to padding being used or not
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

- (SlimMPSCNNFullyConnected *) initWithKernelWidth:(uint)width
                                      kernelHeight:(uint)height
                              inputFeatureChannels:(uint)inChannels
                             outputFeatureChannels:(uint)outChannels
                                            neuron:(MPSCNNNeuron *)neuron
                                            device:(id <MTLDevice>)device
                                           weights:(const float *)weights
                                              bias:(const float *)bias
                   destinationFeatureChannelOffset:(uint)offset {
    
    // create appropriate convolution descriptor (in fully connected, stride is always 1)
    MPSCNNConvolutionDescriptor *convDesc;
    convDesc = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:width
                                                                       kernelHeight:height
                                                               inputFeatureChannels:inChannels
                                                              outputFeatureChannels:outChannels
                                                                       neuronFilter:neuron];
    
    // initialize the convolution layer by calling the parent's (MPSCNNFullyConnected's) initializer
    self = [super initWithDevice:device
           convolutionDescriptor:convDesc
                   kernelWeights:weights
                       biasTerms:bias
                           flags:MPSCNNConvolutionFlagsNone];
    self.destinationFeatureChannelOffset = offset;
    
    return self;
}

- (SlimMPSCNNFullyConnected *) initWithKernelSize:(uint)kernelSize
                             inputFeatureChannels:(uint)inChannels
                            outputFeatureChannels:(uint)outChannels
                                           neuron:(MPSCNNNeuron *)neuron
                                           device:(id <MTLDevice>)device
                                          weights:(const float *)weights
                                             bias:(const float *)bias
                  destinationFeatureChannelOffset:(uint)offset{
    
    // create appropriate convolution descriptor (in fully connected, stride is always 1)
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

@implementation SlimMPSCNNPoolingMax

- (SlimMPSCNNPoolingMax *) initWithDevice:(id <MTLDevice>)device
                              kernelWidth:(NSUInteger)kernelWidth
                             kernelHeight:(NSUInteger)kernelHeight
                          strideInPixelsX:(NSUInteger)strideInPixelsX
                          strideInPixelsY:(NSUInteger)strideInPixelsY
                                  willPad:(BOOL)willPad {
    if (self = [super initWithDevice:device
                         kernelWidth:kernelWidth
                        kernelHeight:kernelHeight
                     strideInPixelsX:strideInPixelsX
                     strideInPixelsY:strideInPixelsY]) {
        _padding = willPad;
    }
    
    return self;
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
    
    // select offset according to padding being used or not
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
                                        kernelWidth:(NSUInteger)kernelWidth
                                       kernelHeight:(NSUInteger)kernelHeight {
    if (self = [super initWithDevice:device
                         kernelWidth:kernelWidth
                        kernelHeight:kernelHeight
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
                                                localSize:(uint)localSize
                                                    alpha:(float)alpha
                                                     beta:(float)beta {
    if (self = [super initWithDevice:device kernelSize:localSize]) {
        self.alpha = alpha;
        self.beta = beta;
    }
    
    return self;
}

@end

