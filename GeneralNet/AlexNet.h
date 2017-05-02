//
//  AlexNet.h
//  GeneralNet
//
//  Created by Lun on 2017/4/18.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <sys/mman.h>
#import "GeneralNet.h"
#import "SlimMPSCNN.h"

@interface AlexNet : GeneralNet

@property (nonatomic) float *basePtr;
@property (nonatomic) int fd;

@property (nonatomic) id <MTLComputePipelineState> pipelineRGB;
@property (nonatomic) MPSImageLanczosScale *lanczos;
@property (nonatomic) MPSImageDescriptor *input_id;
@property (nonatomic) MPSImage *sftImage;

- (instancetype)initWithCommandQueue:(id <MTLCommandQueue>)commandQueue;
- (void)forwardWithCommandBuffer:(id <MTLCommandBuffer>)commandBuffer
                   sourceTexture:(id <MTLTexture>)sourceTexture;
- (NSString *) getLabels;

@end
