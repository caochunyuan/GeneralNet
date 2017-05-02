//
//  GeneralNet.h
//  GeneralNet
//
//  Created by Lun on 2017/4/18.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface GeneralLayer : NSObject

@property (strong, nonatomic) MPSImageDescriptor *imageDescriptor;
@property (strong, nonatomic) MPSImage *outputImage;
@property (assign, nonatomic) NSUInteger readCount;
@property (strong, nonatomic) MPSCNNKernel *kernel;
@property (strong, nonatomic) GeneralLayer *concatLayer;

- (instancetype)initWithImageDescriptor:(MPSImageDescriptor *)imageDescritor
                              readCount:(NSUInteger)readCount
                            outputImage:(MPSImage *)outputImage
                                 kernel:(MPSCNNKernel *)kernel;

@end

@interface GeneralNet : NSObject

@property (strong, nonatomic) id <MTLDevice> device;
@property (strong, nonatomic) id <MTLCommandQueue> commandQueue;
@property (strong, nonatomic) NSMutableArray *layers;
@property (strong, nonatomic) NSMutableArray *descriptors;
@property (strong, nonatomic) NSMutableArray *encodeList;
@property (readonly, nonatomic) uint textureFormat;

- (instancetype)initWithCommandQueue:(id <MTLCommandQueue>)commandQueue;
- (void)addLayer:(GeneralLayer *)layer;
- (MPSImage *)getOutputImage;
- (void)forwardWithCommandBuffer:(id <MTLCommandBuffer>)commandBuffer
                     sourceImage:(MPSImage *)sourceImage;

@end
