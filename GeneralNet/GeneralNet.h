//
//  GeneralNet.h
//  GeneralNet
//
//  Created by Lun on 2017/5/9.
//  Copyright © 2017年 Lun. All rights reserved.
//

#pragma mark - if use Metal
#if USE_METAL

#import <Foundation/Foundation.h>
#import <MetalKit/MetalKit.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>
#import <sys/mman.h>
#import "SlimMPSCNN.h"

@interface GeneralLayer : NSObject

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

@interface GeneralNet : NSObject

@property (assign, nonatomic) float *basePtr;
@property (assign, nonatomic) int fd;

@property (strong, nonatomic) id <MTLDevice> device;
@property (strong, nonatomic) id <MTLCommandQueue> commandQueue;
@property (strong, nonatomic) id <MTLTexture> sourceTexture;
@property (strong, nonatomic) id <MTLComputePipelineState> pipelineRGB;
@property (strong, nonatomic) MTKTextureLoader *textureLoader;
@property (strong, nonatomic) MPSImageLanczosScale *lanczos;
@property (strong, nonatomic) MPSImageDescriptor *input_id;

@property (strong, nonatomic) NSString *firstLayerName;
@property (strong, nonatomic) NSString *lastLayerName;
@property (strong, nonatomic) NSArray *labels;
@property (strong, nonatomic) NSMutableDictionary *layersDict;
@property (strong, nonatomic) NSMutableArray *encodeSequence;
@property (strong, nonatomic) NSMutableArray *prefetchList;
@property (strong, nonatomic) NSMutableArray *tempImageList;

- (instancetype)initWithDescriptionFile:(NSString *)descriptionFile
                               dataFile:(NSString *)dataFile;
- (NSString *)forwardWithImage:(UIImage *)image;

@end

#pragma mark - if not use Metal
#else

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <Accelerate/Accelerate.h>
#import <sys/mman.h>
#import "CPULayer.h"

@interface GeneralNet : NSObject

@property (assign, nonatomic) float *basePtr;
@property (assign, nonatomic) int fd;
@property (assign, nonatomic) size_t fileSize;
@property (assign, nonatomic) int inputSize;
@property (assign, nonatomic) unsigned char *imageRawData;
@property (assign, nonatomic) float *imageData;

@property (strong, nonatomic) NSString *firstLayerName;
@property (strong, nonatomic) NSString *lastLayerName;
@property (strong, nonatomic) NSArray *labels;
@property (strong, nonatomic) NSMutableDictionary *layersDict;
@property (strong, nonatomic) NSMutableArray *encodeSequence;

- (instancetype)initWithDescriptionFile:(NSString *)descriptionFile
                               dataFile:(NSString *)dataFile;
- (NSString *)forwardWithImage:(UIImage *)image;

@end

#endif
