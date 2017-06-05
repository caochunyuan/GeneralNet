//
//  GeneralNet.h
//  GeneralNet
//
//  Created by Lun on 2017/5/9.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <sys/mman.h>

#pragma mark - protocol for both GPU and CPU
@protocol GeneralNetProtocol

- (instancetype)initWithDescriptionFile:(NSString *)descriptionFile
                               dataFile:(NSString *)dataFile;
- (void)forwardWithImage:(UIImage *)image
              completion:(void (^)())completion;
- (NSString *)getTopProbs;

@end

#pragma mark - if use Metal
#if USE_METAL

#import <MetalKit/MetalKit.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface GeneralNet : NSObject <GeneralNetProtocol>

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

@end

#pragma mark - if not use Metal
#else

@interface GeneralNet : NSObject <GeneralNetProtocol>

@property (assign, nonatomic) float *basePtr;
@property (assign, nonatomic) int fd;
@property (assign, nonatomic) size_t fileSize;
@property (assign, nonatomic) int inputSize;
@property (assign, nonatomic) unsigned char *imageRawData;
@property (assign, nonatomic) float *imageData;
@property (assign, nonatomic) float *colData;

@property (strong, nonatomic) NSString *firstLayerName;
@property (strong, nonatomic) NSString *lastLayerName;
@property (strong, nonatomic) NSArray *labels;
@property (strong, nonatomic) NSMutableDictionary *layersDict;
@property (strong, nonatomic) NSMutableArray *encodeSequence;

@end

#endif
