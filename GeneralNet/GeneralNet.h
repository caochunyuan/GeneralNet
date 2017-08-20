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

#pragma mark - protocol for both GPU and CPU implemention
@protocol GeneralNetProtocol

+ (id <GeneralNetProtocol>)netWithDescriptionFilename:(NSString *)descriptionFilename
                                         dataFilename:(NSString *)dataFilename;
- (void)forwardWithImage:(UIImage *)image
              completion:(void (^)())completion;
- (NSString *)labelsOfTopProbs;

@end

#pragma mark - if use Metal
#if USE_METAL

#import <MetalKit/MetalKit.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@class MPSLayer;

@interface MPSNet : NSObject <GeneralNetProtocol> {
@protected
    
    float *m_BasePtr;
    int m_Fd;
    
    id <MTLDevice>  m_Device;
    id <MTLCommandQueue> m_CommandQueue;
    id <MTLTexture> m_SourceTexture;
    id <MTLComputePipelineState> m_PipelineRGB;
    MTKTextureLoader *m_TextureLoader;
    MPSImageLanczosScale *m_Lanczos;
    MPSImageDescriptor *m_InputImageDescriptor;
    
    MPSLayer *m_FirstLayer;
    MPSLayer *m_LastLayer;
    NSDictionary *m_LayersDict;
    NSArray *m_EncodeSequence;
    NSArray *m_PrefetchList;
    NSArray *m_TempImageList;
    NSArray *m_Labels;
}

@end

#pragma mark - if not use Metal
#else

@class CPULayer;

@interface CPUNet : NSObject <GeneralNetProtocol> {
@protected
    
    float *m_BasePtr;
    int m_Fd;
    size_t m_FileSize;
    int m_InputSize;
    unsigned char *m_ImageRawData;
    float *m_ImageData;
    float *m_ColData;
    
    CPULayer *m_FirstLayer;
    CPULayer *m_LastLayer;
    NSDictionary *m_LayersDict;
    NSArray *m_EncodeSequence;
    NSArray *m_Labels;
}

@end

#endif
