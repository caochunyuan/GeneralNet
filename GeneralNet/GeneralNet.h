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

@interface GeneralNet : NSObject <GeneralNetProtocol> {
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
    
    NSString *m_FirstLayerName;
    NSString *m_LastLayerName;
    NSArray *m_Labels;
    NSDictionary *m_LayersDict;
    NSArray *m_EncodeSequence;
    NSArray *m_PrefetchList;
    NSArray *m_TempImageList;
}

@end

#pragma mark - if not use Metal
#else

@interface GeneralNet : NSObject <GeneralNetProtocol> {
@protected
    
    float *m_BasePtr;
    int m_Fd;
    size_t m_FileSize;
    int m_InputSize;
    unsigned char *m_ImageRawData;
    float *m_ImageData;
    float *m_ColData;
    
    NSString *m_FirstLayerName;
    NSString *m_LastLayerName;
    NSArray *m_Labels;
    NSDictionary *m_LayersDict;
    NSArray *m_EncodeSequence;
}

@end

#endif
