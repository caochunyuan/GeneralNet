//
//  MPSNet.h
//  GeneralNet
//
//  Created by Lun on 2017/8/19.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import "GeneralNetProtocol.h"
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
