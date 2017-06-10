//
//  CPULayer.h
//  GeneralNet
//
//  Created by Lun on 2017/5/23.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface CPULayer : NSObject

@property (readonly, nonatomic) NSString *name;
@property (assign, nonatomic) float *output;
@property (assign, nonatomic) int destinationOffset;
@property (assign, nonatomic) int outputNum;

- (instancetype)initWithName:(NSString *)name;

// subclass of CPULayer should overwrite this method
- (void)forwardWithInput:(const float *)input
                  output:(float *)output;

@end

@interface CPUConvolutionLayer : CPULayer {
@protected
    float *m_Weight;
    float *m_Biases;
    int m_InputChannel;
    int m_OutputChannel;
    int m_InputSize;
    int m_OutputSize;
    int m_KernelSize;
    int m_Pad;
    int m_Stride;
    int m_Group;
    BOOL m_ReLU;
    float m_Zero;
    float *m_ColData;
    int m_M;
    int m_N;
    int m_K;
    int m_InputPerGroup;
    int m_OutputPerGroup;
    int m_WeightPerGroup;
}

- (instancetype)initWithName:(NSString *)name
                      weight:(float *)weight
                        bias:(float *)bias
                       group:(int)group
                inputChannel:(int)inputChannel
               outputChannel:(int)outputChannel
                   inputSize:(int)inputSize
                  outputSize:(int)outputSize
                  kernelSize:(int)kernelSize
                         pad:(int)pad
                      stride:(int)stride
                      doReLU:(BOOL)doReLU
                     colData:(float *)colData;

@end

@interface CPUFullyConnectedLayer : CPULayer {
@protected
    float *m_Weight;
    float *m_Biases;
    int m_InputChannel;
    int m_OutputChannel;
    int m_InputSize;
    BOOL m_ReLU;
    float m_Zero;
    int m_M;
    int m_N;
}

- (instancetype)initWithName:(NSString *)name
                      weight:(float *)weight
                        bias:(float *)bias
                inputChannel:(int)inputChannel
               outputChannel:(int)outputChannel
                   inputSize:(int)inputSize
                      doReLU:(BOOL)doReLU;

@end

@interface CPUPoolingLayer : CPULayer

typedef NS_ENUM (NSInteger, PoolingLayerTypes) {
    ePoolingMax = 0,
    ePoolingAverage,
    ePoolingGlobalAverage
};

- (instancetype)initWithName:(NSString *)name
                 poolingType:(PoolingLayerTypes)poolingType
                inputChannel:(int)inputChannel
               outputChannel:(int)outputChannel
                   inputSize:(int)inputSize
                  outputSize:(int)outputSize
                  kernelSize:(int)kernelSize
                         pad:(int)pad
                      stride:(int)stride;

@end

@interface CPULocalResponseNormalizationLayer : CPULayer {
@protected
    int m_InputChannel;
    int m_InputSize;
    float m_AlphaOverN;
    float *m_Beta;
    float m_Delta;
    int m_LocalSize;
    int m_Pad;
    float m_One;
    int m_InputPerChannel;
    int m_PaddedPerChannel;
    float *m_MidShort;
    float *m_MidLong;
}

- (instancetype)initWithName:(NSString *)name
                inputChannel:(int)inputChannel
                   inputSize:(int)inputSize
                       alpha:(float)alpha
                        beta:(float)beta
                       delta:(float)delta
                   localSize:(int)localSize;

@end

@interface CPUSoftMaxLayer : CPULayer {
@protected
    int m_InputChannel;
    float *m_Mid;
}

- (instancetype)initWithName:(NSString *)name
                inputChannel:(int)inputChannel;

@end
