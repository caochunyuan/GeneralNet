//
//  CPULayer.h
//  GeneralNet
//
//  Created by Lun on 2017/5/23.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

@interface CPULayer : NSObject

@property (strong, nonatomic) NSString *name;
@property (assign, nonatomic) float *srcPtr;
@property (assign, nonatomic) float *dstPtr;

- (void)forward;

@end

@interface CPUConvolutionLayer : CPULayer

@property (assign, nonatomic) float *weight;
@property (assign, nonatomic) float *bias;
@property (assign, nonatomic) int inputChannel;
@property (assign, nonatomic) int outputChannel;
@property (assign, nonatomic) int inputSize;
@property (assign, nonatomic) int outputSize;
@property (assign, nonatomic) int kernelSize;
@property (assign, nonatomic) int pad;
@property (assign, nonatomic) int stride;
@property (assign, nonatomic) int group;
@property (assign, nonatomic) BOOL doReLU;
@property (assign, nonatomic) float zero;
@property (assign, nonatomic) float *colData;
@property (assign, nonatomic) int M;
@property (assign, nonatomic) int N;
@property (assign, nonatomic) int K;
@property (assign, nonatomic) int inputPerGroup;
@property (assign, nonatomic) int outputPerGroup;
@property (assign, nonatomic) int weightPerGroup;

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
                      doReLU:(BOOL)doReLU;

@end

@interface CPUFullyConnectedLayer : CPULayer

@property (assign, nonatomic) float *weight;
@property (assign, nonatomic) float *bias;
@property (assign, nonatomic) int inputChannel;
@property (assign, nonatomic) int outputChannel;
@property (assign, nonatomic) int inputSize;
@property (assign, nonatomic) BOOL doReLU;
@property (assign, nonatomic) float zero;
@property (assign, nonatomic) int M;
@property (assign, nonatomic) int N;

- (instancetype)initWithName:(NSString *)name
                      weight:(float *)weight
                        bias:(float *)bias
                inputChannel:(int)inputChannel
               outputChannel:(int)outputChannel
                   inputSize:(int)inputSize
                      doReLU:(BOOL)doReLU;

@end

@interface CPUPoolingMaxLayer : CPULayer

@property (assign, nonatomic) BNNSFilter filter;

- (instancetype)initWithName:(NSString *)name
                inputChannel:(int)inputChannel
               outputChannel:(int)outputChannel
                   inputSize:(int)inputSize
                  outputSize:(int)outputSize
                  kernelSize:(int)kernelSize
                         pad:(int)pad
                      stride:(int)stride;

@end
