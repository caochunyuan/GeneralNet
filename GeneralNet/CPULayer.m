//
//  CPULayer.m
//  GeneralNet
//
//  Created by Lun on 2017/5/23.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import "CPULayer.h"
#import <Accelerate/Accelerate.h>
#import "gemmHandler.h"

@implementation CPULayer

- (instancetype)initWithName:(NSString *)name {
    if (self = [super init]) {
        _name = name;
    }
    
    return self;
}

- (void)forwardWithInput:(const float *)input
                  output:(float *)output {
    // subclass of CPULayer should overwrite this method
}

- (void)dealloc {
    if (_output) free(_output);
}

@end

@implementation CPUConvolutionLayer

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
                     colData:(float *)colData {
    if (self = [super initWithName:name]) {
        m_Weight = weight;
        m_Biases = bias;
        m_Group = group;
        m_InputChannel = inputChannel / m_Group;
        m_OutputChannel = outputChannel / m_Group;
        m_InputSize = inputSize;
        m_OutputSize = outputSize;
        m_KernelSize = kernelSize;
        m_Pad = pad;
        m_Stride = stride;
        m_ReLU = doReLU;
        m_Zero = 0.0f;
        m_ColData = colData;
        m_M = m_OutputChannel;
        m_N = m_OutputSize * m_OutputSize;
        m_K = m_InputChannel * m_KernelSize * m_KernelSize;
        m_InputPerGroup = m_InputChannel * m_InputSize * m_InputSize;
        m_OutputPerGroup = m_OutputChannel * m_OutputSize * m_OutputSize;
        m_WeightPerGroup = m_OutputChannel * m_InputChannel * m_KernelSize * m_KernelSize;
    }
    
    return self;
}

- (void)forwardWithInput:(const float *)input
                  output:(float *)output {
    for (int groupIndex = 0; groupIndex < m_Group; groupIndex++) {
        const float *src = input + groupIndex * m_InputPerGroup;
        float *dst = output + groupIndex * m_OutputPerGroup;
        im2col(src, m_InputChannel, m_InputSize, m_InputSize, m_OutputSize, m_OutputSize, m_KernelSize, m_KernelSize, 1, 1, m_Pad, m_Pad, m_Pad, m_Pad, m_Stride, m_Stride, m_ColData);
        for (int featureIndex = 0; featureIndex < m_N; featureIndex++) {
            memcpy(dst + featureIndex * m_M, m_Biases + groupIndex * m_M, m_M * sizeof(float));
        }
        
        [gemmHandler gemmWithTransA:gemmNoTrans
                             transB:gemmNoTrans
                                  M:m_M
                                  N:m_N
                                  K:m_K
                              alpha:1
                                  A:m_Weight + groupIndex * m_WeightPerGroup
                                  B:m_ColData
                               beta:1
                                  C:dst];
    }
    if (m_ReLU) vDSP_vthres(output, 1, &m_Zero, output, 1, m_OutputPerGroup * m_Group);
}

static void im2col (const float* data_im,
                    const int channels,
                    const int input_h,
                    const int input_w,
                    const int output_h,
                    const int output_w,
                    const int kernel_h,
                    const int kernel_w,
                    const int dilation_h,
                    const int dilation_w,
                    const int pad_t,
                    const int pad_l,
                    const int pad_b,
                    const int pad_r,
                    const int stride_h,
                    const int stride_w,
                    float* data_col) {
    
    // Fast path for zero padding and no dilation
    // From Torch, THNN_(unfolded_copy)
    if (dilation_h == 1 && dilation_w == 1 && pad_l == 0 && pad_r == 0 &&
        pad_t == 0 && pad_b == 0) {
        for (int k = 0; k < channels * kernel_h * kernel_w; k++) {
            const int nip = k / (kernel_h * kernel_w);
            const int rest = k % (kernel_h * kernel_w);
            const int kh = rest / kernel_w;
            const int kw = rest % kernel_w;
            float* dst = data_col + nip * (kernel_h * kernel_w * output_h * output_w) +
            kh * (kernel_w * output_h * output_w) + kw * (output_h * output_w);
            const float* src = data_im + nip * (input_h * input_w);
            for (int y = 0; y < output_h; y++) {
                const int iy = y * stride_h + kh;
                const int ix = kw;
                if (stride_w == 1) {
                    memcpy(
                           dst + (y * output_w),
                           src + (iy * input_w + ix),
                           sizeof(float) * output_w);
                } else {
                    for (int x = 0; x < output_w; x++) {
                        memcpy(
                               dst + (y * output_w + x),
                               src + (iy * input_w + ix + x * stride_w),
                               sizeof(float));
                    }
                }
            }
        }
        return;
    }
    
    // Fast path for equal padding
    if (pad_l == pad_r && pad_t == pad_b) {
        // From Intel, https://github.com/BVLC/caffe/pull/3536
        const int pad_h = pad_t;
        const int pad_w = pad_l;
        const int channel_size = input_h * input_w;
        for (int channel = channels; channel--; data_im += channel_size) {
            for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
                for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                    int input_row = -pad_h + kernel_row * dilation_h;
                    for (int output_rows = output_h; output_rows; output_rows--) {
                        if (!((unsigned int)input_row < (unsigned int)input_h)) {
                            for (int output_cols = output_w; output_cols; output_cols--) {
                                *(data_col++) = 0;
                            }
                        } else {
                            int input_col = -pad_w + kernel_col * dilation_w;
                            for (int output_col = output_w; output_col; output_col--) {
                                if ((unsigned int)input_col < (unsigned int)input_w) {
                                    *(data_col++) = data_im[input_row * input_w + input_col];
                                } else {
                                    *(data_col++) = 0;
                                }
                                input_col += stride_w;
                            }
                        }
                        input_row += stride_h;
                    }
                }
            }
        }
        return;
    }
    
    // Baseline
    const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
    const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
    
    int height_col = (input_h + pad_t + pad_b - dkernel_h) / stride_h + 1;
    int width_col = (input_w + pad_l + pad_r - dkernel_w) / stride_w + 1;
    
    int channels_col = channels * kernel_h * kernel_w;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int h_pad = h * stride_h - pad_t + h_offset * dilation_h;
                int w_pad = w * stride_w - pad_l + w_offset * dilation_w;
                if (h_pad >= 0 && h_pad < input_h && w_pad >= 0 && w_pad < input_w)
                    data_col[(c * height_col + h) * width_col + w] =
                    data_im[(c_im * input_h + h_pad) * input_w + w_pad];
                else
                    data_col[(c * height_col + h) * width_col + w] = 0;
            }
        }
    }
}

@end

@implementation CPUFullyConnectedLayer

- (instancetype)initWithName:(NSString *)name
                      weight:(float *)weight
                        bias:(float *)bias
                inputChannel:(int)inputChannel
               outputChannel:(int)outputChannel
                   inputSize:(int)inputSize
                      doReLU:(BOOL)doReLU {
    if (self = [super initWithName:name]) {
        m_Weight = weight;
        m_Biases = bias;
        m_InputChannel = inputChannel;
        m_OutputChannel = outputChannel;
        m_InputSize = inputSize;
        m_ReLU = doReLU;
        m_Zero = 0.0f;
        m_M = m_OutputChannel;
        m_N = m_InputSize * m_InputSize * m_InputChannel;
    }
    
    return self;
}

- (void)forwardWithInput:(const float *)input
                  output:(float *)output {
    memcpy(output, m_Biases, m_OutputChannel * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m_M, m_N, 1, m_Weight, m_N, input, 1, 1,
                output, 1);
    if (m_ReLU) vDSP_vthres(output, 1, &m_Zero, output, 1, m_OutputChannel);
}

@end

@implementation CPUPoolingLayer

- (instancetype)initWithName:(NSString *)name
                 poolingType:(PoolingLayerTypes)poolingType
                inputChannel:(int)inputChannel
               outputChannel:(int)outputChannel
                   inputSize:(int)inputSize
                  outputSize:(int)outputSize
                  kernelSize:(int)kernelSize
                         pad:(int)pad
                      stride:(int)stride {
    if (self = [super initWithName:name]) {
        m_PoolingType = poolingType;
        
        switch (m_PoolingType) {
            case ePoolingMax:
            case ePoolingAverage:
                m_InputSize = inputSize;
                m_OutputSize = outputSize;
                m_InputChannel = inputChannel;
                m_KernelSize = kernelSize;
                m_Pad = pad;
                m_Stride = stride;
                break;
            case ePoolingGlobalAverage:
                m_InputChannel = inputChannel;
                m_InputSize = inputSize;
                break;
            default:
                assert("Unknown pooling layer type!");
                break;
        }
    }
    
    return self;
}

- (void)forwardWithInput:(const float *)input
                  output:(float *)output {
    switch (m_PoolingType) {
        case ePoolingMax:
            for (int channelIndex = 0; channelIndex < m_InputChannel; channelIndex++) {
                computeMaxPooling(input + channelIndex * m_InputSize * m_InputSize, output + channelIndex * m_OutputSize * m_OutputSize, m_InputSize, m_InputSize, m_Pad, m_Pad, m_OutputSize, m_OutputSize, m_Stride, m_Stride, m_KernelSize, m_KernelSize);
            }
            break;
        case ePoolingAverage:
            for (int channelIndex = 0; channelIndex < m_InputChannel; channelIndex++) {
                computeAveragePooling(input + channelIndex * m_InputSize * m_InputSize, output + channelIndex * m_OutputSize * m_OutputSize, m_InputSize, m_InputSize, m_Pad, m_Pad, m_OutputSize, m_OutputSize, m_Stride, m_Stride, m_KernelSize, m_KernelSize);
            }
            break;
        case ePoolingGlobalAverage:
            computeGlobalAveragePooling(input, output, m_InputSize, m_InputSize, m_InputChannel);
            break;
        default:
            break;
    }
}

static void computeMaxPooling(const float *input_pointer,
                              float *output_pointer,
                              size_t input_height,
                              size_t input_width,
                              size_t padding_top,
                              size_t padding_left,
                              size_t output_height,
                              size_t output_width,
                              size_t stride_height,
                              size_t stride_width,
                              size_t pooling_height,
                              size_t pooling_width) {
    const float (*input)[input_width] = (const float(*)[input_width]) input_pointer;
    float (*output)[output_width] = (float(*)[output_width]) output_pointer;
    
    for (size_t y = 0; y < output_height; y++) {
        for (size_t x = 0; x < output_width; x++) {
            float v = -__builtin_inff();
            for (size_t i = 0; i < pooling_height; i++) {
                const size_t s = y * stride_height + i - padding_top;
                if (s < input_height) {
                    for (size_t j = 0; j < pooling_width; j++) {
                        const size_t t = x * stride_width + j - padding_left;
                        if (t < input_width) {
                            v = fmaxf(input[s][t], v);
                        }
                    }
                }
            }
            output[y][x] = v;
        }
    }
}

static void computeAveragePooling(const float *input_pointer,
                                  float *output_pointer,
                                  size_t input_height,
                                  size_t input_width,
                                  size_t padding_top,
                                  size_t padding_left,
                                  size_t output_height,
                                  size_t output_width,
                                  size_t stride_height,
                                  size_t stride_width,
                                  size_t pooling_height,
                                  size_t pooling_width) {
    const float (*input)[input_width] = (const float(*)[input_width]) input_pointer;
    float (*output)[output_width] = (float(*)[output_width]) output_pointer;
    
    for (size_t y = 0; y < output_height; y++) {
        for (size_t x = 0; x < output_width; x++) {
            float sum = 0;
            for (size_t i = 0; i < pooling_height; i++) {
                const size_t s = y * stride_height + i - padding_top;
                if (s < input_height) {
                    for (size_t j = 0; j < pooling_width; j++) {
                        const size_t t = x * stride_width + j - padding_left;
                        if (t < input_width) {
                            sum += input[s][t];
                        }
                    }
                }
            }
            output[y][x] = sum / (float)(pooling_width * pooling_height);
        }
    }
}

static void computeGlobalAveragePooling(const float *input_pointer,
                                        float *output_pointer,
                                        size_t input_height,
                                        size_t input_width,
                                        size_t input_channel) {
    for (int channelIndex = 0; channelIndex < input_channel; channelIndex++) {
        vDSP_sve(input_pointer + channelIndex * input_width * input_height, 1,
                 output_pointer + channelIndex, input_width * input_height);
    }
    float denom = input_width * input_height;
    vDSP_vsdiv(output_pointer, 1, &denom, output_pointer, 1, input_channel);
}

@end

@implementation CPULocalResponseNormalizationLayer

- (instancetype)initWithName:(NSString *)name
                inputChannel:(int)inputChannel
                   inputSize:(int)inputSize
                       alpha:(float)alpha
                        beta:(float)beta
                       delta:(float)delta
                   localSize:(int)localSize {
    if (self = [super initWithName:name]) {
        m_InputChannel = inputChannel;
        m_InputSize = inputSize;
        m_InputPerChannel = inputSize * inputSize;
        m_LocalSize = localSize;
        m_AlphaOverN = alpha / m_LocalSize;
        m_Beta = (float *)malloc(m_InputPerChannel * sizeof(float));
        vDSP_vfill(&beta, m_Beta, 1, m_InputPerChannel);
        m_Delta = delta;
        m_Pad =  (localSize - 1) / 2;
        m_PaddedPerChannel = m_InputPerChannel + 2 * m_Pad;
        m_MidShort = (float *)malloc(m_InputPerChannel * sizeof(float));
        m_MidLong = (float *)malloc(m_PaddedPerChannel * sizeof(float));
        m_One = 1.0f;
    }
    
    return self;
}

- (void)forwardWithInput:(const float *)input
                  output:(float *)output {
    for (int channelIndex = 0; channelIndex < m_InputChannel; channelIndex++) {
        const float *src = input + channelIndex * m_InputPerChannel;
        float *dst = output + channelIndex * m_InputPerChannel;
        vDSP_vsq(src, 1, m_MidShort, 1, m_InputPerChannel);                                           // square of each element
        memset(m_MidLong, 0, m_PaddedPerChannel * sizeof(float));
        for (int regionIndex = 0; regionIndex < m_LocalSize; regionIndex++) {                        // sum up nearby channels
            vDSP_vadd(m_MidLong + regionIndex, 1, m_MidShort, 1, m_MidLong + regionIndex, 1, m_InputPerChannel);
        }
        vDSP_vsmsa(m_MidLong + m_Pad, 1, &m_AlphaOverN, &m_Delta, m_MidShort, 1, m_InputPerChannel);      // denom = delta + (alpha / N) * sum
        vvpowf(m_MidShort, m_Beta, m_MidShort, &m_InputPerChannel);                                     // denom = denom ^ beta
        vDSP_vdiv(m_MidShort, 1, src, 1, dst, 1, m_InputPerChannel);                                  // norm_result = origin / denom
    }
}

- (void)dealloc {
    free(m_MidShort);
    free(m_MidLong);
}

@end

@implementation CPUSoftMaxLayer

- (instancetype)initWithName:(NSString *)name
                inputChannel:(int)inputChannel{
    if (self = [super initWithName:name]) {
        m_InputChannel = inputChannel;
        m_Mid = (float *)malloc(m_InputChannel * sizeof(float));
    }
    
    return self;
}

- (void)forwardWithInput:(const float *)input
                  output:(float *)output {
    float max;
    vDSP_maxv(input, 1, &max, m_InputChannel);                 // find maximum
    max *= -1;
    vDSP_vsadd(input, 1, &max, m_Mid, 1, m_InputChannel);       // subtract the maximum
    vvexpf(m_Mid, m_Mid, &m_InputChannel);                       // exponential of each element
    float sum;
    vDSP_sve(m_Mid, 1, &sum, m_InputChannel);                   // sum of exponential of all elements
    vDSP_vsdiv(m_Mid, 1, &sum, output, 1, m_InputChannel);      // divide by the sum of exponential
}

- (void)dealloc {
    free(m_Mid);
}

@end
