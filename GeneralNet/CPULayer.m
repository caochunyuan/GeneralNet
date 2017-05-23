//
//  CPULayer.m
//  GeneralNet
//
//  Created by Lun on 2017/5/23.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import "CPULayer.h"

@implementation CPULayer

- (void)forward {
    // subclass of CPULayer should overwrite this method
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
                      doReLU:(BOOL)doReLU {
    if (self = [super init]) {
        self.name = name;
        _weight = weight;
        _bias = bias;
        _group = group;
        _inputChannel = inputChannel / _group;
        _outputChannel = outputChannel / _group;
        _inputSize = inputSize;
        _outputSize = outputSize;
        _kernelSize = kernelSize;
        _pad = pad;
        _stride = stride;
        _doReLU = doReLU;
        _zero = 0;
        _colData = malloc(sizeof(float) * _inputSize * _inputSize * _inputChannel
                          * _kernelSize * _kernelSize);
        _M = _outputChannel;
        _N = _outputSize * _outputSize;
        _K = _inputChannel * _kernelSize * _kernelSize;
        _inputPerGroup = _inputChannel * _inputSize * _inputSize;
        _outputPerGroup = _outputChannel * _outputSize * _outputSize;
        _weightPerGroup = _outputChannel * _inputChannel * _outputSize * _outputSize;
    }
    
    return self;
}

- (void)forward {
    for (int groupIndex = 0; groupIndex < _group; groupIndex++) {
        float *src = self.srcPtr + groupIndex * _inputPerGroup;
        float *dst = self.dstPtr + groupIndex * _outputPerGroup;
        im2col(src, _inputChannel, _inputSize, _inputSize, _kernelSize, _kernelSize, 1, 1,
               _pad, _pad, _pad, _pad, _stride, _stride, _colData);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _M, _N, _K, 1,
                    _weight + groupIndex * _weightPerGroup, _K, _colData, _N, 0, dst, _N);
        for (int featureIndex = 0; featureIndex < _N; featureIndex++) {
            vDSP_vadd(dst + featureIndex * _M, 1, _bias + groupIndex * _M, 1,
                      dst + featureIndex * _M, 1, _M);
        }
    }
    if (_doReLU) vDSP_vthres(self.dstPtr, 1, &_zero, self.dstPtr, 1, _outputPerGroup * _group);
}

void im2col (const float* data_im,
             const int channels,
             const int height,
             const int width,
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
    const int output_h =
    (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
    1;
    const int output_w =
    (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
    1;
    
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
            const float* src = data_im + nip * (height * width);
            for (int y = 0; y < output_h; y++) {
                const int iy = y * stride_h + kh;
                const int ix = kw;
                if (stride_w == 1) {
                    memcpy(
                           dst + (y * output_w),
                           src + (iy * width + ix),
                           sizeof(float) * output_w);
                } else {
                    for (int x = 0; x < output_w; x++) {
                        memcpy(
                               dst + (y * output_w + x),
                               src + (iy * width + ix + x * stride_w),
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
        const int channel_size = height * width;
        for (int channel = channels; channel--; data_im += channel_size) {
            for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
                for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                    int input_row = -pad_h + kernel_row * dilation_h;
                    for (int output_rows = output_h; output_rows; output_rows--) {
                        if (!((unsigned int)input_row < (unsigned int)height)) {
                            for (int output_cols = output_w; output_cols; output_cols--) {
                                *(data_col++) = 0;
                            }
                        } else {
                            int input_col = -pad_w + kernel_col * dilation_w;
                            for (int output_col = output_w; output_col; output_col--) {
                                if ((unsigned int)input_col < (unsigned int)width) {
                                    *(data_col++) = data_im[input_row * width + input_col];
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
    
    int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
    int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
    
    int channels_col = channels * kernel_h * kernel_w;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int h_pad = h * stride_h - pad_t + h_offset * dilation_h;
                int w_pad = w * stride_w - pad_l + w_offset * dilation_w;
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                    data_col[(c * height_col + h) * width_col + w] =
                    data_im[(c_im * height + h_pad) * width + w_pad];
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
    if (self = [super init]) {
        self.name = name;
        _weight = weight;
        _bias = bias;
        _inputChannel = inputChannel;
        _outputChannel = outputChannel;
        _inputSize = inputSize;
        _doReLU = doReLU;
        _zero = 0;
        _M = _outputChannel;
        _N = _inputSize * _inputSize * _inputChannel;
    }
    
    return self;
}

- (void)forward {
    memcpy(self.dstPtr, _bias, _outputChannel);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, _M, _N, 1, _weight, _N, self.dstPtr, 1, 1,
                self.dstPtr, 1);
    if (_doReLU) vDSP_vthres(self.dstPtr, 1, &_zero, self.dstPtr, 1, _outputChannel);
}

@end

@implementation CPUPoolingMaxLayer

- (instancetype)initWithName:(NSString *)name
                inputChannel:(int)inputChannel
               outputChannel:(int)outputChannel
                   inputSize:(int)inputSize
                  outputSize:(int)outputSize
                  kernelSize:(int)kernelSize
                         pad:(int)pad
                      stride:(int)stride {
    if (self = [super init]) {
        self.name = name;
        
        BNNSImageStackDescriptor input_desc;
        bzero(&input_desc, sizeof(input_desc));
        input_desc.width = inputSize;
        input_desc.height = inputSize;
        input_desc.channels = inputChannel;
        input_desc.row_stride = inputSize;
        input_desc.image_stride = inputSize * inputSize;
        input_desc.data_type = BNNSDataTypeFloat32;
        
        BNNSImageStackDescriptor output_desc;
        bzero(&output_desc, sizeof(output_desc));
        output_desc.width = outputSize;
        output_desc.height = outputSize;
        output_desc.channels = outputChannel;
        output_desc.row_stride = outputSize;
        output_desc.image_stride = outputSize * outputSize;
        output_desc.data_type = BNNSDataTypeFloat32;
        
        BNNSPoolingLayerParameters params;
        bzero(&params, sizeof(params));
        params.x_stride = stride;
        params.y_stride = stride;
        params.x_padding = pad;
        params.y_padding = pad;
        params.k_width = kernelSize;
        params.k_height = kernelSize;
        params.in_channels = inputChannel;
        params.out_channels = outputChannel;
        params.pooling_function = BNNSPoolingFunctionMax;
        
        BNNSFilterParameters filter_params;
        bzero(&filter_params, sizeof(filter_params));
        
        _filter = BNNSFilterCreatePoolingLayer(&input_desc, &output_desc,
                                               &params, &filter_params);
    }
    
    return self;
}

- (void)forward {
    BNNSFilterApply(_filter, self.srcPtr, self.dstPtr);
}

@end
