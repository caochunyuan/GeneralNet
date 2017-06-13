//
//  gemmHandler.m
//  GeneralNet
//
//  Created by Lun on 2017/6/13.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import "gemmHandler.h"
#if USE_NNPACK_FOR_GEMM
#import "nnpackGemm.h"
#elif USE_EIGEN_FOR_GEMM
#import "eigenGemm.hpp"
#else
#import <Accelerate/Accelerate.h>
#endif

@implementation gemmHandler

+ (void)gemmWithTransA:(const enum GEMM_TRANSPOSE)transA
                transB:(const enum GEMM_TRANSPOSE)transB
                     M:(const int)M N:(const int)N
                     K:(const int)K
                 alpha:(const float)alpha
                     A:(const float *)A
                     B:(const float *)B
                  beta:(const float)beta
                     C:(float *)C{
#if USE_NNPACK_FOR_GEMM
    nnpack_gemm(nnpackNoTrans, nnpackNoTrans, M, N, K, 1, A, B, 1, C);
#elif USE_EIGEN_FOR_GEMM
    eigen_gemm(eigenNoTrans, eigenNoTrans, M, N, K, 1, A, B, 1, C);
#else
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1,
                A, K, B, N, 1, C, N);
#endif
}

@end
