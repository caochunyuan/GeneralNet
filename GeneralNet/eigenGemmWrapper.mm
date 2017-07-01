//
//  eigenGemmWrapper.m
//  OperationCluster
//
//  Created by Lun on 2017/6/13.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import "eigenGemmWrapper.h"
#import "eigenGemm.hpp"

@implementation eigenGemmWrapper

+ (void)gemmWithTransA:(const BOOL)transA
                transB:(const BOOL)transB
                     M:(const int)M
                     N:(const int)N
                     K:(const int)K
                 alpha:(const float)alpha
                     A:(const float *)A
                     B:(const float *)B
                  beta:(const float)beta
                     C:(float *)C {
    eigen_gemm(transA? eigenTrans : eigenNoTrans, transB? eigenTrans : eigenNoTrans, M, N, K, alpha, A, B, beta, C);
}

@end
