//
//  gemmHandler.h
//  GeneralNet
//
//  Created by Lun on 2017/6/13.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface gemmHandler : NSObject

enum GEMM_TRANSPOSE {
    gemmNoTrans = 111,
    gemmTrans   = 112
};

+ (void)gemmWithTransA:(const enum GEMM_TRANSPOSE)transA
                transB:(const enum GEMM_TRANSPOSE)transB
                     M:(const int)M N:(const int)N
                     K:(const int)K
                 alpha:(const float)alpha
                     A:(const float *)A
                     B:(const float *)B
                  beta:(const float)beta
                     C:(float *)C;

@end
