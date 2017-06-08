//
//  nnpackGemm.h
//  GeneralNet
//
//  Created by Lun on 2017/6/8.
//  Copyright © 2017年 Lun. All rights reserved.
//

#ifndef nnpackGemm_h
#define nnpackGemm_h

#include "pthreadpool.h"

void nnpack_gemm(const int m,
                 const int n,
                 const int k,
                 const float* A,
                 const float* B,
                 float* C,
                 pthreadpool_t threadpool);

#endif /* nnpackGemm_h */
