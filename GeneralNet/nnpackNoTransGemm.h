//
//  nnpackNoTransGemm.h
//  NNPACK_GEMM
//
//  Created by Lun on 2017/8/11.
//  Copyright © 2017年 Lun. All rights reserved.
//

#ifndef nnpackNoTransGemm_h
#define nnpackNoTransGemm_h

#include <stdio.h>

void nnpack_no_trans_gemm(const int M,
                          const int N,
                          const int K,
                          const float alpha,
                          const float* A,
                          const float* B,
                          const float beta,
                          float* C);

#endif /* nnpackNoTransGemm_h */
