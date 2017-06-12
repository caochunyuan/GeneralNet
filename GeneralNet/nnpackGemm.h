//
//  nnpackGemm.h
//  GeneralNet
//
//  Created by Lun on 2017/6/8.
//  Copyright © 2017年 Lun. All rights reserved.
//

#ifndef nnpackGemm_h
#define nnpackGemm_h

enum NNPACK_TRANSPOSE {
    nnpackNoTrans = 111,
    nnpackTrans   = 112
};

void nnpack_gemm(const enum NNPACK_TRANSPOSE transA,
                 const enum NNPACK_TRANSPOSE transB,
                 const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 const float* A,
                 const float* B,
                 const float beta,
                 float* C);

#endif /* nnpackGemm_h */
