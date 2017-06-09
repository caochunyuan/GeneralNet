//
//  nnpackGemm.h
//  GeneralNet
//
//  Created by Lun on 2017/6/8.
//  Copyright © 2017年 Lun. All rights reserved.
//

#ifndef nnpackGemm_h
#define nnpackGemm_h

#define USE_ACCELERATE_FOR_TRANSPOSE 1

enum NNPACK_TRANSPOSE {
    NNPACKNoTrans = 111,
    NNPACKTrans   = 112
};

void nnpack_gemm(const int m,
                 const int n,
                 const int k,
                 const float alpha,
                 const float beta,
                 enum NNPACK_TRANSPOSE transA,
                 enum NNPACK_TRANSPOSE transB,
                 const float* A,
                 const float* B,
                 float* C);

#endif /* nnpackGemm_h */
