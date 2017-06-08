//
//  eigenGemm.hpp
//  GeneralNet
//
//  Created by Lun on 2017/6/8.
//  Copyright © 2017年 Lun. All rights reserved.
//

#ifndef eigenGemm_hpp
#define eigenGemm_hpp

enum BLAS_ORDER { blasRowMajor = 101, blasColMajor = 102 };
enum BLAS_TRANSPOSE { blasNoTrans = 111, blasTrans = 112, blasConjTrans = 113 };

void eigen_gemm(const enum BLAS_TRANSPOSE TransA,
                const enum BLAS_TRANSPOSE TransB,
                const int M,
                const int N,
                const int K,
                const float alpha,
                const float* A,
                const float* B,
                const float beta,
                float* C);

#endif /* eigenGemm_hpp */
