//
//  eigenGemm.hpp
//  GeneralNet
//
//  Created by Lun on 2017/6/8.
//  Copyright © 2017年 Lun. All rights reserved.
//

#ifndef eigenGemm_hpp
#define eigenGemm_hpp

enum EIGEN_TRANSPOSE {
    eigenNoTrans = 111,
    eigenTrans   = 112,
};

void eigen_gemm(const enum EIGEN_TRANSPOSE TransA,
                const enum EIGEN_TRANSPOSE TransB,
                const int M,
                const int N,
                const int K,
                const float alpha,
                const float* A,
                const float* B,
                const float beta,
                float* C);

#endif /* eigenGemm_hpp */
