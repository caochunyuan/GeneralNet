//
//  eigenGemm.cpp
//  GeneralNet
//
//  Created by Lun on 2017/6/8.
//  Copyright © 2017年 Lun. All rights reserved.
//

#include "eigenGemm.hpp"
#include "Eigen/Dense"

using Eigen::Map;
using Eigen::VectorXf;
using Eigen::VectorXd;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::RowMajor;
using Eigen::InnerStride;

template <typename T>
using EigenMatrixMap =
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >;
template <typename T>
using ConstEigenMatrixMap =
Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >;

void eigen_gemm(const enum BLAS_TRANSPOSE TransA,
                const enum BLAS_TRANSPOSE TransB,
                const int M,
                const int N,
                const int K,
                const float alpha,
                const float* A,
                const float* B,
                const float beta,
                float* C) {
    auto C_mat = EigenMatrixMap<float>(C, N, M);
    if (beta == 0) {
        C_mat.setZero();
    } else {
        C_mat *= beta;
    }
    switch (TransA) {
        case blasNoTrans: {
            switch (TransB) {
                case blasNoTrans:
                    C_mat.noalias() += alpha * (ConstEigenMatrixMap<float>(B, N, K) *
                                                ConstEigenMatrixMap<float>(A, K, M));
                    return;
                case blasTrans:
                    C_mat.noalias() += alpha * (ConstEigenMatrixMap<float>(B, K, N).transpose() *
                                                ConstEigenMatrixMap<float>(A, K, M));
                    return;
                default:
                    return;
            }
        }
        case blasTrans: {
            switch (TransB) {
                case blasNoTrans:
                    C_mat.noalias() += alpha * (ConstEigenMatrixMap<float>(B, N, K) *
                                                ConstEigenMatrixMap<float>(A, M, K).transpose());
                    return;
                case blasTrans:
                    C_mat.noalias() += alpha * (ConstEigenMatrixMap<float>(B, K, N).transpose() *
                                                ConstEigenMatrixMap<float>(A, M, K).transpose());
                    return;
                default:
                    return;
            }
        }
        default:
            return;
    }
}
