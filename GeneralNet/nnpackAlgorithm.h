//
//  nnpackAlgorithm.h
//  GeneralNet
//
//  Created by Lun on 2017/6/26.
//  Copyright © 2017年 Lun. All rights reserved.
//

#ifndef nnpackAlgorithm_h
#define nnpackAlgorithm_h

#include <stdbool.h>

void nnp_sgemm_only_4x12(size_t k,
                         size_t update,
                         size_t output_row,
                         size_t output_col,
                         size_t reduction_size,
                         const bool trans_a,
                         const bool trans_b,
                         const float alpha,
                         const float beta,
                         const float *a,
                         const float *b,
                         float *c);

void nnp_sgemm_upto_4x12(size_t mr,
                         size_t nr,
                         size_t k,
                         size_t update,
                         size_t output_row,
                         size_t output_col,
                         size_t reduction_size,
                         const bool trans_a,
                         const bool trans_b,
                         const float alpha,
                         const float beta,
                         const float *a,
                         const float *b,
                         float *c);

void nnp_sgemm_only_8x8(size_t k,
                        size_t update,
                        size_t output_row,
                        size_t output_col,
                        size_t reduction_size,
                        const bool trans_a,
                        const bool trans_b,
                        const float alpha,
                        const float beta,
                        const float *a,
                        const float *b,
                        float *c);

void nnp_sgemm_upto_8x8(size_t mr,
                        size_t nr,
                        size_t k,
                        size_t update,
                        size_t output_row,
                        size_t output_col,
                        size_t reduction_size,
                        const bool trans_a,
                        const bool trans_b,
                        const float alpha,
                        const float beta,
                        const float *a,
                        const float *b,
                        float *c);

#endif /* nnpackAlgorithm_h */
