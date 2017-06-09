//
//  nnpackGemm.c
//  GeneralNet
//
//  Created by Lun on 2017/6/8.
//  Copyright © 2017年 Lun. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <arm_neon.h>
#include "nnpackGemm.h"
#include "pthreadpool.h"

#if USE_ACCELERATE_FOR_TRANSPOSE
#include <Accelerate/Accelerate.h>
#endif

#define NNP_ALIGN(alignment) __attribute__((__aligned__(alignment)))
#define NNP_CACHE_ALIGN NNP_ALIGN(64)

static inline float32x4_t vmuladdq_f32(float32x4_t c, float32x4_t a, float32x4_t b)
{
#if defined(__aarch64__)
    return vfmaq_f32(c, a, b);
#else
    return vmlaq_f32(c, a, b);
#endif
}

static inline size_t min(size_t a, size_t b)
{
    return a > b ? b : a;
}

static inline void transpose(const float *src, float *dst, size_t row, size_t col)
{
    // calculate the transpose of "src"
    // "row" and "col" are row and column numbers of "dst"
#if USE_ACCELERATE_FOR_TRANSPOSE
    vDSP_mtrans(src, 1, dst, 1, row, col);
#endif
}

struct NNP_CACHE_ALIGN gemm_context
{
    const float alpha;
    const float beta;
    const float* matrix_A;
    const float* matrix_B;
    float* matrix_C;
    
    size_t reduction_block_start;
    size_t reduction_block_size;
    size_t output_row;
    size_t output_col;
    size_t col_block_start;
    size_t col_subblock_max;
    size_t row_subblock_max;
};

void nnp_sgemm_only_4x12(size_t k,
                         size_t update,
                         size_t output_row,
                         size_t output_col,
                         const float alpha,
                         const float beta,
                         const float* a,
                         const float* b,
                         float* c)
{
    float32x4_t vc00 = vdupq_n_f32(0.0f), vc01 = vdupq_n_f32(0.0f), vc02 = vdupq_n_f32(0.0f);
    float32x4_t vc10 = vdupq_n_f32(0.0f), vc11 = vdupq_n_f32(0.0f), vc12 = vdupq_n_f32(0.0f);
    float32x4_t vc20 = vdupq_n_f32(0.0f), vc21 = vdupq_n_f32(0.0f), vc22 = vdupq_n_f32(0.0f);
    float32x4_t vc30 = vdupq_n_f32(0.0f), vc31 = vdupq_n_f32(0.0f), vc32 = vdupq_n_f32(0.0f);
    
    do {
        const float32x4_t va = vld1q_f32(a);
        a += output_row;
        
        const float32x4_t vb0 = vld1q_f32(b + 0);
        const float32x4_t vb1 = vld1q_f32(b + 4);
        const float32x4_t vb2 = vld1q_f32(b + 8);
        b += output_col;
        
#if defined(__aarch64__)
        vc00 = vfmaq_lane_f32(vc00, vb0, vget_low_f32(va), 0);
        vc10 = vfmaq_lane_f32(vc10, vb0, vget_low_f32(va), 1);
        vc20 = vfmaq_lane_f32(vc20, vb0, vget_high_f32(va), 0);
        vc30 = vfmaq_lane_f32(vc30, vb0, vget_high_f32(va), 1);
        vc01 = vfmaq_lane_f32(vc01, vb1, vget_low_f32(va), 0);
        vc11 = vfmaq_lane_f32(vc11, vb1, vget_low_f32(va), 1);
        vc21 = vfmaq_lane_f32(vc21, vb1, vget_high_f32(va), 0);
        vc31 = vfmaq_lane_f32(vc31, vb1, vget_high_f32(va), 1);
        vc02 = vfmaq_lane_f32(vc02, vb2, vget_low_f32(va), 0);
        vc12 = vfmaq_lane_f32(vc12, vb2, vget_low_f32(va), 1);
        vc22 = vfmaq_lane_f32(vc22, vb2, vget_high_f32(va), 0);
        vc32 = vfmaq_lane_f32(vc32, vb2, vget_high_f32(va), 1);
#else
        vc00 = vmlaq_lane_f32(vc00, vb0, vget_low_f32(va), 0);
        vc10 = vmlaq_lane_f32(vc10, vb0, vget_low_f32(va), 1);
        vc20 = vmlaq_lane_f32(vc20, vb0, vget_high_f32(va), 0);
        vc30 = vmlaq_lane_f32(vc30, vb0, vget_high_f32(va), 1);
        vc01 = vmlaq_lane_f32(vc01, vb1, vget_low_f32(va), 0);
        vc11 = vmlaq_lane_f32(vc11, vb1, vget_low_f32(va), 1);
        vc21 = vmlaq_lane_f32(vc21, vb1, vget_high_f32(va), 0);
        vc31 = vmlaq_lane_f32(vc31, vb1, vget_high_f32(va), 1);
        vc02 = vmlaq_lane_f32(vc02, vb2, vget_low_f32(va), 0);
        vc12 = vmlaq_lane_f32(vc12, vb2, vget_low_f32(va), 1);
        vc22 = vmlaq_lane_f32(vc22, vb2, vget_high_f32(va), 0);
        vc32 = vmlaq_lane_f32(vc32, vb2, vget_high_f32(va), 1);
#endif
    } while (--k);
    
    // c = alpha * a * b + beta * c
    if (alpha == 1.0f) {
        // alpha == 1 (nothing to do with alpha)
        if (update) {
            // update (nothing to do with beta)
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc00));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc01));
            vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vc02));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc10));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc11));
            vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vc12));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc20));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc21));
            vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vc22));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc30));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc31));
            vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vc32));
            
        } else {
            // !update (should consider beta)
            if (beta == 0.0f) {
                // beta == 0 (nothing to do with beta)
                vst1q_f32(c + 0, vc00);
                vst1q_f32(c + 4, vc01);
                vst1q_f32(c + 8, vc02);
                c += output_col;
                vst1q_f32(c + 0, vc10);
                vst1q_f32(c + 4, vc11);
                vst1q_f32(c + 8, vc12);
                c += output_col;
                vst1q_f32(c + 0, vc20);
                vst1q_f32(c + 4, vc21);
                vst1q_f32(c + 8, vc22);
                c += output_col;
                vst1q_f32(c + 0, vc30);
                vst1q_f32(c + 4, vc31);
                vst1q_f32(c + 8, vc32);
            } else if (beta == 1.0f) {
                // beta == 1 (do not need to multiply with beta)
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc00));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc01));
                vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vc02));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc10));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc11));
                vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vc12));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc20));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc21));
                vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vc22));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc30));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc31));
                vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vc32));
            } else {
                // beta != 0 (should consider beta)
                float32x4_t beta_t = vdupq_n_f32(beta);
                
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vc00));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vc01));
                vst1q_f32(c + 8, vaddq_f32(vmulq_f32(vld1q_f32(c + 8), beta_t), vc02));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vc10));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vc11));
                vst1q_f32(c + 8, vaddq_f32(vmulq_f32(vld1q_f32(c + 8), beta_t), vc12));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vc20));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vc21));
                vst1q_f32(c + 8, vaddq_f32(vmulq_f32(vld1q_f32(c + 8), beta_t), vc22));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vc30));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vc31));
                vst1q_f32(c + 8, vaddq_f32(vmulq_f32(vld1q_f32(c + 8), beta_t), vc32));
            }
        }
    } else {
        // alpha != 1 (should consider alpha)
        float32x4_t alpha_t = vdupq_n_f32(alpha);
        
        if (update) {
            // update (nothing to do with beta)
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc00, alpha_t)));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc01, alpha_t)));
            vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vmulq_f32(vc02, alpha_t)));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc10, alpha_t)));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc11, alpha_t)));
            vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vmulq_f32(vc12, alpha_t)));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc20, alpha_t)));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc21, alpha_t)));
            vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vmulq_f32(vc22, alpha_t)));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc30, alpha_t)));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc31, alpha_t)));
            vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vmulq_f32(vc32, alpha_t)));
        } else {
            // !update (should consider beta)
            if (beta == 0.0f) {
                // beta == 0 (nothing to do with beta)
                vst1q_f32(c + 0, vmulq_f32(vc00, alpha_t));
                vst1q_f32(c + 4, vmulq_f32(vc01, alpha_t));
                vst1q_f32(c + 8, vmulq_f32(vc02, alpha_t));
                c += output_col;
                vst1q_f32(c + 0, vmulq_f32(vc10, alpha_t));
                vst1q_f32(c + 4, vmulq_f32(vc11, alpha_t));
                vst1q_f32(c + 8, vmulq_f32(vc12, alpha_t));
                c += output_col;
                vst1q_f32(c + 0, vmulq_f32(vc20, alpha_t));
                vst1q_f32(c + 4, vmulq_f32(vc21, alpha_t));
                vst1q_f32(c + 8, vmulq_f32(vc22, alpha_t));
                c += output_col;
                vst1q_f32(c + 0, vmulq_f32(vc30, alpha_t));
                vst1q_f32(c + 4, vmulq_f32(vc31, alpha_t));
                vst1q_f32(c + 8, vmulq_f32(vc32, alpha_t));
            } else if (beta == 1.0f) {
                // beta == 1 (do not need to multiply with beta)
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc00, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc01, alpha_t)));
                vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vmulq_f32(vc02, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc10, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc11, alpha_t)));
                vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vmulq_f32(vc12, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc20, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc21, alpha_t)));
                vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vmulq_f32(vc22, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc30, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc31, alpha_t)));
                vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vmulq_f32(vc32, alpha_t)));
            } else {
                // beta != 0 (should consider beta)
                float32x4_t beta_t = vdupq_n_f32(beta);
                
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vmulq_f32(vc00, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vmulq_f32(vc01, alpha_t)));
                vst1q_f32(c + 8, vaddq_f32(vmulq_f32(vld1q_f32(c + 8), beta_t), vmulq_f32(vc02, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vmulq_f32(vc10, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vmulq_f32(vc11, alpha_t)));
                vst1q_f32(c + 8, vaddq_f32(vmulq_f32(vld1q_f32(c + 8), beta_t), vmulq_f32(vc12, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vmulq_f32(vc20, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vmulq_f32(vc21, alpha_t)));
                vst1q_f32(c + 8, vaddq_f32(vmulq_f32(vld1q_f32(c + 8), beta_t), vmulq_f32(vc22, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vmulq_f32(vc30, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vmulq_f32(vc31, alpha_t)));
                vst1q_f32(c + 8, vaddq_f32(vmulq_f32(vld1q_f32(c + 8), beta_t), vmulq_f32(vc32, alpha_t)));
            }
        }
    }
}

void nnp_sgemm_upto_4x12(size_t mr,
                         size_t nr,
                         size_t k,
                         size_t update,
                         size_t output_row,
                         size_t output_col,
                         const float alpha,
                         const float beta,
                         const float* a,
                         const float* b,
                         float* c)
{
    float32x4_t vc00 = vdupq_n_f32(0.0f), vc01 = vdupq_n_f32(0.0f), vc02 = vdupq_n_f32(0.0f);
    float32x4_t vc10 = vdupq_n_f32(0.0f), vc11 = vdupq_n_f32(0.0f), vc12 = vdupq_n_f32(0.0f);
    float32x4_t vc20 = vdupq_n_f32(0.0f), vc21 = vdupq_n_f32(0.0f), vc22 = vdupq_n_f32(0.0f);
    float32x4_t vc30 = vdupq_n_f32(0.0f), vc31 = vdupq_n_f32(0.0f), vc32 = vdupq_n_f32(0.0f);
    
    do {
        float32x4_t vb0, vb1, vb2;
        
        vb0 = vld1q_f32(b + 0);
        if (nr > 4) {
            vb1 = vld1q_f32(b + 4);
            vb2 = nr > 8 ? vld1q_f32(b + 8) : vdupq_n_f32(0.0f);
        } else {
            vb1 = vdupq_n_f32(0.0f);
            vb2 = vdupq_n_f32(0.0f);
        }
        b += output_col;
        
        const float32x4_t va0 = vld1q_dup_f32(a + 0);
        vc00 = vmuladdq_f32(vc00, va0, vb0);
        vc01 = vmuladdq_f32(vc01, va0, vb1);
        vc02 = vmuladdq_f32(vc02, va0, vb2);
        
        if (mr > 1) {
            const float32x4_t va1 = vld1q_dup_f32(a + 1);
            vc10 = vmuladdq_f32(vc10, va1, vb0);
            vc11 = vmuladdq_f32(vc11, va1, vb1);
            vc12 = vmuladdq_f32(vc12, va1, vb2);
            
            if (mr > 2) {
                const float32x4_t va2 = vld1q_dup_f32(a + 2);
                vc20 = vmuladdq_f32(vc20, va2, vb0);
                vc21 = vmuladdq_f32(vc21, va2, vb1);
                vc22 = vmuladdq_f32(vc22, va2, vb2);
                
                if (mr > 3) {
                    const float32x4_t va3 = vld1q_dup_f32(a + 3);
                    vc30 = vmuladdq_f32(vc30, va3, vb0);
                    vc31 = vmuladdq_f32(vc31, va3, vb1);
                    vc32 = vmuladdq_f32(vc32, va3, vb2);
                }
            }
        }
        a += output_row;
        
    } while (--k);
    
    // c = alpha * a * b + beta * c
    float32x4_t alpha_t4 = vdupq_n_f32(alpha);
    float32x2_t alpha_t2 = vdup_n_f32(alpha);
    float32x4_t beta_t4 = vdupq_n_f32(beta);
    float32x2_t beta_t2 = vdup_n_f32(beta);
    
    if (update) {
        // update (nothing to do with beta)
        float32x4_t vc0n = vc00;
        size_t nr0 = nr;
        float* c0n = c;
        if (nr0 > 4) {
            vst1q_f32(c0n, vaddq_f32(vld1q_f32(c0n), vmulq_f32(vc0n, alpha_t4)));
            c0n += 4;
            nr0 -= 4;
            vc0n = vc01;
            if (nr0 > 4) {
                vst1q_f32(c0n, vaddq_f32(vld1q_f32(c0n), vmulq_f32(vc0n, alpha_t4)));
                c0n += 4;
                nr0 -= 4;
                vc0n = vc02;
            }
        }
        switch (nr0) {
            case 4:
                vst1q_f32(c0n, vaddq_f32(vld1q_f32(c0n), vmulq_f32(vc0n, alpha_t4)));
                break;
            case 3:
                vst1_lane_f32(c0n + 2, vadd_f32(vld1_dup_f32(c0n + 2), vmul_f32(vget_high_f32(vc0n), alpha_t2)), 0);
            case 2:
                vst1_f32(c0n, vadd_f32(vld1_f32(c0n), vmul_f32(vget_low_f32(vc0n), alpha_t2)));
                break;
            case 1:
                vst1_lane_f32(c0n, vadd_f32(vld1_dup_f32(c0n), vmul_f32(vget_low_f32(vc0n), alpha_t2)), 0);
                break;
            default:
                break;
        }
        if (mr > 1) {
            c += output_col;
            float32x4_t vc1n = vc10;
            size_t nr1 = nr;
            float* c1n = c;
            if (nr1 > 4) {
                vst1q_f32(c1n, vaddq_f32(vld1q_f32(c1n), vmulq_f32(vc1n, alpha_t4)));
                c1n += 4;
                nr1 -= 4;
                vc1n = vc11;
                if (nr1 > 4) {
                    vst1q_f32(c1n, vaddq_f32(vld1q_f32(c1n), vmulq_f32(vc1n, alpha_t4)));
                    c1n += 4;
                    nr1 -= 4;
                    vc1n = vc12;
                }
            }
            switch (nr1) {
                case 4:
                    vst1q_f32(c1n, vaddq_f32(vld1q_f32(c1n), vmulq_f32(vc1n, alpha_t4)));
                    break;
                case 3:
                    vst1_lane_f32(c1n + 2, vadd_f32(vld1_dup_f32(c1n + 2), vmul_f32(vget_high_f32(vc1n), alpha_t2)), 0);
                case 2:
                    vst1_f32(c1n, vadd_f32(vld1_f32(c1n), vmul_f32(vget_low_f32(vc1n), alpha_t2)));
                    break;
                case 1:
                    vst1_lane_f32(c1n, vadd_f32(vld1_dup_f32(c1n), vmul_f32(vget_low_f32(vc1n), alpha_t2)), 0);
                    break;
                default:
                    break;
            }
            if (mr > 2) {
                c += output_col;
                float32x4_t vc2n = vc20;
                size_t nr2 = nr;
                float* c2n = c;
                if (nr2 > 4) {
                    vst1q_f32(c2n, vaddq_f32(vld1q_f32(c2n), vmulq_f32(vc2n, alpha_t4)));
                    c2n += 4;
                    nr2 -= 4;
                    vc2n = vc21;
                    if (nr2 > 4) {
                        vst1q_f32(c2n, vaddq_f32(vld1q_f32(c2n), vmulq_f32(vc2n, alpha_t4)));
                        c2n += 4;
                        nr2 -= 4;
                        vc2n = vc22;
                    }
                }
                switch (nr2) {
                    case 4:
                        vst1q_f32(c2n, vaddq_f32(vld1q_f32(c2n), vmulq_f32(vc2n, alpha_t4)));
                        break;
                    case 3:
                        vst1_lane_f32(c2n + 2, vadd_f32(vld1_dup_f32(c2n + 2), vmul_f32(vget_high_f32(vc2n), alpha_t2)), 0);
                    case 2:
                        vst1_f32(c2n, vadd_f32(vld1_f32(c2n), vmul_f32(vget_low_f32(vc2n), alpha_t2)));
                        break;
                    case 1:
                        vst1_lane_f32(c2n, vadd_f32(vld1_dup_f32(c2n), vmul_f32(vget_low_f32(vc2n), alpha_t2)), 0);
                        break;
                    default:
                        break;
                }
                if (mr > 3) {
                    c += output_col;
                    float32x4_t vc3n = vc30;
                    size_t nr3 = nr;
                    float* c3n = c;
                    if (nr3 > 4) {
                        vst1q_f32(c3n, vaddq_f32(vld1q_f32(c3n), vmulq_f32(vc3n, alpha_t4)));
                        c3n += 4;
                        nr3 -= 4;
                        vc3n = vc31;
                        if (nr3 > 4) {
                            vst1q_f32(c3n, vaddq_f32(vld1q_f32(c3n), vmulq_f32(vc3n, alpha_t4)));
                            c3n += 4;
                            nr3 -= 4;
                            vc3n = vc32;
                        }
                    }
                    switch (nr3) {
                        case 4:
                            vst1q_f32(c3n, vaddq_f32(vld1q_f32(c3n), vmulq_f32(vc3n, alpha_t4)));
                            break;
                        case 3:
                            vst1_lane_f32(c3n + 2, vadd_f32(vld1_dup_f32(c3n + 2), vmul_f32(vget_high_f32(vc3n), alpha_t2)), 0);
                        case 2:
                            vst1_f32(c3n, vadd_f32(vld1_f32(c3n), vmul_f32(vget_low_f32(vc3n), alpha_t2)));
                            break;
                        case 1:
                            vst1_lane_f32(c3n, vadd_f32(vld1_dup_f32(c3n), vmul_f32(vget_low_f32(vc3n), alpha_t2)), 0);
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    } else {
        // !update (should consider beta)
        float32x4_t vc0n = vc00;
        size_t nr0 = nr;
        float* c0n = c;
        if (nr0 > 4) {
            vst1q_f32(c0n, vaddq_f32(vmulq_f32(vld1q_f32(c0n), beta_t4), vmulq_f32(vc0n, alpha_t4)));
            c0n += 4;
            nr0 -= 4;
            vc0n = vc01;
            if (nr0 > 4) {
                vst1q_f32(c0n, vaddq_f32(vmulq_f32(vld1q_f32(c0n), beta_t4), vmulq_f32(vc0n, alpha_t4)));
                c0n += 4;
                nr0 -= 4;
                vc0n = vc02;
            }
        }
        switch (nr0) {
            case 4:
                vst1q_f32(c0n, vaddq_f32(vmulq_f32(vld1q_f32(c0n), beta_t4), vmulq_f32(vc0n, alpha_t4)));
                break;
            case 3:
                vst1_lane_f32(c0n + 2, vadd_f32(vmul_f32(vld1_f32(c0n + 2), beta_t2), vmul_f32(vget_high_f32(vc0n), alpha_t2)), 0);
            case 2:
                vst1_f32(c0n, vadd_f32(vmul_f32(vld1_f32(c0n), beta_t2), vmul_f32(vget_low_f32(vc0n), alpha_t2)));
                break;
            case 1:
                vst1_lane_f32(c0n, vadd_f32(vmul_f32(vld1_f32(c0n), beta_t2), vmul_f32(vget_low_f32(vc0n), alpha_t2)), 0);
                break;
            default:
                break;
        }
        if (mr > 1) {
            c += output_col;
            float32x4_t vc1n = vc10;
            size_t nr1 = nr;
            float* c1n = c;
            if (nr1 > 4) {
                vst1q_f32(c1n, vaddq_f32(vmulq_f32(vld1q_f32(c1n), beta_t4), vmulq_f32(vc1n, alpha_t4)));
                c1n += 4;
                nr1 -= 4;
                vc1n = vc11;
                if (nr1 > 4) {
                    vst1q_f32(c1n, vaddq_f32(vmulq_f32(vld1q_f32(c1n), beta_t4), vmulq_f32(vc1n, alpha_t4)));
                    c1n += 4;
                    nr1 -= 4;
                    vc1n = vc12;
                }
            }
            switch (nr1) {
                case 4:
                    vst1q_f32(c1n, vaddq_f32(vmulq_f32(vld1q_f32(c1n), beta_t4), vmulq_f32(vc1n, alpha_t4)));
                    break;
                case 3:
                    vst1_lane_f32(c1n + 2, vadd_f32(vmul_f32(vld1_f32(c1n + 2), beta_t2), vmul_f32(vget_high_f32(vc1n), alpha_t2)), 0);
                case 2:
                    vst1_f32(c1n, vadd_f32(vmul_f32(vld1_f32(c1n), beta_t2), vmul_f32(vget_low_f32(vc1n), alpha_t2)));
                    break;
                case 1:
                    vst1_lane_f32(c1n, vadd_f32(vmul_f32(vld1_f32(c1n), beta_t2), vmul_f32(vget_low_f32(vc1n), alpha_t2)), 0);
                    break;
                default:
                    break;
            }
            if (mr > 2) {
                c += output_col;
                float32x4_t vc2n = vc20;
                size_t nr2 = nr;
                float* c2n = c;
                if (nr2 > 4) {
                    vst1q_f32(c2n, vaddq_f32(vmulq_f32(vld1q_f32(c2n), beta_t4), vmulq_f32(vc2n, alpha_t4)));
                    c2n += 4;
                    nr2 -= 4;
                    vc2n = vc21;
                    if (nr2 > 4) {
                        vst1q_f32(c2n, vaddq_f32(vmulq_f32(vld1q_f32(c2n), beta_t4), vmulq_f32(vc2n, alpha_t4)));
                        c2n += 4;
                        nr2 -= 4;
                        vc2n = vc22;
                    }
                }
                switch (nr2) {
                    case 4:
                        vst1q_f32(c2n, vaddq_f32(vmulq_f32(vld1q_f32(c2n), beta_t4), vmulq_f32(vc2n, alpha_t4)));
                        break;
                    case 3:
                        vst1_lane_f32(c2n + 2, vadd_f32(vmul_f32(vld1_f32(c2n + 2), beta_t2), vmul_f32(vget_high_f32(vc2n), alpha_t2)), 0);
                    case 2:
                        vst1_f32(c2n, vadd_f32(vmul_f32(vld1_f32(c2n), beta_t2), vmul_f32(vget_low_f32(vc2n), alpha_t2)));
                        break;
                    case 1:
                        vst1_lane_f32(c2n, vadd_f32(vmul_f32(vld1_f32(c2n), beta_t2), vmul_f32(vget_low_f32(vc2n), alpha_t2)), 0);
                        break;
                    default:
                        break;
                }
                if (mr > 3) {
                    c += output_col;
                    float32x4_t vc3n = vc30;
                    size_t nr3 = nr;
                    float* c3n = c;
                    if (nr3 > 4) {
                        vst1q_f32(c3n, vaddq_f32(vmulq_f32(vld1q_f32(c3n), beta_t4), vmulq_f32(vc3n, alpha_t4)));
                        c3n += 4;
                        nr3 -= 4;
                        vc3n = vc31;
                        if (nr3 > 4) {
                            vst1q_f32(c3n, vaddq_f32(vmulq_f32(vld1q_f32(c3n), beta_t4), vmulq_f32(vc3n, alpha_t4)));
                            c3n += 4;
                            nr3 -= 4;
                            vc3n = vc32;
                        }
                    }
                    switch (nr3) {
                        case 4:
                            vst1q_f32(c3n, vaddq_f32(vmulq_f32(vld1q_f32(c3n), beta_t4), vmulq_f32(vc3n, alpha_t4)));
                            break;
                        case 3:
                            vst1_lane_f32(c3n + 2, vadd_f32(vmul_f32(vld1_f32(c3n + 2), beta_t2), vmul_f32(vget_high_f32(vc3n), alpha_t2)), 0);
                        case 2:
                            vst1_f32(c3n, vadd_f32(vmul_f32(vld1_f32(c3n), beta_t2), vmul_f32(vget_low_f32(vc3n), alpha_t2)));
                            break;
                        case 1:
                            vst1_lane_f32(c3n, vadd_f32(vmul_f32(vld1_f32(c3n), beta_t2), vmul_f32(vget_low_f32(vc3n), alpha_t2)), 0);
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    }
}

void compute_gemm(const struct gemm_context context[1],
                  size_t row_block_start, size_t col_subblock_start,
                  size_t row_block_size,  size_t col_subblock_size)
{
    const float  alpha                  = context->alpha;
    const float  beta                   = context->beta;
    const size_t reduction_block_start  = context->reduction_block_start;
    const size_t reduction_block_size   = context->reduction_block_size;
    const size_t output_row             = context->output_row;
    const size_t output_col             = context->output_col;
    const size_t col_block_start        = context->col_block_start;
    const size_t col_subblock_max       = context->col_subblock_max;
    const size_t row_subblock_max       = context->row_subblock_max;
    
    const float* matrix_A = context->matrix_A + reduction_block_start * output_row + row_block_start;
    const float* matrix_B = context->matrix_B + reduction_block_start * output_col + col_block_start + col_subblock_start;
    float* matrix_C       = context->matrix_C + row_block_start       * output_col + col_block_start + col_subblock_start;
    
    if (col_subblock_size == col_subblock_max) {
        while (row_block_size >= row_subblock_max) {
            row_block_size -= row_subblock_max;
            nnp_sgemm_only_4x12(
                                reduction_block_size, reduction_block_start,
                                output_row, output_col,
                                alpha, beta,
                                matrix_A, matrix_B, matrix_C
                                );
            
            matrix_A += row_subblock_max;
            matrix_C += output_col * row_subblock_max;
        }
    }
    
    while (row_block_size != 0) {
        const size_t row_subblock_size = min(row_block_size, row_subblock_max);
        row_block_size -= row_subblock_size;
        
        nnp_sgemm_upto_4x12(
                            row_subblock_size, col_subblock_size,
                            reduction_block_size, reduction_block_start,
                            output_row, output_col,
                            alpha, beta,
                            matrix_A, matrix_B, matrix_C
                            );
        
        matrix_A += row_subblock_max;
        matrix_C += output_col * row_subblock_max;
    }
}

static const size_t cache_l1_size = 16 * 1024;
static const size_t cache_l2_size = 128 * 1024;
static const size_t cache_l3_size = 2 * 1024 * 1024;

/* Compute high-level cache blocking parameters */
static const size_t blocking_l1 = cache_l1_size;
static const size_t blocking_l2 = cache_l2_size - cache_l1_size;
static const size_t blocking_l3 = cache_l3_size - cache_l2_size;

/* Calculate cache blocking parameters */
static const size_t cache_elements_l1 = blocking_l1 / sizeof(float);
static const size_t cache_elements_l2 = blocking_l2 / sizeof(float);
static const size_t cache_elements_l3 = blocking_l3 / sizeof(float);

static const size_t row_subblock_max = 4;
static const size_t col_subblock_max = 12;

static const size_t reduction_block_max = cache_elements_l1 / (row_subblock_max + col_subblock_max) / (2 * 2);
static const size_t row_block_max = cache_elements_l2 / reduction_block_max / (row_subblock_max * row_subblock_max);
static const size_t col_block_max = cache_elements_l3 / reduction_block_max / (col_subblock_max * col_subblock_max);

typedef struct nnpack_context {
    bool initialized;
    pthreadpool_t threadpool;
    float *ptr_a;
    float *ptr_b;
    size_t size_a;
    size_t size_b;
} nnpack_context;

static nnpack_context global_context = {
    .initialized = false
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
                 float* C)
{
    // initialization
    if (!global_context.initialized) {
        global_context.initialized = true;
        global_context.threadpool = pthreadpool_create(0);
        global_context.size_a = 0;
        global_context.size_b = 0;
    }
    
    // to check how many threads is NNPACK using, uncomment the next lines
    //    printf("NNPACK is using %zu threads\n",
    //           pthreadpool_get_threads_count(global_context.threadpool));
    
    if (transA == NNPACKNoTrans) {
        if (m * k > global_context.size_a) {
            if (global_context.ptr_a) free(global_context.ptr_a);
            global_context.ptr_a = malloc(m * k * sizeof(float));
            global_context.size_a = m * k;
        }
        transpose(A, global_context.ptr_a, k, m);
    }
    
    if (transB == NNPACKTrans) {
        if (k * n > global_context.size_b) {
            if (global_context.ptr_b) free(global_context.ptr_b);
            global_context.ptr_b = malloc(k * n * sizeof(float));
            global_context.size_b = k * n;
        }
        transpose(B, global_context.ptr_b, k, n);
    }
    
    const size_t output_row = m;
    const size_t output_col = n;
    const size_t reduction_size = k;
    
    for (size_t reduction_block_start = 0; reduction_block_start < reduction_size; reduction_block_start += reduction_block_max) {
        const size_t reduction_block_size = min(reduction_size - reduction_block_start, reduction_block_max);
        
        for (size_t col_block_start = 0; col_block_start < output_col; col_block_start += col_block_max) {
            const size_t col_block_size = min(output_col - col_block_start, col_block_max);
            
            struct gemm_context gemm_context = {
                .alpha = alpha,
                .beta = beta,
                .matrix_A = transA == NNPACKTrans ? A : global_context.ptr_a,
                .matrix_B = transB == NNPACKNoTrans ? B : global_context.ptr_b,
                .matrix_C = C,
                .reduction_block_start = reduction_block_start,
                .reduction_block_size = reduction_block_size,
                .output_row = output_row,
                .output_col = output_col,
                .col_block_start = col_block_start,
                .col_subblock_max = col_subblock_max,
                .row_subblock_max = row_subblock_max
            };
            pthreadpool_compute_2d_tiled(global_context.threadpool,
                                         (pthreadpool_function_2d_tiled_t) compute_gemm,
                                         &gemm_context,
                                         output_row,    col_block_size,
                                         row_block_max, col_subblock_max);
        }
    }
}
