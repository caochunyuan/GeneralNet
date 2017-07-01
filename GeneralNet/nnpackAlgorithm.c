//
//  nnpackAlgorithm.c
//  OperationCluster
//
//  Created by Lun on 2017/6/26.
//  Copyright © 2017年 Lun. All rights reserved.
//

#include <arm_neon.h>
#include "nnpackAlgorithm.h"

static inline float32x4_t vmuladdq_f32(float32x4_t c, float32x4_t a, float32x4_t b)
{
#if defined(__aarch64__)
    return vfmaq_f32(c, a, b);
#else
    return vmlaq_f32(c, a, b);
#endif
}

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
                         float *c)
{
    float32x4_t vc00 = vdupq_n_f32(0.0f), vc01 = vdupq_n_f32(0.0f), vc02 = vdupq_n_f32(0.0f);
    float32x4_t vc10 = vdupq_n_f32(0.0f), vc11 = vdupq_n_f32(0.0f), vc12 = vdupq_n_f32(0.0f);
    float32x4_t vc20 = vdupq_n_f32(0.0f), vc21 = vdupq_n_f32(0.0f), vc22 = vdupq_n_f32(0.0f);
    float32x4_t vc30 = vdupq_n_f32(0.0f), vc31 = vdupq_n_f32(0.0f), vc32 = vdupq_n_f32(0.0f);
    
    do {
        float32x4_t va;
        if (trans_a) {
            va = vld1q_f32(a);
            a += output_row;
        } else {
            va = (float32x4_t) {
                *(a + reduction_size * 0),
                *(a + reduction_size * 1),
                *(a + reduction_size * 2),
                *(a + reduction_size * 3),
            };
            a += 1;
        }
        
        float32x4_t vb0, vb1, vb2;
        if (trans_b) {
            vb0 = (float32x4_t) {
                *(b + reduction_size * 0),
                *(b + reduction_size * 1),
                *(b + reduction_size * 2),
                *(b + reduction_size * 3),
            };
            vb1 = (float32x4_t) {
                *(b + reduction_size * 4),
                *(b + reduction_size * 5),
                *(b + reduction_size * 6),
                *(b + reduction_size * 7),
            };
            vb2 = (float32x4_t) {
                *(b + reduction_size * 8),
                *(b + reduction_size * 9),
                *(b + reduction_size * 10),
                *(b + reduction_size * 11),
            };
            b += 1;
        } else {
            vb0 = vld1q_f32(b + 0);
            vb1 = vld1q_f32(b + 4);
            vb2 = vld1q_f32(b + 8);
            b += output_col;
        }
        
#if defined(__aarch64__)
        vc00 = vfmaq_lane_f32(vc00, vb0, vget_low_f32(va),  0);
        vc10 = vfmaq_lane_f32(vc10, vb0, vget_low_f32(va),  1);
        vc20 = vfmaq_lane_f32(vc20, vb0, vget_high_f32(va), 0);
        vc30 = vfmaq_lane_f32(vc30, vb0, vget_high_f32(va), 1);
        vc01 = vfmaq_lane_f32(vc01, vb1, vget_low_f32(va),  0);
        vc11 = vfmaq_lane_f32(vc11, vb1, vget_low_f32(va),  1);
        vc21 = vfmaq_lane_f32(vc21, vb1, vget_high_f32(va), 0);
        vc31 = vfmaq_lane_f32(vc31, vb1, vget_high_f32(va), 1);
        vc02 = vfmaq_lane_f32(vc02, vb2, vget_low_f32(va),  0);
        vc12 = vfmaq_lane_f32(vc12, vb2, vget_low_f32(va),  1);
        vc22 = vfmaq_lane_f32(vc22, vb2, vget_high_f32(va), 0);
        vc32 = vfmaq_lane_f32(vc32, vb2, vget_high_f32(va), 1);
#else
        vc00 = vmlaq_lane_f32(vc00, vb0, vget_low_f32(va),  0);
        vc10 = vmlaq_lane_f32(vc10, vb0, vget_low_f32(va),  1);
        vc20 = vmlaq_lane_f32(vc20, vb0, vget_high_f32(va), 0);
        vc30 = vmlaq_lane_f32(vc30, vb0, vget_high_f32(va), 1);
        vc01 = vmlaq_lane_f32(vc01, vb1, vget_low_f32(va),  0);
        vc11 = vmlaq_lane_f32(vc11, vb1, vget_low_f32(va),  1);
        vc21 = vmlaq_lane_f32(vc21, vb1, vget_high_f32(va), 0);
        vc31 = vmlaq_lane_f32(vc31, vb1, vget_high_f32(va), 1);
        vc02 = vmlaq_lane_f32(vc02, vb2, vget_low_f32(va),  0);
        vc12 = vmlaq_lane_f32(vc12, vb2, vget_low_f32(va),  1);
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
                         size_t reduction_size,
                         const bool trans_a,
                         const bool trans_b,
                         const float alpha,
                         const float beta,
                         const float *a,
                         const float *b,
                         float *c)
{
    float32x4_t vc00 = vdupq_n_f32(0.0f), vc01 = vdupq_n_f32(0.0f), vc02 = vdupq_n_f32(0.0f);
    float32x4_t vc10 = vdupq_n_f32(0.0f), vc11 = vdupq_n_f32(0.0f), vc12 = vdupq_n_f32(0.0f);
    float32x4_t vc20 = vdupq_n_f32(0.0f), vc21 = vdupq_n_f32(0.0f), vc22 = vdupq_n_f32(0.0f);
    float32x4_t vc30 = vdupq_n_f32(0.0f), vc31 = vdupq_n_f32(0.0f), vc32 = vdupq_n_f32(0.0f);
    
    do {
        float32x4_t vb0 = vdupq_n_f32(0.0f), vb1 = vdupq_n_f32(0.0f), vb2 = vdupq_n_f32(0.0f);
        
        if (trans_b) {
            vb0 = (float32x4_t) {
                         *(b + reduction_size * 0),
                nr > 1 ? *(b + reduction_size * 1) : 0.0f,
                nr > 2 ? *(b + reduction_size * 2) : 0.0f,
                nr > 3 ? *(b + reduction_size * 3) : 0.0f,
            };
            if (nr > 4) {
                vb1 = (float32x4_t) {
                             *(b + reduction_size * 4),
                    nr > 5 ? *(b + reduction_size * 5) : 0.0f,
                    nr > 6 ? *(b + reduction_size * 6) : 0.0f,
                    nr > 7 ? *(b + reduction_size * 7) : 0.0f,
                };
                if (nr > 8) {
                    vb2 = (float32x4_t) {
                                  *(b + reduction_size * 8),
                        nr > 9  ? *(b + reduction_size * 9)  : 0.0f,
                        nr > 10 ? *(b + reduction_size * 10) : 0.0f,
                        nr > 11 ? *(b + reduction_size * 11) : 0.0f,
                    };
                }
            }
            b += 1;
        } else {
            vb0 = vld1q_f32(b + 0);
            if (nr > 4) {
                vb1 = vld1q_f32(b + 4);
                if (nr > 8) {
                    vb2 = vld1q_f32(b + 8);
                }
            }
            b += output_col;
        }
        
        const float32x4_t va0 = trans_a ? vld1q_dup_f32(a + 0) : vld1q_dup_f32(a + reduction_size * 0);
        vc00 = vmuladdq_f32(vc00, va0, vb0);
        vc01 = vmuladdq_f32(vc01, va0, vb1);
        vc02 = vmuladdq_f32(vc02, va0, vb2);
        
        if (mr > 1) {
            const float32x4_t va1 = trans_a ? vld1q_dup_f32(a + 1) : vld1q_dup_f32(a + reduction_size * 1);
            vc10 = vmuladdq_f32(vc10, va1, vb0);
            vc11 = vmuladdq_f32(vc11, va1, vb1);
            vc12 = vmuladdq_f32(vc12, va1, vb2);
            
            if (mr > 2) {
                const float32x4_t va2 = trans_a ? vld1q_dup_f32(a + 2) : vld1q_dup_f32(a + reduction_size * 2);
                vc20 = vmuladdq_f32(vc20, va2, vb0);
                vc21 = vmuladdq_f32(vc21, va2, vb1);
                vc22 = vmuladdq_f32(vc22, va2, vb2);
                
                if (mr > 3) {
                    const float32x4_t va3 = trans_a ? vld1q_dup_f32(a + 3) : vld1q_dup_f32(a + reduction_size * 3);
                    vc30 = vmuladdq_f32(vc30, va3, vb0);
                    vc31 = vmuladdq_f32(vc31, va3, vb1);
                    vc32 = vmuladdq_f32(vc32, va3, vb2);
                }
            }
        }
        a += trans_a ? output_row : 1;
        
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
                        float *c)
{
    float32x4_t vc00 = vdupq_n_f32(0.0f), vc01 = vdupq_n_f32(0.0f);
    float32x4_t vc10 = vdupq_n_f32(0.0f), vc11 = vdupq_n_f32(0.0f);
    float32x4_t vc20 = vdupq_n_f32(0.0f), vc21 = vdupq_n_f32(0.0f);
    float32x4_t vc30 = vdupq_n_f32(0.0f), vc31 = vdupq_n_f32(0.0f);
    float32x4_t vc40 = vdupq_n_f32(0.0f), vc41 = vdupq_n_f32(0.0f);
    float32x4_t vc50 = vdupq_n_f32(0.0f), vc51 = vdupq_n_f32(0.0f);
    float32x4_t vc60 = vdupq_n_f32(0.0f), vc61 = vdupq_n_f32(0.0f);
    float32x4_t vc70 = vdupq_n_f32(0.0f), vc71 = vdupq_n_f32(0.0f);
    
    do {
        float32x4_t va0, va1;
        if (trans_a) {
            va0 = vld1q_f32(a + 0);
            va1 = vld1q_f32(a + 4);
            a += output_row;
        } else {
            va0 = (float32x4_t) {
                *(a + reduction_size * 0),
                *(a + reduction_size * 1),
                *(a + reduction_size * 2),
                *(a + reduction_size * 3),
            };
            va1 = (float32x4_t) {
                *(a + reduction_size * 4),
                *(a + reduction_size * 5),
                *(a + reduction_size * 6),
                *(a + reduction_size * 7),
            };
            a += 1;
        }
        
        float32x4_t vb0, vb1;
        if (trans_b) {
            vb0 = (float32x4_t) {
                *(b + reduction_size * 0),
                *(b + reduction_size * 1),
                *(b + reduction_size * 2),
                *(b + reduction_size * 3),
            };
            vb1 = (float32x4_t) {
                *(b + reduction_size * 4),
                *(b + reduction_size * 5),
                *(b + reduction_size * 6),
                *(b + reduction_size * 7),
            };
            b += 1;
        } else {
            vb0 = vld1q_f32(b + 0);
            vb1 = vld1q_f32(b + 4);
            b += output_col;
        }
        
#if defined(__aarch64__)
        vc00 = vfmaq_lane_f32(vc00, vb0, vget_low_f32(va0),  0);
        vc10 = vfmaq_lane_f32(vc10, vb0, vget_low_f32(va0),  1);
        vc20 = vfmaq_lane_f32(vc20, vb0, vget_high_f32(va0), 0);
        vc30 = vfmaq_lane_f32(vc30, vb0, vget_high_f32(va0), 1);
        vc01 = vfmaq_lane_f32(vc01, vb1, vget_low_f32(va0),  0);
        vc11 = vfmaq_lane_f32(vc11, vb1, vget_low_f32(va0),  1);
        vc21 = vfmaq_lane_f32(vc21, vb1, vget_high_f32(va0), 0);
        vc31 = vfmaq_lane_f32(vc31, vb1, vget_high_f32(va0), 1);
        vc40 = vfmaq_lane_f32(vc40, vb0, vget_low_f32(va1),  0);
        vc50 = vfmaq_lane_f32(vc50, vb0, vget_low_f32(va1),  1);
        vc60 = vfmaq_lane_f32(vc60, vb0, vget_high_f32(va1), 0);
        vc70 = vfmaq_lane_f32(vc70, vb0, vget_high_f32(va1), 1);
        vc41 = vfmaq_lane_f32(vc41, vb1, vget_low_f32(va1),  0);
        vc51 = vfmaq_lane_f32(vc51, vb1, vget_low_f32(va1),  1);
        vc61 = vfmaq_lane_f32(vc61, vb1, vget_high_f32(va1), 0);
        vc71 = vfmaq_lane_f32(vc71, vb1, vget_high_f32(va1), 1);
#else
        vc00 = vmlaq_lane_f32(vc00, vb0, vget_low_f32(va0),  0);
        vc10 = vmlaq_lane_f32(vc10, vb0, vget_low_f32(va0),  1);
        vc20 = vmlaq_lane_f32(vc20, vb0, vget_high_f32(va0), 0);
        vc30 = vmlaq_lane_f32(vc30, vb0, vget_high_f32(va0), 1);
        vc01 = vmlaq_lane_f32(vc01, vb1, vget_low_f32(va0),  0);
        vc11 = vmlaq_lane_f32(vc11, vb1, vget_low_f32(va0),  1);
        vc21 = vmlaq_lane_f32(vc21, vb1, vget_high_f32(va0), 0);
        vc31 = vmlaq_lane_f32(vc31, vb1, vget_high_f32(va0), 1);
        vc40 = vmlaq_lane_f32(vc40, vb0, vget_low_f32(va1),  0);
        vc50 = vmlaq_lane_f32(vc50, vb0, vget_low_f32(va1),  1);
        vc60 = vmlaq_lane_f32(vc60, vb0, vget_high_f32(va1), 0);
        vc70 = vmlaq_lane_f32(vc70, vb0, vget_high_f32(va1), 1);
        vc41 = vmlaq_lane_f32(vc41, vb1, vget_low_f32(va1),  0);
        vc51 = vmlaq_lane_f32(vc51, vb1, vget_low_f32(va1),  1);
        vc61 = vmlaq_lane_f32(vc61, vb1, vget_high_f32(va1), 0);
        vc71 = vmlaq_lane_f32(vc71, vb1, vget_high_f32(va1), 1);
#endif
    } while (--k);
    
    // c = alpha * a * b + beta * c
    if (alpha == 1.0f) {
        // alpha == 1 (nothing to do with alpha)
        if (update) {
            // update (nothing to do with beta)
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc00));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc01));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc10));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc11));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc20));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc21));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc30));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc31));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc40));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc41));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc50));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc51));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc60));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc61));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc70));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc71));
        } else {
            // !update (should consider beta)
            if (beta == 0.0f) {
                // beta == 0 (nothing to do with beta)
                vst1q_f32(c + 0, vc00);
                vst1q_f32(c + 4, vc01);
                c += output_col;
                vst1q_f32(c + 0, vc10);
                vst1q_f32(c + 4, vc11);
                c += output_col;
                vst1q_f32(c + 0, vc20);
                vst1q_f32(c + 4, vc21);
                c += output_col;
                vst1q_f32(c + 0, vc30);
                vst1q_f32(c + 4, vc31);
                c += output_col;
                vst1q_f32(c + 0, vc40);
                vst1q_f32(c + 4, vc41);
                c += output_col;
                vst1q_f32(c + 0, vc50);
                vst1q_f32(c + 4, vc51);
                c += output_col;
                vst1q_f32(c + 0, vc60);
                vst1q_f32(c + 4, vc61);
                c += output_col;
                vst1q_f32(c + 0, vc70);
                vst1q_f32(c + 4, vc71);
            } else if (beta == 1.0f) {
                // beta == 1 (do not need to multiply with beta)
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc00));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc01));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc10));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc11));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc20));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc21));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc30));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc31));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc40));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc41));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc50));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc51));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc60));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc61));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc70));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc71));
            } else {
                // beta != 0 (should consider beta)
                float32x4_t beta_t = vdupq_n_f32(beta);
                
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vc00));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vc01));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vc10));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vc11));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vc20));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vc21));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vc30));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vc31));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vc40));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vc41));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vc50));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vc51));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vc60));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vc61));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vc70));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vc71));
            }
        }
    } else {
        // alpha != 1 (should consider alpha)
        float32x4_t alpha_t = vdupq_n_f32(alpha);
        
        if (update) {
            // update (nothing to do with beta)
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc00, alpha_t)));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc01, alpha_t)));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc10, alpha_t)));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc11, alpha_t)));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc20, alpha_t)));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc21, alpha_t)));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc30, alpha_t)));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc31, alpha_t)));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc40, alpha_t)));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc41, alpha_t)));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc50, alpha_t)));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc51, alpha_t)));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc60, alpha_t)));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc61, alpha_t)));
            c += output_col;
            vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc70, alpha_t)));
            vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc71, alpha_t)));
        } else {
            // !update (should consider beta)
            if (beta == 0.0f) {
                // beta == 0 (nothing to do with beta)
                vst1q_f32(c + 0, vmulq_f32(vc00, alpha_t));
                vst1q_f32(c + 4, vmulq_f32(vc01, alpha_t));
                c += output_col;
                vst1q_f32(c + 0, vmulq_f32(vc10, alpha_t));
                vst1q_f32(c + 4, vmulq_f32(vc11, alpha_t));
                c += output_col;
                vst1q_f32(c + 0, vmulq_f32(vc20, alpha_t));
                vst1q_f32(c + 4, vmulq_f32(vc21, alpha_t));
                c += output_col;
                vst1q_f32(c + 0, vmulq_f32(vc30, alpha_t));
                vst1q_f32(c + 4, vmulq_f32(vc31, alpha_t));
                c += output_col;
                vst1q_f32(c + 0, vmulq_f32(vc40, alpha_t));
                vst1q_f32(c + 4, vmulq_f32(vc41, alpha_t));
                c += output_col;
                vst1q_f32(c + 0, vmulq_f32(vc50, alpha_t));
                vst1q_f32(c + 4, vmulq_f32(vc51, alpha_t));
                c += output_col;
                vst1q_f32(c + 0, vmulq_f32(vc60, alpha_t));
                vst1q_f32(c + 4, vmulq_f32(vc61, alpha_t));
                c += output_col;
                vst1q_f32(c + 0, vmulq_f32(vc70, alpha_t));
                vst1q_f32(c + 4, vmulq_f32(vc71, alpha_t));
            } else if (beta == 1.0f) {
                // beta == 1 (do not need to multiply with beta)
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc00, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc01, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc10, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc11, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc20, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc21, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc30, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc31, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc40, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc41, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc50, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc51, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc60, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc61, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vmulq_f32(vc70, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vmulq_f32(vc71, alpha_t)));
            } else {
                // beta != 0 (should consider beta)
                float32x4_t beta_t = vdupq_n_f32(beta);
                
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vmulq_f32(vc00, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vmulq_f32(vc01, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vmulq_f32(vc10, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vmulq_f32(vc11, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vmulq_f32(vc20, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vmulq_f32(vc21, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vmulq_f32(vc30, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vmulq_f32(vc31, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vmulq_f32(vc40, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vmulq_f32(vc41, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vmulq_f32(vc50, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vmulq_f32(vc51, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vmulq_f32(vc60, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vmulq_f32(vc61, alpha_t)));
                c += output_col;
                vst1q_f32(c + 0, vaddq_f32(vmulq_f32(vld1q_f32(c + 0), beta_t), vmulq_f32(vc70, alpha_t)));
                vst1q_f32(c + 4, vaddq_f32(vmulq_f32(vld1q_f32(c + 4), beta_t), vmulq_f32(vc71, alpha_t)));
            }
        }
    }
}

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
                        float *c)
{
    float32x4_t vc00 = vdupq_n_f32(0.0f), vc01 = vdupq_n_f32(0.0f);
    float32x4_t vc10 = vdupq_n_f32(0.0f), vc11 = vdupq_n_f32(0.0f);
    float32x4_t vc20 = vdupq_n_f32(0.0f), vc21 = vdupq_n_f32(0.0f);
    float32x4_t vc30 = vdupq_n_f32(0.0f), vc31 = vdupq_n_f32(0.0f);
    float32x4_t vc40 = vdupq_n_f32(0.0f), vc41 = vdupq_n_f32(0.0f);
    float32x4_t vc50 = vdupq_n_f32(0.0f), vc51 = vdupq_n_f32(0.0f);
    float32x4_t vc60 = vdupq_n_f32(0.0f), vc61 = vdupq_n_f32(0.0f);
    float32x4_t vc70 = vdupq_n_f32(0.0f), vc71 = vdupq_n_f32(0.0f);
    
    do {
        float32x4_t vb0 = vdupq_n_f32(0.0f), vb1 = vdupq_n_f32(0.0f);
        
        if (trans_b) {
            vb0 = (float32x4_t) {
                         *(b + reduction_size * 0),
                nr > 1 ? *(b + reduction_size * 1) : 0.0f,
                nr > 2 ? *(b + reduction_size * 2) : 0.0f,
                nr > 3 ? *(b + reduction_size * 3) : 0.0f,
            };
            if (nr > 4) {
                vb1 = (float32x4_t) {
                             *(b + reduction_size * 4),
                    nr > 5 ? *(b + reduction_size * 5) : 0.0f,
                    nr > 6 ? *(b + reduction_size * 6) : 0.0f,
                    nr > 7 ? *(b + reduction_size * 7) : 0.0f,
                };
            }
            b += 1;
        } else {
            vb0 = vld1q_f32(b + 0);
            if (nr > 4) {
                vb1 = vld1q_f32(b + 4);
            }
            b += output_col;
        }
        
        const float32x4_t va0 = trans_a ? vld1q_dup_f32(a + 0) : vld1q_dup_f32(a + reduction_size * 0);
        vc00 = vmuladdq_f32(vc00, va0, vb0);
        vc01 = vmuladdq_f32(vc01, va0, vb1);
        
        if (mr > 1) {
            const float32x4_t va1 = trans_a ? vld1q_dup_f32(a + 1) : vld1q_dup_f32(a + reduction_size * 1);
            vc10 = vmuladdq_f32(vc10, va1, vb0);
            vc11 = vmuladdq_f32(vc11, va1, vb1);
            
            if (mr > 2) {
                const float32x4_t va2 = trans_a ? vld1q_dup_f32(a + 2) : vld1q_dup_f32(a + reduction_size * 2);
                vc20 = vmuladdq_f32(vc20, va2, vb0);
                vc21 = vmuladdq_f32(vc21, va2, vb1);
                
                if (mr > 3) {
                    const float32x4_t va3 = trans_a ? vld1q_dup_f32(a + 3) : vld1q_dup_f32(a + reduction_size * 3);
                    vc30 = vmuladdq_f32(vc30, va3, vb0);
                    vc31 = vmuladdq_f32(vc31, va3, vb1);
                    
                    if (mr > 4) {
                        const float32x4_t va4 = trans_a ? vld1q_dup_f32(a + 4) : vld1q_dup_f32(a + reduction_size * 4);
                        vc40 = vmuladdq_f32(vc40, va4, vb0);
                        vc41 = vmuladdq_f32(vc41, va4, vb1);
                        
                        if (mr > 5) {
                            const float32x4_t va5 = trans_a ? vld1q_dup_f32(a + 5) : vld1q_dup_f32(a + reduction_size * 5);
                            vc50 = vmuladdq_f32(vc50, va5, vb0);
                            vc51 = vmuladdq_f32(vc51, va5, vb1);
                            
                            if (mr > 6) {
                                const float32x4_t va6 = trans_a ? vld1q_dup_f32(a + 6) : vld1q_dup_f32(a + reduction_size * 6);
                                vc60 = vmuladdq_f32(vc60, va6, vb0);
                                vc61 = vmuladdq_f32(vc61, va6, vb1);
                                
                                if (mr > 7) {
                                    const float32x4_t va7 = trans_a ? vld1q_dup_f32(a + 7) : vld1q_dup_f32(a + reduction_size * 7);
                                    vc70 = vmuladdq_f32(vc70, va7, vb0);
                                    vc71 = vmuladdq_f32(vc71, va7, vb1);
                                }
                            }
                        }
                    }
                }
            }
        }
        a += trans_a ? output_row : 1;
        
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
                    if (mr > 4) {
                        c += output_col;
                        float32x4_t vc4n = vc40;
                        size_t nr4 = nr;
                        float* c4n = c;
                        if (nr4 > 4) {
                            vst1q_f32(c4n, vaddq_f32(vld1q_f32(c4n), vmulq_f32(vc4n, alpha_t4)));
                            c4n += 4;
                            nr4 -= 4;
                            vc4n = vc41;
                        }
                        switch (nr4) {
                            case 4:
                                vst1q_f32(c4n, vaddq_f32(vld1q_f32(c4n), vmulq_f32(vc4n, alpha_t4)));
                                break;
                            case 3:
                                vst1_lane_f32(c4n + 2, vadd_f32(vld1_dup_f32(c4n + 2), vmul_f32(vget_high_f32(vc4n), alpha_t2)), 0);
                            case 2:
                                vst1_f32(c4n, vadd_f32(vld1_f32(c4n), vmul_f32(vget_low_f32(vc4n), alpha_t2)));
                                break;
                            case 1:
                                vst1_lane_f32(c4n, vadd_f32(vld1_dup_f32(c4n), vmul_f32(vget_low_f32(vc4n), alpha_t2)), 0);
                                break;
                            default:
                                break;
                        }
                        if (mr > 5) {
                            c += output_col;
                            float32x4_t vc5n = vc50;
                            size_t nr5 = nr;
                            float* c5n = c;
                            if (nr5 > 4) {
                                vst1q_f32(c5n, vaddq_f32(vld1q_f32(c5n), vmulq_f32(vc5n, alpha_t4)));
                                c5n += 4;
                                nr5 -= 4;
                                vc5n = vc51;
                            }
                            switch (nr5) {
                                case 4:
                                    vst1q_f32(c5n, vaddq_f32(vld1q_f32(c5n), vmulq_f32(vc5n, alpha_t4)));
                                    break;
                                case 3:
                                    vst1_lane_f32(c5n + 2, vadd_f32(vld1_dup_f32(c5n + 2), vmul_f32(vget_high_f32(vc5n), alpha_t2)), 0);
                                case 2:
                                    vst1_f32(c5n, vadd_f32(vld1_f32(c5n), vmul_f32(vget_low_f32(vc5n), alpha_t2)));
                                    break;
                                case 1:
                                    vst1_lane_f32(c5n, vadd_f32(vld1_dup_f32(c5n), vmul_f32(vget_low_f32(vc5n), alpha_t2)), 0);
                                    break;
                                default:
                                    break;
                            }
                            if (mr > 6) {
                                c += output_col;
                                float32x4_t vc6n = vc60;
                                size_t nr6 = nr;
                                float* c6n = c;
                                if (nr6 > 4) {
                                    vst1q_f32(c6n, vaddq_f32(vld1q_f32(c6n), vmulq_f32(vc6n, alpha_t4)));
                                    c6n += 4;
                                    nr6 -= 4;
                                    vc6n = vc61;
                                }
                                switch (nr6) {
                                    case 4:
                                        vst1q_f32(c6n, vaddq_f32(vld1q_f32(c6n), vmulq_f32(vc6n, alpha_t4)));
                                        break;
                                    case 3:
                                        vst1_lane_f32(c6n + 2, vadd_f32(vld1_dup_f32(c6n + 2), vmul_f32(vget_high_f32(vc6n), alpha_t2)), 0);
                                    case 2:
                                        vst1_f32(c6n, vadd_f32(vld1_f32(c6n), vmul_f32(vget_low_f32(vc6n), alpha_t2)));
                                        break;
                                    case 1:
                                        vst1_lane_f32(c6n, vadd_f32(vld1_dup_f32(c6n), vmul_f32(vget_low_f32(vc6n), alpha_t2)), 0);
                                        break;
                                    default:
                                        break;
                                }
                                if (mr > 7) {
                                    c += output_col;
                                    float32x4_t vc7n = vc70;
                                    size_t nr7 = nr;
                                    float* c7n = c;
                                    if (nr7 > 4) {
                                        vst1q_f32(c7n, vaddq_f32(vld1q_f32(c7n), vmulq_f32(vc7n, alpha_t4)));
                                        c7n += 4;
                                        nr7 -= 4;
                                        vc7n = vc71;
                                    }
                                    switch (nr7) {
                                        case 4:
                                            vst1q_f32(c7n, vaddq_f32(vld1q_f32(c7n), vmulq_f32(vc7n, alpha_t4)));
                                            break;
                                        case 3:
                                            vst1_lane_f32(c7n + 2, vadd_f32(vld1_dup_f32(c7n + 2), vmul_f32(vget_high_f32(vc7n), alpha_t2)), 0);
                                        case 2:
                                            vst1_f32(c7n, vadd_f32(vld1_f32(c7n), vmul_f32(vget_low_f32(vc7n), alpha_t2)));
                                            break;
                                        case 1:
                                            vst1_lane_f32(c7n, vadd_f32(vld1_dup_f32(c7n), vmul_f32(vget_low_f32(vc7n), alpha_t2)), 0);
                                            break;
                                        default:
                                            break;
                                    }
                                }
                            }
                        }
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
                    if (mr > 4) {
                        c += output_col;
                        float32x4_t vc4n = vc40;
                        size_t nr4 = nr;
                        float* c4n = c;
                        if (nr4 > 4) {
                            vst1q_f32(c4n, vaddq_f32(vmulq_f32(vld1q_f32(c4n), beta_t4), vmulq_f32(vc4n, alpha_t4)));
                            c4n += 4;
                            nr4 -= 4;
                            vc4n = vc41;
                        }
                        switch (nr4) {
                            case 4:
                                vst1q_f32(c4n, vaddq_f32(vmulq_f32(vld1q_f32(c4n), beta_t4), vmulq_f32(vc4n, alpha_t4)));
                                break;
                            case 3:
                                vst1_lane_f32(c4n + 2, vadd_f32(vmul_f32(vld1_f32(c4n + 2), beta_t2), vmul_f32(vget_high_f32(vc4n), alpha_t2)), 0);
                            case 2:
                                vst1_f32(c4n, vadd_f32(vmul_f32(vld1_f32(c4n), beta_t2), vmul_f32(vget_low_f32(vc4n), alpha_t2)));
                                break;
                            case 1:
                                vst1_lane_f32(c4n, vadd_f32(vmul_f32(vld1_f32(c4n), beta_t2), vmul_f32(vget_low_f32(vc4n), alpha_t2)), 0);
                                break;
                            default:
                                break;
                        }
                        if (mr > 5) {
                            c += output_col;
                            float32x4_t vc5n = vc50;
                            size_t nr5 = nr;
                            float* c5n = c;
                            if (nr5 > 4) {
                                vst1q_f32(c5n, vaddq_f32(vmulq_f32(vld1q_f32(c5n), beta_t4), vmulq_f32(vc5n, alpha_t4)));
                                c5n += 4;
                                nr5 -= 4;
                                vc5n = vc51;
                            }
                            switch (nr5) {
                                case 4:
                                    vst1q_f32(c5n, vaddq_f32(vmulq_f32(vld1q_f32(c5n), beta_t4), vmulq_f32(vc5n, alpha_t4)));
                                    break;
                                case 3:
                                    vst1_lane_f32(c5n + 2, vadd_f32(vmul_f32(vld1_f32(c5n + 2), beta_t2), vmul_f32(vget_high_f32(vc5n), alpha_t2)), 0);
                                case 2:
                                    vst1_f32(c5n, vadd_f32(vmul_f32(vld1_f32(c5n), beta_t2), vmul_f32(vget_low_f32(vc5n), alpha_t2)));
                                    break;
                                case 1:
                                    vst1_lane_f32(c5n, vadd_f32(vmul_f32(vld1_f32(c5n), beta_t2), vmul_f32(vget_low_f32(vc5n), alpha_t2)), 0);
                                    break;
                                default:
                                    break;
                            }
                            if (mr > 6) {
                                c += output_col;
                                float32x4_t vc6n = vc60;
                                size_t nr6 = nr;
                                float* c6n = c;
                                if (nr6 > 4) {
                                    vst1q_f32(c6n, vaddq_f32(vmulq_f32(vld1q_f32(c6n), beta_t4), vmulq_f32(vc6n, alpha_t4)));
                                    c6n += 4;
                                    nr6 -= 4;
                                    vc6n = vc61;
                                }
                                switch (nr6) {
                                    case 4:
                                        vst1q_f32(c6n, vaddq_f32(vmulq_f32(vld1q_f32(c6n), beta_t4), vmulq_f32(vc6n, alpha_t4)));
                                        break;
                                    case 3:
                                        vst1_lane_f32(c6n + 2, vadd_f32(vmul_f32(vld1_f32(c6n + 2), beta_t2), vmul_f32(vget_high_f32(vc6n), alpha_t2)), 0);
                                    case 2:
                                        vst1_f32(c6n, vadd_f32(vmul_f32(vld1_f32(c6n), beta_t2), vmul_f32(vget_low_f32(vc6n), alpha_t2)));
                                        break;
                                    case 1:
                                        vst1_lane_f32(c6n, vadd_f32(vmul_f32(vld1_f32(c6n), beta_t2), vmul_f32(vget_low_f32(vc6n), alpha_t2)), 0);
                                        break;
                                    default:
                                        break;
                                }
                                if (mr > 7) {
                                    c += output_col;
                                    float32x4_t vc7n = vc70;
                                    size_t nr7 = nr;
                                    float* c7n = c;
                                    if (nr7 > 4) {
                                        vst1q_f32(c7n, vaddq_f32(vmulq_f32(vld1q_f32(c7n), beta_t4), vmulq_f32(vc7n, alpha_t4)));
                                        c7n += 4;
                                        nr7 -= 4;
                                        vc7n = vc71;
                                    }
                                    switch (nr7) {
                                        case 4:
                                            vst1q_f32(c7n, vaddq_f32(vmulq_f32(vld1q_f32(c7n), beta_t4), vmulq_f32(vc7n, alpha_t4)));
                                            break;
                                        case 3:
                                            vst1_lane_f32(c7n + 2, vadd_f32(vmul_f32(vld1_f32(c7n + 2), beta_t2), vmul_f32(vget_high_f32(vc7n), alpha_t2)), 0);
                                        case 2:
                                            vst1_f32(c7n, vadd_f32(vmul_f32(vld1_f32(c7n), beta_t2), vmul_f32(vget_low_f32(vc7n), alpha_t2)));
                                            break;
                                        case 1:
                                            vst1_lane_f32(c7n, vadd_f32(vmul_f32(vld1_f32(c7n), beta_t2), vmul_f32(vget_low_f32(vc7n), alpha_t2)), 0);
                                            break;
                                        default:
                                            break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
