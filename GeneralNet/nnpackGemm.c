//
//  nnpackGemm.c
//  GeneralNet
//
//  Created by Lun on 2017/6/8.
//  Copyright © 2017年 Lun. All rights reserved.
//

#include "pthreadpool.h"
#include "nnpackAlgorithm.h"
#include "nnpackGemm.h"

#define NNP_ALIGN(alignment) __attribute__((__aligned__(alignment)))
#define NNP_CACHE_ALIGN NNP_ALIGN(64)

typedef void (*nnpack_gemm_func_only)(size_t, size_t, size_t, size_t, size_t, const bool, const bool, const float, const float, const float*, const float*, float*);
typedef void (*nnpack_gemm_func_upto)(size_t, size_t, size_t, size_t, size_t, size_t, size_t, const bool, const bool, const float, const float, const float*, const float*, float*);

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

typedef struct nnpack_context {
    bool initialized;
    pthreadpool_t threadpool;
} nnpack_context;

static nnpack_context global_context = {
    .initialized = false
};

struct NNP_CACHE_ALIGN gemm_context
{
    const bool trans_a;
    const bool trans_b;
    const float alpha;
    const float beta;
    const float *matrix_a;
    const float *matrix_b;
    float *matrix_c;
    
    size_t reduction_block_start;
    size_t reduction_block_size;
    size_t output_row;
    size_t output_col;
    size_t reduction_size;
    size_t col_block_start;
    size_t col_subblock_max;
    size_t row_subblock_max;
    
    nnpack_gemm_func_only func_only;
    nnpack_gemm_func_upto func_upto;
};

static inline size_t min(size_t a, size_t b)
{
    return a > b ? b : a;
}

static inline size_t round_down(size_t number, size_t factor)
{
    return number / factor * factor;
}

void nnpack_init()
{
    global_context.initialized = true;
    global_context.threadpool = pthreadpool_create(0);
    
    // to check how many threads is NNPACK using, uncomment the next lines
    //    printf("NNPACK is using %zu threads\n",
    //           pthreadpool_get_threads_count(global_context.threadpool));
}

void compute_gemm(const struct gemm_context context[1],
                  size_t row_block_start, size_t col_subblock_start,
                  size_t row_block_size,  size_t col_subblock_size)
{
    const bool   trans_a                  = context->trans_a;
    const bool   trans_b                  = context->trans_b;
    const float  alpha                    = context->alpha;
    const float  beta                     = context->beta;
    const size_t reduction_block_start    = context->reduction_block_start;
    const size_t reduction_block_size     = context->reduction_block_size;
    const size_t output_row               = context->output_row;
    const size_t output_col               = context->output_col;
    const size_t reduction_size           = context->reduction_size;
    const size_t col_block_start          = context->col_block_start;
    const size_t col_subblock_max         = context->col_subblock_max;
    const size_t row_subblock_max         = context->row_subblock_max;
    const nnpack_gemm_func_only func_only = context->func_only;
    const nnpack_gemm_func_upto func_upto = context->func_upto;
    
    const float *matrix_a = trans_a ? context->matrix_a + reduction_block_start * output_row + row_block_start :
                                      context->matrix_a + row_block_start * reduction_size + reduction_block_start;
    const float *matrix_b = trans_b ? context->matrix_b + (col_block_start + col_subblock_start) * reduction_size
                                                        + reduction_block_start :
                                      context->matrix_b + reduction_block_start * output_col + col_block_start
                                                        + col_subblock_start;
    float *matrix_c       =           context->matrix_c + row_block_start * output_col + col_block_start
                                                        + col_subblock_start;
    
    if (col_subblock_size == col_subblock_max) {
        while (row_block_size >= row_subblock_max) {
            row_block_size -= row_subblock_max;
            func_only(
                      reduction_block_size, reduction_block_start,
                      output_row, output_col, reduction_size,
                      trans_a, trans_b,
                      alpha, beta,
                      matrix_a, matrix_b, matrix_c
                      );
            
            matrix_a += trans_a ? row_subblock_max : row_subblock_max * reduction_size;
            matrix_c += row_subblock_max * output_col;
        }
    }
    
    while (row_block_size != 0) {
        const size_t row_subblock_size = min(row_block_size, row_subblock_max);
        row_block_size -= row_subblock_size;
        
        func_upto(
                  row_subblock_size, col_subblock_size,
                  reduction_block_size, reduction_block_start,
                  output_row, output_col, reduction_size,
                  trans_a, trans_b,
                  alpha, beta,
                  matrix_a, matrix_b, matrix_c
                  );
        
        matrix_a += trans_a ? row_subblock_max : row_subblock_max * reduction_size;
        matrix_c += row_subblock_max * output_col;
    }
}

void nnpack_gemm(const enum NNPACK_ALGORITHM algorithm,
                 const enum NNPACK_TRANSPOSE transA,
                 const enum NNPACK_TRANSPOSE transB,
                 const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 const float* A,
                 const float* B,
                 const float beta,
                 float* C)
{
    if (!global_context.initialized) nnpack_init();
    
    const size_t output_row = M;
    const size_t output_col = N;
    const size_t reduction_size = K;
    
    nnpack_gemm_func_only algorithm_only;
    nnpack_gemm_func_upto algorithm_upto;
    
    size_t algorithm_row;
    size_t algorithm_col;
    
    switch (algorithm) {
        case nnpackGemmAuto:
            if (transA == nnpackNoTrans && transB == nnpackNoTrans) {
                algorithm_only = nnp_sgemm_only_4x12;
                algorithm_upto = nnp_sgemm_upto_4x12;
                algorithm_row  =  4;
                algorithm_col  = 12;
            } else {
                algorithm_only = nnp_sgemm_only_8x8;
                algorithm_upto = nnp_sgemm_upto_8x8;
                algorithm_row  =  8;
                algorithm_col  =  8;
            }
            break;
        case nnpackGemm4x12:
            algorithm_only = nnp_sgemm_only_4x12;
            algorithm_upto = nnp_sgemm_upto_4x12;
            algorithm_row  =  4;
            algorithm_col  = 12;
            break;
        case nnpackGemm8x8:
            algorithm_only = nnp_sgemm_only_8x8;
            algorithm_upto = nnp_sgemm_upto_8x8;
            algorithm_row  =  8;
            algorithm_col  =  8;
            break;
    }
    
    const nnpack_gemm_func_only func_only = algorithm_only;
    const nnpack_gemm_func_upto func_upto = algorithm_upto;
    
    const size_t row_subblock_max = algorithm_row;
    const size_t col_subblock_max = algorithm_col;
    
    const size_t reduction_block_max = round_down(cache_elements_l1 / (row_subblock_max + col_subblock_max), 2);
    const size_t row_block_max = round_down(cache_elements_l2 / reduction_block_max, row_subblock_max);
    const size_t col_block_max = round_down(cache_elements_l3 / reduction_block_max, col_subblock_max);
    
    for (size_t reduction_block_start = 0; reduction_block_start < reduction_size; reduction_block_start += reduction_block_max) {
        const size_t reduction_block_size = min(reduction_size - reduction_block_start, reduction_block_max);
        
        for (size_t col_block_start = 0; col_block_start < output_col; col_block_start += col_block_max) {
            const size_t col_block_size = min(output_col - col_block_start, col_block_max);
            
            struct gemm_context gemm_context = {
                .trans_a = transA == nnpackTrans,
                .trans_b = transB == nnpackTrans,
                .alpha = alpha,
                .beta = beta,
                .matrix_a = A,
                .matrix_b = B,
                .matrix_c = C,
                .reduction_block_start = reduction_block_start,
                .reduction_block_size = reduction_block_size,
                .output_row = output_row,
                .output_col = output_col,
                .reduction_size = reduction_size,
                .col_block_start = col_block_start,
                .col_subblock_max = col_subblock_max,
                .row_subblock_max = row_subblock_max,
                .func_only = func_only,
                .func_upto = func_upto,
            };
            pthreadpool_compute_2d_tiled(global_context.threadpool,
                                         (pthreadpool_function_2d_tiled_t) compute_gemm,
                                         &gemm_context,
                                         output_row,    col_block_size,
                                         row_block_max, col_subblock_max);
        }
    }
}
