#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <unistd.h>
#include <assert.h>

#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// nvcc macro.cu --std=c++11 -o cargo -arch=sm_80 -g -lineinfo

using namespace nvcuda;
using namespace cooperative_groups;

#define BLOCK_SIZE 256
typedef double real_t;

#define checkCudaError(call)                                                \
{                                                                           \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess)                                                 \
    {                                                                       \
        fprintf(stderr, "CUDA Error: %s (code: %d), at %s:%d\n",            \
                cudaGetErrorString(err), err, __FILE__, __LINE__);          \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
}

#define ifLastErrorExists(msg)                                         \
{                                                                      \
    cudaError_t err = cudaGetLastError();                              \
    if (err != cudaSuccess)                                            \
    {                                                                  \
        fprintf(stderr, "CUDA Error: %s, at %s:%d - %s\n",             \
                msg, __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
}

real_t determinant_3x3(real_t *m) {
    // computes the inverse of a matrix m
    double det = m[0*3+0] * (m[1*3+1] * m[2*3+2] - m[2*3+1] * m[1*3+2]) -
        m[0*3+1] * (m[1*3+0] * m[2*3+2] - m[1*3+2] * m[2*3+0]) +
        m[0*3+2] * (m[1*3+0] * m[2*3+1] - m[1*3+1] * m[2*3+0]);
    // print_matrix(m, 3, 3);
    // printf("det(m) = %lf\n", det);
    return det;
}

void inverse_3x3_T(real_t *m, real_t *m_inv)
{
    real_t det_inv = 1.0 / determinant_3x3(m);

    m_inv[0*3+0] = (m[1*3+1] * m[2*3+2] - m[2*3+1] * m[1*3+2]) * det_inv;
    m_inv[1*3+0] = (m[0*3+2] * m[2*3+1] - m[0*3+1] * m[2*3+2]) * det_inv;
    m_inv[2*3+0] = (m[0*3+1] * m[1*3+2] - m[0*3+2] * m[1*3+1]) * det_inv;
    m_inv[0*3+1] = (m[1*3+2] * m[2*3+0] - m[1*3+0] * m[2*3+2]) * det_inv;
    m_inv[1*3+1] = (m[0*3+0] * m[2*3+2] - m[0*3+2] * m[2*3+0]) * det_inv;
    m_inv[2*3+1] = (m[1*3+0] * m[0*3+2] - m[0*3+0] * m[1*3+2]) * det_inv;
    m_inv[0*3+2] = (m[1*3+0] * m[2*3+1] - m[2*3+0] * m[1*3+1]) * det_inv;
    m_inv[1*3+2] = (m[2*3+0] * m[0*3+1] - m[0*3+0] * m[2*3+1]) * det_inv;
    m_inv[2*3+2] = (m[0*3+0] * m[1*3+1] - m[1*3+0] * m[0*3+1]) * det_inv;
}

void jacobian_to_laplacian(real_t *micro_J, real_t *d_micro_L) {
    real_t J_inv_trans[9];
    real_t local_M[32];

    inverse_3x3_T(micro_J, J_inv_trans);

    real_t grad_ref_phi[4][3] = {
        {-1, -1, -1},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };

    real_t grad_phi[4][3];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            grad_phi[i][j] = 0;
            for (int k = 0; k < 3; k++) {
                grad_phi[i][j] += J_inv_trans[j * 3 + k] * grad_ref_phi[i][k];
            }
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            real_t dot_product = 0;
            for (int k = 0; k < 3; k++) {
                dot_product += grad_phi[i][k] * grad_phi[j][k];
            }
            local_M[i * 4 + j] = dot_product * determinant_3x3(micro_J) / 6.0;
        }
    }

    checkCudaError(cudaMemcpy(d_micro_L, local_M, 32 * sizeof(real_t), cudaMemcpyHostToDevice));

}

__global__ void macro_tet4_laplacian_apply_category_0(int level, real_t *local_M, real_t *vecX, real_t *vecY) {
    __shared__ real_t vals_gathered[1024];
    __shared__ real_t vals_to_scatter[1024];
    int vals_iter = 0;
    real_t results[64];

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, local_M, 4);

    thread_block g = this_thread_block();
    // Choose a leader in the thread block
    if (g.thread_rank() == 0) {
        int p = 0;
        vals_iter = 0;
        for (int i = 0; i < level - 1; i++)
        {
            int layer_items = (level - i + 1) * (level - i) / 2;
            for (int j = 0; j < level - i - 1; j++)
            {
                for (int k = 0; k < level - i - j - 1; k++)
                {
                    int e0 = p;
                    int e1 = p + 1;
                    int e2 = p + level - i - j;
                    int e3 = p + layer_items - j;

                    vals_gathered[vals_iter + 0] = vecX[e0];
                    vals_gathered[vals_iter + 1] = vecX[e3];
                    vals_gathered[vals_iter + 2] = vecX[e2];
                    vals_gathered[vals_iter + 3] = vecX[e1];
                    vals_iter += 4;

                    p++;
                }
                p++;
            }
            p++;
        }
    }
    g.sync();

    // TODO: check if there's a transpose missing; can change row order
    // TODO: think about the case where we don't have enough sub tetrahedrons (not divisible by 32)
    for (int i = 0; i < vals_iter; i += 32) {
        // Load the inputs
        wmma::load_matrix_sync(b_frag, &vals_gathered[i], 8);
        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        // Store the output (sync is necessary for &vals_to_scatter[i] due to padding overwritting memory)
        wmma::store_matrix_sync(results, c_frag, 8, wmma::mem_row_major);
        for (int j = 0; j < 8; j += 1) {
            vals_to_scatter[i+j*4+0] = results[0*8+j];
            vals_to_scatter[i+j*4+1] = results[1*8+j];
            vals_to_scatter[i+j*4+2] = results[2*8+j];
            vals_to_scatter[i+j*4+3] = results[3*8+j];
        }
    }

    if (g.thread_rank() == 0) {
        int p = 0;
        vals_iter = 0;
        for (int i = 0; i < level - 1; i++)
        {
            int layer_items = (level - i + 1) * (level - i) / 2;
            for (int j = 0; j < level - i - 1; j++)
            {
                for (int k = 0; k < level - i - j - 1; k++)
                {
                    int e0 = p;
                    int e1 = p + 1;
                    int e2 = p + level - i - j;
                    int e3 = p + layer_items - j;

                    vecY[e0] = vals_to_scatter[vals_iter + 0];
                    vecY[e3] = vals_to_scatter[vals_iter + 1];
                    vecY[e2] = vals_to_scatter[vals_iter + 2];
                    vecY[e1] = vals_to_scatter[vals_iter + 3];
                    vals_iter += 4;

                    p++;
                }
                p++;
            }
            p++;
        }
    }
}

__global__ void macro_tet4_laplacian_apply_category_1(int level, real_t *local_M, real_t *vecX, real_t *vecY) {
    __shared__ real_t vals_gathered[1024];
    __shared__ real_t vals_to_scatter[1024];
    int vals_iter = 0;
    real_t results[64];

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, local_M, 4);

    thread_block g = this_thread_block();
    // Choose a leader in the thread block
    if (g.thread_rank() == 0) {
        int p = 0;
        vals_iter = 0;
        for (int i = 0; i < level - 1; i++)
        {
            int layer_items = (level - i) * (level - i - 1) / 2;
            for (int j = 0; j < level - i - 1; j++)
            {
                p++;
                for (int k = 1; k < level - i - j - 1; k++)
                {
                    int e0 = p;
                    int e1 = p + layer_items + level - i - j - 1;
                    int e2 = p + layer_items + level - i - j;
                    int e3 = p + layer_items + level - i - j - 1 + level - i - j - 1;

                    vals_gathered[vals_iter + 0] = vecX[e0];
                    vals_gathered[vals_iter + 1] = vecX[e3];
                    vals_gathered[vals_iter + 2] = vecX[e2];
                    vals_gathered[vals_iter + 3] = vecX[e1];
                    vals_iter += 4;

                    p++;
                }
                p++;
            }
            p++;
        }
    }
    g.sync();

    // TODO: check if there's a transpose missing; can change row order
    // TODO: think about the case where we don't have enough sub tetrahedrons (not divisible by 32)
    for (int i = 0; i < vals_iter; i += 32) {
        // Load the inputs
        wmma::load_matrix_sync(b_frag, &vals_gathered[i], 8);
        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        // Store the output (sync is necessary for &vals_to_scatter[i] due to padding overwritting memory)
        wmma::store_matrix_sync(results, c_frag, 8, wmma::mem_row_major);
        for (int j = 0; j < 8; j += 1) {
            vals_to_scatter[i+j*4+0] = results[0*8+j];
            vals_to_scatter[i+j*4+1] = results[1*8+j];
            vals_to_scatter[i+j*4+2] = results[2*8+j];
            vals_to_scatter[i+j*4+3] = results[3*8+j];
        }
    }

    if (g.thread_rank() == 0) {
        int p = 0;
        vals_iter = 0;
        for (int i = 0; i < level - 1; i++)
        {
            int layer_items = (level - i) * (level - i - 1) / 2;
            for (int j = 0; j < level - i - 1; j++)
            {
                p++;
                for (int k = 1; k < level - i - j - 1; k++)
                {
                    int e0 = p;
                    int e1 = p + layer_items + level - i - j - 1;
                    int e2 = p + layer_items + level - i - j;
                    int e3 = p + layer_items + level - i - j - 1 + level - i - j - 1;

                    vecY[e0] = vals_to_scatter[vals_iter + 0];
                    vecY[e3] = vals_to_scatter[vals_iter + 1];
                    vecY[e2] = vals_to_scatter[vals_iter + 2];
                    vecY[e1] = vals_to_scatter[vals_iter + 3];
                    vals_iter += 4;

                    p++;
                }
                p++;
            }
            p++;
        }
    }
}

__global__ void macro_tet4_laplacian_apply_category_2(int level, real_t *local_M, real_t *vecX, real_t *vecY) {
    __shared__ real_t vals_gathered[1024];
    __shared__ real_t vals_to_scatter[1024];
    int vals_iter = 0;
    real_t results[64];

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, local_M, 4);

    thread_block g = this_thread_block();
    // Choose a leader in the thread block
    if (g.thread_rank() == 0) {
        int p = 0;
        vals_iter = 0;
        for (int i = 0; i < level - 1; i++)
        {
            int layer_items = (level - i) * (level - i - 1) / 2;
            for (int j = 0; j < level - i - 1; j++)
            {
                p++;
                for (int k = 1; k < level - i - j - 1; k++)
                {
                    int e0 = p;
                    int e1 = p + level - i - j;
                    int e3 = p + layer_items + level - i - j;
                    int e2 = p + layer_items + level - i - j - 1 + level - i - j - 1;

                    vals_gathered[vals_iter + 0] = vecX[e0];
                    vals_gathered[vals_iter + 1] = vecX[e3];
                    vals_gathered[vals_iter + 2] = vecX[e2];
                    vals_gathered[vals_iter + 3] = vecX[e1];
                    vals_iter += 4;

                    p++;
                }
                p++;
            }
            p++;
        }
    }
    g.sync();

    // TODO: check if there's a transpose missing; can change row order
    // TODO: think about the case where we don't have enough sub tetrahedrons (not divisible by 32)
    for (int i = 0; i < vals_iter; i += 32) {
        // Load the inputs
        wmma::load_matrix_sync(b_frag, &vals_gathered[i], 8);
        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        // Store the output (sync is necessary for &vals_to_scatter[i] due to padding overwritting memory)
        wmma::store_matrix_sync(results, c_frag, 8, wmma::mem_row_major);
        for (int j = 0; j < 8; j += 1) {
            vals_to_scatter[i+j*4+0] = results[0*8+j];
            vals_to_scatter[i+j*4+1] = results[1*8+j];
            vals_to_scatter[i+j*4+2] = results[2*8+j];
            vals_to_scatter[i+j*4+3] = results[3*8+j];
        }
    }

    if (g.thread_rank() == 0) {
        int p = 0;
        vals_iter = 0;
        for (int i = 0; i < level - 1; i++)
        {
            int layer_items = (level - i) * (level - i - 1) / 2;
            for (int j = 0; j < level - i - 1; j++)
            {
                p++;
                for (int k = 1; k < level - i - j - 1; k++)
                {
                    int e0 = p;
                    int e1 = p + level - i - j;
                    int e3 = p + layer_items + level - i - j;
                    int e2 = p + layer_items + level - i - j - 1 + level - i - j - 1;

                    vecY[e0] = vals_to_scatter[vals_iter + 0];
                    vecY[e3] = vals_to_scatter[vals_iter + 1];
                    vecY[e2] = vals_to_scatter[vals_iter + 2];
                    vecY[e1] = vals_to_scatter[vals_iter + 3];
                    vals_iter += 4;

                    p++;
                }
                p++;
            }
            p++;
        }
    }
}

__global__ void macro_tet4_laplacian_apply_category_3(int level, real_t *local_M, real_t *vecX, real_t *vecY) {
    __shared__ real_t vals_gathered[1024];
    __shared__ real_t vals_to_scatter[1024];
    int vals_iter = 0;
    real_t results[64];

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, local_M, 4);

    thread_block g = this_thread_block();
    // Choose a leader in the thread block
    if (g.thread_rank() == 0) {
        int p = 0;
        vals_iter = 0;
        for (int i = 0; i < level - 1; i++)
        {
            int layer_items = (level - i) * (level - i - 1) / 2;
            for (int j = 0; j < level - i - 1; j++)
            {
                p++;
                for (int k = 1; k < level - i - j - 1; k++)
                {
                    int e0 = p;
                    int e1 = p + level - i - j - 1;
                    int e2 = p + layer_items + level - i - j - 1;
                    int e3 = p + layer_items + level - i - j - 1 + level - i - j - 1;

                    vals_gathered[vals_iter + 0] = vecX[e0];
                    vals_gathered[vals_iter + 1] = vecX[e3];
                    vals_gathered[vals_iter + 2] = vecX[e2];
                    vals_gathered[vals_iter + 3] = vecX[e1];
                    vals_iter += 4;

                    p++;
                }
                p++;
            }
            p++;
        }
    }
    g.sync();

    // TODO: check if there's a transpose missing; can change row order
    // TODO: think about the case where we don't have enough sub tetrahedrons (not divisible by 32)
    for (int i = 0; i < vals_iter; i += 32) {
        // Load the inputs
        wmma::load_matrix_sync(b_frag, &vals_gathered[i], 8);
        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        // Store the output (sync is necessary for &vals_to_scatter[i] due to padding overwritting memory)
        wmma::store_matrix_sync(results, c_frag, 8, wmma::mem_row_major);
        for (int j = 0; j < 8; j += 1) {
            vals_to_scatter[i+j*4+0] = results[0*8+j];
            vals_to_scatter[i+j*4+1] = results[1*8+j];
            vals_to_scatter[i+j*4+2] = results[2*8+j];
            vals_to_scatter[i+j*4+3] = results[3*8+j];
        }
    }

    if (g.thread_rank() == 0) {
        int p = 0;
        vals_iter = 0;
        for (int i = 0; i < level - 1; i++)
        {
            int layer_items = (level - i) * (level - i - 1) / 2;
            for (int j = 0; j < level - i - 1; j++)
            {
                p++;
                for (int k = 1; k < level - i - j - 1; k++)
                {
                    int e0 = p;
                    int e1 = p + level - i - j - 1;
                    int e2 = p + layer_items + level - i - j - 1;
                    int e3 = p + layer_items + level - i - j - 1 + level - i - j - 1;

                    vecY[e0] = vals_to_scatter[vals_iter + 0];
                    vecY[e3] = vals_to_scatter[vals_iter + 1];
                    vecY[e2] = vals_to_scatter[vals_iter + 2];
                    vecY[e1] = vals_to_scatter[vals_iter + 3];
                    vals_iter += 4;

                    p++;
                }
                p++;
            }
            p++;
        }
    }
}


__global__ void macro_tet4_laplacian_apply_category_4(int level, real_t *local_M, real_t *vecX, real_t *vecY) {
    __shared__ real_t vals_gathered[1024];
    __shared__ real_t vals_to_scatter[1024];
    int vals_iter = 0;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

    real_t results[64];

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, local_M, 4);

    thread_block g = this_thread_block();
    // Choose a leader in the thread block
    if (g.thread_rank() == 0) {
        int p = 0;
        vals_iter = 0;
        for (int i = 1; i < level - 1; i++)
        {
            p = p + level - i + 1;
            int layer_items = (level - i) * (level - i - 1) / 2;
            for (int j = 0; j < level - i - 1; j++)
            {
                p++;
                for (int k = 1; k < level - i - j - 1; k++)
                {
                    int e0 = p;
                    int e1 = p + layer_items + level - i;
                    int e2 = p + layer_items + level - i - j + level - i;
                    int e3 = p + layer_items + level - i - j + level - i - 1;

                    vals_gathered[vals_iter + 0] = vecX[e0];
                    vals_gathered[vals_iter + 1] = vecX[e3];
                    vals_gathered[vals_iter + 2] = vecX[e2];
                    vals_gathered[vals_iter + 3] = vecX[e1];
                    vals_iter += 4;

                    p++;
                }
                p++;
            }
            p++;
        }
    }
    g.sync();

    // TODO: think about the case where we don't have enough sub tetrahedrons (not divisible by 32)
    for (int i = 0; i < vals_iter; i += 32) {
        // Load the inputs
        wmma::load_matrix_sync(b_frag, &vals_gathered[i], 8);
        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        // Store the output (sync is necessary for &vals_to_scatter[i] due to padding overwritting memory)
        wmma::store_matrix_sync(results, c_frag, 8, wmma::mem_row_major);
        for (int j = 0; j < 8; j += 1) {
            vals_to_scatter[i+j*4+0] = results[0*8+j];
            vals_to_scatter[i+j*4+1] = results[1*8+j];
            vals_to_scatter[i+j*4+2] = results[2*8+j];
            vals_to_scatter[i+j*4+3] = results[3*8+j];
        }
    }

    if (g.thread_rank() == 0) {
        int p = 0;
        vals_iter = 0;
        for (int i = 1; i < level - 1; i++)
        {
            p = p + level - i + 1;
            int layer_items = (level - i) * (level - i - 1) / 2;
            for (int j = 0; j < level - i - 1; j++)
            {
                p++;
                for (int k = 1; k < level - i - j - 1; k++)
                {
                    int e0 = p;
                    int e1 = p + layer_items + level - i;
                    int e2 = p + layer_items + level - i - j + level - i;
                    int e3 = p + layer_items + level - i - j + level - i - 1;

                    vecY[e0] = vals_to_scatter[vals_iter + 0];
                    vecY[e3] = vals_to_scatter[vals_iter + 1];
                    vecY[e2] = vals_to_scatter[vals_iter + 2];
                    vecY[e1] = vals_to_scatter[vals_iter + 3];
                    vals_iter += 4;

                    p++;
                }
                p++;
            }
            p++;
        }
    }
}

__global__ void macro_tet4_laplacian_apply_category_5(int level, real_t *local_M, real_t *vecX, real_t *vecY) {
    __shared__ real_t vals_gathered[1024];
    __shared__ real_t vals_to_scatter[1024];
    int vals_iter = 0;
    real_t results[64];

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, local_M, 4);

    thread_block g = this_thread_block();
    // Choose a leader in the thread block
    if (g.thread_rank() == 0) {
        int p = 0;
        vals_iter = 0;
        for (int i = 0; i < level - 1; i++)
        {
            int layer_items = (level - i) * (level - i - 1) / 2;
            for (int j = 0; j < level - i - 1; j++)
            {
                p++;
                for (int k = 1; k < level - i - j - 1; k++)
                {
                    int e0 = p;
                    int e1 = p + level - i - j - 1;
                    int e2 = p + layer_items + level - i - j - 1 + level - i - j - 1;
                    int e3 = p + level - i - j;

                    vals_gathered[vals_iter + 0] = vecX[e0];
                    vals_gathered[vals_iter + 1] = vecX[e3];
                    vals_gathered[vals_iter + 2] = vecX[e2];
                    vals_gathered[vals_iter + 3] = vecX[e1];
                    vals_iter += 4;

                    p++;
                }
                p++;
            }
            p++;
        }
    }
    g.sync();

    // TODO: check if there's a transpose missing; can change row order
    // TODO: think about the case where we don't have enough sub tetrahedrons (not divisible by 32)
    for (int i = 0; i < vals_iter; i += 32) {
        // Load the inputs
        wmma::load_matrix_sync(b_frag, &vals_gathered[i], 8);
        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        // Store the output (sync is necessary for &vals_to_scatter[i] due to padding overwritting memory)
        wmma::store_matrix_sync(results, c_frag, 8, wmma::mem_row_major);
        for (int j = 0; j < 8; j += 1) {
            vals_to_scatter[i+j*4+0] = results[0*8+j];
            vals_to_scatter[i+j*4+1] = results[1*8+j];
            vals_to_scatter[i+j*4+2] = results[2*8+j];
            vals_to_scatter[i+j*4+3] = results[3*8+j];
        }
    }

    if (g.thread_rank() == 0) {
        int p = 0;
        vals_iter = 0;
        for (int i = 0; i < level - 1; i++)
        {
            int layer_items = (level - i) * (level - i - 1) / 2;
            for (int j = 0; j < level - i - 1; j++)
            {
                p++;
                for (int k = 1; k < level - i - j - 1; k++)
                {
                    int e0 = p;
                    int e1 = p + level - i - j - 1;
                    int e2 = p + layer_items + level - i - j - 1 + level - i - j - 1;
                    int e3 = p + level - i - j;

                    vecY[e0] = vals_to_scatter[vals_iter + 0];
                    vecY[e3] = vals_to_scatter[vals_iter + 1];
                    vecY[e2] = vals_to_scatter[vals_iter + 2];
                    vecY[e1] = vals_to_scatter[vals_iter + 3];
                    vals_iter += 4;

                    p++;
                }
                p++;
            }
            p++;
        }
    }
}

int compute_nodes_number(int tetra_level)
{
    int num_nodes = 0;
    if (tetra_level % 2 == 0)
    {
        for (int i = 0; i < floor(tetra_level / 2); i++)
        {
            num_nodes += (tetra_level - i + 1) * (i + 1) * 2;
        }
        num_nodes += (tetra_level / 2 + 1) * (tetra_level / 2 + 1);
    }
    else 
    {
        for (int i = 0; i < floor(tetra_level / 2) + 1; i++)
        {
            num_nodes += (tetra_level - i + 1) * (i + 1) * 2;
        }
    }
    return num_nodes;
}

// CUDA kernel to sum six arrays
__global__ void sumUpVecY(real_t * a, real_t * b, real_t * c, real_t * d, real_t * e, real_t * f, real_t * vecY, int n) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not access out of bounds
    if (idx < n) {
        // Perform element-wise sum of six arrays
        vecY[idx] = a[idx] + b[idx] + c[idx] + d[idx] + e[idx] + f[idx];
    }
}

__host__ real_t *apply_cuda_macro_kernel(real_t *macro_J, int tetra_level, int num_nodes, real_t *d_vecX)
{
    cudaStream_t stream[6];
    for (int stream_idx = 0; stream_idx < 6; stream_idx += 1) {
        cudaStreamCreate(&stream[stream_idx]);
    }

    // TODO: think about this
    int level = tetra_level + 1;

    dim3 grid(1);
    dim3 block(32);

    // real_t *d_vecX;
    // cudaMalloc(&d_vecX, num_nodes * sizeof(real_t));
    // cudaMemcpy(d_vecX, vecX, num_nodes * sizeof(real_t), cudaMemcpyHostToDevice);

    // only the first 4x4 = 16 entries are used
    // the rest serves as padding to fit in a 8x4 matrix
    real_t *d_micro_L;
    checkCudaError(cudaMalloc(&d_micro_L, 32 * sizeof(real_t)));

    // real_t lapl_0[32] = {0}, lapl_1[32] = {0}, lapl_2[32] = {0};
    // real_t lapl_3[32] = {0}, lapl_4[32] = {0}, lapl_5[32] = {0};

    // real_t *vecY_host = (real_t *)malloc(num_nodes * sizeof(real_t *));
    // memset(vecY_host, 0, num_nodes * sizeof(real_t *));

    real_t *vecY_0, *vecY_1, *vecY_2, *vecY_3, *vecY_4, *vecY_5, *vecY;
    checkCudaError(cudaMalloc(&vecY_0, num_nodes * sizeof(real_t))));
    checkCudaError(cudaMemset(vecY_0, 0, num_nodes * sizeof(real_t))));

    checkCudaError(cudaMalloc(&vecY_1, num_nodes * sizeof(real_t)));
    checkCudaError(cudaMemset(vecY_1, 0, num_nodes * sizeof(real_t)));

    checkCudaError(cudaMalloc(&vecY_2, num_nodes * sizeof(real_t)));
    checkCudaError(cudaMemset(vecY_2, 0, num_nodes * sizeof(real_t)));

    checkCudaError(cudaMalloc(&vecY_3, num_nodes * sizeof(real_t)));
    checkCudaError(cudaMemset(vecY_3, 0, num_nodes * sizeof(real_t)));

    checkCudaError(cudaMalloc(&vecY_4, num_nodes * sizeof(real_t)));
    checkCudaError(cudaMemset(vecY_4, 0, num_nodes * sizeof(real_t)));

    checkCudaError(cudaMalloc(&vecY_5, num_nodes * sizeof(real_t)));
    checkCudaError(cudaMemset(vecY_5, 0, num_nodes * sizeof(real_t)));

    checkCudaError(cudaMalloc(&vecY, num_nodes * sizeof(real_t)));
    checkCudaError(cudaMemset(vecY, 0, num_nodes * sizeof(real_t)));

    real_t micro_J[9];
    // have to match the row/col order of compute_A
    real_t u[3] = {macro_J[0], macro_J[1], macro_J[2]};
    real_t v[3] = {macro_J[3], macro_J[4], macro_J[5]};
    real_t w[3] = {macro_J[6], macro_J[7], macro_J[8]};

    // [u | v | w]
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            micro_J[i * 3 + j] = macro_J[i * 3 + j] / level;
        }
    }
    assert(determinant_3x3(micro_J) > 0);
    jacobian_to_laplacian(micro_J, d_micro_L);

    // 2048 * 8 B / 1024 = 16 KB
    macro_tet4_laplacian_apply_category_0<<<grid, block, 16, stream[0]>>>(level, d_micro_L, d_vecX, vecY_0);
    ifLastErrorExists("Kernel launch failed");

        // [-u + w | w | -u + v + w]
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 0] = (-u[i] + w[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 1] = (w[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 2] = (-u[i] + v[i] + w[i]) / level;
        }
    
    assert(determinant_3x3(micro_J) > 0);
    jacobian_to_laplacian(micro_J, d_micro_L);
    macro_tet4_laplacian_apply_category_1<<<grid, block, 16, stream[1]>>>(level, d_micro_L, d_vecX, vecY_1);
    ifLastErrorExists("Kernel launch failed");

        // [v | -u + v + w | w]
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 0] = v[i] / level;
        }
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 1] = (-u[i] + v[i] + w[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 2] = (w[i]) / level;
        }

    assert(determinant_3x3(micro_J) > 0);
    jacobian_to_laplacian(micro_J, d_micro_L);
    macro_tet4_laplacian_apply_category_2<<<grid, block, 16, stream[2]>>>(level, d_micro_L, d_vecX, vecY_2);
    ifLastErrorExists("Kernel launch failed");

        // [-u + v | -u + w | -u + v + w]
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 0] = (-u[i] + v[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 1] = (-u[i] + w[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 2] = (-u[i] + v[i] + w[i]) / level;
        }

    assert(determinant_3x3(micro_J) > 0);
    jacobian_to_laplacian(micro_J, d_micro_L);
    macro_tet4_laplacian_apply_category_3<<<grid, block, 16, stream[3]>>>(level, d_micro_L, d_vecX, vecY_3);
    ifLastErrorExists("Kernel launch failed");

        // [-v + w | w | -u + w]
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 0] = (-v[i] + w[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 1] = (w[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 2] = (-u[i] + w[i]) / level;
        }

    assert(determinant_3x3(micro_J) > 0);
    jacobian_to_laplacian(micro_J, d_micro_L);
    macro_tet4_laplacian_apply_category_4<<<grid, block, 16, stream[4]>>>(level, d_micro_L, d_vecX, vecY_4);
    ifLastErrorExists("Kernel launch failed");

        // [-u + v | -u + v + w | v]
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 0] = (-u[i] + v[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 1] = (-u[i] + v[i] + w[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 2] = (v[i]) / level;
        }

    assert(determinant_3x3(micro_J) > 0);
    jacobian_to_laplacian(micro_J, d_micro_L);
    macro_tet4_laplacian_apply_category_5<<<grid, block, 16, stream[5]>>>(level, d_micro_L, d_vecX, vecY_5);
    ifLastErrorExists("Kernel launch failed");

    for (int stream_idx = 0; stream_idx < 6; stream_idx += 1) {
        cudaStreamSynchronize(stream[stream_idx]);
    }

    // Define block size and grid size
    int blockSize = 256;
    int gridSize = (num_nodes + blockSize - 1) / blockSize;
    // Launch the kernel
    sumUpVecY<<<gridSize, blockSize, 0, stream[0]>>>(vecY_0, vecY_1, vecY_2, vecY_3, vecY_4, vecY_5, vecY, num_nodes);
    ifLastErrorExists("Kernel launch failed");

    cudaStreamSynchronize(stream[0]);

    // cudaMemcpy(vecY_host, vecY, 1024 * sizeof(double), cudaMemcpyDeviceToHost);

    return vecY;
}

int compute_tets_number(int tetra_level)
{
    return (int) pow(tetra_level, 3);
}

// Kernel to apply Dirichlet boundary conditions
__global__ void applyDirichlet(real_t *Ax, real_t *x, int *dirichlet_nodes, int num_dirichlet_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_dirichlet_nodes) {
        int dirichlet_node = dirichlet_nodes[idx];
        Ax[dirichlet_node] = x[dirichlet_node];
    }
}

// Kernel to compute the residual r = rhs - Ax
__global__ void computeResidual(real_t *r, real_t *rhs, real_t *Ax, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_nodes) {
        r[idx] = rhs[idx] - Ax[idx];
    }
}

// Kernel to compute the norm of the residual
__global__ void computeNorm(real_t *r, real_t *norm_r, int num_nodes) {
    __shared__ real_t shared_norm[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Initialize shared memory
    shared_norm[tid] = 0.0f;

    // Compute partial sums of squares in parallel
    if (idx < num_nodes) {
        shared_norm[tid] = r[idx] * r[idx];
    }
    __syncthreads();

    // Reduce within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_norm[tid] += shared_norm[tid + stride];
        }
        __syncthreads();
    }

    // Accumulate the results from all blocks
    if (tid == 0) {
        atomicAdd(norm_r, shared_norm[0]);
    }
}

// Kernel for vector dot product: result = sum(a[i] * b[i])
__global__ void dotProduct(const real_t* a, const real_t* b, real_t* result, int N) {
    __shared__ real_t shared_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    shared_data[tid] = (idx < N) ? a[idx] * b[idx] : 0.0;
    __syncthreads();

    // Parallel reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Atomic add to accumulate the result across all blocks
    if (tid == 0) {
        atomicAdd(result, shared_data[0]);
    }
}

// Kernel for vector update: x = x + alpha * p
__global__ void vectorAdd(real_t* x, const real_t* p, real_t alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        x[idx] += alpha * p[idx];
    }
}

// CUDA Kernel to set the Dirichlet boundary conditions
__global__ void setDirichletBoundaryConditions(int *dirichlet_nodes, real_t *rhs, real_t *x, real_t *dirichlet_values, int num_dirichlet_nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_dirichlet_nodes) {
        int idx = dirichlet_nodes[i];
        rhs[idx] = dirichlet_values[i];
        x[idx] = dirichlet_values[i];
    }
}

void set_boundary_conditions_cuda(int num_nodes, real_t *rhs, real_t *x, size_t **dirichlet_nodes, size_t *num_dirichlet_nodes)
{
    *num_dirichlet_nodes = 2;
    checkCudaError(cudaMalloc(dirichlet_nodes, (*num_dirichlet_nodes) * sizeof(size_t)));

    // Set the Dirichlet nodes (e.g., first and last nodes)
    size_t h_dirichlet_nodes[] = {0, num_nodes - 1};
    checkCudaError(cudaMemcpy(*dirichlet_nodes, h_dirichlet_nodes, (*num_dirichlet_nodes) * sizeof(size_t), cudaMemcpyHostToDevice));

    // Set the Dirichlet values corresponding to the Dirichlet nodes
    real_t h_dirichlet_values[] = {1.0, 0.0};

    real_t *d_dirichlet_values;
    checkCudaError(cudaMalloc(&d_dirichlet_values, (*num_dirichlet_nodes) * sizeof(real_t)));
    checkCudaError(cudaMemcpy(d_dirichlet_values, h_dirichlet_values, (*num_dirichlet_nodes) * sizeof(real_t), cudaMemcpyHostToDevice));

    // Launch the kernel to set the Dirichlet boundary conditions
    int blockSize = 256;
    int gridSize = (*num_dirichlet_nodes + blockSize - 1) / blockSize;
    setDirichletBoundaryConditions<<<gridSize, blockSize>>>(*dirichlet_nodes, rhs, x, d_dirichlet_values, *num_dirichlet_nodes);
    ifLastErrorExists("Kernel launch failed");

    // Free the temporary device memory for Dirichlet values
    checkCudaError(cudaFree(d_dirichlet_values));
}

__host__ real_t *solve_using_conjugate_gradient(int tetra_level, int num_nodes, int num_tets, real_t *macro_J)
{
    // Allocate variables for boundary conditions
    int max_iter = 10000;
    double tol = 1e-7;
    real_t *h_x;
    checkCudaError(cudaMallocHost(&h_x, sizeof(real_t) * num_nodes));

    #define N 1024

    // Allocate GPU memory
    real_t *d_b, *d_x, *d_r, *d_p, *d_Ap;
    real_t *d_dot_r0, *d_dot_r1, *d_dot_pAp;
    checkCudaError(cudaMalloc(&d_b, num_nodes * sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_x, num_nodes * sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_r, num_nodes * sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_p, num_nodes * sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_Ap, num_nodes * sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_dot_r0, sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_dot_r1, sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_dot_pAp, sizeof(real_t)));

    real_t *d_norm_r;
    checkCudaError(cudaMalloc(&d_norm_r, sizeof(real_t)));
    int *d_dirichlet_nodes;
    int num_dirichlet_nodes;

    // Set boundary conditions
    set_boundary_conditions_cuda(num_nodes, d_b, d_x, &d_dirichlet_nodes, &num_dirichlet_nodes);

    // Define grid and block sizes
    int blockSize = BLOCK_SIZE;
    int gridSizeDirichlet = (num_dirichlet_nodes + blockSize - 1) / blockSize;
    int gridSizeNodes = (num_nodes + blockSize - 1) / blockSize;

    // Initialize r = b - A * x and p = r
    real_t *d_Ax = apply_cuda_macro_kernel(macro_J, tetra_level, num_nodes, d_x);
    applyDirichlet<<<gridSizeDirichlet, blockSize>>>(d_Ax, d_x, d_dirichlet_nodes, num_dirichlet_nodes);
    ifLastErrorExists("Kernel launch failed");

    computeResidual<<<gridSizeNodes, blockSize>>>(d_r, d_b, d_Ax, num_nodes);
    ifLastErrorExists("Kernel launch failed");

    // Calculate the initial dot product r0 = r^T * r
    checkCudaError(cudaMemset(d_dot_r0, 0, sizeof(real_t)));

    dotProduct<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_r, d_r, d_dot_r0, N);
    ifLastErrorExists("Kernel launch failed");

    real_t h_dot_r0, h_dot_r1, h_dot_pAp;
    checkCudaError(cudaMemcpy(&h_dot_r0, d_dot_r0, sizeof(real_t), cudaMemcpyDeviceToHost));

    // Start Conjugate Gradient iterations
    int iter = 0;
    while (iter < max_iter && sqrt(h_dot_r0) > tol) {
        // Ap = A * p
        d_Ap = apply_cuda_macro_kernel(macro_J, tetra_level, num_nodes, d_p);
        applyDirichlet<<<gridSizeDirichlet, blockSize>>>(d_Ap, d_p, d_dirichlet_nodes, num_dirichlet_nodes);
        ifLastErrorExists("Kernel launch failed");

        // Calculate p^T * Ap
        checkCudaError(cudaMemset(d_dot_pAp, 0, sizeof(real_t)));
        dotProduct<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_p, d_Ap, d_dot_pAp, num_nodes);
        ifLastErrorExists("Kernel launch failed");

        checkCudaError(cudaMemcpy(&h_dot_pAp, d_dot_pAp, sizeof(real_t), cudaMemcpyDeviceToHost));

        // alpha = r^T * r / p^T * Ap
        real_t alpha = h_dot_r0 / h_dot_pAp;

        // Update x = x + alpha * p
        vectorAdd<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_p, alpha, num_nodes);
        ifLastErrorExists("Kernel launch failed");

        // Update r = r - alpha * Ap
        vectorAdd<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_r, d_Ap, -alpha, num_nodes);
        ifLastErrorExists("Kernel launch failed");

        // Calculate new r^T * r
        checkCudaError(cudaMemset(d_dot_r1, 0, sizeof(real_t)));
        dotProduct<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_r, d_r, d_dot_r1, num_nodes);
        ifLastErrorExists("Kernel launch failed");

        checkCudaError(cudaMemcpy(&h_dot_r1, d_dot_r1, sizeof(real_t), cudaMemcpyDeviceToHost));

        printf("Iteration %d, Residual norm: %lf\n", iter, h_dot_r1);

        // Check for convergence
        if (sqrt(h_dot_r1) < tol) {
            checkCudaError(cudaMemcpy(&h_x, d_x, sizeof(real_t) * num_nodes, cudaMemcpyDeviceToHost));
            printf("Converged after %d iterations\nSolution: [", iter + 1);
            for (int k = 0; k < num_nodes; k++)
            {
                printf("%lf ", h_x[k]);
            }
            printf("]\n");
            break;
        }

        // beta = r1^T * r1 / r0^T * r0
        real_t beta = h_dot_r1 / h_dot_r0;

        // Update p = r + beta * p
        vectorAdd<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_p, d_r, beta, N);
        ifLastErrorExists("Kernel launch failed");

        // Update r0 = r1
        h_dot_r0 = h_dot_r1;

        iter++;
    }

    // Free GPU memory
    checkCudaError(cudaFree(d_b));
    checkCudaError(cudaFree(d_x));
    checkCudaError(cudaFree(d_r));
    checkCudaError(cudaFree(d_p));
    checkCudaError(cudaFree(d_Ap));
    checkCudaError(cudaFree(d_dot_r0));
    checkCudaError(cudaFree(d_dot_r1));
    checkCudaError(cudaFree(d_dot_pAp));

    checkCudaError(cudaFree(d_Ax));

    // Free allocated memory
    checkCudaError(cudaFree(d_dirichlet_nodes));

    return h_x;
}

// int solve_using_gradient_descent(int tetra_level, int nodes, int tets, real_t *macro_J)
// {
//     // Allocate variables for boundary conditions
//     real_t *rhs;          // = (real_t *)malloc(nodes * sizeof(real_t));
//     real_t *x;            // = (real_t *)malloc(nodes * sizeof(real_t));
//     int *dirichlet_nodes; // = (int *)malloc(nodes * sizeof(int));
//     int num_dirichlet_nodes;

//     // Set boundary conditions
//     set_boundary_conditions(nodes, &rhs, &x, &dirichlet_nodes, &num_dirichlet_nodes);

//     printf("Number of coordinate triplets: %d, Number of nodes: %d\n", num_coords, nodes);

//     // Maximum number of iterations
//     int max_iters = 100000;
//     real_t gamma = 2*1e-1;

//     real_t *r;
//     cudaMalloc(&r, nodes * sizeof(real_t));

//     for (int i = 0; i < max_iters; i++)
//     {
//         // Compute residual
//         norm_r = cuda_residual(macro_J, tetra_level, nodes, dirichlet_nodes, num_dirichlet_nodes, rhs, x, r);

//         // Print the norm of r
//         printf("Iteration %d, Residual norm: %lf\n", i, norm_r);

//         // Update x
//         for (int j = 0; j < nodes; j++)
//         {
//             x[j] += gamma * r[j];
//         }

//         // printf("nodes: %d coords: %d\n", nodes, num_coords);

// #ifdef GENERATE_VTK
//         // Write the result to construct the VTK file
//         FILE *f = fopen("solution.raw", "wb");
//         fwrite(x, sizeof(real_t), nodes, f);
//         fclose(f);

//         // Change directory
//         chdir("/Users/bolema/Documents/sfem/");
//         const char *command = "source venv/bin/activate && cd python/sfem/mesh/ && "
//         "python3 raw_to_db.py /Users/bolema/Documents/hpcfem/a64fx /Users/bolema/Documents/hpcfem/a64fx/test.vtk " 
//         "-c /Users/bolema/Documents/hpcfem/a64fx/category.raw "
//         "-p /Users/bolema/Documents/hpcfem/a64fx/solution.raw";

//         // Execute the command
//         int ret = system(command);
//         if (ret == -1) {
//             perror("system() call failed");
//         }
// #endif

//         // Check for convergence
//         if (norm_r < 1e-8)
//         {
//             printf("Converged after %d iterations\nSolution:", i + 1);
//             for (int k = 0; k < nodes; k++)
//             {
//                 printf("%lf \n", x[k]);
//             }
//             printf("\n");

//             // // Write the result to construct the VTK file
//             // FILE *f = fopen("solution.raw", "wb");
//             // fwrite(x, sizeof(real_t), nodes, f);
//             // fclose(f);

//             // // Change directory
//             // chdir("/Users/bolema/Documents/sfem/");
//             // const char *command = "source venv/bin/activate && cd python/sfem/mesh/ && "
//             // "python3 raw_to_db.py /Users/bolema/Documents/hpcfem/a64fx /Users/bolema/Documents/hpcfem/a64fx/test.vtk " 
//             // "-c /Users/bolema/Documents/hpcfem/a64fx/category.raw "
//             // "-p /Users/bolema/Documents/hpcfem/a64fx/solution.raw";

//             // // Execute the command
//             // int ret = system(command);
//             // if (ret == -1) {
//             //     perror("system() call failed");
//             // }

//             free(r);
//             break;
//         }
//     }

//     // Free allocated memory
//     free(rhs);
//     free(x);
//     free(dirichlet_nodes);

//     return 0;
// }

void compute_A(real_t *p0, real_t *p1, real_t *p2, real_t *p3, real_t *A)
{
    for (int i = 0; i < 3; i++)
    {
        A[i] = p1[i] - p0[i];
        A[3 + i] = p2[i] - p0[i];
        A[6 + i] = p3[i] - p0[i];
    }
    assert(determinant_3x3(A) > 0);
}

int main(void) {
    int tetra_level = 8;

    // Compute the number of nodes
    int num_nodes = compute_nodes_number(tetra_level);
    int num_tets = compute_tets_number(tetra_level);

    real_t macro_J[9];
    real_t p0[3] = {0, 0, 0};
    real_t p1[3] = {1, 0, 0};
    real_t p2[3] = {0, 1, 0};
    real_t p3[3] = {0, 0, 1};
    compute_A(p0, p1, p2, p3, macro_J);

    real_t *h_x = solve_using_conjugate_gradient(tetra_level, num_nodes, num_tets, macro_J);
    checkCudaError(cudaFreeHost(h_x));
    // solve_using_gradient_descent(tetra_level, num_nodes, num_tets, macro_J);

    return 0;

}