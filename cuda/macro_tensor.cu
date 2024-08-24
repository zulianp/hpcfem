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
#include <cublas_v2.h>

// nvcc macro.cu --std=c++11 -o cargo -arch=sm_75 -g -G -lcublas
using namespace nvcuda;
using namespace cooperative_groups;

#define BLOCK_SIZE 128
#define BLOCK_Y_SIZE 8
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

__device__ void print_matrix(real_t *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

__device__ real_t determinant_3x3(real_t *m) {
    // computes the inverse of a matrix m
    double det = m[0*3+0] * (m[1*3+1] * m[2*3+2] - m[2*3+1] * m[1*3+2]) -
        m[0*3+1] * (m[1*3+0] * m[2*3+2] - m[1*3+2] * m[2*3+0]) +
        m[0*3+2] * (m[1*3+0] * m[2*3+1] - m[1*3+1] * m[2*3+0]);
    // print_matrix(m, 3, 3);
    // printf("det(m) = %lf\n", det);
    return det;
}

__device__ void inverse_3x3_T(real_t *m, real_t *m_inv)
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

__device__ void jacobian_to_laplacian(real_t *macro_J, real_t *micro_L, int tetra_level, int category) {
    real_t J_inv_trans[9];
    real_t micro_J[9];
    const real_t grad_ref_phi[4][3] = {
        {-1, -1, -1},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };
    real_t grad_phi[4][3];

    // have to match the row/col order of compute_A
    real_t u[3] = {macro_J[0], macro_J[1], macro_J[2]};
    real_t v[3] = {macro_J[3], macro_J[4], macro_J[5]};
    real_t w[3] = {macro_J[6], macro_J[7], macro_J[8]};

    if (category == 0) {
        // [u | v | w]
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                micro_J[i * 3 + j] = macro_J[i * 3 + j] / tetra_level;
            }
        }
        assert(determinant_3x3(micro_J) > 0);
    } else if (category == 1) {
        // [-u + w | w | -u + v + w]
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 0] = (-u[i] + w[i]) / tetra_level;
            micro_J[i * 3 + 1] = (w[i]) / tetra_level;
            micro_J[i * 3 + 2] = (-u[i] + v[i] + w[i]) / tetra_level;
        }
        assert(determinant_3x3(micro_J) > 0);
    } else if (category == 2) {
        // [v | -u + v + w | w]
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 0] = v[i] / tetra_level;
            micro_J[i * 3 + 1] = (-u[i] + v[i] + w[i]) / tetra_level;
            micro_J[i * 3 + 2] = (w[i]) / tetra_level;
        }
        assert(determinant_3x3(micro_J) > 0);
    } else if (category == 3) {
        // [-u + v | -u + w | -u + v + w]
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 0] = (-u[i] + v[i]) / tetra_level;
            micro_J[i * 3 + 1] = (-u[i] + w[i]) / tetra_level;
            micro_J[i * 3 + 2] = (-u[i] + v[i] + w[i]) / tetra_level;
        }
        assert(determinant_3x3(micro_J) > 0);
    } else if (category == 4) {
        // [-v + w | w | -u + w]
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 0] = (-v[i] + w[i]) / tetra_level;
            micro_J[i * 3 + 1] = (w[i]) / tetra_level;
            micro_J[i * 3 + 2] = (-u[i] + w[i]) / tetra_level;
        }
        assert(determinant_3x3(micro_J) > 0);
    } else if (category == 5) {
        // [-u + v | -u + v + w | v]
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 0] = (-u[i] + v[i]) / tetra_level;
            micro_J[i * 3 + 1] = (-u[i] + v[i] + w[i]) / tetra_level;
            micro_J[i * 3 + 2] = (v[i]) / tetra_level;
        }
        assert(determinant_3x3(micro_J) > 0);
    }

    inverse_3x3_T(micro_J, J_inv_trans);

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
            micro_L[i * 4 + j] = dot_product * determinant_3x3(micro_J) / 6.0;
        }
    }

}

template <typename real_t>
__global__ void cu_macro_tet4_laplacian_apply_kernel(
        const size_t nelements,
        const size_t stride,  // Stride here represents the number of macro-elements (aligned to 256 bytes?)
        int tetra_level, 
        const real_t *const macro_jacobians,
        const real_t *const vecX,
        real_t *const vecY) {

    int level = tetra_level + 1;

    real_t macro_J[9];
    real_t micro_L[32];

    __shared__ real_t results[BLOCK_Y_SIZE][64];
    __shared__ real_t vals_gathered[BLOCK_Y_SIZE][256];
    __shared__ real_t vals_to_scatter[BLOCK_Y_SIZE][256];

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0);

    for (size_t macro_idx = blockIdx.x * blockDim.x + threadIdx.y; macro_idx < nelements;
         macro_idx += blockDim.x * gridDim.x) {

// #pragma unroll(9)
        for (int d = 0; d < 9; d++) {
            macro_J[d] = macro_jacobians[d * stride + macro_idx];
        }

        jacobian_to_laplacian(macro_J, micro_L, tetra_level, 0);
        // wmma::load_matrix_sync(a_frag, micro_L, 4);

        int vals_iter = 0;
        int p = 0;

        if (threadIdx.x < 4) {
            for (int i = 0; i < level - 1; i++)
            {
                int layer_items = (level - i + 1) * (level - i) / 2;
                for (int j = 0; j < level - i - 1; j++)
                {
                    for (int k = 0; k < level - i - j - 1; k++)
                    {
                        int e[4] = {p, p + layer_items - j, p + level - i - j, p + 1};
                        vals_gathered[threadIdx.y][4 * vals_iter + threadIdx.x] = vecX[e[threadIdx.x] * stride + macro_idx];
                        vals_iter += 1;

                        // printf("First: %d %d %d %d\n", e[0], e[1], e[2], e[3]);

                        if (macro_idx == 0 && threadIdx.x == 0 && p < 2) {
                            printf("p:%d vals_gathered[%d, %d]: %lf\n", p, threadIdx.x, threadIdx.y, vals_gathered[threadIdx.y][4 * vals_iter + threadIdx.x]);
                        }

                        p++;
                    }
                    p++;
                }
                p++;
            }
        }

        __syncwarp();
        // TODO: think about the case where we don't have enough sub tetrahedrons (not divisible by 32)
        for (int i = 0; i < vals_iter; i += 32) {
            // Load the inputs
            // wmma::load_matrix_sync(b_frag, &vals_gathered[threadIdx.y][i], 8);
            // Perform the matrix multiplication
            // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            // Store the output (sync is necessary for &vals_to_scatter[i] due to padding overwritting memory)
            // wmma::store_matrix_sync(results[threadIdx.y], c_frag, 8, // wmma::mem_row_major);
            if (threadIdx.x < 4) {
                for (int j = 0; j < 8; j += 1) {
                    vals_to_scatter[threadIdx.y][i + j * 4 + threadIdx.x] = results[threadIdx.y][threadIdx.x * 8 + j];

                    if (vals_iter == 0 && threadIdx.x == 0) {
                        printf("vals_iter:%d vals_to_scatter[%d, %d]: %lf\n", vals_iter, threadIdx.x, threadIdx.y, vals_to_scatter[threadIdx.y][i + j * 4 + threadIdx.x]);
                    }
                }
            }
        }
        __syncwarp();
        if (threadIdx.x < 4) {
            p = 0;
            vals_iter = 0;
            for (int i = 0; i < level - 1; i++)
            {
                int layer_items = (level - i) * (level - i - 1) / 2;
                for (int j = 0; j < level - i - 1; j++)
                {
                    p++;
                    for (int k = 1; k < level - i - j - 1; k++)
                    {
                        int e[4] = {p, p + layer_items - j, p + level - i - j, p + 1};
                        vecY[e[threadIdx.x] * stride + macro_idx] = vals_to_scatter[threadIdx.y][4 * vals_iter + threadIdx.x];
                        vals_iter += 1;
                        p++;
                    }
                    p++;
                }
                p++;
            }
        }

        // Second case
        jacobian_to_laplacian(macro_J, micro_L, tetra_level, 1);
        // wmma::load_matrix_sync(a_frag, micro_L, 4);

        // if (e == 0) {
        //     printf("Laplacian of Category %d\n", 1);
        //     print_matrix(micro_L, 4, 4);
        // }

        p = 0;
        if (threadIdx.x < 4) {
            for (int i = 0; i < level - 1; i++)
            {
                int layer_items = (level - i) * (level - i - 1) / 2;
                for (int j = 0; j < level - i - 1; j++)
                {
                    p++;
                    for (int k = 1; k < level - i - j - 1; k++)
                    {
                        int e[4] = {
                            p, 
                            p + layer_items + level - i - j - 1 + level - i - j - 1,
                            p + layer_items + level - i - j, 
                            p + layer_items + level - i - j - 1
                        };
                        vals_gathered[threadIdx.y][4 * vals_iter + threadIdx.x] = vecX[e[threadIdx.x] * stride + macro_idx];
                        vals_iter += 1;
                        p++;
                    }
                    p++;
                }
                p++;
            }
        }

        __syncwarp();
        for (int i = 0; i < vals_iter; i += 32) {
            // Load the inputs
            // wmma::load_matrix_sync(b_frag, &vals_gathered[threadIdx.y][i], 8);
            // Perform the matrix multiplication
            // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            // Store the output (sync is necessary for &vals_to_scatter[i] due to padding overwritting memory)
            // wmma::store_matrix_sync(results[threadIdx.y], c_frag, 8, // wmma::mem_row_major);
            if (threadIdx.x < 4) {
                for (int j = 0; j < 8; j += 1) {
                    vals_to_scatter[threadIdx.y][i + j * 4 + threadIdx.x] = results[threadIdx.y][threadIdx.x * 8 + j];
                }
            }
        }
        __syncwarp();
        if (threadIdx.x < 4) {
            p = 0;
            vals_iter = 0;

            for (int i = 0; i < level - 1; i++)
            {
                int layer_items = (level - i) * (level - i - 1) / 2;
                for (int j = 0; j < level - i - 1; j++)
                {
                    p++;
                    for (int k = 1; k < level - i - j - 1; k++)
                    {
                        int e[4] = {
                            p, 
                            p + layer_items + level - i - j - 1 + level - i - j - 1,
                            p + layer_items + level - i - j, 
                            p + layer_items + level - i - j - 1
                        };
                        vecY[e[threadIdx.x] * stride + macro_idx] = vals_to_scatter[threadIdx.y][4 * vals_iter + threadIdx.x];
                        vals_iter += 1;
                        p++;
                    }
                    p++;
                }
                p++;
            }
        }

        jacobian_to_laplacian(macro_J, micro_L, tetra_level, 2);
        // wmma::load_matrix_sync(a_frag, micro_L, 4);

        // if (e == 0) {
        //     printf("Laplacian of Category %d\n", 2);
        //     print_matrix(micro_L, 4, 4);
        // }

        // Third case
        p = 0;
        if (threadIdx.x < 4) {
            for (int i = 0; i < level - 1; i++)
            {
                int layer_items = (level - i) * (level - i - 1) / 2;
                for (int j = 0; j < level - i - 1; j++)
                {
                    p++;
                    for (int k = 1; k < level - i - j - 1; k++)
                    {
                        int e[4] = {
                            p, 
                            p + layer_items + level - i - j,
                            p + layer_items + level - i - j - 1 + level - i - j - 1,
                            p + level - i - j
                        };
                        vals_gathered[threadIdx.y][4 * vals_iter + threadIdx.x] = vecX[e[threadIdx.x] * stride + macro_idx];
                        vals_iter += 1;
                        p++;
                    }
                    p++;
                }
                p++;
            }
        }

        __syncwarp();
        for (int i = 0; i < vals_iter; i += 32) {
            // Load the inputs
            // wmma::load_matrix_sync(b_frag, &vals_gathered[threadIdx.y][i], 8);
            // Perform the matrix multiplication
            // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            // Store the output (sync is necessary for &vals_to_scatter[i] due to padding overwritting memory)
            // wmma::store_matrix_sync(results[threadIdx.y], c_frag, 8, // wmma::mem_row_major);
            if (threadIdx.x < 4) {
                for (int j = 0; j < 8; j += 1) {
                    vals_to_scatter[threadIdx.y][i + j * 4 + threadIdx.x] = results[threadIdx.y][threadIdx.x * 8 + j];
                }
            }
        }
        __syncwarp();
        if (threadIdx.x < 4) {
            p = 0;
            vals_iter = 0;

            for (int i = 0; i < level - 1; i++)
            {
                int layer_items = (level - i) * (level - i - 1) / 2;
                for (int j = 0; j < level - i - 1; j++)
                {
                    p++;
                    for (int k = 1; k < level - i - j - 1; k++)
                    {
                        int e[4] = {
                            p, 
                            p + layer_items + level - i - j,
                            p + layer_items + level - i - j - 1 + level - i - j - 1,
                            p + level - i - j
                        };
                        vecY[e[threadIdx.x] * stride + macro_idx] = vals_to_scatter[threadIdx.y][4 * vals_iter + threadIdx.x];
                        vals_iter += 1;

                        p++;
                    }
                    p++;
                }
                p++;
            }
        }

        jacobian_to_laplacian(macro_J, micro_L, tetra_level, 3);
        // wmma::load_matrix_sync(a_frag, micro_L, 4);
        // if (e == 0) {
        //     printf("Laplacian of Category %d\n", 3);
        //     print_matrix(micro_L, 4, 4);
        // }

        // Fourth case
        p = 0;

        if (threadIdx.x < 4) {
            for (int i = 0; i < level - 1; i++)
            {
                int layer_items = (level - i) * (level - i - 1) / 2;
                for (int j = 0; j < level - i - 1; j++)
                {
                    p++;
                    for (int k = 1; k < level - i - j - 1; k++)
                    {
                        int e[4] = {
                            p, 
                            p + layer_items + level - i - j - 1 + level - i - j - 1,
                            p + layer_items + level - i - j - 1,
                            p + level - i - j - 1
                        };
                        vals_gathered[threadIdx.y][4 * vals_iter + threadIdx.x] = vecX[e[threadIdx.x] * stride + macro_idx];
                        vals_iter += 1;
                        p++;
                    }
                    p++;
                }
                p++;
            }
        }

        __syncwarp();
        for (int i = 0; i < vals_iter; i += 32) {
            // Load the inputs
            // wmma::load_matrix_sync(b_frag, &vals_gathered[threadIdx.y][i], 8);
            // Perform the matrix multiplication
            // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            // Store the output (sync is necessary for &vals_to_scatter[i] due to padding overwritting memory)
            // wmma::store_matrix_sync(results[threadIdx.y], c_frag, 8, // wmma::mem_row_major);
            if (threadIdx.x < 4) {
                for (int j = 0; j < 8; j += 1) {
                    vals_to_scatter[threadIdx.y][i + j * 4 + threadIdx.x] = results[threadIdx.y][threadIdx.x * 8 + j];
                }
            }
        }
        __syncwarp();
        if (threadIdx.x < 4) {
            p = 0;
            vals_iter = 0;

            for (int i = 0; i < level - 1; i++)
            {
                int layer_items = (level - i) * (level - i - 1) / 2;
                for (int j = 0; j < level - i - 1; j++)
                {
                    p++;
                    for (int k = 1; k < level - i - j - 1; k++)
                    {
                        int e[4] = {
                            p, 
                            p + layer_items + level - i - j - 1 + level - i - j - 1,
                            p + layer_items + level - i - j - 1,
                            p + level - i - j - 1
                        };
                        vecY[e[threadIdx.x] * stride + macro_idx] = vals_to_scatter[threadIdx.y][4 * vals_iter + threadIdx.x];
                        vals_iter += 1;
                        p++;
                    }
                    p++;
                }
                p++;
            }
        }

        jacobian_to_laplacian(macro_J, micro_L, tetra_level, 4);
        // wmma::load_matrix_sync(a_frag, micro_L, 4);
        // if (e == 0) {
        //     printf("Laplacian of Category %d\n", 4);
        //     print_matrix(micro_L, 4, 4);
        // }

        // Fifth case
        p = 0;
        if (threadIdx.x < 4) {

            for (int i = 1; i < level - 1; i++)
            {
                p = p + level - i + 1;
                int layer_items = (level - i) * (level - i - 1) / 2;
                for (int j = 0; j < level - i - 1; j++)
                {
                    p++;
                    for (int k = 1; k < level - i - j - 1; k++)
                    {
                        int e[4] = {
                            p, 
                            p + layer_items + level - i - j + level - i,
                            p + layer_items + level - i,
                            p + layer_items + level - i - j + level - i - 1
                        };
                        vals_gathered[threadIdx.y][4 * vals_iter + threadIdx.x] = vecX[e[threadIdx.x] * stride + macro_idx];
                        vals_iter += 1;
                        p++;
                    }
                    p++;
                }
                p++;
            }
        }

        __syncwarp();
        for (int i = 0; i < vals_iter; i += 32) {
            // Load the inputs
            // wmma::load_matrix_sync(b_frag, &vals_gathered[threadIdx.y][i], 8);
            // Perform the matrix multiplication
            // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            // Store the output (sync is necessary for &vals_to_scatter[i] due to padding overwritting memory)
            // wmma::store_matrix_sync(results[threadIdx.y], c_frag, 8, // wmma::mem_row_major);
            if (threadIdx.x < 4) {
                for (int j = 0; j < 8; j += 1) {
                    vals_to_scatter[threadIdx.y][i + j * 4 + threadIdx.x] = results[threadIdx.y][threadIdx.x * 8 + j];
                }
            }
        }
        __syncwarp();
        if (threadIdx.x < 4) {
            p = 0;
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
                        int e[4] = {
                            p, 
                            p + layer_items + level - i - j + level - i,
                            p + layer_items + level - i,
                            p + layer_items + level - i - j + level - i - 1
                        };
                        vecY[e[threadIdx.x] * stride + macro_idx] = vals_to_scatter[threadIdx.y][4 * vals_iter + threadIdx.x];
                        vals_iter += 1;
                        p++;
                    }
                    p++;
                }
                p++;
            }
        }

        jacobian_to_laplacian(macro_J, micro_L, tetra_level, 5);
        // wmma::load_matrix_sync(a_frag, micro_L, 4);
        // if (e == 0) {
        //     printf("Laplacian of Category %d\n", 5);
        //     print_matrix(micro_L, 4, 4);
        // }

        // Sixth case
        p = 0;
        if (threadIdx.x < 4) {

            for (int i = 0; i < level - 1; i++)
            {
                int layer_items = (level - i) * (level - i - 1) / 2;
                for (int j = 0; j < level - i - 1; j++)
                {
                    p++;
                    for (int k = 1; k < level - i - j - 1; k++)
                    {
                        int e[4] = {
                            p, 
                            p + level - i - j,
                            p + layer_items + level - i - j - 1 + level - i - j - 1,
                            p + level - i - j - 1
                        };
                        vals_gathered[threadIdx.y][4 * vals_iter + threadIdx.x] = vecX[e[threadIdx.x] * stride + macro_idx];
                        vals_iter += 1;
                        p++;
                    }
                    p++;
                }
                p++;
            }
        }

        __syncwarp();
        for (int i = 0; i < vals_iter; i += 32) {
            // Load the inputs
            // wmma::load_matrix_sync(b_frag, &vals_gathered[threadIdx.y][i], 8);
            // Perform the matrix multiplication
            // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            // Store the output (sync is necessary for &vals_to_scatter[i] due to padding overwritting memory)
            // wmma::store_matrix_sync(results[threadIdx.y], c_frag, 8, // wmma::mem_row_major);
            if (threadIdx.x < 4) {
                for (int j = 0; j < 8; j += 1) {
                    vals_to_scatter[threadIdx.y][i + j * 4 + threadIdx.x] = results[threadIdx.y][threadIdx.x * 8 + j];
                }
            }
        }
        __syncwarp();
        if (threadIdx.x < 4) {
            p = 0;
            vals_iter = 0;

            for (int i = 0; i < level - 1; i++)
            {
                int layer_items = (level - i) * (level - i - 1) / 2;
                for (int j = 0; j < level - i - 1; j++)
                {
                    p++;
                    for (int k = 1; k < level - i - j - 1; k++)
                    {
                        int e[4] = {
                            p, 
                            p + level - i - j,
                            p + layer_items + level - i - j - 1 + level - i - j - 1,
                            p + level - i - j - 1
                        };
                        vecY[e[threadIdx.x] * stride + macro_idx] = vals_to_scatter[threadIdx.y][4 * vals_iter + threadIdx.x];
                        vals_iter += 1;
                        p++;
                    }
                    p++;
                }
                p++;
            }

        }
    }
}

int compute_nodes_number_aux(int tetra_level)
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

int compute_nodes_number(int tetra_level)
{
    // 1 layer = 4
    // 2 layer = 10
    return (tetra_level + 3) * (tetra_level + 2) * (tetra_level + 1) / 6;
}

int compute_tets_number(int tetra_level)
{
    return (int) pow(tetra_level, 3);
}

// Kernel to apply Dirichlet boundary conditions
__global__ void applyDirichlet(real_t *Ax, real_t *rhs, size_t num_macro_tets, size_t stride, size_t *dirichlet_nodes, size_t num_dirichlet_nodes) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_macro_tets;
         idx += blockDim.x * gridDim.x) {
            for (int j = 0; j < num_dirichlet_nodes; j += 1) {
                size_t dirichlet_node_idx = dirichlet_nodes[j];
                Ax[dirichlet_node_idx * stride + idx] = rhs[dirichlet_node_idx * stride + idx];
            }
    }
}

// Kernel to compute the residual r = rhs - Ax
__global__ void computeResidual(real_t *r, real_t *rhs, real_t *Ax, size_t num_macro_tets, size_t stride, size_t num_local_nodes) {
    // iterate over some tetrahedrons
    for (size_t macro_tet_idx = blockIdx.x * blockDim.x + threadIdx.x; macro_tet_idx < num_macro_tets;
         macro_tet_idx += blockDim.x * gridDim.x) {
            // iterate over the local nodes
            for (size_t node_idx = 0; node_idx < num_local_nodes; node_idx += 1) {
                r[node_idx * stride + macro_tet_idx] = rhs[node_idx * stride + macro_tet_idx] - Ax[node_idx * stride + macro_tet_idx];
            }
    }
}

// Kernel for vector dot product: result = sum(a[i] * b[i])
__global__ void dotProduct(const real_t* a, const real_t* b, real_t* result, size_t num_macro_tets, size_t stride, size_t num_local_nodes) {
    for (size_t macro_tet_idx = blockIdx.x * blockDim.x + threadIdx.x; macro_tet_idx < num_macro_tets;
         macro_tet_idx += blockDim.x * gridDim.x) {
            // iterate over the local nodes
            for (size_t node_idx = 0; node_idx < num_local_nodes; node_idx += 1) {
                result[macro_tet_idx] += a[node_idx * stride + macro_tet_idx] * b[node_idx * stride + macro_tet_idx];
            }
            if (macro_tet_idx == 0) {
                printf("dotProduct of %d: %lf\n", macro_tet_idx, result[macro_tet_idx]);
            }
    }
}

// Kernel for vector update: y = alpha * x + b
__global__ void vectorAdd(real_t *y, const real_t *alpha, const real_t *x, const real_t *b, size_t stride, size_t num_macro_tets, size_t num_local_nodes) {
    // iterate over some tetrahedrons
    for (size_t macro_tet_idx = blockIdx.x * blockDim.x + threadIdx.x; macro_tet_idx < num_macro_tets;
         macro_tet_idx += blockDim.x * gridDim.x) {
            // iterate over the local nodes
            for (size_t node_idx = 0; node_idx < num_local_nodes; node_idx += 1) {
                y[node_idx * stride + macro_tet_idx] = alpha[macro_tet_idx] * x[node_idx * stride + macro_tet_idx] + b[node_idx * stride + macro_tet_idx];
            }

            if (macro_tet_idx == 0) {
                printf("vecX after vectorAdd: \n");
                for (int n = 0; n < 100; n += 1) {
                    printf("%lf ", y[n * stride + macro_tet_idx]);
                }
                printf("\n");
            }
    }

}

// Kernel for vector update: x += alpha * r 
__global__ void vectorUpdate(real_t *x, const real_t alpha, const real_t *r, size_t stride, size_t num_macro_tets, size_t num_local_nodes) {
    // iterate over some tetrahedrons
    for (size_t macro_tet_idx = blockIdx.x * blockDim.x + threadIdx.x; macro_tet_idx < num_macro_tets;
         macro_tet_idx += blockDim.x * gridDim.x) {
            // iterate over the local nodes
            for (size_t node_idx = 0; node_idx < num_local_nodes; node_idx += 1) {
                x[node_idx * stride + macro_tet_idx] = alpha * r[node_idx * stride + macro_tet_idx];
            }
    }

}

// Kernel for vector update: x = x - alpha * p
__global__ void vectorMinus(real_t* x, const real_t* p, real_t *alpha, size_t stride, size_t num_macro_tets, size_t num_local_nodes) {
    // iterate over some tetrahedrons
    for (size_t macro_tet_idx = blockIdx.x * blockDim.x + threadIdx.x; macro_tet_idx < num_macro_tets;
         macro_tet_idx += blockDim.x * gridDim.x) {
            // iterate over the local nodes
            for (size_t node_idx = 0; node_idx < num_local_nodes; node_idx += 1) {
                x[node_idx * stride + macro_tet_idx] -= alpha[macro_tet_idx] * p[node_idx * stride + macro_tet_idx];
            }

            if (macro_tet_idx == 0) {
                printf("p in vectorMinus: \n");
                for (int n = 0; n < 100; n += 1) {
                    printf("%lf ", p[n * stride + macro_tet_idx]);
                }
                printf("\n");
                printf("alpha: %lf\n", alpha[macro_tet_idx]);
            }
    }

}

// Kernel for division update: alpha = up / down
__global__ void scalarDivide(real_t* alpha, const real_t* up, real_t *down, size_t num_macro_tets) {
    // iterate over some tetrahedrons
    for (size_t macro_tet_idx = blockIdx.x * blockDim.x + threadIdx.x; macro_tet_idx < num_macro_tets;
         macro_tet_idx += blockDim.x * gridDim.x) {
            // iterate over the local nodes
            alpha[macro_tet_idx] = up[macro_tet_idx] / down[macro_tet_idx];
            if (macro_tet_idx == 0) {
                printf("scalarDivide of %lf/%lf: %lf\n", up[macro_tet_idx], down[macro_tet_idx], alpha[macro_tet_idx]);
            }
    }

}

__global__ void checkConvergence(const real_t tol, const real_t* residual, int num_macro_tets, size_t* converged) {
    for (size_t macro_tet_idx = blockIdx.x * blockDim.x + threadIdx.x; macro_tet_idx < num_macro_tets;
         macro_tet_idx += blockDim.x * gridDim.x) {
        if (residual[macro_tet_idx] >= tol * tol) {
            *converged = 0;
            return;
        }
    }
}

// CUDA Kernel to set the Dirichlet boundary conditions
__global__ void setDirichletBoundaryConditions(size_t *dirichlet_nodes, real_t *rhs, real_t *x, size_t num_macro_tets, size_t stride, real_t *dirichlet_values, size_t num_dirichlet_nodes) {
    for (size_t macro_tet_idx = blockIdx.x * blockDim.x + threadIdx.x; macro_tet_idx < num_macro_tets;
         macro_tet_idx += blockDim.x * gridDim.x) {
            // iterate over the local nodes
            for (size_t i = 0; i < num_dirichlet_nodes; i += 1) {
                size_t local_node_idx = dirichlet_nodes[i];
                rhs[local_node_idx * stride + macro_tet_idx] = dirichlet_values[i];
                x[local_node_idx * stride + macro_tet_idx] = dirichlet_values[i];
            }
    }
}

void set_boundary_conditions_cuda(size_t num_nodes, real_t *rhs, real_t *x, size_t num_macro_tets, size_t stride, size_t **dirichlet_nodes, size_t *num_dirichlet_nodes)
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
    int threadsPerBlock = BLOCK_SIZE;
    int numBlocks = (num_macro_tets + threadsPerBlock - 1) / threadsPerBlock;
    setDirichletBoundaryConditions<<<numBlocks, threadsPerBlock>>>(*dirichlet_nodes, rhs, x, num_macro_tets, stride, d_dirichlet_values, *num_dirichlet_nodes);

    ifLastErrorExists("Kernel launch failed");

    // Free the temporary device memory for Dirichlet values
    checkCudaError(cudaFree(d_dirichlet_values));
}

__host__ real_t *solve_using_gradient_descent(int tetra_level, int num_macro_tets, int num_nodes, real_t *macro_jacobians)
{
    // Allocate variables for boundary conditions
    int max_iter = 100;
    double tol = 1e-2;
    real_t *h_x, *h_r;
    checkCudaError(cudaMallocHost(&h_x, num_macro_tets * sizeof(real_t) * num_nodes));
    checkCudaError(cudaMallocHost(&h_r, num_macro_tets * sizeof(real_t) * num_nodes));

    // Allocate GPU memory
    real_t *d_b, *d_x, *d_r, *d_Ax;
    checkCudaError(cudaMalloc(&d_b, num_macro_tets * num_nodes * sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_x, num_macro_tets * num_nodes * sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_Ax, num_macro_tets * num_nodes * sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_r, num_macro_tets * num_nodes * sizeof(real_t)));

    dim3 tensorCoreBlockDim(32, BLOCK_Y_SIZE, 1);
    int numTensorCoreBlocks = (num_macro_tets + tensorCoreBlockDim.y - 1) / tensorCoreBlockDim.y;

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    size_t *d_dirichlet_nodes;
    size_t num_dirichlet_nodes;

    int stride = num_macro_tets;

    set_boundary_conditions_cuda(num_nodes, d_b, d_x, num_macro_tets, stride, &d_dirichlet_nodes, &num_dirichlet_nodes);
    checkCudaError(cudaMemcpy(h_x, d_x, sizeof(real_t *) * num_macro_tets * num_nodes, cudaMemcpyDeviceToHost));

    int threadsPerBlock = BLOCK_SIZE;
    int numBlocks = (num_macro_tets + threadsPerBlock - 1) / threadsPerBlock;

    // Start Gradient Descent iterations
    int iter = 0;
    real_t gamma = 7 * 1e-1;
    while (iter < max_iter) {

        // Initialize r = b - A * x
        cu_macro_tet4_laplacian_apply_kernel<<<numTensorCoreBlocks, tensorCoreBlockDim>>>(num_macro_tets, stride, tetra_level, macro_jacobians, d_x, d_Ax);
        ifLastErrorExists("Kernel launch failed");

        applyDirichlet<<<numBlocks, threadsPerBlock>>>(d_Ax, d_b, num_macro_tets, stride, d_dirichlet_nodes, num_dirichlet_nodes);
        ifLastErrorExists("Kernel launch failed");

        computeResidual<<<numBlocks, threadsPerBlock>>>(d_r, d_b, d_Ax, num_macro_tets, stride, num_nodes);
        ifLastErrorExists("Kernel launch failed");

        printf("Iteration: %d, computeResidual finished\n", iter);
        cudaDeviceSynchronize();

        // cuBLAS for reduction
        // minSquareError computeNorm
        double norm_r = 0;
        cublasDnrm2(cublas_handle, num_macro_tets * num_nodes, d_r, 1, &norm_r);
        ifLastErrorExists("Kernel launch failed");

        printf("Iteration: %d, Global 2-norm = %lf\n", iter, norm_r);

        // Check for convergence
        if (norm_r < tol) {
            checkCudaError(cudaMemcpy(&h_x, d_x, sizeof(real_t) * num_nodes * num_macro_tets, cudaMemcpyDeviceToHost));
            for (int n = 0; n < num_nodes * num_macro_tets; n += num_macro_tets) {
                printf("%lf ", h_x[n]);
            }
            printf("Converged after %d iterations, 2-norm: %lf\n", iter, norm_r);
            // cudaFree(converged);
            break;
        }

        // Update x = x + alpha * p
        vectorUpdate<<<numBlocks, threadsPerBlock>>>(d_x, gamma, d_r, stride, num_macro_tets, num_nodes);
        ifLastErrorExists("Kernel launch failed");
        
        // checkCudaError(cudaMemcpy(h_x, d_x, sizeof(real_t *) * num_macro_tets * num_nodes, cudaMemcpyDeviceToHost));
        // printf("resulting x from vectorAdd: \n");
        // for (int n = 0; n < num_nodes * num_macro_tets; n += num_macro_tets) {
        //     printf("%lf ", h_x[n]);
        // }
        // printf("\n");

        iter++;
    }

    // Free GPU memory
    checkCudaError(cudaFree(d_b));
    checkCudaError(cudaFree(d_x));
    checkCudaError(cudaFree(d_r));
    checkCudaError(cudaFree(d_Ax));

    // Free allocated memory
    checkCudaError(cudaFree(d_dirichlet_nodes));

    checkCudaError(cudaFreeHost(h_r));

    cublasDestroy(cublas_handle);

    return h_x;
}

void compute_A(real_t *p0, real_t *p1, real_t *p2, real_t *p3, real_t *A)
{
    for (int i = 0; i < 3; i++)
    {
        A[i] = p1[i] - p0[i];
        A[3 + i] = p2[i] - p0[i];
        A[6 + i] = p3[i] - p0[i];
    }
    // assert(determinant_3x3(A) > 0);
}

int main(void) {
    int tetra_level = 8;

    // Compute the number of nodes
    int num_nodes = compute_nodes_number(tetra_level);
    int num_micro_tets = compute_tets_number(tetra_level);

    int num_macro_tets = 1000;

    real_t *macro_jacobians, *h_macro_jacobians;
    checkCudaError(cudaMallocHost(&h_macro_jacobians, sizeof(real_t) * 9 * num_macro_tets));
    checkCudaError(cudaMalloc(&macro_jacobians, sizeof(real_t) * 9 * num_macro_tets));

    for (int i = 0; i < num_macro_tets; i += 1) {
        real_t macro_J[9];
        real_t p0[3] = {0, 0, 0};
        real_t p1[3] = {1, 0, 0};
        real_t p2[3] = {0, 1, 0};
        real_t p3[3] = {0, 0, 1};
        compute_A(p0, p1, p2, p3, macro_J);
        for (int j = 0; j < 9; j += 1) {
            h_macro_jacobians[j * num_macro_tets + i] = macro_J[j];
        }
    }

    checkCudaError(cudaMemcpy(macro_jacobians, h_macro_jacobians, 9 * sizeof(real_t) * num_macro_tets, cudaMemcpyHostToDevice));
    real_t *h_x = solve_using_gradient_descent(tetra_level, num_macro_tets, num_nodes, macro_jacobians);
    checkCudaError(cudaFreeHost(h_x));

    checkCudaError(cudaFreeHost(h_macro_jacobians));
    checkCudaError(cudaFree(macro_jacobians));
    // solve_using_gradient_descent(tetra_level, num_nodes, num_micro_tets, macro_J);

    return 0;

}