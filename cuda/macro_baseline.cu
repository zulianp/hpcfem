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

// nvcc macro.cu --std=c++11 -o cargo -arch=sm_75 -g -lineinfo
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

__global__ void print_matrix(real_t *matrix, int rows, int cols)
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

__global__ real_t determinant_3x3(real_t *m) {
    // computes the inverse of a matrix m
    double det = m[0*3+0] * (m[1*3+1] * m[2*3+2] - m[2*3+1] * m[1*3+2]) -
        m[0*3+1] * (m[1*3+0] * m[2*3+2] - m[1*3+2] * m[2*3+0]) +
        m[0*3+2] * (m[1*3+0] * m[2*3+1] - m[1*3+1] * m[2*3+0]);
    // print_matrix(m, 3, 3);
    // printf("det(m) = %lf\n", det);
    return det;
}

__global__ void inverse_3x3_T(real_t *m, real_t *m_inv)
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

    int layer_level = tetra_level + 1;

    // have to match the row/col order of compute_A
    real_t u[3] = {macro_J[0], macro_J[1], macro_J[2]};
    real_t v[3] = {macro_J[3], macro_J[4], macro_J[5]};
    real_t w[3] = {macro_J[6], macro_J[7], macro_J[8]};

    if (category == 0) {
        // [u | v | w]
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                micro_J[i * 3 + j] = macro_J[i * 3 + j] / layer_level;
            }
        }
        assert(determinant_3x3(micro_J) > 0);
    } else if (category == 1) {
        // [-u + w | w | -u + v + w]
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 0] = (-u[i] + w[i]) / layer_level;
            micro_J[i * 3 + 1] = (w[i]) / layer_level;
            micro_J[i * 3 + 2] = (-u[i] + v[i] + w[i]) / layer_level;
        }
        assert(determinant_3x3(micro_J) > 0);
    } else if (category == 2) {
        // [v | -u + v + w | w]
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 0] = v[i] / layer_level;
            micro_J[i * 3 + 1] = (-u[i] + v[i] + w[i]) / layer_level;
            micro_J[i * 3 + 2] = (w[i]) / layer_level;
        }
        assert(determinant_3x3(micro_J) > 0);
    } else if (category == 3) {
        // [-u + v | -u + w | -u + v + w]
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 0] = (-u[i] + v[i]) / layer_level;
            micro_J[i * 3 + 1] = (-u[i] + w[i]) / layer_level;
            micro_J[i * 3 + 2] = (-u[i] + v[i] + w[i]) / layer_level;
        }
        assert(determinant_3x3(micro_J) > 0);
    } else if (category == 4) {
        // [-v + w | w | -u + w]
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 0] = (-v[i] + w[i]) / layer_level;
            micro_J[i * 3 + 1] = (w[i]) / layer_level;
            micro_J[i * 3 + 2] = (-u[i] + w[i]) / layer_level;
        }
        assert(determinant_3x3(micro_J) > 0);
    } else if (category == 5) {
        // [-u + v | -u + v + w | v]
        for (int i = 0; i < 3; i++) {
            micro_J[i * 3 + 0] = (-u[i] + v[i]) / layer_level;
            micro_J[i * 3 + 1] = (-u[i] + v[i] + w[i]) / layer_level;
            micro_J[i * 3 + 2] = (v[i]) / layer_level;
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

    for (size_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        real_t macro_J[9];
        real_t micro_L[16];
#pragma unroll(9)
        for (int d = 0; d < 9; d++) {
            macro_J[d] = macro_jacobians[d * stride + e];
        }

    jacobian_to_laplacian(macro_J, micro_L, tetra_level, 0);


    // printf("Laplacian of Category %d\n", 0);
    // print_matrix(micro_L, 4, 4);

        int p = 0;
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

      real_t vals_gathered[4];
      real_t vals_to_scatter[4];

                    // printf("First: %d %d %d %d\n", e0, e3, e2, e1);

                    vals_gathered[0] = vecX[e0 * stride + e];
                    vals_gathered[1] = vecX[e3 * stride + e];
                    vals_gathered[2] = vecX[e2 * stride + e];
                    vals_gathered[3] = vecX[e1 * stride + e];

                    for (int n = 0; n < 4; n++) {
                        for (int m = 0; m < 4; m++) {
                            vals_to_scatter[m] += micro_L[n * 4 + m] * vals_gathered[n];
                            // assert(!isnan(micro_L[i * 4 + j]));
                            // assert(!isnan(vecX[es[i]]));
                            // assert(!isnan(vecY[es[j]]));
                        }
                    }

                    vecY[e0 * stride + e] += vals_to_scatter[0];
                    vecY[e3 * stride + e] += vals_to_scatter[1];
                    vecY[e2 * stride + e] += vals_to_scatter[2];
                    vecY[e1 * stride + e] += vals_to_scatter[3];

                    p++;
                }
                p++;
            }
            p++;
        }

        // Second case

    jacobian_to_laplacian(macro_J, micro_L, tetra_level, 1);

    // printf("Laplacian of Category %d\n", 1);
    // print_matrix(micro_L, 4, 4);

        p = 0;
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

      real_t vals_gathered[4];
      real_t vals_to_scatter[4];

                    vals_gathered[0] = vecX[e0 * stride + e];
                    vals_gathered[1] = vecX[e3 * stride + e];
                    vals_gathered[2] = vecX[e2 * stride + e];
                    vals_gathered[3] = vecX[e1 * stride + e];

                    for (int n = 0; n < 4; n++) {
                        for (int m = 0; m < 4; m++) {
                            vals_to_scatter[m] += micro_L[n * 4 + m] * vals_gathered[n];
                            // assert(!isnan(micro_L[i * 4 + j]));
                            // assert(!isnan(vecX[es[i]]));
                            // assert(!isnan(vecY[es[j]]));
                        }
                    }

                    vecY[e0 * stride + e] += vals_to_scatter[0];
                    vecY[e3 * stride + e] += vals_to_scatter[1];
                    vecY[e2 * stride + e] += vals_to_scatter[2];
                    vecY[e1 * stride + e] += vals_to_scatter[3];

                    p++;
                }
                p++;
            }
            p++;
        }

    jacobian_to_laplacian(macro_J, micro_L, tetra_level, 2);

    // printf("Laplacian of Category %d\n", 2);
    // print_matrix(micro_L, 4, 4);

        // Third case
        p = 0;

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

      real_t vals_gathered[4];
      real_t vals_to_scatter[4];

                    vals_gathered[0] = vecX[e0 * stride + e];
                    vals_gathered[1] = vecX[e3 * stride + e];
                    vals_gathered[2] = vecX[e2 * stride + e];
                    vals_gathered[3] = vecX[e1 * stride + e];

                    for (int n = 0; n < 4; n++) {
                        for (int m = 0; m < 4; m++) {
                            vals_to_scatter[m] += micro_L[n * 4 + m] * vals_gathered[n];
                            // assert(!isnan(micro_L[i * 4 + j]));
                            // assert(!isnan(vecX[es[i]]));
                            // assert(!isnan(vecY[es[j]]));
                        }
                    }

                    vecY[e0 * stride + e] += vals_to_scatter[0];
                    vecY[e3 * stride + e] += vals_to_scatter[1];
                    vecY[e2 * stride + e] += vals_to_scatter[2];
                    vecY[e1 * stride + e] += vals_to_scatter[3];

                    p++;
                }
                p++;
            }
            p++;
        }

    jacobian_to_laplacian(macro_J, micro_L, tetra_level, 3);

    // printf("Laplacian of Category %d\n", 3);
    // print_matrix(micro_L, 4, 4);

        // Fourth case
        p = 0;

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

      real_t vals_gathered[4];
      real_t vals_to_scatter[4];

                    vals_gathered[0] = vecX[e0 * stride + e];
                    vals_gathered[1] = vecX[e3 * stride + e];
                    vals_gathered[2] = vecX[e2 * stride + e];
                    vals_gathered[3] = vecX[e1 * stride + e];

                    for (int n = 0; n < 4; n++) {
                        for (int m = 0; m < 4; m++) {
                            vals_to_scatter[m] += micro_L[n * 4 + m] * vals_gathered[n];
                            // assert(!isnan(micro_L[i * 4 + j]));
                            // assert(!isnan(vecX[es[i]]));
                            // assert(!isnan(vecY[es[j]]));
                        }
                    }

                    vecY[e0 * stride + e] += vals_to_scatter[0];
                    vecY[e3 * stride + e] += vals_to_scatter[1];
                    vecY[e2 * stride + e] += vals_to_scatter[2];
                    vecY[e1 * stride + e] += vals_to_scatter[3];

                    p++;
                }
                p++;
            }
            p++;
        }

    jacobian_to_laplacian(macro_J, micro_L, tetra_level, 4);

    // printf("Laplacian of Category %d\n", 0);
    // print_matrix(micro_L, 4, 4);

        // Fifth case
        p = 0;

        for (int i = 1; i < level - 1; i++)
        {
            p = p + level - i + 1;
            for (int j = 0; j < level - i - 1; j++)
            {
                p++;
                for (int k = 1; k < level - i - j - 1; k++)
                {
                    int layer_items = (level - i) * (level - i - 1) / 2;
                    int e0 = p;
                    int e1 = p + layer_items + level - i;
                    int e2 = p + layer_items + level - i - j + level - i;
                    int e3 = p + layer_items + level - i - j + level - i - 1;

      real_t vals_gathered[4];
      real_t vals_to_scatter[4];

                    vals_gathered[0] = vecX[e0 * stride + e];
                    vals_gathered[1] = vecX[e3 * stride + e];
                    vals_gathered[2] = vecX[e2 * stride + e];
                    vals_gathered[3] = vecX[e1 * stride + e];

                    for (int n = 0; n < 4; n++) {
                        for (int m = 0; m < 4; m++) {
                            vals_to_scatter[m] += micro_L[n * 4 + m] * vals_gathered[n];
                            // assert(!isnan(micro_L[i * 4 + j]));
                            // assert(!isnan(vecX[es[i]]));
                            // assert(!isnan(vecY[es[j]]));
                        }
                    }

                    vecY[e0 * stride + e] += vals_to_scatter[0];
                    vecY[e3 * stride + e] += vals_to_scatter[1];
                    vecY[e2 * stride + e] += vals_to_scatter[2];
                    vecY[e1 * stride + e] += vals_to_scatter[3];

                    p++;
                }
                p++;
            }
            p++;
        }

    jacobian_to_laplacian(macro_J, micro_L, tetra_level, 5);

    // printf("Laplacian of Category %d\n", 5);
    // print_matrix(micro_L, 4, 4);

        // Sixth case
        p = 0;
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

      real_t vals_gathered[4];
      real_t vals_to_scatter[4];

                    vals_gathered[0] = vecX[e0 * stride + e];
                    vals_gathered[1] = vecX[e3 * stride + e];
                    vals_gathered[2] = vecX[e2 * stride + e];
                    vals_gathered[3] = vecX[e1 * stride + e];

                    for (int n = 0; n < 4; n++) {
                        for (int m = 0; m < 4; m++) {
                            vals_to_scatter[m] += micro_L[n * 4 + m] * vals_gathered[n];
                            // assert(!isnan(micro_L[i * 4 + j]));
                            // assert(!isnan(vecX[es[i]]));
                            // assert(!isnan(vecY[es[j]]));
                        }
                    }

                    vecY[e0 * stride + e] += vals_to_scatter[0];
                    vecY[e3 * stride + e] += vals_to_scatter[1];
                    vecY[e2 * stride + e] += vals_to_scatter[2];
                    vecY[e1 * stride + e] += vals_to_scatter[3];

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

int compute_tets_number(int tetra_level)
{
    return (int) pow(tetra_level, 3);
}

// Kernel to apply Dirichlet boundary conditions
__global__ void applyDirichlet(real_t *Ax, real_t *x, size_t num_macro_tets, size_t stride, size_t *dirichlet_nodes, size_t num_dirichlet_nodes) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_macro_tets;
         idx += blockDim.x * gridDim.x) {
            for (int j = 0; j < num_dirichlet_nodes; j += 1) {
                size_t dirichlet_node_idx = dirichlet_nodes[j];
                Ax[dirichlet_node_idx * stride + idx] = x[dirichlet_node_idx * stride + idx];
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
    }
}

// Kernel for vector update: x = x + alpha * p
__global__ void vectorAdd(real_t* x, const real_t* p, real_t *alpha, size_t stride, size_t num_macro_tets, size_t num_local_nodes) {
    // iterate over some tetrahedrons
    for (size_t macro_tet_idx = blockIdx.x * blockDim.x + threadIdx.x; macro_tet_idx < num_macro_tets;
         macro_tet_idx += blockDim.x * gridDim.x) {
            // iterate over the local nodes
            for (size_t node_idx = 0; node_idx < num_local_nodes; node_idx += 1) {
                x[node_idx * stride + macro_tet_idx] += alpha[macro_tet_idx] * p[node_idx * stride + macro_tet_idx];
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
    }

}

// Kernel for division update: alpha = up / down
__global__ void scalarDivide(real_t* alpha, const real_t* up, real_t *down, size_t num_macro_tets) {
    // iterate over some tetrahedrons
    for (size_t macro_tet_idx = blockIdx.x * blockDim.x + threadIdx.x; macro_tet_idx < num_macro_tets;
         macro_tet_idx += blockDim.x * gridDim.x) {
            // iterate over the local nodes
            alpha[macro_tet_idx] = up[macro_tet_idx] / down[macro_tet_idx];
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
    int blockSize = 256;
    int gridSize = (num_macro_tets + blockSize - 1) / blockSize;
    setDirichletBoundaryConditions<<<gridSize, blockSize>>>(*dirichlet_nodes, rhs, x, num_macro_tets, stride, d_dirichlet_values, *num_dirichlet_nodes);

    ifLastErrorExists("Kernel launch failed");

    // Free the temporary device memory for Dirichlet values
    checkCudaError(cudaFree(d_dirichlet_values));
}

__host__ real_t *solve_using_conjugate_gradient(int tetra_level, int num_macro_tets, int num_nodes, real_t *macro_jacobians)
{
    // Allocate variables for boundary conditions
    int max_iter = 10000;
    double tol = 1e-7;
    real_t *h_x;
    checkCudaError(cudaMallocHost(&h_x, num_macro_tets * sizeof(real_t) * num_nodes));

    // Allocate GPU memory
    real_t *d_b, *d_x, *d_r, *d_p, *d_Ap, *d_Ax;
    real_t *d_dot_r0, *d_dot_r1, *d_dot_pAp;
    checkCudaError(cudaMalloc(&d_b, num_macro_tets * num_nodes * sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_x, num_macro_tets * num_nodes * sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_Ax, num_macro_tets * num_nodes * sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_r, num_macro_tets * num_nodes * sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_p, num_macro_tets * num_nodes * sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_Ap, num_macro_tets * num_nodes * sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_dot_r0, num_macro_tets * sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_dot_r1, num_macro_tets * sizeof(real_t)));
    checkCudaError(cudaMalloc(&d_dot_pAp, num_macro_tets * sizeof(real_t)));

    real_t *alpha, *beta;
    checkCudaError(cudaMalloc(&alpha, num_macro_tets * sizeof(real_t)));
    checkCudaError(cudaMalloc(&beta, num_macro_tets * sizeof(real_t)));

    // real_t *d_norm_r;
    // checkCudaError(cudaMalloc(&d_norm_r, sizeof(real_t)));
    size_t *d_dirichlet_nodes;
    size_t num_dirichlet_nodes;

    int stride = num_macro_tets;

    // TODO: Set boundary conditions
    set_boundary_conditions_cuda(num_nodes, d_b, d_x, num_macro_tets, stride, &d_dirichlet_nodes, &num_dirichlet_nodes);

    // Define grid and block sizes
    // int blockSize = BLOCK_SIZE;
    // int gridSizeNodes = (num_nodes + blockSize - 1) / blockSize;

    // Initialize r = b - A * x and p = r
    int blockSizeMacroTets = BLOCK_SIZE;
    int gridSizeMacroTets = (num_macro_tets + blockSizeMacroTets - 1) / blockSizeMacroTets;
    cu_macro_tet4_laplacian_apply_kernel<<<blockSizeMacroTets, gridSizeMacroTets>>>(num_macro_tets, stride, tetra_level, macro_jacobians, d_x, d_Ax);

    applyDirichlet<<<blockSizeMacroTets, gridSizeMacroTets>>>(d_Ax, d_x, num_macro_tets, stride, d_dirichlet_nodes, num_dirichlet_nodes);
    ifLastErrorExists("Kernel launch failed");

    computeResidual<<<blockSizeMacroTets, gridSizeMacroTets>>>(d_r, d_b, d_Ax, num_macro_tets, stride, num_nodes);
    ifLastErrorExists("Kernel launch failed");

    // Calculate the initial dot product r0 = r^T * r
    checkCudaError(cudaMemset(d_dot_r0, 0, num_macro_tets * sizeof(real_t)));

    dotProduct<<<blockSizeMacroTets, gridSizeMacroTets>>>(d_r, d_r, d_dot_r0, num_macro_tets, stride, num_nodes);

    ifLastErrorExists("Kernel launch failed");

    size_t *converged = nullptr;
    if (converged != nullptr) {
        checkCudaError(cudaMallocManaged(&converged, sizeof(size_t)));
    }
    *converged = 0;

    checkConvergence<<<blockSizeMacroTets, gridSizeMacroTets>>>(tol, d_dot_r0, num_macro_tets, converged);

    //real_t h_dot_r0, h_dot_r1, h_dot_pAp;
    //checkCudaError(cudaMemcpy(&h_dot_r0, d_dot_r0, sizeof(real_t), cudaMemcpyDeviceToHost));

    // Start Conjugate Gradient iterations
    int iter = 0;
    while (iter < max_iter && converged == 0) {
        // Ap = A * p
        cu_macro_tet4_laplacian_apply_kernel<<<blockSizeMacroTets, gridSizeMacroTets>>>(num_macro_tets, stride, tetra_level, macro_jacobians, d_p, d_Ap);

        applyDirichlet<<<blockSizeMacroTets, gridSizeMacroTets>>>(d_Ap, d_p, num_macro_tets, stride, d_dirichlet_nodes, num_dirichlet_nodes);
        ifLastErrorExists("Kernel launch failed");

        // Calculate p^T * Ap
        checkCudaError(cudaMemset(d_dot_pAp, 0, num_macro_tets * sizeof(real_t)));

        dotProduct<<<blockSizeMacroTets, gridSizeMacroTets>>>(d_p, d_Ap, d_dot_pAp, num_macro_tets, stride, num_nodes);
        ifLastErrorExists("Kernel launch failed");

        // checkCudaError(cudaMemcpy(&h_dot_pAp, d_dot_pAp, sizeof(real_t), cudaMemcpyDeviceToHost));

        // alpha = r^T * r / p^T * Ap
        // real_t alpha = h_dot_r0 / h_dot_pAp;
        scalarDivide<<<blockSizeMacroTets, gridSizeMacroTets>>>(alpha, d_dot_r0, d_dot_pAp, num_macro_tets);

        // Update x = x + alpha * p
        vectorAdd<<<blockSizeMacroTets, gridSizeMacroTets>>>(d_x, d_p, alpha, stride, num_macro_tets, num_nodes);
        ifLastErrorExists("Kernel launch failed");

        // Update r = r - alpha * Ap
        vectorMinus<<<blockSizeMacroTets, gridSizeMacroTets>>>(d_r, d_Ap, alpha, stride, num_macro_tets, num_nodes);
        ifLastErrorExists("Kernel launch failed");

        // Calculate new r^T * r
        checkCudaError(cudaMemset(d_dot_r1, 0, sizeof(real_t) * num_macro_tets));

        dotProduct<<<blockSizeMacroTets, gridSizeMacroTets>>>(d_r, d_r, d_dot_r1, num_macro_tets, stride, num_nodes);
        ifLastErrorExists("Kernel launch failed");

        // checkCudaError(cudaMemcpy(&h_dot_r1, d_dot_r1, sizeof(real_t), cudaMemcpyDeviceToHost));
        // printf("Iteration %d, Residual norm: %lf\n", iter, h_dot_r1);

        *converged = 0;
        checkConvergence<<<blockSizeMacroTets, gridSizeMacroTets>>>(tol, d_dot_r1, num_macro_tets, converged);
        // Check for convergence
        if (*converged == 1) {
            checkCudaError(cudaMemcpy(&h_x, d_x, sizeof(real_t) * num_nodes * num_macro_tets, cudaMemcpyDeviceToHost));
            printf("Converged after %d iterations\n", iter + 1);
            cudaFree(converged);

            // for (int k = 0; k < num_nodes; k++)
            // {
            //     printf("%lf ", h_x[k]);
            // }
            // printf("]\n"); //printf("Solution: [\n");
            break;
        }

        // beta = r1^T * r1 / r0^T * r0
        // real_t beta = h_dot_r1 / h_dot_r0;
        scalarDivide<<<blockSizeMacroTets, gridSizeMacroTets>>>(beta, d_dot_r1, d_dot_r0, num_macro_tets);

        // Update p = r + beta * p
        vectorAdd<<<blockSizeMacroTets, gridSizeMacroTets>>>(d_p, d_r, beta, stride, num_macro_tets, num_nodes);

        ifLastErrorExists("Kernel launch failed");

        // Update r0 = r1
        d_dot_r0 = d_dot_r1;

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

    checkCudaError(cudaFree(alpha));
    checkCudaError(cudaFree(beta));

    checkCudaError(cudaFree(d_Ax));

    // Free allocated memory
    checkCudaError(cudaFree(d_dirichlet_nodes));

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
    assert(determinant_3x3(A) > 0);
}

int main(void) {
    int tetra_level = 8;

    // Compute the number of nodes
    int num_nodes = compute_nodes_number(tetra_level);
    int num_micro_tets = compute_tets_number(tetra_level);

    int num_macro_tets = 100000;

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
    real_t *h_x = solve_using_conjugate_gradient(tetra_level, num_macro_tets, num_nodes, macro_jacobians);
    checkCudaError(cudaFreeHost(h_x));

    checkCudaError(cudaFreeHost(h_macro_jacobians));
    checkCudaError(cudaFree(macro_jacobians));
    // solve_using_gradient_descent(tetra_level, num_nodes, num_micro_tets, macro_J);

    return 0;

}