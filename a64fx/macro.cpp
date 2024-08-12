#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <unistd.h>
#include <assert.h>
// #include <omp.h>

#define MAX_NODES 100000
#define POW2(x) ((x) * (x))

// #define GENERATE_VTK

typedef double real_t;
typedef float geom_t;

void matrix_inverse(real_t *A, real_t *invA, int n);

void print_matrix(real_t *matrix, int rows, int cols)
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
real_t determinant(real_t *A, int n)
{
    int i, j, k;
    real_t det = 1.0;
    for (i = 0; i < n; i++)
    {
        for (j = i + 1; j < n; j++)
        {
            real_t ratio = A[j * n + i] / A[i * n + i];
            for (k = i; k < n; k++)
            {
                A[j * n + k] -= ratio * A[i * n + k];
            }
        }
        det *= A[i * n + i];
    }
    return det;
}

real_t determinant_3x3(real_t *m) {
    // computes the inverse of a matrix m
    double det = m[0*3+0] * (m[1*3+1] * m[2*3+2] - m[2*3+1] * m[1*3+2]) -
        m[0*3+1] * (m[1*3+0] * m[2*3+2] - m[1*3+2] * m[2*3+0]) +
        m[0*3+2] * (m[1*3+0] * m[2*3+1] - m[1*3+1] * m[2*3+0]);
    print_matrix(m, 3, 3);
    printf("det(m) = %lf\n", det);
    return det;
}

void inverse_3x3(real_t *m, real_t *m_inv)
{
    real_t det_inv = 1.0 / determinant_3x3(m);

    m_inv[0*3+0] = (m[1*3+1] * m[2*3+2] - m[2*3+1] * m[1*3+2]) * det_inv;
    m_inv[0*3+1] = (m[0*3+2] * m[2*3+1] - m[0*3+1] * m[2*3+2]) * det_inv;
    m_inv[0*3+2] = (m[0*3+1] * m[1*3+2] - m[0*3+2] * m[1*3+1]) * det_inv;
    m_inv[1*3+0] = (m[1*3+2] * m[2*3+0] - m[1*3+0] * m[2*3+2]) * det_inv;
    m_inv[1*3+1] = (m[0*3+0] * m[2*3+2] - m[0*3+2] * m[2*3+0]) * det_inv;
    m_inv[1*3+2] = (m[1*3+0] * m[0*3+2] - m[0*3+0] * m[1*3+2]) * det_inv;
    m_inv[2*3+0] = (m[1*3+0] * m[2*3+1] - m[2*3+0] * m[1*3+1]) * det_inv;
    m_inv[2*3+1] = (m[2*3+0] * m[0*3+1] - m[0*3+0] * m[2*3+1]) * det_inv;
    m_inv[2*3+2] = (m[0*3+0] * m[1*3+1] - m[1*3+0] * m[0*3+1]) * det_inv;
}

void macro_tet4_laplacian_apply(int level, int category, real_t *macro_J, real_t *vecX, real_t *vecY) {
    real_t J_inv[9];
    real_t J_inv_trans[9];
    real_t mat_J[9];

    // have to match the row/col order of compute_A
    real_t u[3] = {macro_J[0], macro_J[1], macro_J[2]};
    real_t v[3] = {macro_J[3], macro_J[4], macro_J[5]};
    real_t w[3] = {macro_J[6], macro_J[7], macro_J[8]};

    if (category == 0) {
        // [u | v | w]
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                mat_J[i * 3 + j] = macro_J[i * 3 + j] / level;
            }
        }
        assert(determinant_3x3(mat_J) > 0);
    } else if (category == 1) {
        // [-u + w | w | -u + v + w]
        for (int i = 0; i < 3; i++) {
            mat_J[i * 3 + 0] = (-u[i] + w[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            mat_J[i * 3 + 1] = (w[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            mat_J[i * 3 + 2] = (-u[i] + v[i] + w[i]) / level;
        }
        assert(determinant_3x3(mat_J) > 0);
    } else if (category == 2) {
        // [v | -u + v + w | w]
        for (int i = 0; i < 3; i++) {
            mat_J[i * 3 + 0] = v[i] / level;
        }
        for (int i = 0; i < 3; i++) {
            mat_J[i * 3 + 1] = (-u[i] + v[i] + w[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            mat_J[i * 3 + 2] = (w[i]) / level;
        }
        assert(determinant_3x3(mat_J) > 0);
    } else if (category == 3) {
        // [-u + v | -u + w | -u + v + w]
        for (int i = 0; i < 3; i++) {
            mat_J[i * 3 + 0] = (-u[i] + v[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            mat_J[i * 3 + 1] = (-u[i] + w[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            mat_J[i * 3 + 2] = (-u[i] + v[i] + w[i]) / level;
        }
        assert(determinant_3x3(mat_J) > 0);
    } else if (category == 4) {
        // [-v + w | w | -u + w]
        for (int i = 0; i < 3; i++) {
            mat_J[i * 3 + 0] = (-v[i] + w[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            mat_J[i * 3 + 1] = (w[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            mat_J[i * 3 + 2] = (-u[i] + w[i]) / level;
        }
        assert(determinant_3x3(mat_J) > 0);
    } else if (category == 5) {
        // [-u + v | -u + v + w | v]
        for (int i = 0; i < 3; i++) {
            mat_J[i * 3 + 0] = (-u[i] + v[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            mat_J[i * 3 + 1] = (-u[i] + v[i] + w[i]) / level;
        }
        for (int i = 0; i < 3; i++) {
            mat_J[i * 3 + 2] = (v[i]) / level;
        }
        assert(determinant_3x3(mat_J) > 0);
    }

    inverse_3x3(mat_J, J_inv);

    // Transpose J_inv
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            J_inv_trans[i * 3 + j] = J_inv[j * 3 + i];
        }
    }

    real_t grad_ref_phi[4][3] = {
        {-1, -1, -1},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };

    real_t local_M[16] = {0};
    // real_t mat_A[4][4] = {
    //     {0, 0, 0, 0},
    //     {0, 0, 0, 0},
    //     {0, 0, 0, 0},
    //     {0, 0, 0, 0}
    // };

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
            local_M[i * 4 + j] = dot_product * determinant(mat_J, 3) / 6.0;
        }
    }

    if (category == 0)
    {
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

                    int es[4] = {e0, e1, e2, e3};

                    // printf("First: %d %d %d %d\n", e0, e1, e2, e3);

                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            vecY[es[j]] += local_M[i * 4 + j] * vecX[es[i]];
                        }
                    }
                    p++;
                }
                p++;
            }
            p++;
        }
    } else if (category == 1) {
        // Second case
        int p = 0;
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
                    int es[4] = {e0, e1, e2, e3};

                    // printf("Second: %d %d %d %d\n", e0, e1, e2, e3);
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            vecY[es[j]] += local_M[i * 4 + j] * vecX[es[i]];
                        }
                    }

                    p++;
                }
                p++;
            }
            p++;
        }
    } else if (category == 2) {
        // Third case
        int p = 0;

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
                    int es[4] = {e0, e1, e2, e3};

                    // printf("Third: %d %d %d %d\n", e0, e1, e2, e3);

                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            vecY[es[j]] += local_M[i * 4 + j] * vecX[es[i]];
                        }
                    }

                    p++;
                }
                p++;
            }
            p++;
        }
    } else if (category == 3) {
        // Fourth case
        int p = 0;

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
                    int es[4] = {e0, e1, e2, e3};

                    // printf("Fourth: %d %d %d %d\n", e0, e1, e2, e3);
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            vecY[es[j]] += local_M[i * 4 + j] * vecX[es[i]];
                        }
                    }

                    p++;
                }
                p++;
            }
            p++;
        }
    } else if (category == 4) {

        // Fifth case
        int p = 0;

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
                    int es[4] = {e0, e1, e2, e3};

                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            vecY[es[j]] += local_M[i * 4 + j] * vecX[es[i]];
                        }
                    }
                    // printf("Fifth: %d %d %d %d\n", e0, e1, e2, e3);

                    p++;
                }
                p++;
            }
            p++;
        }
    } else if (category == 5) {
        // Sixth case
        int p = 0;
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
                    int es[4] = {e0, e1, e2, e3};

                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            vecY[es[j]] += local_M[i * 4 + j] * vecX[es[i]];
                        }
                    }
                    // printf("Sixth: %d %d %d %d\n", e0, e1, e2, e3);

                    p++;
                }
                p++;
            }
            p++;
        }
    }

}

// Note: taken from Dr. Zulian's code
// FFF < u, v >_FFF = u^T * FFF * v
static inline void
tet4_laplacian_hessian_fff(const real_t *const __restrict__ fff,
                           real_t *const __restrict__ element_matrix) {
  const real_t x0 = -fff[0] - fff[1] - fff[2];
  const real_t x1 = -fff[1] - fff[3] - fff[4];
  const real_t x2 = -fff[2] - fff[4] - fff[5];
  element_matrix[0] =
      fff[0] + 2 * fff[1] + 2 * fff[2] + fff[3] + 2 * fff[4] + fff[5];
  element_matrix[1] = x0;
  element_matrix[2] = x1;
  element_matrix[3] = x2;
  element_matrix[4] = x0;
  element_matrix[5] = fff[0];
  element_matrix[6] = fff[1];
  element_matrix[7] = fff[2];
  element_matrix[8] = x1;
  element_matrix[9] = fff[1];
  element_matrix[10] = fff[3];
  element_matrix[11] = fff[4];
  element_matrix[12] = x2;
  element_matrix[13] = fff[2];
  element_matrix[14] = fff[4];
  element_matrix[15] = fff[5];
}

static inline void
tet4_fff(const real_t px0, const real_t px1, const real_t px2, const real_t px3,
         const real_t py0, const real_t py1, const real_t py2, const real_t py3,
         const real_t pz0, const real_t pz1, const real_t pz2, const real_t pz3,
         real_t *const fff) {
  const real_t x0 = -px0 + px1;
  const real_t x1 = -py0 + py2;
  const real_t x2 = -pz0 + pz3;
  const real_t x3 = x1 * x2;
  const real_t x4 = x0 * x3;
  const real_t x5 = -py0 + py3;
  const real_t x6 = -pz0 + pz2;
  const real_t x7 = x5 * x6;
  const real_t x8 = x0 * x7;
  const real_t x9 = -py0 + py1;
  const real_t x10 = -px0 + px2;
  const real_t x11 = x10 * x2;
  const real_t x12 = x11 * x9;
  const real_t x13 = -pz0 + pz1;
  const real_t x14 = x10 * x5;
  const real_t x15 = x13 * x14;
  const real_t x16 = -px0 + px3;
  const real_t x17 = x16 * x6 * x9;
  const real_t x18 = x1 * x16;
  const real_t x19 = x13 * x18;
  const real_t x20 = -1.0 / 6.0 * x12 + (1.0 / 6.0) * x15 + (1.0 / 6.0) * x17 -
                     1.0 / 6.0 * x19 + (1.0 / 6.0) * x4 - 1.0 / 6.0 * x8;
  const real_t x21 = x14 - x18;
  const real_t x22 = 1. / POW2(-x12 + x15 + x17 - x19 + x4 - x8);
  const real_t x23 = -x11 + x16 * x6;
  const real_t x24 = x3 - x7;
  const real_t x25 = -x0 * x5 + x16 * x9;
  const real_t x26 = x21 * x22;
  const real_t x27 = x0 * x2 - x13 * x16;
  const real_t x28 = x22 * x23;
  const real_t x29 = x13 * x5 - x2 * x9;
  const real_t x30 = x22 * x24;
  const real_t x31 = x0 * x1 - x10 * x9;
  const real_t x32 = -x0 * x6 + x10 * x13;
  const real_t x33 = -x1 * x13 + x6 * x9;
  fff[0] = x20 * (POW2(x21) * x22 + x22 * POW2(x23) + x22 * POW2(x24));
  fff[1] = x20 * (x25 * x26 + x27 * x28 + x29 * x30);
  fff[2] = x20 * (x26 * x31 + x28 * x32 + x30 * x33);
  fff[3] = x20 * (x22 * POW2(x25) + x22 * POW2(x27) + x22 * POW2(x29));
  fff[4] = x20 * (x22 * x25 * x31 + x22 * x27 * x32 + x22 * x29 * x33);
  fff[5] = x20 * (x22 * POW2(x31) + x22 * POW2(x32) + x22 * POW2(x33));
}

static inline real_t tet4_det_fff(const real_t *const fff) {
  return fff[0] * fff[3] * fff[5] - fff[0] * POW2(fff[4]) -
         POW2(fff[1]) * fff[5] + 2 * fff[1] * fff[2] * fff[4] -
         POW2(fff[2]) * fff[3];
}

void tet4_laplacian_hessian(real_t *element_matrix, const real_t x0,
                            const real_t x1, const real_t x2, const real_t x3,
                            const real_t y0, const real_t y1, const real_t y2,
                            const real_t y3, const real_t z0, const real_t z1,
                            const real_t z2, const real_t z3) {
#ifndef NDEBUG
  real_t J[9] = {
      x1 - x0, x2 - x0, x3 - x0, //
      y1 - y0, y2 - y0, y3 - y0, //
      z1 - z0, z2 - z0, z3 - z0
  };

  // print_matrix(J, 3, 3);

  assert(determinant(J, 3) > 0);
#endif

  real_t fff[6];
  tet4_fff(x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3, fff);
  // printf("%g\n", tet4_det_fff(fff));
  assert(tet4_det_fff(fff) > 0);
  tet4_laplacian_hessian_fff(fff, element_matrix);
}

// void tet4_laplacian_hessian(real_t *element_matrix,
//                             const real_t x0, const real_t x1, const real_t x2, const real_t x3,
//                             const real_t y0, const real_t y1, const real_t y2, const real_t y3,
//                             const real_t z0, const real_t z1, const real_t z2, const real_t z3)
// {
//     // FLOATING POINT OPS!
//     //    - Result: 4*ADD + 16*ASSIGNMENT + 16*MUL + 12*POW
//     //    - Subexpressions: 16*ADD + 9*DIV + 56*MUL + 7*NEG + POW + 32*SUB
//     const real_t x4 = z0 - z3;
//     const real_t x5 = x0 - x1;
//     const real_t x6 = y0 - y2;
//     const real_t x7 = x5 * x6;
//     const real_t x8 = z0 - z1;
//     const real_t x9 = x0 - x2;
//     const real_t x10 = y0 - y3;
//     const real_t x11 = x10 * x9;
//     const real_t x12 = z0 - z2;
//     const real_t x13 = x0 - x3;
//     const real_t x14 = y0 - y1;
//     const real_t x15 = x13 * x14;
//     const real_t x16 = x10 * x5;
//     const real_t x17 = x14 * x9;
//     const real_t x18 = x13 * x6;
//     const real_t x19 = x11 * x8 + x12 * x15 - x12 * x16 - x17 * x4 - x18 * x8 + x4 * x7;
//     assert(x19 > 0);
//     const real_t x20 = 1.0 / x19;
//     const real_t x21 = x11 - x18;
//     const real_t x22 = -x17 + x7;
//     const real_t x23 = x15 - x16 + x21 + x22;
//     const real_t x24 = -x12 * x13 + x4 * x9;
//     const real_t x25 = x12 * x5 - x8 * x9;
//     const real_t x26 = x13 * x8;
//     const real_t x27 = x4 * x5;
//     const real_t x28 = x26 - x27;
//     const real_t x29 = -x24 - x25 - x28;
//     const real_t x30 = x10 * x8;
//     const real_t x31 = x14 * x4;
//     const real_t x32 = -x10 * x12 + x4 * x6;
//     const real_t x33 = x12 * x14 - x6 * x8;
//     const real_t x34 = x30 - x31 + x32 + x33;
//     const real_t x35 = -x12;
//     const real_t x36 = -x9;
//     const real_t x37 = x19 * (x13 * x35 + x28 - x35 * x5 - x36 * x4 + x36 * x8);
//     const real_t x38 = -x19;
//     const real_t x39 = -x23;
//     const real_t x40 = -x34;
//     const real_t x41 = (1.0 / 6.0) / pow(x19, 2);
//     const real_t x42 = x41 * (x24 * x37 + x38 * (x21 * x39 + x32 * x40));
//     const real_t x43 = -x15 + x16;
//     const real_t x44 = (1.0 / 6.0) * x43;
//     const real_t x45 = -x26 + x27;
//     const real_t x46 = -x30 + x31;
//     const real_t x47 = (1.0 / 6.0) * x46;
//     const real_t x48 = x20 * (-x23 * x44 + (1.0 / 6.0) * x29 * x45 - x34 * x47);
//     const real_t x49 = x41 * (x25 * x37 + x38 * (x22 * x39 + x33 * x40));
//     const real_t x50 = (1.0 / 6.0) * x45;
//     const real_t x51 = x20 * (x21 * x44 + x24 * x50 + x32 * x47);
//     const real_t x52 =
//         x20 * (-1.0 / 6.0 * x21 * x22 - 1.0 / 6.0 * x24 * x25 - 1.0 / 6.0 * x32 * x33);
//     const real_t x53 = x20 * (x22 * x44 + x25 * x50 + x33 * x47);

//     element_matrix[0] =
//         x20 * (-1.0 / 6.0 * pow(x23, 2) - 1.0 / 6.0 * pow(x29, 2) - 1.0 / 6.0 * pow(x34, 2));
//     element_matrix[1] = x42;
//     element_matrix[2] = x48;
//     element_matrix[3] = x49;
//     element_matrix[4] = x42;
//     element_matrix[5] =
//         x20 * (-1.0 / 6.0 * pow(x21, 2) - 1.0 / 6.0 * pow(x24, 2) - 1.0 / 6.0 * pow(x32, 2));
//     element_matrix[6] = x51;
//     element_matrix[7] = x52;
//     element_matrix[8] = x48;
//     element_matrix[9] = x51;
//     element_matrix[10] =
//         x20 * (-1.0 / 6.0 * pow(x43, 2) - 1.0 / 6.0 * pow(x45, 2) - 1.0 / 6.0 * pow(x46, 2));
//     element_matrix[11] = x53;
//     element_matrix[12] = x49;
//     element_matrix[13] = x52;
//     element_matrix[14] = x53;
//     element_matrix[15] =
//         x20 * (-1.0 / 6.0 * pow(x22, 2) - 1.0 / 6.0 * pow(x25, 2) - 1.0 / 6.0 * pow(x33, 2));
// }

// void matrix_inverse(real_t *A, real_t *invA, int n) {
//     int *ipiv = (int *)malloc(n * sizeof(int));
//     int *info = (int *)malloc(sizeof(int));
//     int lwork = n * n;
//     real_t *work = (real_t *)malloc(lwork * sizeof(real_t));

//     // Copy A to invA because the LAPACK routine will overwrite it
//     for (int i = 0; i < n * n; i++) {
//         invA[i] = A[i];
//     }

//     // LAPACK routine to compute the LU factorization of A
//     dgetrf_(&n, &n, invA, &n, ipiv, info);
//     assert(*info == 0);

//     // LAPACK routine to compute the inverse of A given its LU factorization
//     dgetri_(&n, invA, &n, ipiv, work, &lwork, info);
//     assert(*info == 0);

//     free(ipiv);
//     free(info);
//     free(work);
// }

void matrix_inverse(real_t *A, real_t *invA, int n)
{
    int i, j, k;
    real_t ratio;

    // Create an augmented matrix [A|I]
    real_t augmented[n][2 * n];
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            augmented[i][j] = A[i * n + j];
            augmented[i][j + n] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Apply Gauss-Jordan elimination
    for (i = 0; i < n; i++)
    {
        // Make the diagonal element 1
        ratio = augmented[i][i];
        if (ratio == 0)
        {
            printf("Matrix is singular and cannot be inverted.\n");
            return;
        }
        for (j = 0; j < 2 * n; j++)
        {
            augmented[i][j] /= ratio;
        }

        // Make the other elements in the current column 0
        for (k = 0; k < n; k++)
        {
            if (k != i)
            {
                ratio = augmented[k][i];
                for (j = 0; j < 2 * n; j++)
                {
                    augmented[k][j] -= ratio * augmented[i][j];
                }
            }
        }
    }

    // Extract the inverse matrix from the augmented matrix
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            invA[i * n + j] = augmented[i][j + n];
        }
    }
}

real_t tetrahedron_volume(real_t *A, real_t *B, real_t *C, real_t *D)
{
    real_t mat[9] = {
        B[0] - A[0], C[0] - A[0], D[0] - A[0],
        B[1] - A[1], C[1] - A[1], D[1] - A[1],
        B[2] - A[2], C[2] - A[2], D[2] - A[2]};
    real_t det = determinant(mat, 3);
    return det / 6.0;
}

void compute_A(real_t *p0, real_t *p1, real_t *p2, real_t *p3, real_t *A)
{
    for (int i = 0; i < 3; i++)
    {
        A[i] = p1[i] - p0[i];
        A[3 + i] = p2[i] - p0[i];
        A[6 + i] = p3[i] - p0[i];
    }
    assert(determinant(A, 3) > 0);
}

void gather_and_scatter(int **micro_tets, int num_micro_tets, geom_t *x_coords, geom_t *y_coords, geom_t *z_coords, real_t *vecX, real_t *vecY)
{
    // pick a random micro tetrahedron
    int *e0 = micro_tets[0]; // num_micro_tets - 1
    // geom_t p0[3] = {x_coords[e0[0]], y_coords[e0[0]], z_coords[e0[0]]};
    // geom_t p1[3] = {x_coords[e0[1]], y_coords[e0[1]], z_coords[e0[1]]};
    // geom_t p2[3] = {x_coords[e0[2]], y_coords[e0[2]], z_coords[e0[2]]};
    // geom_t p3[3] = {x_coords[e0[3]], y_coords[e0[3]], z_coords[e0[3]]};

    real_t local_M[16];
    // printf("p0: %lf %lf %lf\n", x_coords[e0[0]], y_coords[e0[0]], z_coords[e0[0]]);
    // printf("p1: %lf %lf %lf\n", x_coords[e0[1]], y_coords[e0[1]], z_coords[e0[1]]);
    // printf("p2: %lf %lf %lf\n", x_coords[e0[2]], y_coords[e0[2]], z_coords[e0[2]]);
    // printf("p3: %lf %lf %lf\n", x_coords[e0[3]], y_coords[e0[3]], z_coords[e0[3]]);

    // tet4_laplacian_hessian(local_M,
    //                        x_coords[e0[0]], x_coords[e0[3]], x_coords[e0[2]], x_coords[e0[1]],
    //                        y_coords[e0[0]], y_coords[e0[3]], y_coords[e0[2]], y_coords[e0[1]],
    //                        z_coords[e0[0]], z_coords[e0[3]], z_coords[e0[2]], z_coords[e0[1]]);
    tet4_laplacian_hessian(local_M,
                           x_coords[e0[0]], x_coords[e0[1]], x_coords[e0[2]], x_coords[e0[3]],
                           y_coords[e0[0]], y_coords[e0[1]], y_coords[e0[2]], y_coords[e0[3]],
                           z_coords[e0[0]], z_coords[e0[1]], z_coords[e0[2]], z_coords[e0[3]]);
    // print_matrix(local_M, 4, 4);

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int t = 0; t < num_micro_tets; t++)
            {
                vecY[micro_tets[t][j]] += local_M[i * 4 + j] * vecX[micro_tets[t][i]];
            }
        }
    }
}

void assemble_macro_elem(int **micro_elems, int tetra_level, int nodes, int tets, 
    geom_t *x_coords, geom_t *y_coords, geom_t *z_coords, real_t *vecX, real_t *vecY)
{
    int level = tetra_level + 1;
    int n_macro_elems = 1;
    int *dofs = (int *)malloc(nodes * sizeof(int));
    // int global_iter = 0;

    // printf("Creating i0-i3 each containing %d micro tets\n", tets);

    int *i0 = (int *)malloc(tets * sizeof(int));
    int *i1 = (int *)malloc(tets * sizeof(int));
    int *i2 = (int *)malloc(tets * sizeof(int));
    int *i3 = (int *)malloc(tets * sizeof(int));
    real_t *category = (real_t *)malloc(tets * sizeof(real_t));

    // Initialize vecY to zero
    for (int i = 0; i < nodes; i++)
    {
        vecY[i] = 0;
    }

    for (int elem = 0; elem < n_macro_elems; elem++)
    {
        for (int i = 0; i < nodes; i++)
        {
            dofs[i] = micro_elems[i][elem];
        }

        int **micro_tets = (int **)malloc(tets * sizeof(int *));
        for (int i = 0; i < tets; i++)
        {
            micro_tets[i] = (int *)malloc(4 * sizeof(int));
        }

        int p = 0;
        int local_iter = 0, global_iter = 0;
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

                    // printf("First: %d %d %d %d\n", e0, e1, e2, e3);

                    micro_tets[local_iter][0] = dofs[e0];
                    micro_tets[local_iter][1] = dofs[e3];
                    micro_tets[local_iter][2] = dofs[e2];
                    micro_tets[local_iter][3] = dofs[e1];
                    local_iter += 1;

                    i0[global_iter] = e0;
                    i1[global_iter] = e3;
                    i2[global_iter] = e2;
                    i3[global_iter] = e1;
                    category[global_iter] = 1;
                    global_iter += 1;

                    p++;
                }
                p++;
            }
            p++;
        }
        // printf("Gathering category 1, processed %d tets\n", local_iter);
        gather_and_scatter(micro_tets, local_iter, x_coords, y_coords, z_coords, vecX, vecY);

        // Repeat the process for the remaining subtetrahedrons
        // Second case
        p = 0;
        for (int i = 0; i < local_iter; i++)
        {
            // cleanup
            for (int j = 0; j < 4; j++)
            {
                micro_tets[i][j] = 0;
            }
        }
        local_iter = 0;

        for (int i = 0; i < level - 1; i++)
        {
            int layer_items = (level - i) * (level - i - 1) / 2;
            for (int j = 0; j < level - i - 1; j++)
            {
                p++;
                for (int k = 1; k < level - i - j - 1; k++)
                {
                    /*
                    e0 = p
                    e1 = p-i-j-1+layer_items+level
                    e2 = p-i-j+layer_items+level
                    e3 = p+level-i-j-1+layer_items+level-i-j-1
                    */
                    int e0 = p;
                    int e1 = p + layer_items + level - i - j - 1;
                    int e2 = p + layer_items + level - i - j;
                    int e3 = p + layer_items + level - i - j - 1 + level - i - j - 1;

                    // printf("Second: %d %d %d %d\n", e0, e1, e2, e3);

                    micro_tets[local_iter][0] = dofs[e0];
                    micro_tets[local_iter][1] = dofs[e3];
                    micro_tets[local_iter][2] = dofs[e2];
                    micro_tets[local_iter][3] = dofs[e1];
                    local_iter += 1;

                    i0[global_iter] = e0;
                    i1[global_iter] = e3;
                    i2[global_iter] = e2;
                    i3[global_iter] = e1;
                    category[global_iter] = 2;
                    global_iter += 1;

                    p++;
                }
                p++;
            }
            p++;
        }
        // printf("Gathering category 2, processed %d tets\n", local_iter);
        gather_and_scatter(micro_tets, local_iter, x_coords, y_coords, z_coords, vecX, vecY);

        // Third case
        p = 0;
        for (int i = 0; i < local_iter; i++)
        {
            // cleanup
            for (int j = 0; j < 4; j++)
            {
                micro_tets[i][j] = 0;
            }
        }
        local_iter = 0;

        for (int i = 0; i < level - 1; i++)
        {
            int layer_items = (level - i) * (level - i - 1) / 2;
            for (int j = 0; j < level - i - 1; j++)
            {
                p++;
                for (int k = 1; k < level - i - j - 1; k++)
                {
                    /*
                    e0 = p
                    e1 = p+level-i-j
                    e3 = p+level-i-j+layer_items
                    e2 = p+level-i-j-1+layer_items+level-i-j-1
                    */
                    int e0 = p;
                    int e1 = p + level - i - j;
                    int e3 = p + layer_items + level - i - j;
                    int e2 = p + layer_items + level - i - j - 1 + level - i - j - 1;

                    // printf("Third: %d %d %d %d\n", e0, e1, e2, e3);

                    micro_tets[local_iter][0] = dofs[e0];
                    micro_tets[local_iter][1] = dofs[e3];
                    micro_tets[local_iter][2] = dofs[e2];
                    micro_tets[local_iter][3] = dofs[e1];
                    local_iter += 1;

                    i0[global_iter] = e0;
                    i1[global_iter] = e3;
                    i2[global_iter] = e2;
                    i3[global_iter] = e1;
                    category[global_iter] = 3;
                    global_iter += 1;

                    p++;
                }
                p++;
            }
            p++;
        }
        // printf("Gathering category 3, processed %d tets\n", local_iter);
        gather_and_scatter(micro_tets, local_iter, x_coords, y_coords, z_coords, vecX, vecY);

        // Fourth case
        p = 0;
        for (int i = 0; i < local_iter; i++)
        {
            // cleanup
            for (int j = 0; j < 4; j++)
            {
                micro_tets[i][j] = 0;
            }
        }
        local_iter = 0;

        for (int i = 0; i < level - 1; i++)
        {
            int layer_items = (level - i) * (level - i - 1) / 2;
            for (int j = 0; j < level - i - 1; j++)
            {
                p++;
                for (int k = 1; k < level - i - j - 1; k++)
                {
                    /*
                    e0 = p
                    e1 = p+level-i-j-1
                    e2 = p-i-j-1+layer_items+level
                    e3 = p+level-i-j-1+layer_items+level-i-j-1
                    */
                    int e0 = p;
                    int e1 = p + level - i - j - 1;
                    int e2 = p + layer_items + level - i - j - 1;
                    int e3 = p + layer_items + level - i - j - 1 + level - i - j - 1;

                    // printf("Fourth: %d %d %d %d\n", e0, e1, e2, e3);

                    micro_tets[local_iter][0] = dofs[e0];
                    micro_tets[local_iter][1] = dofs[e3];
                    micro_tets[local_iter][2] = dofs[e2];
                    micro_tets[local_iter][3] = dofs[e1];
                    local_iter += 1;

                    i0[global_iter] = e0;
                    i1[global_iter] = e3;
                    i2[global_iter] = e2;
                    i3[global_iter] = e1;
                    category[global_iter] = 4;
                    global_iter += 1;

                    p++;
                }
                p++;
            }
            p++;
        }
        // printf("Gathering category 4, processed %d tets\n", local_iter);
        gather_and_scatter(micro_tets, local_iter, x_coords, y_coords, z_coords, vecX, vecY);

        // Fifth case
        p = 0;
        for (int i = 0; i < local_iter; i++)
        {
            // cleanup
            for (int j = 0; j < 4; j++)
            {
                micro_tets[i][j] = 0;
            }
        }
        local_iter = 0;

        for (int i = 1; i < level - 1; i++)
        {
            p = p + level - i + 1;
            for (int j = 0; j < level - i - 1; j++)
            {
                p++;
                for (int k = 1; k < level - i - j - 1; k++)
                {
                    int layer_items = (level - i) * (level - i - 1) / 2;
                    /*
                    e0 = p
                    e1 = p+layer_items+level-i
                    e2 = p+layer_items+level-i-j+level-i
                    e3 = p+layer_items+level-i-j+level-i-1
                    */
                    int e0 = p;
                    int e1 = p + layer_items + level - i;
                    int e2 = p + layer_items + level - i - j + level - i;
                    int e3 = p + layer_items + level - i - j + level - i - 1;

                    // printf("Fifth: %d %d %d %d\n", e0, e1, e2, e3);

                    micro_tets[local_iter][0] = dofs[e0];
                    micro_tets[local_iter][1] = dofs[e1];
                    micro_tets[local_iter][2] = dofs[e3];
                    micro_tets[local_iter][3] = dofs[e2];
                    local_iter += 1;

                    i0[global_iter] = e0;
                    i1[global_iter] = e2;
                    i2[global_iter] = e1;
                    i3[global_iter] = e3;
                    category[global_iter] = 5;
                    global_iter += 1;

                    p++;
                }
                p++;
            }
            p++;
        }
        // printf("Gathering category 5, processed %d tets\n", local_iter);
        gather_and_scatter(micro_tets, local_iter, x_coords, y_coords, z_coords, vecX, vecY);

        // Sixth case
        p = 0;
        for (int i = 0; i < local_iter; i++)
        {
            // cleanup
            for (int j = 0; j < 4; j++)
            {
                micro_tets[i][j] = 0;
            }
        }
        local_iter = 0;

        for (int i = 0; i < level - 1; i++)
        {
            int layer_items = (level - i) * (level - i - 1) / 2;
            for (int j = 0; j < level - i - 1; j++)
            {
                p++;
                for (int k = 1; k < level - i - j - 1; k++)
                {
                    /*
                    e0 = p
                    e1 = p+level-i-j-1
                    e3 = p+level-i-j
                    e2 = p+level-i-j-1+layer_items+level-i-j-1
                    */
                    int e0 = p;
                    int e1 = p + level - i - j - 1;
                    int e2 = p + layer_items + level - i - j - 1 + level - i - j - 1;
                    int e3 = p + level - i - j;

                    // printf("Sixth: %d %d %d %d\n", e0, e1, e2, e3);

                    micro_tets[local_iter][0] = dofs[e0];
                    micro_tets[local_iter][1] = dofs[e2];
                    micro_tets[local_iter][2] = dofs[e1];
                    micro_tets[local_iter][3] = dofs[e3];
                    local_iter += 1;

                    i0[global_iter] = e0;
                    i1[global_iter] = e2;
                    i2[global_iter] = e1;
                    i3[global_iter] = e3;
                    category[global_iter] = 6;
                    global_iter += 1;

                    p++;
                }
                p++;
            }
            p++;
        }
        // printf("Gathering category 6, processed %d tets\n", local_iter);
        gather_and_scatter(micro_tets, local_iter, x_coords, y_coords, z_coords, vecX, vecY);

#ifdef GENERATE_VTK
        FILE *f = fopen("i0.raw", "wb");
        fwrite(i0, sizeof(int32_t), tets, f);
        fclose(f);

        f = fopen("i1.raw", "wb");
        fwrite(i1, sizeof(int32_t), tets, f);
        fclose(f);

        f = fopen("i2.raw", "wb");
        fwrite(i2, sizeof(int32_t), tets, f);
        fclose(f);

        f = fopen("i3.raw", "wb");
        fwrite(i3, sizeof(int32_t), tets, f);
        fclose(f);

        f = fopen("category.raw", "wb");
        fwrite(category, sizeof(real_t), tets, f);
        fclose(f);
#endif

        // Free memory for micro_tets
        for (int i = 0; i < tets; i++)
        {
            free(micro_tets[i]);
        }
        free(micro_tets);
    }

    // Free memory for dofs
    free(dofs);
}

int **create_tetn_mesh(int nodes_in_macro_elem, int num_macro_elements)
{
    // [[0, 5], [1, 7], [4, 3], [6, 7]]
    int **in_list = (int **)malloc(nodes_in_macro_elem * sizeof(int *));
    for (int i = 0; i < nodes_in_macro_elem; i++)
    {
        in_list[i] = (int *)malloc(num_macro_elements * sizeof(int));
        in_list[i][0] = i;
    }
    return in_list;
}

int compute_nodes_number(int tetra_level)
{
    int nodes = 0;
    if (tetra_level % 2 == 0)
    {
        for (int i = 0; i < floor(tetra_level / 2); i++)
        {
            nodes += (tetra_level - i + 1) * (i + 1) * 2;
        }
        nodes += (tetra_level / 2 + 1) * (tetra_level / 2 + 1);
    }
    else 
    {
        for (int i = 0; i < floor(tetra_level / 2) + 1; i++)
        {
            nodes += (tetra_level - i + 1) * (i + 1) * 2;
        }
    }
    return nodes;
}

int compute_tets_number(int tetra_level)
{
    return (int) pow(tetra_level, 3);
}

int **create_macro_tet4_mesh(int tetra_level)
{
    int nodes_in_macro_elem = compute_nodes_number(tetra_level);
    return create_tetn_mesh(nodes_in_macro_elem, 1);
}

void check_laplacian_matrix(real_t *L, int n)
{
// Check if the matrix is square
// assert(rows == cols);

// Check that the row sum is zero
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        real_t row_sum = 0;
        for (int j = 0; j < n; j++)
        {
            row_sum += L[i * n + j];
        }
        assert(fabs(row_sum) < 1e-8);
    }

// Check that the matrix is symmetric
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            assert(fabs(L[i * n + j] - L[j * n + i]) < 1e-8);
        }
    }

// Check that the nonzero diagonal entries have opposite signs to their corresponding off-diagonal entries
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        real_t diag_entry = L[i * n + i];
        if (diag_entry != 0)
        {
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    real_t off_diag_entry = L[i * n + j];
                    if (diag_entry < 0)
                    {
                        assert(off_diag_entry >= 0);
                    }
                    else
                    {
                        assert(off_diag_entry <= 0);
                    }
                }
            }
        }
    }
}

void compute_Lapl(real_t *J, real_t *A)
{
    // real_t J_inv[9];
    // real_t J_inv_trans[9];

    // matrix_inverse(J, J_inv, 3);

    // // Transpose J_inv
    // for (int i = 0; i < 3; i++) {
    //     for (int j = 0; j < 3; j++) {
    //         J_inv_trans[i * 3 + j] = J_inv[j * 3 + i];
    //     }
    // }

    // real_t grad_ref_phi[4][3] = {
    //     {-1, -1, -1},
    //     {1, 0, 0},
    //     {0, 1, 0},
    //     {0, 0, 1}
    // };

    // real_t grad_phi[4][3];
    // for (int i = 0; i < 4; i++) {
    //     for (int j = 0; j < 3; j++) {
    //         grad_phi[i][j] = 0;
    //         for (int k = 0; k < 3; k++) {
    //             grad_phi[i][j] += J_inv_trans[j * 3 + k] * grad_ref_phi[i][k];
    //         }
    //     }
    // }

    // for (int i = 0; i < 4; i++) {
    //     for (int j = 0; j < 4; j++) {
    //         real_t dot_product = 0;
    //         for (int k = 0; k < 3; k++) {
    //             dot_product += grad_phi[i][k] * grad_phi[j][k];
    //         }
    //         A[i * 4 + j] = dot_product * determinant(J, 3) / 2.0;
    //     }
    // }

    check_laplacian_matrix(A, 4);
}

int generate_coords(int tetra_level, geom_t *x_coords, geom_t *y_coords, geom_t *z_coords)
{
    int node_index = 0;

    for (int i = 0; i <= tetra_level; i++)
    {
        for (int j = 0; j <= tetra_level - i; j++)
        {
            for (int k = 0; k <= tetra_level - i - j; k++)
            {
                x_coords[node_index] = (geom_t)i / tetra_level;
                y_coords[node_index] = (geom_t)j / tetra_level;
                z_coords[node_index] = (geom_t)k / tetra_level;
                // printf("%d %lf %lf %lf\n", node_index, x_coords[node_index], y_coords[node_index], z_coords[node_index]);
                node_index++;
            }
        }
    }

#ifdef GENERATE_VTK
    FILE *f = fopen("x.raw", "wb");
    fwrite(x_coords, sizeof(geom_t), node_index, f);
    fclose(f);

    f = fopen("y.raw", "wb");
    fwrite(y_coords, sizeof(geom_t), node_index, f);
    fclose(f);

    f = fopen("z.raw", "wb");
    fwrite(z_coords, sizeof(geom_t), node_index, f);
    fclose(f);
#endif

    return node_index;
}

real_t *apply_macro_kernel(int **in_list, int tetra_level, int nodes, int tets, geom_t *x_coords, geom_t *y_coords, geom_t *z_coords, real_t *vecX)
{
    real_t *vecY = (real_t *)malloc(nodes * sizeof(real_t *));
    memset(vecY, 0, nodes * sizeof(real_t *));
    assemble_macro_elem(in_list, tetra_level, nodes, tets, x_coords, y_coords, z_coords, vecX, vecY);
    return vecY;
}

real_t *apply_new_macro_kernel(real_t *macro_J, int tetra_level, int nodes, real_t *vecX)
{
    real_t *vecY = (real_t *)malloc(nodes * sizeof(real_t *));
    memset(vecY, 0, nodes * sizeof(real_t *));

    for (int category = 0; category < 6; category += 1) {
        macro_tet4_laplacian_apply(tetra_level, category, macro_J, vecX, vecY);
    }
    return vecY;
}

void residual(int **in_list, real_t *macro_J, int tetra_level, int nodes, int tets, geom_t *x_coords, geom_t *y_coords, geom_t *z_coords, int *dirichlet_nodes, int num_dirichlet_nodes, real_t *rhs, real_t *x, real_t *r)
{
    // Call apply_macro_kernel to compute Ax
    // real_t *Ax = apply_macro_kernel(in_list, tetra_level, nodes, tets, x_coords, y_coords, z_coords, x);
    real_t *Ax = apply_new_macro_kernel(macro_J, tetra_level, nodes, x);

    // Apply Dirichlet boundary conditions
    for (int i = 0; i < num_dirichlet_nodes; i++)
    {
        int dirichlet_node = dirichlet_nodes[i];
        Ax[dirichlet_node] = x[dirichlet_node];
        // printf("dirichlet_node: %d %lf\n", dirichlet_node, x[dirichlet_node]);
    }

    // Compute residual r = rhs - Ax.flatten()
    for (int i = 0; i < nodes; i++)
    {
        r[i] = rhs[i] - Ax[i];
    }

    // Free the memory allocated for Ax
    free(Ax);
}

void set_boundary_conditions(int num_nodes, real_t **rhs, real_t **x, int **dirichlet_nodes, int *num_dirichlet_nodes)
{
    // Set boundary conditions
    *rhs = (real_t *)malloc(num_nodes * sizeof(real_t));
    *x = (real_t *)malloc(num_nodes * sizeof(real_t));

    *num_dirichlet_nodes = 2;
    *dirichlet_nodes = (int *)malloc((*num_dirichlet_nodes) * sizeof(int));
    (*dirichlet_nodes)[0] = 0;
    (*dirichlet_nodes)[1] = num_nodes - 1;

    real_t dirichlet_values[] = {1, 0};

    for (int i = 0; i < *num_dirichlet_nodes; i++)
    {
        int idx = (*dirichlet_nodes)[i];
        (*rhs)[idx] = dirichlet_values[i];
        (*x)[idx] = dirichlet_values[i];
    }

#ifdef GENERATE_VTK
    FILE *f = fopen("dirichlet_nodes.raw", "wb");
    fwrite(*dirichlet_nodes, sizeof(int), *num_dirichlet_nodes, f);
    fclose(f);

    f = fopen("dirichlet_values.raw", "wb");
    fwrite(dirichlet_values, sizeof(real_t), *num_dirichlet_nodes, f);
    fclose(f);    
#endif
}

int main(void)
{
    int tetra_level = 4;

    // Compute the number of nodes
    int nodes = compute_nodes_number(tetra_level);
    int tets = compute_tets_number(tetra_level);

    printf("Generating %d micro-tetrahedrons\n", tets);

    // Allocate memory for coordinates
    geom_t *x_coords = (geom_t *)malloc(nodes * sizeof(geom_t));
    geom_t *y_coords = (geom_t *)malloc(nodes * sizeof(geom_t));
    geom_t *z_coords = (geom_t *)malloc(nodes * sizeof(geom_t));

    real_t macro_J[9];
    real_t p0[3] = {0, 0, 0};
    real_t p1[3] = {1, 0, 0};
    real_t p2[3] = {0, 1, 0};
    real_t p3[3] = {0, 0, 1};
    compute_A(p0, p1, p2, p3, macro_J);

    // Generate coordinates
    int num_coords = generate_coords(tetra_level, x_coords, y_coords, z_coords);

    // Create macro tetrahedral mesh
    int **in_list = create_macro_tet4_mesh(tetra_level);

    // Allocate variables for boundary conditions
    real_t *rhs;          // = (real_t *)malloc(nodes * sizeof(real_t));
    real_t *x;            // = (real_t *)malloc(nodes * sizeof(real_t));
    int *dirichlet_nodes; // = (int *)malloc(nodes * sizeof(int));
    int num_dirichlet_nodes;

    // Set boundary conditions
    set_boundary_conditions(nodes, &rhs, &x, &dirichlet_nodes, &num_dirichlet_nodes);

    printf("Number of coordinate triplets: %d, Number of nodes: %d\n", num_coords, nodes);

    // Check the length of nodes against the length of x_coords
    assert(nodes == num_coords);

    // Maximum number of iterations
    int max_iters = 2;
    real_t gamma = 8*1e-1;

    real_t *r = (real_t *)malloc(nodes * sizeof(real_t));

    for (int i = 0; i < max_iters; i++)
    {
        // Compute residual
        residual(in_list, macro_J, tetra_level, nodes, tets, x_coords, y_coords, z_coords, dirichlet_nodes, num_dirichlet_nodes, rhs, x, r);

        // Compute the norm of r
        real_t norm_r = 0.0;
        for (int j = 0; j < nodes; j++)
        {
            norm_r += r[j] * r[j];
        }
        norm_r = sqrt(norm_r);

        // Print the norm of r
        printf("Iteration %d, Residual norm: %lf\n", i, norm_r);

        // Update x
        for (int j = 0; j < nodes; j++)
        {
            x[j] += gamma * r[j];
        }

        // printf("nodes: %d coords: %d\n", nodes, num_coords);

#ifdef GENERATE_VTK
        // Write the result to construct the VTK file
        FILE *f = fopen("solution.raw", "wb");
        fwrite(x, sizeof(real_t), nodes, f);
        fclose(f);

        // Change directory
        chdir("/Users/bolema/Documents/sfem/");
        const char *command = "source venv/bin/activate && cd python/sfem/mesh/ && "
        "python3 raw_to_db.py /Users/bolema/Documents/hpcfem/a64fx /Users/bolema/Documents/hpcfem/a64fx/test.vtk " 
        "-c /Users/bolema/Documents/hpcfem/a64fx/category.raw "
        "-p /Users/bolema/Documents/hpcfem/a64fx/solution.raw";

        // Execute the command
        int ret = system(command);
        if (ret == -1) {
            perror("system() call failed");
        }
#endif

        // Check for convergence
        if (norm_r < 1e-8)
        {
            printf("Converged after %d iterations\nSolution:", i + 1);
            for (int k = 0; k < nodes; k++)
            {
                printf("%lf \n", x[k]);
            }
            printf("\n");

            // Write the result to construct the VTK file
            FILE *f = fopen("solution.raw", "wb");
            fwrite(x, sizeof(real_t), nodes, f);
            fclose(f);

            // Change directory
            chdir("/Users/bolema/Documents/sfem/");
            const char *command = "source venv/bin/activate && cd python/sfem/mesh/ && "
            "python3 raw_to_db.py /Users/bolema/Documents/hpcfem/a64fx /Users/bolema/Documents/hpcfem/a64fx/test.vtk " 
            "-c /Users/bolema/Documents/hpcfem/a64fx/category.raw "
            "-p /Users/bolema/Documents/hpcfem/a64fx/solution.raw";

            // Execute the command
            int ret = system(command);
            if (ret == -1) {
                perror("system() call failed");
            }

            free(r);
            break;
        }
    }

    // Free allocated memory
    free(x_coords);
    free(y_coords);
    free(z_coords);
    for (int i = 0; i < nodes; i++)
    {
        free(in_list[i]);
    }
    free(in_list);
    free(rhs);
    free(x);
    free(dirichlet_nodes);

    return 0;
}
