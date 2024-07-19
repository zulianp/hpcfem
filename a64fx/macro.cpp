#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
// #include <omp.h>

#define MAX_NODES 100000

typedef double real_t;
typedef float geom_t;

void matrix_inverse(real_t *A, real_t *invA, int n);
void print_matrix(real_t *matrix, int n);

#define POW2(x) ((x) * (x))

void print_matrix(real_t *matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%f ", matrix[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void print_coords(geom_t *matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%f ", matrix[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}

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

  print_matrix(J, 3, 3);

  // assert(det3(J) > 0);
#endif

  real_t fff[6];
  tet4_fff(x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3, fff);
  printf("%g\n", tet4_det_fff(fff));
  assert(tet4_det_fff(fff) > 0);
  tet4_laplacian_hessian_fff(fff, element_matrix);
}

real_t determinant(real_t *A, int n) {
  int i, j, k;
  real_t det = 1.0;
  for (i = 0; i < n; i++) {
    for (j = i + 1; j < n; j++) {
      real_t ratio = A[j * n + i] / A[i * n + i];
      for (k = i; k < n; k++) {
        A[j * n + k] -= ratio * A[i * n + k];
      }
    }
    det *= A[i * n + i];
  }
  return det;
}

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

void matrix_inverse(real_t *A, real_t *invA, int n) {
  int i, j, k;
  real_t ratio;

  // Create an augmented matrix [A|I]
  real_t augmented[n][2 * n];
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      augmented[i][j] = A[i * n + j];
      augmented[i][j + n] = (i == j) ? 1.0 : 0.0;
    }
  }

  // Apply Gauss-Jordan elimination
  for (i = 0; i < n; i++) {
    // Make the diagonal element 1
    ratio = augmented[i][i];
    if (ratio == 0) {
      printf("Matrix is singular and cannot be inverted.\n");
      return;
    }
    for (j = 0; j < 2 * n; j++) {
      augmented[i][j] /= ratio;
    }

    // Make the other elements in the current column 0
    for (k = 0; k < n; k++) {
      if (k != i) {
        ratio = augmented[k][i];
        for (j = 0; j < 2 * n; j++) {
          augmented[k][j] -= ratio * augmented[i][j];
        }
      }
    }
  }

  // Extract the inverse matrix from the augmented matrix
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      invA[i * n + j] = augmented[i][j + n];
    }
  }
}

real_t tetrahedron_volume(real_t *A, real_t *B, real_t *C, real_t *D) {
  real_t mat[9] = {B[0] - A[0], C[0] - A[0], D[0] - A[0],
                   B[1] - A[1], C[1] - A[1], D[1] - A[1],
                   B[2] - A[2], C[2] - A[2], D[2] - A[2]};
  real_t det = determinant(mat, 3);
  return det / 6.0;
}

void compute_A(real_t *p0, real_t *p1, real_t *p2, real_t *p3, real_t *A) {
  for (int i = 0; i < 3; i++) {
    A[i] = p1[i] - p0[i];
    A[3 + i] = p2[i] - p0[i];
    A[6 + i] = p3[i] - p0[i];
  }
  assert(determinant(A, 3) > 0);
}

void gather_and_scatter(int **micro_tets, int num_micro_tets, geom_t *x_coords,
                        geom_t *y_coords, geom_t *z_coords, real_t *vecX,
                        real_t *vecY) {
  // pick a random micro tetrahedron
  int *e0 = micro_tets[num_micro_tets - 1];
  // real_t p0[3] = {x_coords[e0[0]], y_coords[e0[0]], z_coords[e0[0]]};
  // real_t p1[3] = {x_coords[e0[1]], y_coords[e0[1]], z_coords[e0[1]]};
  // real_t p2[3] = {x_coords[e0[2]], y_coords[e0[2]], z_coords[e0[2]]};
  // real_t p3[3] = {x_coords[e0[3]], y_coords[e0[3]], z_coords[e0[3]]};

  assert(e0[0] >= 0);
  assert(e0[1] >= 0);
  assert(e0[2] >= 0);
  assert(e0[3] >= 0);

  real_t local_M[16];
  tet4_laplacian_hessian(
      local_M, x_coords[e0[0]], x_coords[e0[1]], x_coords[e0[2]],
      x_coords[e0[3]],                                                    //
      y_coords[e0[0]], y_coords[e0[1]], y_coords[e0[2]], y_coords[e0[3]], //
      z_coords[e0[0]], z_coords[e0[1]], z_coords[e0[2]], z_coords[e0[3]]);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      for (int t = 0; t < num_micro_tets; t++) {
        // assert(idx < len(vecX));
        vecY[micro_tets[t][j]] += local_M[i * 4 + j] * vecX[micro_tets[t][i]];
      }
    }
  }
}

// GOAL (later)
// void assemble_macro_elem_subparametric(
//     const eid,
//     const ptrdiff_t fff_stride,
//     const geom_t *const fff,
//     const real_t *const vecX, real_t *const vecY)
// {
//     fff[0 * fff_stride];
//     fff[1 * fff_stride];
//     fff[2 * fff_stride];
//     fff[3 * fff_stride];
//     fff[4 * fff_stride];
//     fff[5 * fff_stride];

// }

// GOAL
void static inline assemble_macro_elem_subparametric(
    const ptrdiff_t eid,
    const ptrdiff_t jacobian_stride,
    const geom_t *const jacobian,
    const ptrdiff_t vec_stride,
    const real_t *const vecX, 
    real_t *const vecY)
{
    // how you Access the jacobian
    // jacobian[0 * jacobian_stride];
    // jacobian[1 * jacobian_stride];
    // jacobian[2 * jacobian_stride];
    // jacobian[3 * jacobian_stride];
    // jacobian[4 * jacobian_stride];
    // jacobian[5 * jacobian_stride];
    // jacobian[6 * jacobian_stride];
    // jacobian[7 * jacobian_stride];
    // jacobian[8 * jacobian_stride];

    // loop on all sub elements p0 = [0, 0, 0], p1 = J[:,0], ....

    // TODO:
    // 1) Identify indexing per category
    // 2) From jacobian create sub-jacobian for category
    //    -> J_category = J * J_ref_category

    // vecX[maco_element_node_idx * vec_stride] 
    // vecY[maco_element_node_idx * vec_stride] 

}

// nxe nodes_per_element
void assemble_macro_elem(int **micro_elems, int tetra_level, int nodes,
                         geom_t *x_coords, geom_t *y_coords, geom_t *z_coords,
                         real_t *vecX, real_t *vecY) {
  int level = tetra_level + 1;
  int n_macro_elems = 1;
  int *dofs = (int *)malloc(nodes * sizeof(int));

  int *i0 = (int *)malloc(nodes * sizeof(int));
  int *i1 = (int *)malloc(nodes * sizeof(int));
  int *i2 = (int *)malloc(nodes * sizeof(int));
  int *i3 = (int *)malloc(nodes * sizeof(int));
  int *category = (int *)malloc(nodes * sizeof(int));

  // Initialize vecY to zero
  for (int i = 0; i < nodes; i++) {
    vecY[i] = 0;
  }

  for (int elem = 0; elem < n_macro_elems; elem++) {
    for (int i = 0; i < nodes; i++) {
      dofs[i] = micro_elems[i][elem];
      assert(micro_elems[i][elem] >= 0);
    }

    int num_tets = (level * (level + 1) * (level + 2)) / 6;
    int num_coords = num_tets * 4;
    

    int **micro_tets = (int **)malloc(num_coords * sizeof(int *));
    for (int i = 0; i < num_coords; i++) {
      micro_tets[i] = (int *)calloc(4,  sizeof(int));
    }

    int p = 0;
    int micro_tets_iter = 0;
    for (int i = 0; i < level - 1; i++) {
      int layer_items = (level - i + 1) * (level - i) / 2;
      for (int j = 0; j < level - i - 1; j++) {
        for (int k = 0; k < level - i - j - 1; k++) {
          int e0 = p;
          int e1 = p + 1;
          int e2 = p + level - i - j;
          int e3 = p + layer_items - j;

          printf("%d %d %d %d\n", e0, e1, e2, e3);
          printf("%d %d %d %d\n", dofs[e0], dofs[e1], dofs[e2], dofs[e3]);
          fflush(stdout);

          assert(e0 >= 0);
          assert(e1 >= 0);
          assert(e2 >= 0);
          assert(e3 >= 0);

          assert(dofs[e0] >= 0);
          assert(dofs[e1] >= 0);
          assert(dofs[e2] >= 0);
          assert(dofs[e3] >= 0);

          assert(dofs[e0] == e0);
          assert(dofs[e1] == e1);
          assert(dofs[e2] == e2);
          assert(dofs[e3] == e3);

          micro_tets[micro_tets_iter][0] = dofs[e0];
          micro_tets[micro_tets_iter][1] = dofs[e1];
          micro_tets[micro_tets_iter][2] = dofs[e2];
          micro_tets[micro_tets_iter][3] = dofs[e3];

          assert(e0 >= 0);
          assert(e1 >= 0);
          assert(e2 >= 0);
          assert(e3 >= 0);

          assert(dofs[e0] >= 0);
          assert(dofs[e1] >= 0);
          assert(dofs[e2] >= 0);
          assert(dofs[e3] >= 0);

          assert(dofs[e0] == e0);
          assert(dofs[e1] == e1);
          assert(dofs[e2] == e2);
          assert(dofs[e3] == e3);

          i0[micro_tets_iter] = e0;
          i1[micro_tets_iter] = e1;
          i2[micro_tets_iter] = e2;
          i3[micro_tets_iter] = e3;
          micro_tets_iter += 1;
          p++;
        }
        p++;
      }
      p++;
    }

    gather_and_scatter(micro_tets, micro_tets_iter, x_coords, y_coords,
                       z_coords, vecX, vecY);
    // // pick a random micro tetrahedron
    // int *e0 = micro_tets[num_tets - 1];
    // real_t p0[3] = {x_coords[e0[0]], y_coords[e0[0]], z_coords[e0[0]]};
    // real_t p1[3] = {x_coords[e0[1]], y_coords[e0[1]], z_coords[e0[1]]};
    // real_t p2[3] = {x_coords[e0[2]], y_coords[e0[2]], z_coords[e0[2]]};
    // real_t p3[3] = {x_coords[e0[3]], y_coords[e0[3]], z_coords[e0[3]]};

    // real_t local_M[16];
    // tet4_laplacian_hessian(local_M,
    //     x_coords[e0[0]], x_coords[e0[1]], x_coords[e0[2]], x_coords[e0[3]],
    //     y_coords[e0[0]], y_coords[e0[1]], y_coords[e0[2]], y_coords[e0[3]],
    //     z_coords[e0[0]], z_coords[e0[1]], z_coords[e0[2]], z_coords[e0[3]]);

    // for (int i = 0; i < 4; i++) {
    //     for (int j = 0; j < 4; j++) {
    //         for (int t = 0; t < num_tets; t++) {
    //             vecY[micro_tets[t][j]] += local_M[i][j] *
    //             vecX[micro_tets[t][i]];
    //         }
    //     }
    // }

    // Repeat the process for the remaining subtetrahedrons
    // Second case
    p = 0;
    for (int i = 0; i < micro_tets_iter; i++) {
      // cleanup
      for (int j = 0; j < 4; j++) {
        micro_tets[i][j] = 0;
      }
    }
    micro_tets_iter = 0;
    
    // Harcode ref_jacobian of category 
    // ( G(p) = G2(G1(p)), chain rule. Special case Affine transform: A2 * (A1 * p + b1 ) + b2 = A2*A1*p + c)
    // J_c = J * J_ref_c 
    // assemble A

    // tet4_laplacian_hessian(element_matrix, 0,
    //                            J_c[0*3+0],  J_c[0*3+1], x3,
    //                            0, J_c[1*3+0], y2,
    //                            y3, 0, J_c[2*3+0],
    //                            z2, z3);

    for (int i = 0; i < level - 1; i++) {
      int layer_items = (level - i) * (level - i - 1) / 2;
      for (int j = 0; j < level - i - 1; j++) {
        p++;
        for (int k = 1; k < level - i - j - 1; k++) {
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

          // TODOS
          // Gather (CPU stride = nxe, element_stride = 1)
          // Gather (GPU stride = 1, element_stride = n_macro_elements)
          // real_t x0 = vecX[threadId * stride + e0 * element_stride];
          // real_t x1 = vecX[threadId * stride + e1 * element_stride];
          // real_t x2 = vecX[threadId * stride + e2 * element_stride];
          // real_t x3 = vecX[threadId * stride + e3 * element_stride];

          // const real_t y0 = A00 * x0 + A01 * x1 + A02 * x2 + A00 * x3
          // const real_t y1 = A10 * x0 + A11 * x1 + A12 * x2 + A10 * x3
          // const real_t y2 = A20 * x0 + A21 * x1 + A22 * x2 + A20 * x3
          // const real_t y3 = A30 * x0 + A31 * x1 + A32 * x2 + A30 * x3

          // Scatter
          // vecY[threadId * stride + e0 * element_stride] += y0 
          // vecY[threadId * stride + e1 * element_stride] += y1 
          // vecY[threadId * stride + e2 * element_stride] += y2 
          // vecY[threadId * stride + e3 * element_stride] += y3 

          // REMOVE ME This stuff we do not need
          micro_tets[micro_tets_iter][0] = dofs[e0];
          micro_tets[micro_tets_iter][1] = dofs[e1];
          micro_tets[micro_tets_iter][2] = dofs[e2];
          micro_tets[micro_tets_iter][3] = dofs[e3];
          i0[micro_tets_iter] = e0;
          i1[micro_tets_iter] = e1;
          i2[micro_tets_iter] = e2;
          i3[micro_tets_iter] = e3;


          micro_tets_iter += 1;

          p++;
        }
        p++;
      }
      p++;
    }
    gather_and_scatter(micro_tets, micro_tets_iter, x_coords, y_coords,
                       z_coords, vecX, vecY);

    // Third case
    p = 0;
    for (int i = 0; i < micro_tets_iter; i++) {
      // cleanup
      for (int j = 0; j < 4; j++) {
        micro_tets[i][j] = 0;
      }
    }
    micro_tets_iter = 0;

    for (int i = 0; i < level - 1; i++) {
      int layer_items = (level - i) * (level - i - 1) / 2;
      for (int j = 0; j < level - i - 1; j++) {
        p++;
        for (int k = 1; k < level - i - j - 1; k++) {
          /*
          e0 = p
          e1 = p+level-i-j
          e3 = p+level-i-j+layer_items
          e2 = p+level-i-j-1+layer_items+level-i-j-1
          */
          int e0 = p;
          int e1 = p + level - i - j;
          int e2 = p + layer_items + level - i - j;
          int e3 = p + layer_items + level - i - j - 1 + level - i - j - 1;
          micro_tets[micro_tets_iter][0] = dofs[e0];
          micro_tets[micro_tets_iter][1] = dofs[e1];
          micro_tets[micro_tets_iter][2] = dofs[e2];
          micro_tets[micro_tets_iter][3] = dofs[e3];
          i0[micro_tets_iter] = e0;
          i1[micro_tets_iter] = e1;
          i2[micro_tets_iter] = e2;
          i3[micro_tets_iter] = e3;
          micro_tets_iter += 1;

          p++;
        }
        p++;
      }
      p++;
    }
    gather_and_scatter(micro_tets, micro_tets_iter, x_coords, y_coords,
                       z_coords, vecX, vecY);

    // Fourth case
    p = 0;
    for (int i = 0; i < micro_tets_iter; i++) {
      // cleanup
      for (int j = 0; j < 4; j++) {
        micro_tets[i][j] = 0;
      }
    }
    micro_tets_iter = 0;

    for (int i = 0; i < level - 1; i++) {
      int layer_items = (level - i) * (level - i - 1) / 2;
      for (int j = 0; j < level - i - 1; j++) {
        p++;
        for (int k = 1; k < level - i - j - 1; k++) {
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
          micro_tets[micro_tets_iter][0] = dofs[e0];
          micro_tets[micro_tets_iter][1] = dofs[e1];
          micro_tets[micro_tets_iter][2] = dofs[e2];
          micro_tets[micro_tets_iter][3] = dofs[e3];
          i0[micro_tets_iter] = e0;
          i1[micro_tets_iter] = e1;
          i2[micro_tets_iter] = e2;
          i3[micro_tets_iter] = e3;
          micro_tets_iter += 1;

          p++;
        }
        p++;
      }
      p++;
    }
    gather_and_scatter(micro_tets, micro_tets_iter, x_coords, y_coords,
                       z_coords, vecX, vecY);

    // Fifth case
    p = 0;
    for (int i = 0; i < micro_tets_iter; i++) {
      // cleanup
      for (int j = 0; j < 4; j++) {
        micro_tets[i][j] = 0;
      }
    }
    micro_tets_iter = 0;

    for (int i = 1; i < level - 1; i++) {
      p = p + level - i + 1;
      for (int j = 0; j < level - i - 1; j++) {
        p++;
        for (int k = 1; k < level - i - j - 1; k++) {
          int layer_items = (level - i) * (level - i - 1) / 2;
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
          micro_tets[micro_tets_iter][0] = dofs[e0];
          micro_tets[micro_tets_iter][1] = dofs[e1];
          micro_tets[micro_tets_iter][2] = dofs[e2];
          micro_tets[micro_tets_iter][3] = dofs[e3];
          i0[micro_tets_iter] = e0;
          i1[micro_tets_iter] = e1;
          i2[micro_tets_iter] = e2;
          i3[micro_tets_iter] = e3;
          micro_tets_iter += 1;

          p++;
        }
        p++;
      }
      p++;
    }
    gather_and_scatter(micro_tets, micro_tets_iter, x_coords, y_coords,
                       z_coords, vecX, vecY);

    // Sixth case
    p = 0;
    for (int i = 0; i < micro_tets_iter; i++) {
      // cleanup
      for (int j = 0; j < 4; j++) {
        micro_tets[i][j] = 0;
      }
    }
    micro_tets_iter = 0;

    for (int i = 0; i < level - 1; i++) {
      int layer_items = (level - i) * (level - i - 1) / 2;
      for (int j = 0; j < level - i - 1; j++) {
        p++;
        for (int k = 1; k < level - i - j - 1; k++) {
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

          micro_tets[micro_tets_iter][0] = dofs[e0];
          micro_tets[micro_tets_iter][1] = dofs[e1];
          micro_tets[micro_tets_iter][2] = dofs[e2];
          micro_tets[micro_tets_iter][3] = dofs[e3];
          i0[micro_tets_iter] = e0;
          i1[micro_tets_iter] = e1;
          i2[micro_tets_iter] = e2;
          i3[micro_tets_iter] = e3;
          micro_tets_iter += 1;

          p++;
        }
        p++;
      }
      p++;
    }
    gather_and_scatter(micro_tets, micro_tets_iter, x_coords, y_coords,
                       z_coords, vecX, vecY);

    FILE *f = fopen("i0.raw", "wb");
    fwrite(i0, sizeof(int32_t), nodes, f);
    fclose(f);

    f = fopen("i1.raw", "wb");
    fwrite(i1, sizeof(int32_t), nodes, f);
    fclose(f);

    f = fopen("i2.raw", "wb");
    fwrite(i2, sizeof(int32_t), nodes, f);
    fclose(f);

    f = fopen("i3.raw", "wb");
    fwrite(i3, sizeof(int32_t), nodes, f);
    fclose(f);

    f = fopen("category.raw", "wb");
    fwrite(category, sizeof(int32_t), nodes, f);
    fclose(f);

    // Free memory for micro_tets
    for (int i = 0; i < num_coords; i++) {
      free(micro_tets[i]);
    }
    free(micro_tets);
  }

  // Free memory for dofs
  free(dofs);
}

int **create_tetn_mesh(int nodes_in_macro_elem, int num_macro_elements) {
  // [[0, 5], [1, 7], [4, 3], [6, 7]]
  int **in_list = (int **)malloc(nodes_in_macro_elem * sizeof(int *));
  for (int i = 0; i < nodes_in_macro_elem; i++) {
    in_list[i] = (int *)malloc(num_macro_elements * sizeof(int));
  }
  assert(0); // FIXME fill the mesh
  return in_list;
}

int compute_nodes_number(int tetra_level) {
  int nodes = 0;
  if (tetra_level % 2 == 0) {
    for (int i = 0; i < tetra_level / 2; i++) {
      nodes += (tetra_level - i + 1) * (i + 1) * 2;
    }
    nodes += (tetra_level / 2 + 1) * (tetra_level / 2 + 1);
  } else {
    for (int i = 0; i < tetra_level / 2; i++) {
      nodes += (tetra_level - i + 1) * (i + 1) * 2;
    }
  }
  return nodes;
}

int **create_macro_tet4_mesh(int tetra_level) {
  int nodes_in_macro_elem = compute_nodes_number(tetra_level);
  return create_tetn_mesh(nodes_in_macro_elem, 1);
}

void check_laplacian_matrix(real_t *L, int n) {
// Check if the matrix is square
// assert(rows == cols);

// Check that the row sum is zero
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    real_t row_sum = 0;
    for (int j = 0; j < n; j++) {
      row_sum += L[i * n + j];
    }
    assert(fabs(row_sum) < 1e-8);
  }

// Check that the matrix is symmetric
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      assert(fabs(L[i * n + j] - L[j * n + i]) < 1e-8);
    }
  }

// Check that the nonzero diagonal entries have opposite signs to their
// corresponding off-diagonal entries
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    real_t diag_entry = L[i * n + i];
    if (diag_entry != 0) {
      for (int j = 0; j < n; j++) {
        if (i != j) {
          real_t off_diag_entry = L[i * n + j];
          if (diag_entry < 0) {
            assert(off_diag_entry >= 0);
          } else {
            assert(off_diag_entry <= 0);
          }
        }
      }
    }
  }
}

void compute_Lapl(real_t *J, real_t *A) {
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

int generate_coords(int tetra_level, geom_t *x_coords, geom_t *y_coords,
                    geom_t *z_coords) {
  int node_index = 0;

  for (int i = 0; i <= tetra_level; i++) {
    for (int j = 0; j <= tetra_level - i; j++) {
      for (int k = 0; k <= tetra_level - i - j; k++) {
        x_coords[node_index] = (geom_t)i / tetra_level;
        y_coords[node_index] = (geom_t)j / tetra_level;
        z_coords[node_index] = (geom_t)k / tetra_level;
        node_index++;
      }
    }
  }

  FILE *f = fopen("x.raw", "wb");
  fwrite(x_coords, sizeof(geom_t), node_index, f);
  fclose(f);

  print_coords(x_coords, node_index, 1);

  f = fopen("y.raw", "wb");
  fwrite(y_coords, sizeof(geom_t), node_index, f);
  fclose(f);

  f = fopen("z.raw", "wb");
  fwrite(z_coords, sizeof(geom_t), node_index, f);
  fclose(f);

  return node_index;
}

real_t *apply_macro_kernel(int **in_list, int tetra_level, int nodes,
                           geom_t *x_coords, geom_t *y_coords, geom_t *z_coords,
                           real_t *vecX) {
  real_t *vecY = (real_t *)malloc(nodes * sizeof(real_t *));
  assemble_macro_elem(in_list, tetra_level, nodes, x_coords, y_coords, z_coords,
                      vecX, vecY);
  return vecY;
}

void residual(int **in_list, int tetra_level, int nodes, geom_t *x_coords,
              geom_t *y_coords, geom_t *z_coords, int *dirichlet_nodes,
              int num_dirichlet_nodes, real_t *rhs, real_t *x, real_t *r) {
  // Call apply_macro_kernel to compute Ax
  real_t *Ax = apply_macro_kernel(in_list, tetra_level, nodes, x_coords,
                                  y_coords, z_coords, x);

  // Apply Dirichlet boundary conditions
  for (int i = 0; i < num_dirichlet_nodes; i++) {
    int dirichlet_node = dirichlet_nodes[i];
    Ax[dirichlet_node] = x[dirichlet_node];
  }

  // Compute residual r = rhs - Ax.flatten()
  for (int i = 0; i < nodes; i++) {
    r[i] = rhs[i] - Ax[i];
  }

  // Free the memory allocated for Ax
  free(Ax);
}

void set_boundary_conditions(int num_nodes, real_t **rhs, real_t **x,
                             int **dirichlet_nodes, int *num_dirichlet_nodes) {
  // Set boundary conditions
  *rhs = (real_t *)malloc(num_nodes * sizeof(real_t));
  *x = (real_t *)malloc(num_nodes * sizeof(real_t));

  *num_dirichlet_nodes = 2; // Example number
  *dirichlet_nodes = (int *)malloc(*num_dirichlet_nodes * sizeof(int));
  (*dirichlet_nodes)[0] = 0;
  (*dirichlet_nodes)[1] = num_nodes - 1;

  real_t dirichlet_values[] = {1, 0};

  for (int i = 0; i < *num_dirichlet_nodes; i++) {
    int idx = (*dirichlet_nodes)[i];
    (*rhs)[idx] = dirichlet_values[i];
    (*x)[idx] = dirichlet_values[i];
  }
}


// cc -g -Wall  -fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g  macro.cpp && ./a.out
// cc -g -Wall  -fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g  macro.cpp && lldb -- ./a.out
// -DNDEBUG for removing assertion checking (high-performance/production runs)
int main(void) {
  int tetra_level = 4;

  // Compute the number of nodes
  int nodes = compute_nodes_number(tetra_level);

  // Allocate memory for coordinates
  geom_t *x_coords = (geom_t *)malloc(nodes * sizeof(geom_t));
  geom_t *y_coords = (geom_t *)malloc(nodes * sizeof(geom_t));
  geom_t *z_coords = (geom_t *)malloc(nodes * sizeof(geom_t));

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
  set_boundary_conditions(nodes, &rhs, &x, &dirichlet_nodes,
                          &num_dirichlet_nodes);
  printf("Number of coordinates: %d, Number of nodes: %d\n", num_coords, nodes);

  // Check the length of x_coords and x
  if (nodes != num_coords) {
    printf("Error: The number of coordinates does not match the number of "
           "nodes.\n");
    return 1;
  }

  // Maximum number of iterations
  int max_iters = 20000;
  real_t gamma = 1e-1;

  real_t *r = (real_t *)malloc(nodes * sizeof(real_t));

  for (int i = 0; i < max_iters; i++) {
    // Compute residual
    residual(in_list, tetra_level, nodes, x_coords, y_coords, z_coords,
             dirichlet_nodes, num_dirichlet_nodes, rhs, x, r);

    // Compute the norm of r
    real_t norm_r = 0.0;
    for (int j = 0; j < nodes; j++) {
      norm_r += r[j] * r[j];
    }
    norm_r = sqrt(norm_r);

    // Print the norm of r
    printf("Iteration %d, Residual norm: %lf\n", i, norm_r);

    // Update x
    for (int j = 0; j < nodes; j++) {
      x[j] += gamma * r[j];
    }

    // Check for convergence
    if (norm_r < 1e-8) {
      printf("Converged after %d iterations\n", i + 1);
      free(r);
      break;
    }
  }

  // Free allocated memory
  free(x_coords);
  free(y_coords);
  free(z_coords);
  for (int i = 0; i < nodes; i++) {
    free(in_list[i]);
  }
  free(in_list);
  free(rhs);
  free(x);
  free(dirichlet_nodes);

  return 0;
}
