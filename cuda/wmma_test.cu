#include <iostream>
#include <cuda.h>
#include <mma.h>

using namespace nvcuda;

__global__ void wmma_ker(double *a, double *b, double *c) {
   // For matrix_a the tile takes dimension m x k; for matrix_b the dimension is k x n, and accumulator tiles are m x n.
   // Matrix Size (m-n-k)
   // 8x8x4
   // matrix_a 8x4
   // matrix_b 4x8
   // matrix_c 8x8

   __shared__ double x[32];
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0);

    if (threadIdx.x == 0) {
        for (int i = 0; i < 16; i += 1) {
            x[i] = i + 1;
        }
        for (int i = 16; i < 32; i += 1) {
            x[i] = 0;
        }
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 4; j++) {
                printf("%lf ", x[i * 4 + j]);
            }
            printf("\n");
        }
    }
    __syncwarp();

   wmma::load_matrix_sync(a_frag, x, 4);

   // Load the inputs
   //wmma::load_matrix_sync(a_frag, a, 4);
    for (int i = 0; i < 4; i += 1) {
    wmma::load_matrix_sync(b_frag, b, 8);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(c, c_frag, 8, wmma::mem_row_major);
    }
}

int main(void) {
    // Host matrices
    double C[8*8] = {0}; // Output matrix
    double B[8*4] = {0};
    double A[8*4] = {1};

    double J[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    // // Fill the 8x8 matrix C with blocks of A
    // for (int i = 0; i < 4; i++) {
    //     for (int j = 0; j < 4; j++) {
    //         // Top-left block
    //         A[i * 8 + j] = J[i * 4 + j];
    //         // Bottom-right block
    //         A[(i + 4) * 8 + (j + 4)] = J[i * 4 + j];
    //     }
    // }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            // Top-left block
            A[i * 4 + j] = J[i * 4 + j];
        }
    }

    // Fill the 8x8 matrix C with blocks of A
    // for (int i = 0; i < 8; i++) {
    //     A[i * 8 + i] = 0;
    // }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            // Top-left block
            B[i * 8 + j] = j;
        }
    }

    // for (int i = 0; i < 8; i += 4) {
    //     for (int j = 0; j < 4; j++) {
    //         int k = 1 + (1+j) * (1+i/4);
    //         for (int row = 0; row < 4; row += 1) {
    //             B[(i + row) * 4 + j] = k;
    //         }
    //     }
    // }

    // Device matrices
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, 8*4 * sizeof(double));
    cudaMalloc(&d_B, 8*4 * sizeof(double));
    cudaMalloc(&d_C, 8*8 * sizeof(double));

    // Copy data to the device
    cudaMemcpy(d_A, A, 8*4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, 4*8 * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockSize(1);
    dim3 threadsInBlock(32);
    wmma_ker<<<blockSize, threadsInBlock, 0>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, 8*8 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << A[i * 4 + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Matrix B:" << std::endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            std::cout << B[i * 8 + j] << " ";
        }
        std::cout << std::endl;
    }

    // Print the result
    std::cout << "Result Matrix:" << std::endl;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            std::cout << C[i * 8 + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}