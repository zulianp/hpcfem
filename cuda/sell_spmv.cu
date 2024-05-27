#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int* load_int32_array(const char *filename, int64_t *elements_read) {
    FILE *file;
    long file_size;
    int *buffer;
    size_t num_elements;

    // Open the file for reading in binary mode
    file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    // Seek to the end of the file to get its size
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    rewind(file);

    // Calculate the number of elements based on the file size
    num_elements = file_size / sizeof(int);

    // Allocate managed memory for the buffer
    cudaError_t err = cudaMallocManaged((void**)&buffer, file_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating managed memory: %s\n", cudaGetErrorString(err));
        fclose(file);
        return NULL;
    }

    // Read the entire content of the file into the buffer
    *elements_read = fread(buffer, sizeof(int), num_elements, file);
    if (*elements_read != num_elements) {
        perror("Error reading file");
        cudaFree(buffer);
        fclose(file);
        return NULL;
    }

    // Close the file
    fclose(file);

    // Report the number of elements read
    printf("Number of elements read from %s: %d\n", filename, *elements_read);

    return buffer;
}

double* load_float64_array(const char *filename, int64_t *elements_read) {
    FILE *file;
    long file_size;
    double *buffer;
    size_t num_elements;

    // Open the file for reading in binary mode
    file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    // Seek to the end of the file to get its size
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    rewind(file);

    // Calculate the number of elements based on the file size
    num_elements = file_size / sizeof(double);

    // Allocate managed memory for the buffer
    cudaError_t err = cudaMallocManaged((void**)&buffer, file_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating managed memory: %s\n", cudaGetErrorString(err));
        fclose(file);
        return NULL;
    }

    // Read the entire content of the file into the buffer
    *elements_read = fread(buffer, sizeof(double), num_elements, file);
    if (*elements_read != num_elements) {
        perror("Error reading file");
        cudaFree(buffer);
        fclose(file);
        return NULL;
    }

    // Close the file
    fclose(file);

    // Report the number of elements read
    printf("Number of elements read from %s: %d\n", filename, *elements_read);

    return buffer;
}

int main(void) {
    // Host problem definition
    int A_num_rows      = 1;
    int A_num_cols      = 1;
    int A_nnz           = 1;
    float     alpha           = 1.0f;
    float     beta            = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    // int   *dA_csrOffsets, *dA_columns;
    double *dA_values, *dX, *dY;

    int64_t sellValuesSize = 0;
    int64_t elements_read;

    int *sellSliceOffsets = load_int32_array("sell_slice_offsets.i32", &elements_read);
    double *sellValues = load_float64_array("sell_values.f64", &sellValuesSize);
    int *sellColInd = load_int32_array("sell_column_indices.i32", &elements_read);
    int *sellMetaInfo = load_int32_array("sell_meta.i32", &elements_read);

    int sliceSize = 2;

    A_num_rows = sellMetaInfo[0];
    A_num_cols = sellMetaInfo[0];
    A_nnz = sellMetaInfo[1];
    // sliceSize = sellMetaInfo[2];

    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in SELL format

    CHECK_CUSPARSE( cusparseCreateSlicedEll(&matA, A_num_rows, A_num_cols, A_nnz,
                            sellValuesSize, sliceSize, sellSliceOffsets, sellColInd, sellValues,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )

    CHECK_CUDA( cudaMalloc((void**) &dX, A_num_cols * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &dY, A_num_rows * sizeof(double)) )

    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_64F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_64F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    cudaEventRecord(start);

    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )

    cudaEventRecord(stop);

    // Wait for the event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    double avg_time = milliseconds/1000;
    double avg_throughput = (A_num_rows / avg_time) * 1e-6;

    printf("Time for matrix-vector multiplication: %f milliseconds\n", milliseconds);
    printf("Throughput %g (MDOF/s)\n");

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check

    // CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(float),
    //                        cudaMemcpyDeviceToHost) )
    // int correct = 1;
    // for (int i = 0; i < A_num_rows; i++) {
    //     if (hY[i] != hY_result[i]) { // direct floating point comparison is not
    //         correct = 0;             // reliable
    //         break;
    //     }
    // }
    // if (correct)
    //     printf("spmv_csr_example test PASSED\n");
    // else
    //     printf("spmv_csr_example test FAILED: wrong result\n");

    //--------------------------------------------------------------------------
    // device memory deallocation
    // CHECK_CUDA( cudaFree(dBuffer) )
    // CHECK_CUDA( cudaFree(dA_csrOffsets) )
    // CHECK_CUDA( cudaFree(dA_columns) )
    // CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )
    return EXIT_SUCCESS;
}

