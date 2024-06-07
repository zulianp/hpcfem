#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <assert.h>

// nvcc csr.cu -o csr -lcusparse -std=c++14 -arch=sm_60 -Xptxas=-O3,-v -use_fast_math

inline void cuda_check(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "cuda_check: %s %s:%d\n", cudaGetErrorString(code), file, line);
        assert(!code);
        if (abort) exit(code);
    }
}

#define CUDA_CHECK(ans) \
    { cuda_check((ans), __FILE__, __LINE__); }

#ifndef NDEBUG
#define DEBUG_SYNCHRONIZE()                \
    do {                                        \
        cudaDeviceSynchronize();                \
        CUDA_CHECK(cudaPeekAtLastError()); \
    } while (0)
#else
#define DEBUG_SYNCHRONIZE()
#endif

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
    int *buffer, *device_buffer;
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
    buffer = (int *) malloc(file_size);
    cudaError_t err = cudaMalloc((void**)&device_buffer, file_size);
    DEBUG_SYNCHRONIZE();

    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory: %s\n", cudaGetErrorString(err));
        fclose(file);
        return NULL;
    }

    // Read the entire content of the file into the buffer
    *elements_read = fread(buffer, sizeof(int), num_elements, file);
    if (*elements_read != num_elements) {
        perror("Error reading file");
        // cudaFree(buffer);
        free(buffer);
        fclose(file);
        return NULL;
    }

    // Close the file
    fclose(file);
    cudaMemcpy(device_buffer, buffer, file_size, cudaMemcpyHostToDevice);
    DEBUG_SYNCHRONIZE();

    // Report the number of elements read
    printf("Number of elements read from %s: %d\n", filename, *elements_read);

    return device_buffer;
}

double* load_float64_array(const char *filename, int64_t *elements_read) {
    FILE *file;
    long file_size;
    double *buffer, *device_buffer;
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
    // cudaError_t err = cudaMallocHost((void**)&buffer, file_size);
    buffer = (double *) malloc(file_size);
    cudaError_t err = cudaMalloc((void**)&device_buffer, file_size);
    DEBUG_SYNCHRONIZE();

    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory: %s\n", cudaGetErrorString(err));
        fclose(file);
        return NULL;
    }
    
    // Read the entire content of the file into the buffer
    *elements_read = fread(buffer, sizeof(double), num_elements, file);
    if (*elements_read != num_elements) {
        perror("Error reading file");
        // cudaFree(buffer);
        free(buffer);
        fclose(file);
        return NULL;
    }

    // Close the file
    fclose(file);

    cudaMemcpy(device_buffer, buffer, file_size, cudaMemcpyHostToDevice);
    DEBUG_SYNCHRONIZE();

    // Report the number of elements read
    printf("Number of elements read from %s: %d\n", filename, *elements_read);

    return device_buffer;
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
    int   *dA_csrOffsets, *dA_columns;
    double *dA_values, *dX, *dY;

    // int64_t sellValuesSize = 0;
    int64_t elements_read;

    dA_csrOffsets = load_int32_array("rowptr.raw", &elements_read);
    A_num_rows = elements_read; //sellMetaInfo[0];
    A_num_cols = elements_read; //sellMetaInfo[0];
    // CHECK_CUDA( cudaMemPrefetchAsync(dA_csrOffsets, elements_read * sizeof(int32_t), 0) )

    dA_values = load_float64_array("values.raw", &elements_read);
    A_nnz = elements_read; //sellMetaInfo[1];
    // CHECK_CUDA( cudaMemPrefetchAsync(dA_values, A_nnz * sizeof(double), 0) )

    dA_columns = load_int32_array("colidx.raw", &elements_read);
    // CHECK_CUDA( cudaMemPrefetchAsync(dA_columns, elements_read * sizeof(int32_t), 0) )

    // int *sellMetaInfo = load_int32_array("sell_meta.i32", &elements_read);

    // int sliceSize = 2;


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

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
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

    DEBUG_SYNCHRONIZE();

    cudaDeviceSynchronize();

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
    printf("Time for matrix-vector multiplication: %f milliseconds\n", milliseconds);
    printf("Throughput: %f MDOF/s\n", (A_num_rows / 1e6) / (milliseconds / 1000.0));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )
    return EXIT_SUCCESS;
}