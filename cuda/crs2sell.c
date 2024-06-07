#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>

int main(int argc, char *argv[]) {
    int rows_per_slice = 32;
    if (argc != 2) {
        fprintf(stdout, "Usage: %s ROWS_PER_SLICE\n", argv[0]);
    } else {
        int rows_per_slice = atoi(argv[1]);
        if (rows_per_slice <= 0) {
            fprintf(stderr, "Invalid value for ROWS_PER_SLICE: %d\n", rows_per_slice);
            return EXIT_FAILURE;
        }
    }

    // Load data from files
    FILE* file;
    double* values;
    int32_t* row_offsets;
    int32_t* column_indices;
    int rows, nnz, slices, slice_columns;
    int64_t meta_information[3];

    // Read values
    file = fopen("values.raw", "rb");
    if (file == NULL) {
        perror("Error opening values.raw");
        return EXIT_FAILURE;
    }
    fseek(file, 0, SEEK_END);
    nnz = ftell(file) / sizeof(double);
    fseek(file, 0, SEEK_SET);
    values = malloc(nnz * sizeof(double));
    fread(values, sizeof(double), nnz, file);
    fclose(file);

    // Read row offsets
    file = fopen("rowptr.raw", "rb");
    if (file == NULL) {
        perror("Error opening rowptr.raw");
        return EXIT_FAILURE;
    }
    fseek(file, 0, SEEK_END);
    rows = ftell(file) / sizeof(int32_t) - 1;
    fseek(file, 0, SEEK_SET);
    row_offsets = malloc((rows + 1) * sizeof(int32_t));
    fread(row_offsets, sizeof(int32_t), rows + 1, file);
    fclose(file);

    // Read column indices
    file = fopen("colidx.raw", "rb");
    if (file == NULL) {
        perror("Error opening colidx.raw");
        return EXIT_FAILURE;
    }
    column_indices = malloc(nnz * sizeof(int32_t));
    fread(column_indices, sizeof(int32_t), nnz, file);
    fclose(file);

    // Compute slices
    slices = (int)ceil((double)rows / rows_per_slice);
    printf("Will create %d slices\n", slices);

    // Allocate memory for SELL format
    int64_t* sell_slice_offsets = malloc((slices + 1) * sizeof(int64_t));
    double* sell_values = malloc(2 * nnz * sizeof(double)); // Overestimate size for simplicity
    int64_t* sell_column_indices = malloc(2 * nnz * sizeof(int64_t)); // Overestimate size for simplicity
    int sell_values_count = 0;

    // Initialize slice offsets
    sell_slice_offsets[0] = 0;

    // Process each slice
    for (int slice_idx = 0; slice_idx < slices; ++slice_idx) {
        int start_row = slice_idx * rows_per_slice;
        int end_row = start_row + rows_per_slice;
        slice_columns = 0;

        // Determine the maximum number of columns in this slice
        for (int row_offset_idx = start_row; row_offset_idx < end_row && row_offset_idx < rows; ++row_offset_idx) {
            int row_length = row_offsets[row_offset_idx + 1] - row_offsets[row_offset_idx];
            if (row_length > slice_columns) {
                slice_columns = row_length;
            }
        }

        // Set the next slice offset
        sell_slice_offsets[slice_idx + 1] = sell_slice_offsets[slice_idx] + slice_columns * rows_per_slice;

        // Fill in the values and column indices for this slice
        for (int column_idx_offset = 0; column_idx_offset < slice_columns; ++column_idx_offset) {
            for (int row_offset_idx = start_row; row_offset_idx < end_row; ++row_offset_idx) {
                if (row_offset_idx >= rows) {
                    sell_values[sell_values_count] = 1.23; // "*"
                    sell_column_indices[sell_values_count] = -1;
                } else {
                    int column_idx_head = row_offsets[row_offset_idx];
                    int column_idx_tail = row_offsets[row_offset_idx + 1];
                    int column_idx_curr = column_idx_head + column_idx_offset;
                    if (column_idx_curr >= column_idx_tail) {
                        sell_values[sell_values_count] = 1.23; // "*"
                        sell_column_indices[sell_values_count] = -1;
                    } else {
                        sell_values[sell_values_count] = values[column_idx_curr];
                        sell_column_indices[sell_values_count] = column_indices[column_idx_curr];
                    }
                }
                sell_values_count++;
            }
        }
    }

    // Write meta information
    meta_information[0] = rows;
    meta_information[1] = nnz;
    meta_information[2] = rows_per_slice;

    file = fopen("sell_meta.i64", "wb");
    fwrite(meta_information, sizeof(int64_t), 3, file);
    fclose(file);

    // Determine the data type for indices and offsets based on sell_values_count
    if (sell_values_count > INT_MAX) {
        // Write SELL values
        file = fopen("sell_values.f64", "wb");
        fwrite(sell_values, sizeof(double), sell_values_count, file);
        fclose(file);

        // Write SELL column indices as int64_t
        file = fopen("sell_column_indices.i64", "wb");
        fwrite(sell_column_indices, sizeof(int64_t), sell_values_count, file);
        fclose(file);

        // Write SELL slice offsets as int64_t
        file = fopen("sell_slice_offsets.i64", "wb");
        fwrite(sell_slice_offsets, sizeof(int64_t), slices + 1, file);
        fclose(file);
    } else {
        // Cast to int32_t for writing
        int32_t* sell_column_indices_i32 = malloc(sell_values_count * sizeof(int32_t));
        int32_t* sell_slice_offsets_i32 = malloc((slices + 1) * sizeof(int32_t));

        for (int64_t i = 0; i < sell_values_count; ++i) {
            sell_column_indices_i32[i] = (int32_t)sell_column_indices[i];
        }

        for (int i = 0; i <= slices; ++i) {
            sell_slice_offsets_i32[i] = (int32_t)sell_slice_offsets[i];
        }

        // Write SELL values
        file = fopen("sell_values.f64", "wb");
        fwrite(sell_values, sizeof(double), sell_values_count, file);
        fclose(file);

        // Write SELL column indices as int32_t
        file = fopen("sell_column_indices.i32", "wb");
        fwrite(sell_column_indices_i32, sizeof(int32_t), sell_values_count, file);
        fclose(file);

        // Write SELL slice offsets as int32_t
        file = fopen("sell_slice_offsets.i32", "wb");
        fwrite(sell_slice_offsets_i32, sizeof(int32_t), slices + 1, file);
        fclose(file);

        free(sell_column_indices_i32);
        free(sell_slice_offsets_i32);
    }

    // Clean up
    free(values);
    free(row_offsets);
    free(column_indices);
    free(sell_slice_offsets);
    free(sell_values);
    free(sell_column_indices);

    return 0;
}
