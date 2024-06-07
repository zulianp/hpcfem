import numpy as np
import math

# row_offsets = [0, 2, 3, 3, 5, 8]
# column_indices = [0, 2, 1, 0, 1, 1, 2, 3]
# values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
rows_per_slice = 32

# colidx.raw  meta.yaml  rowptr.raw  values.raw

values = np.fromfile("values.raw", dtype=float)
row_offsets = np.fromfile("rowptr.raw", dtype=np.int32)
column_indices = np.fromfile("colidx.raw", dtype=np.int32)

rows = len(row_offsets) - 1
slices = math.ceil(rows / rows_per_slice)
sell_slice_offsets = []
sell_values = []
sell_column_indices = []
sell_slice_offsets.append(0)

meta_information = []

print(slices)
for slice_idx in range(slices) :
    start_row = slice_idx * rows_per_slice
    end_row = start_row + rows_per_slice
    slice_columns = 0
    # print('start_row', start_row, 'end_row', end_row)
    for row_offset_idx in range(start_row, min(end_row, rows)) :
        # print(slice_idx, row_offset_idx, row_offsets[row_offset_idx+1] - row_offsets[row_offset_idx])
        slice_columns = max(row_offsets[row_offset_idx+1] - row_offsets[row_offset_idx], slice_columns)
    sell_slice_offsets.append(sell_slice_offsets[-1] + slice_columns * rows_per_slice)
    for column_idx_offset in range(slice_columns) :
        # determine if we have enough rows
        for row_offset_idx in range(start_row, end_row) :
            if row_offset_idx >= rows :
                sell_values.append(1.23) # "*"
                sell_column_indices.append(-1)
                continue
            column_idx_head = row_offsets[row_offset_idx]
            column_idx_tail = row_offsets[row_offset_idx+1]
            column_idx_curr = column_idx_head + column_idx_offset
            if column_idx_head + column_idx_offset >= column_idx_tail :
                sell_values.append(1.23) # "*"
                sell_column_indices.append(-1)
            else :
                sell_values.append(values[column_idx_curr])
                sell_column_indices.append(column_indices[column_idx_curr])

# print("Values (Sliced Ellpack) #:")
# print(len(sell_values), np.asarray(sell_values).dtype)
# print("Indices (Sliced Ellpack) #:")
# print(len(sell_column_indices), np.asarray(sell_column_indices).dtype)
# print("Slices (Sliced Ellpack) #:")
# print(len(sell_slice_offsets), np.asarray(sell_slice_offsets).dtype)

# rows
meta_information.append(len(row_offsets)-1)

# nnz
meta_information.append(len(values))

# rows_per_slice
meta_information.append(rows_per_slice)

with open('sell_meta.i32', 'w') as f:
    np.asarray(meta_information, dtype='int32').tofile(f)

with open('sell_values.f64', 'w') as f:
    np.asarray(sell_values, dtype='float64').tofile(f)
    
with open('sell_column_indices.i32', 'w') as f:
    np.asarray(sell_column_indices, dtype='int32').tofile(f)
    
with open('sell_slice_offsets.i32', 'w') as f:
    np.asarray(sell_slice_offsets, dtype='int32').tofile(f)
