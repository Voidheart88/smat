use std::ops::Mul;

use super::matrix::SparseMatrix;

pub(crate) fn scatter<T>(
    matrix_a: &SparseMatrix<T>,
    col_index: usize,
    scalar: T,
    row_marker: &mut [usize],
    row_values: &mut [T],
    marker_value: usize,
    result_matrix: &mut SparseMatrix<T>,
    non_zero_count: usize,
) -> usize
where
    T: Copy + Default + Mul<Output = T> + PartialEq + std::ops::AddAssign,
{
    let mut row_index;
    let mut new_non_zero_count = non_zero_count;

    for p in matrix_a.col_ptr()[col_index]..matrix_a.col_ptr()[col_index + 1] {
        row_index = matrix_a.row_idx()[p];
        if row_marker[row_index] < marker_value {
            row_marker[row_index] = marker_value;
            result_matrix.row_idx_mut()[new_non_zero_count] = row_index;
            new_non_zero_count += 1;
            row_values[row_index] = scalar * matrix_a.values()[p];
        } else {
            row_values[row_index] += scalar * matrix_a.values()[p];
        }
    }

    new_non_zero_count
}