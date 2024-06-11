use crate::sparse::matrix::SparseMatrix;

/// Computes the cumulative sum of the elements in the `values` array and updates
/// the `cumulative_sum` array.
///
/// This function computes the cumulative sum of the first `num_elements` elements
/// in the `values` array and stores the result
/// in the `cumulative_sum` array. The last element of `cumulative_sum` is updated
/// to hold the total sum.
///
/// # Arguments
///
/// * `cumulative_sum` - The array to store the cumulative sum.
/// * `values` - The array containing the values to sum.
/// * `num_elements` - The number of elements to include in the sum.
///
/// # Returns
///
/// The total sum of elements in the `values` array.
pub(crate) fn cumsum(
    cumulative_sum: &mut [isize],
    values: &mut [isize],
    num_elements: usize,
) -> usize {
    let mut total_sum = 0;

    for (cumsum_elem, value_elem) in cumulative_sum
        .iter_mut()
        .zip(values.iter_mut())
        .take(num_elements)
    {
        *cumsum_elem = total_sum;
        total_sum += *value_elem;
        *value_elem = *cumsum_elem;
    }

    cumulative_sum[num_elements] = total_sum;

    total_sum as usize
}

/// Scatters the values of column `col_index` from matrix `a` into vector `values_accum` with a scaling factor `scale`.
///
/// This function scatters the values of column `col_index` from matrix `a` into vector `values_accum` with a scaling factor `scale`.
/// It updates the row indices in the workspace `workspace` and the values in the matrix `result_matrix`. The `marker` parameter is used
/// to mark the visited rows in `workspace`. The function returns the updated number of non-zero elements in `result_matrix`.
///
/// # Arguments
///
/// * `a` - The input matrix in sparse format.
/// * `col_index` - The index of the column in matrix `a` to scatter.
/// * `scale` - The scaling factor to apply to the scattered values.
/// * `workspace` - The workspace to track visited rows during scattering.
/// * `values_accum` - The vector where the scattered values are accumulated.
/// * `marker` - The marker value to indicate visited rows in the workspace `workspace`.
/// * `result_matrix` - The matrix where the scattered values and row indices are stored.
/// * `num_nonzeros` - The current number of non-zero elements in matrix `result_matrix`.
///
/// # Returns
///
/// The updated number of non-zero elements in matrix `result_matrix` after scattering the values from column `col_index` of matrix `a`.
pub(crate) fn scatter(
    a: &SparseMatrix,
    col_index: usize,
    scale: f64,
    workspace: &mut [isize],
    values_accum: &mut [f64],
    marker: usize,
    result_matrix: &mut SparseMatrix,
    num_nonzeros: usize,
) -> usize {
    let mut row_index;
    let mut updated_nonzeros = num_nonzeros;
    for p in a.col_idx[col_index] as usize..a.col_idx[col_index + 1] as usize {
        row_index = a.row_idx[p];
        if workspace[row_index] < marker as isize {
            workspace[row_index] = marker as isize;
            result_matrix.row_idx[updated_nonzeros] = row_index;
            updated_nonzeros += 1;
            values_accum[row_index] = scale * a.values[p];
        } else {
            values_accum[row_index] += scale * a.values[p];
        }
    }

    updated_nonzeros
}

#[cfg(test)]
mod tests;
