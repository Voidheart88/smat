mod graph;

use crate::SparseMatrix;

/// Symbolic analysis for sparse matrices
/// This Module provides methods for symbolic analysis of sparse matrices.

/// The Order for the approximate minimum degree (amd) algorithm.
pub enum Order {
    Natural,  // No reordering
    Cholesky, // Cholesky
    Lu,       // LU decomposition
    Qr,       // QR decomposition
}

/// The structure of the symbolic analysis. It holds the results of the Symbolic
/// analysis.
/// The analysis is lazy calculated. Every entry will be calculated as needed
pub struct Symbolic<'a, T> {
    matrix: &'a SparseMatrix<T>,

    lu_perm: Option<Vec<isize>>, // Fill reducing permutation

    is_symmetric: Option<bool>, // Check if the matrix is symmetric ( Mat = Mat' )
    is_dense: Option<bool>,     // Check if the matrix is dense
}

/// Construct a Symbolic analysis from a reference to a Sparse Matrix
impl<'a, T> Symbolic<'a, T>
where
    T: Copy
        + Default
        + PartialEq
        + std::ops::AddAssign
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    fn is_symmetric(&mut self) -> bool {
        if self.is_symmetric.is_some() {
            return self.is_symmetric.unwrap();
        } else {
            self.is_symmetric = Some(self.check_symmetric());
            return self.is_symmetric.unwrap();
        }
    }

    fn check_symmetric(&self) -> bool {
        if self.matrix.nrows() != self.matrix.ncols() {
            return false; // A non-square matrix cannot be symmetric
        }

        for col in 0..self.matrix.ncols() {
            let start = self.matrix.col_ptr()[col];
            let end = self.matrix.col_ptr()[col + 1];

            for idx in start..end {
                let row = self.matrix.row_idx()[idx as usize];
                let value = self.matrix.values()[idx as usize];

                // Find the corresponding element in the transposed position
                let transposed_start = self.matrix.col_ptr()[row];
                let transposed_end = self.matrix.col_ptr()[row + 1];

                let mut symmetric = false;
                for t_idx in transposed_start..transposed_end {
                    if self.matrix.row_idx()[t_idx as usize] == col {
                        if self.matrix.values()[t_idx as usize] == value {
                            symmetric = true;
                            break;
                        } else {
                            return false;
                        }
                    }
                }

                if !symmetric {
                    return false;
                }
            }
        }
        true
    }

    fn is_dense(&mut self) -> bool {
        if self.is_dense.is_some() {
            return self.is_dense.unwrap();
        } else {
            self.is_dense = Some(self.check_dense());
            return self.is_dense.unwrap();
        }
    }

    // The Sparse/Dense Threshold
    // If nnz/(m*n) > 10%, the matrix is considered as Dense (FIXME = provide a better threshold)
    fn check_dense(&self) -> bool {
        let size = self.matrix.ncols() * self.matrix.nrows();
        let nnz = self.matrix.row_idx().len();

        if (nnz * 10) / size > 1 {
            return true;
        }
        false
    }
}

impl<'a, T> From<&'a SparseMatrix<T>> for Symbolic<'a, T> {
    fn from(value: &'a SparseMatrix<T>) -> Self {
        Self {
            matrix: value,
            lu_perm: None,
            is_symmetric: None,
            is_dense: None,
        }
    }
}

#[cfg(test)]
mod tests;
