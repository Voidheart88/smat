use num::One;

use crate::{SparseMatrix, SparseVector};

use super::Solver;

pub struct LUSolver<'a, 'b, T> {
    matrix: &'a SparseMatrix<T>,
    vector: &'b SparseVector<T>,
}

impl<'a, 'b, T> Solver<T> for LUSolver<'a, 'b, T>
where
    T: Copy
        + Default
        + PartialEq
        + One
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::fmt::Debug,
{
    fn solve(&self) -> crate::SparseVector<T> {
        let (lower_matrix, upper_matrix) = lu_decompose(self.matrix);

        // Forward substitution to solve lower_matrix * y_vector = vector
        let y_vector = forward_substitution(&lower_matrix, self.vector);
        // Backward substitution to solve upper_matrix * x_vector = y_vector
        let x_vector = backward_substitution(&upper_matrix, &y_vector);

        x_vector
    }
}

fn lu_decompose<T>(matrix: &SparseMatrix<T>) -> (SparseMatrix<T>, SparseMatrix<T>)
where
    T: Copy
        + Default
        + PartialEq
        + One
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>,
{
    let mut lower_matrix = matrix.lower_triangular();
    let mut upper_matrix = matrix.upper_triangular();

    let n = matrix.nrows();

    for k in 0..n {
        for i in k + 1..n {
            if let Some(lik) = lower_matrix.get(i, k) {
                if lik != T::default() {
                    let u_kk = upper_matrix.get(k, k).unwrap_or(T::default());
                    lower_matrix.set(i, k, lik / u_kk);

                    for j in k..n {
                        if let Some(u_kj) = upper_matrix.get(k, j) {
                            if u_kj != T::default() {
                                let u_ij = upper_matrix.get(i, j).unwrap_or(T::default());
                                let l_ik = lower_matrix.get(i, k).unwrap_or(T::default());
                                upper_matrix.set(i, j, u_ij - l_ik * u_kj);
                            }
                        }
                    }
                }
            }
        }
        for j in k + 1..n {
            if let Some(ukj) = upper_matrix.get(k, j) {
                if ukj != T::default() {
                    for i in j..n {
                        if let Some(l_ij) = lower_matrix.get(i, j) {
                            if l_ij != T::default() {
                                let l_ik = lower_matrix.get(i, k).unwrap_or(T::default());
                                lower_matrix.set(i, j, l_ij - l_ik * ukj);
                            }
                        }
                    }
                }
            }
        }
    }

    (lower_matrix, upper_matrix)
}

fn forward_substitution<T>(
    lower_matrix: &SparseMatrix<T>,
    vector: &SparseVector<T>,
) -> SparseVector<T>
where
    T: Copy
        + Default
        + PartialEq
        + One
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>,
{
    let mut y_vector = vec![T::default(); vector.len()];

    for row in 0..vector.len() {
        let mut sum = T::default();
        for column in 0..row {
            sum = sum + lower_matrix.get(row, column).unwrap() * y_vector[column];
        }
        y_vector[row] = (vector.get(row).unwrap() - sum) / lower_matrix.get(row, row).unwrap();
    }

    SparseVector::from(y_vector)
}

fn backward_substitution<T>(
    upper_matrix: &SparseMatrix<T>,
    y_vector: &SparseVector<T>,
) -> SparseVector<T>
where
    T: Copy
        + Default
        + PartialEq
        + One
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let dimension = y_vector.len();
    let mut x_vector = vec![T::default(); dimension];

    for row in (0..dimension).rev() {
        let mut sum = T::default();
        for column in row + 1..dimension {
            sum = sum + upper_matrix.get(row, column).unwrap() * x_vector[column];
        }
        x_vector[row] = (y_vector.get(row).unwrap() - sum) / upper_matrix.get(row, row).unwrap();
    }

    SparseVector::from(x_vector)
}

/// Solves Ly = b by forward substitution
fn l_solve<T>(mat_l: SparseMatrix<T>, vec_y: &mut Vec<T> )
where
    T: Copy
        + Default
        + PartialEq
        + One
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    for idx in 0..vec_y.len() {
        vec_y[idx] = vec_y[idx]/mat_l.get(idx, idx).unwrap();
        for idy in (idx+1)..mat_l.nnz() {
            
        }
    }
    todo!()
}

/// Solves Ux = y
fn u_solve<T>() -> Vec<T>
where
    T: Copy
        + Default
        + PartialEq
        + One
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    todo!()
}

#[cfg(test)]
mod tests;
