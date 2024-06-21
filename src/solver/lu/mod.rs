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
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>,
{
    let dimension = matrix.nrows();
    let mut lower_matrix = SparseMatrix::eye(T::default(), dimension); // initialize lower_matrix as the identity matrix
    let mut upper_matrix = matrix.clone();

    for pivot in 0..dimension {
        for row in pivot + 1..dimension {
            let multiplier = upper_matrix.get(row, pivot).unwrap_or(T::default()) / upper_matrix.get(pivot, pivot).unwrap_or(T::default());
            lower_matrix.set(row, pivot, multiplier);
            for column in pivot..dimension {
                let new_value = upper_matrix.get(row, column).unwrap_or(T::default()) - multiplier * upper_matrix.get(pivot, column).unwrap_or(T::default());
                upper_matrix.set(row, column, new_value);
            }
        }
    }

    (lower_matrix, upper_matrix)
}

fn forward_substitution<T>(lower_matrix: &SparseMatrix<T>, vector: &SparseVector<T>) -> SparseVector<T>
where
    T: Copy
        + Default
        + PartialEq
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

fn backward_substitution<T>(upper_matrix: &SparseMatrix<T>, y_vector: &SparseVector<T>) -> SparseVector<T>
where
    T: Copy
        + Default
        + PartialEq
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

#[cfg(test)]
mod tests;