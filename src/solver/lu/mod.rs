use num::One;

use crate::SparseMatrix;

use super::Solver;

pub struct LUSolver<'a, 'b, T> {
    matrix: &'a SparseMatrix<T>,
    vector: &'b Vec<T>,
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
    fn solve(&self) -> Vec<T> {
        let (lower_matrix, upper_matrix) = lu_decompose(self.matrix);

        // Forward substitution to solve lower_matrix * y_vector = vector
        let y_vector = forward_substitution(&lower_matrix, self.vector.into());
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

/// solve Ly = b
/// The Algorithm assumes no empty columns
fn forward_substitution<T>(lower_matrix: &SparseMatrix<T>, vector: &Vec<T>) -> Vec<T>
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
    let mut y_vector = vector.to_vec();

    for col in 0..lower_matrix.ncols() {
        y_vector[col] = y_vector[col] / lower_matrix.values()[lower_matrix.col_ptr()[col] as usize];
        for row in
            (lower_matrix.col_ptr()[col] + 1) as usize..(lower_matrix.col_ptr()[col + 1]) as usize
        {
            y_vector[lower_matrix.row_idx()[row]] = y_vector[lower_matrix.row_idx()[row]]
                - (lower_matrix.values()[row] * y_vector[col]);
        }
    }

    y_vector
}

fn backward_substitution<T>(upper_matrix: &SparseMatrix<T>, y_vector: &Vec<T>) -> Vec<T>
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
        x_vector[row] = (*y_vector.get(row).unwrap() - sum) / upper_matrix.get(row, row).unwrap();
    }

    x_vector
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
