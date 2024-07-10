use num::One;

use crate::SparseMatrix;

use super::Solver;

pub struct LUSolver<'a, 'b, T> {
    matrix: &'a SparseMatrix<T>,
    vector: &'b Vec<T>,
    solution: Vec<T>,

    upper: SparseMatrix<T>,
    lower: SparseMatrix<T>,
}

impl<'a, 'b, T> LUSolver<'a, 'b, T>
where
    T: Copy
        + Default
        + std::fmt::Debug
        + PartialEq
        + One
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>,
{
    /// Create a new Solver.
    /// This method allocates space for the upper and the lower Matrix as wall as the solution vector
    pub fn new(matrix: &'a SparseMatrix<T>, vector: &'b Vec<T>) -> LUSolver<'a, 'b, T> {
        let solution = vec![T::default();vector.len()];
        let upper = matrix.upper_triangular();
        let lower = matrix.lower_triangular();
        LUSolver {
            matrix,
            vector,
            solution,
            upper,
            lower,
        }
    }

    // Todo
    fn decompose(&mut self) {
        self.upper = self.matrix.clone();
        self.lower = self.matrix.lower_triangular();
        let ncols = self.matrix.ncols();
        let nrows = self.matrix.nrows();

        /*for idx_k in 0..ncols-1 {
            //println!("k:{idx_k}");
            for idx_i in (idx_k + 1)..nrows {
                let lik = self.matrix.get_unchecked(idx_i, idx_k);
                let ukk = self.upper.get_unchecked(idx_k, idx_k);
                self.lower.set(idx_i, idx_k, lik / ukk);

                // Update upper row
                for idx_j in idx_i..nrows {
                    let uii = self.upper.get_unchecked(idx_i, idx_j);
                    let uki = self.upper.get_unchecked(idx_k, idx_j);
                    self.upper.set(idx_i, idx_j, uii - uki * lik);
                }
            }

            for idx_i in (idx_k+1)..nrows {
                //println!("i:{idx_k}");
                let ukj = self.upper.get_unchecked(idx_k, idx_i);
                // Update lower col
                for idx_j in (idx_i+1)..nrows {
                    //println!("j:{idx_k}");
                    
                    let ljj = self.lower.get_unchecked(idx_j, idx_i);
                    let ljk = self.lower.get_unchecked(idx_j, idx_k);
                    //println!("row:{idx_j} col:{idx_i}");
                    self.lower.set(idx_j, idx_i, ljj - ljk * ukj);
                }
            }
        }*/

        self.upper = self.matrix.clone();
        self.lower = SparseMatrix::eye(T::one(), ncols);
        for idx_k in 0..nrows {
            let akk = self.upper.get(idx_k,idx_k).unwrap();
            for idx_i in (idx_k+1)..nrows {
                let aik = self.upper.get(idx_i,idx_k).unwrap();
                let lik = aik/akk;
                self.lower.set(idx_i, idx_k, lik);
                for idx_j in (idx_k+1)..ncols {
                    let akj = self.upper.get(idx_k,idx_j).unwrap();
                    let aij = self.upper.get(idx_i,idx_j).unwrap();
                    self.upper.set(idx_i, idx_j, aij-lik*akj );
                }
            }
        }
    }

    /// solve Ly = b
    /// The Algorithm assumes no empty columns and L is a lower triangular matrix
    fn forward_substitution(&mut self) {
        let matrix = &self.lower;
        self.solution.clone_from(self.vector);

        for col in 0..matrix.ncols() {
            self.solution[col] =
                self.solution[col] / matrix.values()[matrix.col_ptr()[col] as usize];
            for row in (matrix.col_ptr()[col] + 1) as usize..(matrix.col_ptr()[col + 1]) as usize {
                self.solution[matrix.row_idx()[row]] = self.solution[matrix.row_idx()[row]]
                    - (matrix.values()[row] * self.solution[col]);
            }
        }
    }

    /// solve Ub = x
    /// The Algorithm assumes no empty columns and U is a upper triangular matrix
    fn backward_substitution(&mut self)
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
        let matrix = &self.upper;
        self.solution.clone_from(self.vector);

        for col in (0..matrix.ncols()).rev() {
            self.solution[col] =
                self.solution[col] / matrix.values()[(matrix.col_ptr()[col + 1] - 1) as usize];
            for row in matrix.col_ptr()[col]..matrix.col_ptr()[col + 1] - 1 {
                self.solution[matrix.row_idx()[row as usize]] = self.solution
                    [matrix.row_idx()[row as usize]]
                    - matrix.values()[row as usize] * self.solution[col]
            }
        }
    }
}

impl<'a, 'b, T> Solver<T> for LUSolver<'a, 'b, T>
where
    T: Copy
        + Default
        + std::fmt::Debug
        + PartialEq
        + One
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
{
    fn solve(&mut self) -> &Vec<T> {
        self.decompose();
        self.solution.clone_from(self.vector);
        self.forward_substitution();
        self.backward_substitution();
        &self.solution
    }
}

#[cfg(test)]
mod tests;
