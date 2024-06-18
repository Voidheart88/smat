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

    lu_perm: Option<Vec<usize>>, // Fill reducing permutation

    is_symmetric: Option<bool>, // Check if the matrix is symmetric ( Mat = Mat' )
    is_dense: Option<bool>,     // Check if the matrix is dense
}

/// Construct a Symbolic analysis from a reference to a Sparse Matrix
impl<'a, T> Symbolic<'a, T>
where
    T: Copy + Default + PartialEq + std::ops::Mul<Output = T> + std::ops::AddAssign,
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
                let row = self.matrix.row_idx()[idx];
                let value = self.matrix.values()[idx];

                // Find the corresponding element in the transposed position
                let transposed_start = self.matrix.col_ptr()[row];
                let transposed_end = self.matrix.col_ptr()[row + 1];

                let mut symmetric = false;
                for t_idx in transposed_start..transposed_end {
                    if self.matrix.row_idx()[t_idx] == col {
                        if self.matrix.values()[t_idx] == value {
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
    // If nnz/(m*n) > 10%, the matrix is considered as Dense
    fn check_dense(&self) -> bool {
        let size = self.matrix.ncols() * self.matrix.nrows();
        let nnz = self.matrix.row_idx().len();

        if (nnz * 10) / size > 1 {
            return true;
        }
        false
    }

    // Returns the LU permutation vector after AMD
    fn lu_perm(&mut self) -> Vec<usize> {
        if self.lu_perm.is_some() {
            return self.lu_perm.clone().unwrap();
        } else {
            self.lu_perm = Some(self.lu_amd());
            return self.lu_perm.clone().unwrap();
        }
    }

    /// calculates the LU permutation vector with the amd algorithm.
    fn lu_amd(&mut self) -> Vec<usize> {

        // calculate the transposed
        let at = if self.is_symmetric() {
            self.matrix.clone()
        } else {
            self.matrix.transpose()
        };

        let c = &at * self.matrix; // C=A'*A
        let n = self.matrix.ncols();
        
        let mut p_v = vec![0; n + 1]; // allocate result
        let mut ww = vec![0; 8 * (n + 1)]; // get workspace
        // offsets of ww (pointers in csparse)

        let len = 0; // of ww
        let nv = n + 1; // of ww
        let next = 2 * (n + 1); // of ww
        let head = 3 * (n + 1); // of ww
        let elen = 4 * (n + 1); // of ww
        let degree = 5 * (n + 1); // of ww
        let w = 6 * (n + 1); // of ww
        let hhead = 7 * (n + 1); // of ww
        let last = 0; // of p_v // use P as workspace for last

        /*
        fkeep(&mut c, &diag); // drop diagonal entries
        let mut cnz = c.p[n];
        // change the max # of entries of C
        let nsz = cnz as usize + cnz as usize / 5 + 2 * n;
        c.nzmax = nsz;
        c.i.resize(nsz, 0);
        c.x.resize(nsz, 0.);

        // --- Initialize quotient graph ----------------------------------------
        for k in 0..n {
            ww[len + k] = c.p[k + 1] - c.p[k];
        }
        ww[len + n] = 0;
        for i in 0..=n {
            ww[head + i] = -1; // degree list i is empty
            p_v[last + i] = -1;
            ww[next + i] = -1;
            ww[hhead + i] = -1; // hash list i is empty
            ww[nv + i] = 1; // node i is just one node
            ww[w + i] = 1; // node i is alive
            ww[elen + i] = 0; // Ek of node i is empty
            ww[degree + i] = ww[len + i]; // degree of node i
        }
        mark_v = wclear(0, 0, &mut ww[..], w, n); // clear w ALERT!!! C implementation passes w (pointer to ww)
        ww[elen + n] = -2; // n is a dead element
        c.p[n] = -1; // n is a root of assembly tree
        ww[w + n] = 0; // n is a dead element
        */
        todo!()
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
