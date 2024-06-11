use utils::{cumsum, scatter};

use std::fmt;

use super::*;
use crate::triple::Triples;

/// Matrix in compressed sparse column (CSC) format
#[derive(Clone, Debug)]
pub struct SparseMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub col_idx: Vec<isize>,
    pub row_idx: Vec<usize>,
    pub values: Vec<f64>,
}

impl SparseMatrix {
    pub fn new(
        nrows: usize,
        ncols: usize,
        col_idx: Vec<isize>,
        row_idx: Vec<usize>,
        values: Vec<f64>,
    ) -> SparseMatrix {
        SparseMatrix {
            nrows,
            ncols,
            col_idx,
            row_idx,
            values,
        }
    }

    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline]
    pub fn column_idx(&self) -> &Vec<isize> {
        &self.col_idx
    }

    #[inline]
    pub fn row_idx(&self) -> &Vec<usize> {
        &self.row_idx
    }

    #[inline]
    pub fn values(&self) -> &Vec<f64> {
        &self.values
    }

    /// Create a new Sparse Matrix filled with Zeros
    /// TODO: Matrix generation should not use this function -- reduce usage
    pub fn zeros(m: usize, n: usize, nzmax: usize) -> SparseMatrix {
        SparseMatrix {
            nrows: m,
            ncols: n,
            col_idx: vec![0; n + 1],
            row_idx: vec![0; nzmax],
            values: vec![0.; nzmax],
        }
    }

    /// Create a sparse eye matrix
    pub fn eye(n: usize) -> SparseMatrix {
        let col_idx = (0..=n).map(|i| i as isize).collect();
        let row_idx = (0..n).collect();
        let values = vec![1.0; n];

        SparseMatrix {
            nrows: n,
            ncols: n,
            col_idx,
            row_idx,
            values,
        }
    }

    /// Get a Value
    pub fn get(&self, row: usize, column: usize) -> Option<f64> {
        self.col_idx
            .iter()
            .zip(self.col_idx.iter().skip(1))
            .enumerate()
            .flat_map(|(j, (&start, &end))| (start..end).map(move |i| (i, j)))
            .find(|&(i, j)| self.row_idx[i as usize] == row && j == column)
            .map(|(i, _)| self.values[i as usize])
    }

    /// Trims the sparse matrix by removing all elements with zero values.
    ///
    /// This method performs the following steps:
    /// 1. Identifies the indices of all zero values in the `values` vector.
    /// 2. Removes the elements at these indices from the `values` and `row_idx` vectors.
    /// 3. Adjusts the `col_idx` vector to reflect the removal of these elements.
    ///
    /// This is useful for keeping the matrix compact and efficient by removing unnecessary zero elements.
    pub fn trim(&mut self) {
        let zero_indices: Vec<usize> = self
            .values
            .iter()
            .enumerate()
            .filter(|&(_, value)| *value == 0.0)
            .map(|(index, _)| index)
            .collect();

        for &index in zero_indices.iter().rev() {
            self.values.remove(index);
            self.row_idx.remove(index);
        }

        for j in (0..self.col_idx.len()).rev() {
            let num_removed = zero_indices
                .iter()
                .filter(|&&index| index < self.col_idx[j] as usize)
                .count();
            if num_removed > 0 {
                self.col_idx[j] -= num_removed as isize;
            }
        }
    }

    /// Quickly trims the sparse matrix by resizing the `row_idx` and `values` vectors.
    ///
    /// This method resizes the `row_idx` and `values` vectors based on the maximum number
    /// of non-zero elements (`nzmax`). This is a faster but less precise method than `trim`,
    /// which completely removes zero elements.
    pub fn quick_trim(&mut self) {
        let nzmax = self.col_idx[self.ncols] as usize;
        self.row_idx.resize(nzmax, 0);
        self.values.resize(nzmax, 0.);
    }

    /// Creates a new sparse matrix by adding a scalar value to each element.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The scalar value to add to each element of the matrix.
    ///
    /// # Returns
    ///
    /// A new `SparseMatrix` where each element is the sum of the original element and `alpha`.
    ///
    /// This method is useful for scalar addition operations on sparse matrices.
    pub(crate) fn scpmat(&self, alpha: f64) -> SparseMatrix {
        SparseMatrix::new(
            self.nrows,
            self.ncols,
            self.col_idx.clone(),
            self.row_idx.clone(),
            self.values.iter().map(|x| x + alpha).collect(),
        )
    }

    /// Creates a new sparse matrix by multiplying each element by a scalar value.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The scalar value to multiply each element of the matrix by.
    ///
    /// # Returns
    ///
    /// A new `SparseMatrix` where each element is the product of the original element and `alpha`.
    ///
    /// This method is useful for scalar multiplication operations on sparse matrices.
    pub(crate) fn scxmat(&self, alpha: f64) -> SparseMatrix {
        SparseMatrix::new(
            self.nrows,
            self.ncols,
            self.col_idx.clone(),
            self.row_idx.clone(),
            self.values.iter().map(|x| x * alpha).collect(),
        )
    }

    /// Returns an iterator over the columns of the sparse matrix.
    ///
    /// The iterator yields `Option<SparseRowIter>`, where each `SparseRowIter` iterates
    /// over the non-zero elements in the corresponding column. If a column has no
    /// non-zero elements, `None` is returned for that column.
    ///
    /// # Returns
    ///
    /// A `SparseColIter` iterator that can be used to traverse the matrix column by column.
    ///
    /// This method is useful for iterating over the elements of the sparse matrix
    /// in a column-major order.
    pub fn iter(&self) -> SparseColIter {
        SparseColIter {
            idx: 0,
            iterable: self,
        }
    }

    /// Computes the norm of a sparse matrix.
    ///
    /// This function computes the norm of the given sparse matrix `matrix`.
    /// The norm of a matrix is defined as the maximum absolute column sum.
    ///
    /// # Returns
    ///
    /// The norm of the sparse matrix `matrix`.
    pub(crate) fn norm(&self) -> f64 {
        let mut max_norm = 0.0;

        for col_option in self.iter() {
            if let Some(mut row_iter) = col_option {
                let mut column_sum = 0.0;
                while let Some((_, value)) = row_iter.next() {
                    column_sum += value.abs();
                }
                max_norm = f64::max(max_norm, column_sum);
            }
        }

        max_norm
    }
}

impl Default for SparseMatrix {
    fn default() -> Self {
        Self::new(0, 0, Vec::new(), Vec::new(), Vec::new())
    }
}

impl From<&[Vec<f64>]> for SparseMatrix {
    fn from(a: &[Vec<f64>]) -> Self {
        let r = a.len();
        let c = a[0].len();
        let mut idxptr = 0;

        let mut sparse = SparseMatrix {
            nrows: r,
            ncols: c,
            col_idx: Vec::with_capacity(c + 1),
            row_idx: Vec::new(),
            values: Vec::new(),
        };

        sparse.col_idx.push(idxptr);

        (0..c).for_each(|i| {
            a.iter().take(r).enumerate().for_each(|(j, aj)| {
                let elem = aj[i];
                if elem != 0.0 {
                    sparse.values.push(elem);
                    sparse.row_idx.push(j);
                    idxptr += 1;
                }
            });
            sparse.col_idx.push(idxptr);
        });

        sparse.trim();
        sparse
    }
}

impl From<Vec<Vec<f64>>> for SparseMatrix {
    fn from(value: Vec<Vec<f64>>) -> Self {
        let r = value.len();
        let c = value[0].len();
        let mut idxptr = 0;

        let mut sparse = SparseMatrix {
            nrows: r,
            ncols: c,
            col_idx: Vec::with_capacity(c + 1),
            row_idx: Vec::new(),
            values: Vec::new(),
        };

        sparse.col_idx.push(idxptr);

        (0..c).for_each(|i| {
            value.iter().take(r).enumerate().for_each(|(j, aj)| {
                let elem = aj[i];
                if elem != 0.0 {
                    sparse.values.push(elem);
                    sparse.row_idx.push(j);
                    idxptr += 1;
                }
            });
            sparse.col_idx.push(idxptr);
        });

        sparse.trim();
        sparse
    }
}

impl From<&Triples> for SparseMatrix {
    fn from(triples: &Triples) -> Self {
        let mut column_counts = vec![0; triples.columns()];
        triples.column_idx().iter().for_each(|&column_index| {
            column_counts[column_index as usize] += 1;
        });

        let mut col_idx = vec![0; triples.columns() + 1];
        cumsum(&mut col_idx, &mut column_counts, triples.columns());

        let nnz = triples.values().len();

        let mut row_idx = vec![0; nnz];
        let mut values = vec![0.0; nnz];
        let mut current_positions = col_idx.clone();

        triples
            .column_idx()
            .iter()
            .enumerate()
            .for_each(|(index, &column_index)| {
                let position = current_positions[column_index as usize];
                row_idx[position as usize] = triples.row_idx()[index];
                values[position as usize] = triples.values()[index];
                current_positions[column_index as usize] += 1;
            });

        SparseMatrix {
            nrows: triples.rows(),
            ncols: triples.columns(),
            col_idx,
            row_idx,
            values,
        }
    }
}

impl From<&SparseMatrix> for Vec<Vec<f64>> {
    fn from(sparse: &SparseMatrix) -> Self {
        let mut result = vec![vec![0.0; sparse.ncols]; sparse.nrows];

        (0..sparse.ncols).for_each(|j| {
            (sparse.col_idx[j] as usize..sparse.col_idx[j + 1] as usize).for_each(|i| {
                result[sparse.row_idx[i]][j] = sparse.values[i];
            });
        });

        result
    }
}

impl PartialEq for SparseMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.nrows == other.nrows
            && self.ncols == other.ncols
            && self.col_idx == other.col_idx
            && self.row_idx == other.row_idx
            && self.values == other.values
    }
}

impl std::ops::Add for SparseMatrix {
    type Output = Self;
    fn add(self, other: SparseMatrix) -> SparseMatrix {
        add(&self, &other, 1., 1.)
    }
}

impl std::ops::Add<&SparseMatrix> for SparseMatrix {
    type Output = Self;
    fn add(self, other: &SparseMatrix) -> SparseMatrix {
        add(&self, other, 1., 1.)
    }
}

impl std::ops::Add for &SparseMatrix {
    type Output = SparseMatrix;
    fn add(self, other: &SparseMatrix) -> SparseMatrix {
        add(self, other, 1., 1.)
    }
}

impl std::ops::Add<SparseMatrix> for &SparseMatrix {
    type Output = SparseMatrix;
    fn add(self, other: SparseMatrix) -> SparseMatrix {
        add(self, &other, 1., 1.)
    }
}

impl std::ops::Sub for SparseMatrix {
    type Output = Self;
    fn sub(self, other: SparseMatrix) -> SparseMatrix {
        add(&self, &other, 1., -1.)
    }
}

impl std::ops::Sub<&SparseMatrix> for SparseMatrix {
    type Output = Self;
    fn sub(self, other: &SparseMatrix) -> SparseMatrix {
        add(&self, other, 1., -1.)
    }
}

impl std::ops::Sub for &SparseMatrix {
    type Output = SparseMatrix;
    fn sub(self, other: &SparseMatrix) -> SparseMatrix {
        add(self, other, 1., -1.)
    }
}

impl std::ops::Sub<SparseMatrix> for &SparseMatrix {
    type Output = SparseMatrix;
    fn sub(self, other: SparseMatrix) -> SparseMatrix {
        add(self, &other, 1., -1.)
    }
}

impl std::ops::Mul for SparseMatrix {
    type Output = Self;
    fn mul(self, other: SparseMatrix) -> SparseMatrix {
        mult(&self, &other)
    }
}

impl std::ops::Mul<&SparseMatrix> for SparseMatrix {
    type Output = Self;
    fn mul(self, other: &SparseMatrix) -> SparseMatrix {
        mult(&self, other)
    }
}

impl std::ops::Mul for &SparseMatrix {
    type Output = SparseMatrix;
    fn mul(self, other: &SparseMatrix) -> SparseMatrix {
        mult(self, other)
    }
}

impl std::ops::Mul<SparseMatrix> for &SparseMatrix {
    type Output = SparseMatrix;
    fn mul(self, other: SparseMatrix) -> SparseMatrix {
        mult(self, &other)
    }
}

impl std::ops::Add<f64> for SparseMatrix {
    type Output = Self;
    fn add(self, other: f64) -> SparseMatrix {
        self.scpmat(other)
    }
}

impl std::ops::Add<f64> for &SparseMatrix {
    type Output = SparseMatrix;
    fn add(self, other: f64) -> SparseMatrix {
        self.scpmat(other)
    }
}

impl std::ops::Sub<f64> for SparseMatrix {
    type Output = Self;
    fn sub(self, other: f64) -> SparseMatrix {
        self.scpmat(-other)
    }
}

impl std::ops::Sub<f64> for &SparseMatrix {
    type Output = SparseMatrix;
    fn sub(self, other: f64) -> SparseMatrix {
        self.scpmat(-other)
    }
}

impl std::ops::Mul<f64> for SparseMatrix {
    type Output = Self;
    fn mul(self, other: f64) -> SparseMatrix {
        self.scxmat(other)
    }
}

impl std::ops::Mul<f64> for &SparseMatrix {
    type Output = SparseMatrix;
    fn mul(self, other: f64) -> SparseMatrix {
        self.scxmat(other)
    }
}

impl std::ops::Div<f64> for SparseMatrix {
    type Output = Self;
    fn div(self, other: f64) -> SparseMatrix {
        self.scxmat(other.recip())
    }
}

impl std::ops::Div<f64> for &SparseMatrix {
    type Output = SparseMatrix;
    fn div(self, other: f64) -> SparseMatrix {
        self.scxmat(other.recip())
    }
}

impl std::ops::Add<SparseMatrix> for f64 {
    type Output = SparseMatrix;
    fn add(self, other: SparseMatrix) -> SparseMatrix {
        other.scpmat(self)
    }
}

impl std::ops::Add<&SparseMatrix> for f64 {
    type Output = SparseMatrix;
    fn add(self, other: &SparseMatrix) -> SparseMatrix {
        other.scpmat(self)
    }
}

impl std::ops::Sub<SparseMatrix> for f64 {
    type Output = SparseMatrix;
    fn sub(self, other: SparseMatrix) -> SparseMatrix {
        other.scpmat(-self)
    }
}

impl std::ops::Sub<&SparseMatrix> for f64 {
    type Output = SparseMatrix;
    fn sub(self, other: &SparseMatrix) -> SparseMatrix {
        other.scpmat(-self)
    }
}

impl std::ops::Mul<SparseMatrix> for f64 {
    type Output = SparseMatrix;
    fn mul(self, other: SparseMatrix) -> SparseMatrix {
        self * &other
    }
}

impl std::ops::Mul<&SparseMatrix> for f64 {
    type Output = SparseMatrix;
    fn mul(self, other: &SparseMatrix) -> SparseMatrix {
        other.scxmat(self as f64)
    }
}

impl fmt::Display for SparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let m = self.nrows;
        let n = self.ncols;
        let nzmax = self.values.len();

        write!(
            f,
            "{}-by-{}, nzmax: {} nnz: {}, 1-norm: {}\n",
            m,
            n,
            nzmax,
            self.col_idx[n],
            self.norm()
        )?;

        for j in 0..n {
            write!(
                f,
                "      col {} : locations {} to {}\n",
                j,
                self.col_idx[j],
                self.col_idx[j + 1] - 1
            )?;

            for p in self.col_idx[j]..self.col_idx[j + 1] {
                write!(
                    f,
                    "            {} : {}\n",
                    self.row_idx[p as usize], self.values[p as usize]
                )?;

                if p > 20 {
                    if false {
                        write!(f, "  ...\n")?;
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }
}

pub fn add(a: &SparseMatrix, b: &SparseMatrix, alpha: f64, beta: f64) -> SparseMatrix {
    let mut nz = 0;
    let m = a.nrows();
    let n = b.ncols();
    let anz = a.column_idx()[n] as usize;
    let bnz = b.column_idx()[n] as usize;
    let mut w = vec![0; m];
    let mut x = vec![0.0; m];
    let mut c = SparseMatrix::zeros(m, n, anz + bnz);

    for j in 0..n {
        c.col_idx[j] = nz as isize;
        nz = scatter(a, j, alpha, &mut w[..], &mut x[..], j + 1, &mut c, nz);
        nz = scatter(b, j, beta, &mut w[..], &mut x[..], j + 1, &mut c, nz);

        for p in c.col_idx[j] as usize..nz {
            c.values[p] = x[c.row_idx()[p]];
        }
    }
    c.col_idx[n] = nz as isize;

    c.quick_trim();

    c
}

pub(crate) fn mult(a: &SparseMatrix, b: &SparseMatrix) -> SparseMatrix {
    let mut nz = 0;
    let mut w = vec![0; a.nrows];
    let mut x = vec![0.0; a.nrows];

    let mut c = SparseMatrix::zeros(
        a.nrows,
        b.ncols(),
        2 * (a.col_idx[a.ncols()] + b.col_idx[b.ncols()]) as usize + a.nrows,
    );

    for (j, col_iter) in b.iter().enumerate() {
        if nz + a.nrows() > c.values.len() {
            let nsz = 2 * c.values.len() + a.nrows();
            c.row_idx.resize(nsz, 0);
            c.values.resize(nsz, 0.);
        }
        c.col_idx[j] = nz as isize;
        if let Some(row_iter) = col_iter {
            for (i, b_val) in row_iter {
                nz = scatter(a, i, b_val, &mut w[..], &mut x[..], j + 1, &mut c, nz);
            }
        }
        for p in c.col_idx[j] as usize..nz {
            c.values[p] = x[c.row_idx[p]];
        }
    }

    c.col_idx[b.ncols()] = nz as isize;
    c.quick_trim();

    c
}

//pub(crate) fn mult2(lhs: &SparseMatrix, rhs: &SparseMatrix) -> SparseMatrix {
//    let lhs_iter = lhs.iter();
//    let rhs_iter = rhs.iter();
//
//    let it = lhs_iter.zip(rhs_iter);
//
//    let res = it.for_each(|(lhs, rhs)| { //Iterate over the rows
//         //match (lhs,rhs) {
//         //    (None, None) => None,
//         //    (None, Some(rhs)) => todo!(),
//         //    (Some(lhs), None) => todo!(),
//         //    (Some(lhs), Some(rhs)) => todo!(),
//         //}
//    });
//
//    SparseMatrix::default()
//}

/// An iterator over the columns of a sparse matrix.
///
/// This iterator yields `Option<SparseRowIter>`, where each `SparseRowIter` iterates
/// over the non-zero elements in the corresponding column. If a column has no
/// non-zero elements, `None` is returned for that column.
pub struct SparseColIter<'a> {
    idx: usize,
    iterable: &'a SparseMatrix,
}

impl<'a> Iterator for SparseColIter<'a> {
    type Item = Option<SparseRowIter<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.iterable.ncols() {
            None
        } else {
            let start = self.iterable.col_idx[self.idx];
            let end = self.iterable.col_idx[self.idx + 1];
            self.idx += 1;
            if start == end {
                Some(None)
            } else {
                Some(Some(SparseRowIter {
                    row_idx: self.iterable.row_idx[start as usize..end as usize].iter(),
                    values: self.iterable.values[start as usize..end as usize].iter(),
                }))
            }
        }
    }
}

/// An iterator over the non-zero elements in a col of a sparse matrix.
///
/// This iterator yields `(usize, f64)` tuples, where the first element is the row index
/// and the second element is the value of the non-zero element in that row.
pub struct SparseRowIter<'a> {
    row_idx: std::slice::Iter<'a, usize>,
    values: std::slice::Iter<'a, f64>,
}

impl<'a> Iterator for SparseRowIter<'a> {
    type Item = (usize, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(&row) = self.row_idx.next() {
            if let Some(&value) = self.values.next() {
                Some((row, value))
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests;
