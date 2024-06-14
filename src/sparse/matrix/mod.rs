mod iterators;

use crate::{sparse::utils::scatter, triple::Triples};
pub use iterators::*;

/// Matrix in compressed sparse column (CSC) format
#[derive(Clone, Debug)]
pub struct SparseMatrix<T> {
    nrows: usize,
    ncols: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<T>,
}

impl<T> SparseMatrix<T>
where
    T: Copy + Default + PartialEq + std::ops::Mul<Output = T>,
{
    pub fn new(
        nrows: usize,
        ncols: usize,
        col_idx: Vec<usize>,
        row_idx: Vec<usize>,
        values: Vec<T>,
    ) -> SparseMatrix<T> {
        SparseMatrix {
            nrows,
            ncols,
            col_ptr: col_idx,
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
    pub fn col_ptr(&self) -> &Vec<usize> {
        &self.col_ptr
    }

    #[inline]
    pub fn col_ptr_mut(&mut self) -> &mut Vec<usize> {
        &mut self.col_ptr
    }

    #[inline]
    pub fn row_idx(&self) -> &Vec<usize> {
        &self.row_idx
    }

    #[inline]
    pub fn row_idx_mut(&mut self) -> &mut Vec<usize> {
        &mut self.row_idx
    }

    #[inline]
    pub fn values(&self) -> &Vec<T> {
        &self.values
    }

    #[inline]
    pub fn values_mut(&mut self) -> &mut Vec<T> {
        &mut self.values
    }

    /// Create a new Sparse Matrix filled with Zeros
    /// TODO: Matrix generation should not use this function -- reduce usage
    pub fn zeros(m: usize, n: usize, nzmax: usize) -> SparseMatrix<T> {
        SparseMatrix {
            nrows: m,
            ncols: n,
            col_ptr: vec![0; n + 1],
            row_idx: vec![0; nzmax],
            values: vec![T::default(); nzmax],
        }
    }

    /// Create a sparse eye matrix
    pub fn eye(val: T, n: usize) -> SparseMatrix<T> {
        let col_ptr = (0..=n).map(|i| i).collect();
        let row_idx = (0..n).collect();
        let values = vec![val; n];

        SparseMatrix {
            nrows: n,
            ncols: n,
            col_ptr,
            row_idx,
            values,
        }
    }

    /// Get a Value
    pub fn get(&self, row: usize, column: usize) -> Option<T> {
        self.col_ptr
            .iter()
            .zip(self.col_ptr.iter().skip(1))
            .enumerate()
            .flat_map(|(j, (&start, &end))| (start..end).map(move |i| (i, j)))
            .find(|&(i, j)| self.row_idx[i] == row && j == column)
            .map(|(i, _)| self.values[i])
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
            .filter(|&(_, value)| *value == T::default())
            .map(|(index, _)| index)
            .collect();

        for &index in zero_indices.iter().rev() {
            self.values.remove(index);
            self.row_idx.remove(index);
        }

        for j in (0..self.col_ptr.len()).rev() {
            let num_removed = zero_indices
                .iter()
                .filter(|&&index| index < self.col_ptr[j] as usize)
                .count();
            if num_removed > 0 {
                self.col_ptr[j] -= num_removed;
            }
        }
    }

    /// Quickly trims the sparse matrix by resizing the `row_idx` and `values` vectors.
    ///
    /// This method resizes the `row_idx` and `values` vectors based on the maximum number
    /// of non-zero elements (`nzmax`). This is a faster but less precise method than `trim`,
    /// which completely removes zero elements.
    pub fn quick_trim(&mut self) {
        let nzmax = self.col_ptr[self.ncols] as usize;
        self.row_idx.resize(nzmax, 0);
        self.values.resize(nzmax, T::default());
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
    pub fn iter(&self) -> SparseIter<T> {
        SparseIter::new(self)
    }

    /// Scales the Matrix by a constant factor
    ///
    fn scale(&mut self, factor: T) {
        for value in &mut self.values {
            *value = *value * factor;
        }
    }
}

impl<T> std::ops::Mul<T> for SparseMatrix<T>
where
    T: Copy + Default + PartialEq + std::ops::Mul<Output = T>,
{
    type Output = SparseMatrix<T>;
    fn mul(self, rhs: T) -> SparseMatrix<T> {
        let mut res = self.clone();
        res.scale(rhs);
        res
    }
}

impl<T> std::ops::Mul<&T> for SparseMatrix<T>
where
    T: Copy + Default + PartialEq + std::ops::Mul<Output = T>,
{
    type Output = SparseMatrix<T>;
    fn mul(self, rhs: &T) -> SparseMatrix<T> {
        let mut res = self.clone();
        res.scale(*rhs);
        res
    }
}

impl<T> std::ops::Add<&SparseMatrix<T>> for &SparseMatrix<T>
where
    T: Copy + std::ops::Add<Output = T>,
{
    type Output = SparseMatrix<T>;

    fn add(self, rhs: &SparseMatrix<T>) -> Self::Output {
        assert!(self.nrows == rhs.nrows && self.ncols == rhs.ncols);

        let mut col_ptr = vec![0; self.ncols + 1];
        let mut row_idx = Vec::new();
        let mut values = Vec::new();

        for col in 0..self.ncols {
            let mut self_pos = self.col_ptr[col];
            let self_end = self.col_ptr[col + 1];
            let mut rhs_pos = rhs.col_ptr[col];
            let rhs_end = rhs.col_ptr[col + 1];

            while self_pos < self_end || rhs_pos < rhs_end {
                if self_pos < self_end
                    && (rhs_pos >= rhs_end || self.row_idx[self_pos] < rhs.row_idx[rhs_pos])
                {
                    row_idx.push(self.row_idx[self_pos]);
                    values.push(self.values[self_pos]);
                    self_pos += 1;
                } else if rhs_pos < rhs_end
                    && (self_pos >= self_end || rhs.row_idx[rhs_pos] < self.row_idx[self_pos])
                {
                    row_idx.push(rhs.row_idx[rhs_pos]);
                    values.push(rhs.values[rhs_pos]);
                    rhs_pos += 1;
                } else {
                    row_idx.push(self.row_idx[self_pos]);
                    values.push(self.values[self_pos] + rhs.values[rhs_pos]);
                    self_pos += 1;
                    rhs_pos += 1;
                }
            }
            col_ptr[col + 1] = row_idx.len();
        }

        SparseMatrix {
            nrows: self.nrows,
            ncols: self.ncols,
            col_ptr,
            row_idx,
            values,
        }
    }
}

impl<T> std::ops::Add<SparseMatrix<T>> for SparseMatrix<T>
where
    T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    type Output = SparseMatrix<T>;

    fn add(self, rhs: SparseMatrix<T>) -> Self::Output {
        (&self) + (&rhs)
    }
}

impl<T> std::ops::Add<&SparseMatrix<T>> for SparseMatrix<T>
where
    T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    type Output = SparseMatrix<T>;

    fn add(self, rhs: &SparseMatrix<T>) -> Self::Output {
        &self + rhs
    }
}

impl<T> std::ops::Add<SparseMatrix<T>> for &SparseMatrix<T>
where
    T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    type Output = SparseMatrix<T>;

    fn add(self, rhs: SparseMatrix<T>) -> Self::Output {
        self + &rhs
    }
}

impl<T> std::ops::Mul<&SparseMatrix<T>> for &SparseMatrix<T>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T> + PartialEq,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: &SparseMatrix<T>) -> Self::Output {
        let mut nz = 0;
        let mut w = vec![0; self.nrows()];
        let mut x = vec![T::default(); self.nrows()];

        let space = 2
            * (self.col_ptr().last().copied().unwrap_or(0)
                + rhs.col_ptr().last().copied().unwrap_or(0))
            + self.nrows();

        let mut acc: SparseMatrix<T> = SparseMatrix::zeros(self.nrows(), rhs.ncols(), space);

        for j in 0..rhs.ncols() {
            if nz + self.nrows() > acc.values().len() {
                let nsz = 2 * acc.values().len() + self.nrows();
                acc.row_idx_mut().resize(nsz, 0);
                acc.values_mut().resize(nsz, T::default());
            }
            acc.col_ptr_mut()[j] = nz;
            for p in rhs.col_ptr()[j]..rhs.col_ptr()[j + 1] {
                nz = scatter(
                    self,
                    rhs.row_idx()[p],
                    rhs.values()[p],
                    &mut w[..],
                    &mut x[..],
                    j + 1,
                    &mut acc,
                    nz,
                );
            }
            for p in acc.col_ptr()[j] as usize..nz {
                acc.values_mut()[p] = x[acc.row_idx()[p]];
            }
        }
        acc.col_ptr_mut()[rhs.ncols()] = nz;
        acc.quick_trim();

        acc
    }
}

impl<T> std::ops::Mul<SparseMatrix<T>> for SparseMatrix<T>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T> + PartialEq,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: SparseMatrix<T>) -> Self::Output {
        &self * &rhs
    }
}

impl<T> std::ops::Mul<&SparseMatrix<T>> for SparseMatrix<T>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T> + PartialEq,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: &SparseMatrix<T>) -> Self::Output {
        &self * rhs
    }
}

impl<T> std::ops::Mul<SparseMatrix<T>> for &SparseMatrix<T>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T> + PartialEq,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: SparseMatrix<T>) -> Self::Output {
        self * &rhs
    }
}

impl<T> PartialEq for SparseMatrix<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.nrows == other.nrows
            && self.ncols == other.ncols
            && self.col_ptr == other.col_ptr
            && self.row_idx == other.row_idx
            && self.values == other.values
    }
}

impl<T> From<Vec<Vec<T>>> for SparseMatrix<T>
where
    T: Copy + Default + PartialEq,
{
    fn from(dense: Vec<Vec<T>>) -> Self {
        if dense.is_empty() {
            return SparseMatrix {
                nrows: 0,
                ncols: 0,
                col_ptr: vec![0],
                row_idx: vec![],
                values: vec![],
            };
        }

        let nrows = dense.len();
        let ncols = dense[0].len();
        let mut col_idx = vec![0; ncols + 1];
        let mut row_idx = vec![];
        let mut values = vec![];

        for col in 0..ncols {
            for row in 0..nrows {
                if dense[row][col] != T::default() {
                    row_idx.push(row);
                    values.push(dense[row][col]);
                }
            }
            col_idx[col + 1] = values.len();
        }

        SparseMatrix {
            nrows,
            ncols,
            col_ptr: col_idx,
            row_idx,
            values,
        }
    }
}

impl From<&Triples<f64>> for SparseMatrix<f64> {
    fn from(triples: &Triples<f64>) -> Self {
        let mut col_idx = vec![0; triples.ncols() + 1];
        let mut row_idx = vec![0; triples.values().len()];
        let mut values = vec![0.0; triples.values().len()];

        let mut count = vec![0; triples.ncols()];

        for &col in triples.column_idx() {
            count[col] += 1;
        }

        for i in 0..triples.ncols() {
            col_idx[i + 1] = col_idx[i] + count[i];
        }

        let mut next = col_idx.clone();

        for i in 0..triples.values().len() {
            let col = triples.column_idx()[i];
            let dest = next[col];

            row_idx[dest] = triples.row_idx()[i];
            values[dest] = triples.values()[i];

            next[col] += 1;
        }

        SparseMatrix {
            nrows: triples.nrows(),
            ncols: triples.ncols(),
            col_ptr: col_idx,
            row_idx,
            values,
        }
    }
}

#[cfg(test)]
mod tests;
