mod iterators;

use crate::triple::Triples;
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
    T: Copy + Default + PartialEq + PartialOrd + std::ops::Mul<Output = T>,
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
    pub fn column_ptr(&self) -> &Vec<usize> {
        &self.col_ptr
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
    pub fn iter(&self) -> SparseColIter<T> {
        SparseColIter::new(0, self)
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
    T: Copy + Default + PartialEq + PartialOrd + std::ops::Mul<Output = T>,
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
    T: Copy + Default + PartialEq + PartialOrd + std::ops::Mul<Output = T>,
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
    T: Copy + Default + PartialOrd + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    type Output = SparseMatrix<T>;

    fn add(self, rhs: &SparseMatrix<T>) -> Self::Output {
        let self_col_iter = SparseColIter::new(0, &self);
        let rhs_col_iter = SparseColIter::new(0, &rhs);

        let mut col_ptr = Vec::new();
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        let mut col_index = 0;

        for (self_col, rhs_col) in self_col_iter.zip(rhs_col_iter) {
            match (self_col, rhs_col) {
                (Some(mut self_iter), Some(mut rhs_iter)) => {
                    let mut rows = Vec::new();
                    let mut vals = Vec::new();

                    let mut self_next = self_iter.next();
                    let mut rhs_next = rhs_iter.next();

                    while let (Some(self_val), Some(rhs_val)) = (self_next, rhs_next) {
                        match self_val.0.cmp(&rhs_val.0) {
                            std::cmp::Ordering::Less => {
                                rows.push(self_val.0);
                                vals.push(self_val.1);
                                self_next = self_iter.next();
                            }
                            std::cmp::Ordering::Greater => {
                                rows.push(rhs_val.0);
                                vals.push(rhs_val.1);
                                rhs_next = rhs_iter.next();
                            }
                            std::cmp::Ordering::Equal => {
                                rows.push(self_val.0);
                                vals.push(self_val.1 + rhs_val.1);
                                self_next = self_iter.next();
                                rhs_next = rhs_iter.next();
                            }
                        }
                    }

                    while let Some(self_val) = self_next {
                        rows.push(self_val.0);
                        vals.push(self_val.1);
                        self_next = self_iter.next();
                    }

                    while let Some(rhs_val) = rhs_next {
                        rows.push(rhs_val.0);
                        vals.push(rhs_val.1);
                        rhs_next = rhs_iter.next();
                    }

                    for (row, val) in rows.into_iter().zip(vals.into_iter()) {
                        row_idx.push(row);
                        values.push(val);
                    }
                }
                (Some(mut self_iter), None) | (None, Some(mut self_iter)) => {
                    while let Some(self_val) = self_iter.next() {
                        row_idx.push(self_val.0);
                        values.push(self_val.1);
                    }
                }
                (None, None) => {}
            }
            col_ptr.push(col_index);
            col_index = values.len();
        }

        col_ptr.push(col_index);

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
    T: Copy + Default + PartialOrd + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    type Output = SparseMatrix<T>;

    fn add(self, rhs: SparseMatrix<T>) -> Self::Output {
        (&self) + (&rhs)
    }
}

impl<T> std::ops::Add<&SparseMatrix<T>> for SparseMatrix<T>
where
    T: Copy + Default + PartialOrd + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    type Output = SparseMatrix<T>;

    fn add(self, rhs: &SparseMatrix<T>) -> Self::Output {
        &self + rhs
    }
}

impl<T> std::ops::Add<SparseMatrix<T>> for &SparseMatrix<T>
where
    T: Copy + Default + PartialOrd + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    type Output = SparseMatrix<T>;

    fn add(self, rhs: SparseMatrix<T>) -> Self::Output {
        self + &rhs
    }
}

impl<T> std::ops::Mul<&SparseMatrix<T>> for &SparseMatrix<T>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T> + PartialOrd,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: &SparseMatrix<T>) -> Self::Output {

        // n = ncols
        // m = nrows
        // p = col_ptr
        // i = row_idx
        // x = val

        let mut nz = 0;
        let mut w = vec![0; self.nrows()];
        let mut x = vec![0.0; self.nrows()];

        for column in rhs.iter() {

        }

        todo!();
    }
}

impl<T> std::ops::Mul<SparseMatrix<T>> for SparseMatrix<T>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T> + PartialOrd,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: SparseMatrix<T>) -> Self::Output {
        &self * &rhs
    }
}

impl<T> std::ops::Mul<&SparseMatrix<T>> for SparseMatrix<T>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T> + PartialOrd,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: &SparseMatrix<T>) -> Self::Output {
        &self * rhs
    }
}

impl<T> std::ops::Mul<SparseMatrix<T>> for &SparseMatrix<T>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T> + PartialOrd,
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
    T: Copy + Default + PartialEq + PartialOrd,
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
