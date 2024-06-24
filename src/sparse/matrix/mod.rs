mod iterators;

use crate::{sparse::utils::scatter, triple::Triples};
pub use iterators::*;

/// Matrix in compressed sparse column (CSC) format
#[derive(Clone, Debug)]
pub struct SparseMatrix<T> {
    nrows: usize,        // m
    ncols: usize,        // n
    col_ptr: Vec<isize>, // p
    row_idx: Vec<usize>, // i
    values: Vec<T>,      // x
}

impl<T> SparseMatrix<T>
where
    T: Copy
        + Default
        + PartialEq
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    pub fn new(
        nrows: usize,
        ncols: usize,
        col_ptr: Vec<isize>,
        row_idx: Vec<usize>,
        values: Vec<T>,
    ) -> SparseMatrix<T> {
        SparseMatrix {
            nrows,
            ncols,
            col_ptr,
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
    pub fn col_ptr(&self) -> &Vec<isize> {
        &self.col_ptr
    }

    #[inline]
    pub fn col_ptr_mut(&mut self) -> &mut Vec<isize> {
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
        let col_ptr = (0..=n).map(|i| i as isize).collect();
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
            .find(|&(i, j)| self.row_idx[i as usize] == row && j == column)
            .map(|(i, _)| self.values[i as usize])
    }

    pub fn permute(&self, row_perm: &[usize], col_perm: &[usize]) -> SparseMatrix<T> {
        assert_eq!(row_perm.len(), self.nrows, "Row permutation vector length must match number of rows");
        assert_eq!(col_perm.len(), self.ncols, "Column permutation vector length must match number of columns");

        let mut new_col_ptr = vec![0; self.ncols + 1];
        let mut new_row_idx = Vec::with_capacity(self.row_idx.len());
        let mut new_values = Vec::with_capacity(self.values.len());

        // Create a mapping from old row indices to new row indices based on the row permutation vector
        let mut row_perm_map = vec![0; self.nrows];
        for (new_idx, &old_idx) in row_perm.iter().enumerate() {
            row_perm_map[old_idx] = new_idx;
        }

        // Create a mapping from old column indices to new column indices based on the column permutation vector
        let mut col_perm_map = vec![0; self.ncols];
        for (new_idx, &old_idx) in col_perm.iter().enumerate() {
            col_perm_map[old_idx] = new_idx;
        }

        // Build the new column pointers, new row indices, and new values based on the permutations
        let mut entries: Vec<(usize, usize, T)> = Vec::new();
        for col in 0..self.ncols {
            for idx in self.col_ptr[col] as usize..self.col_ptr[col + 1] as usize {
                let old_row = self.row_idx[idx];
                let new_row = row_perm_map[old_row];
                let new_col = col_perm_map[col];
                entries.push((new_col, new_row, self.values[idx]));
            }
        }

        entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1))); // Sort by (col, row)

        for &(new_col, new_row, value) in &entries {
            while new_col >= new_col_ptr.len() - 1 {
                new_col_ptr.push(new_row_idx.len() as isize);
            }
            new_row_idx.push(new_row);
            new_values.push(value);
        }
        let len = new_col_ptr.len();
        new_col_ptr[len - 1] = new_row_idx.len() as isize;

        SparseMatrix {
            nrows: self.nrows,
            ncols: self.ncols,
            col_ptr: new_col_ptr,
            row_idx: new_row_idx,
            values: new_values,
        }
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
                self.col_ptr[j] -= num_removed as isize;
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
    fn scale(&mut self, factor: T) {
        for value in &mut self.values {
            *value = *value * factor;
        }
    }

    /// Creates a transpose of the matrix
    pub fn transpose(&self) -> SparseMatrix<T> {
        let mut entries: Vec<(usize, usize, T)> = self.iter().collect();
        entries.sort_by_key(|&(col, row, _)| (row, col));

        let mut col_ptr = vec![0; self.nrows + 1];
        let mut row_idx = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());

        for &(col, row, value) in &entries {
            col_ptr[row + 1] += 1;
            row_idx.push(col);
            values.push(value);
        }

        for i in 1..=self.nrows {
            col_ptr[i] += col_ptr[i - 1];
        }

        SparseMatrix {
            nrows: self.ncols,
            ncols: self.nrows,
            col_ptr,
            row_idx,
            values,
        }
    }

    /// Converts the current sparse matrix to its weighted adjacency matrix representation.
    pub fn to_weighted_adjacency_matrix(&self) -> SparseMatrix<T> {
        let mut adj_col_ptr: Vec<isize> = vec![0; self.ncols + 1];
        let mut adj_row_idx = Vec::with_capacity(self.row_idx.len());
        let mut adj_values = Vec::with_capacity(self.values.len());

        for col in 0..self.ncols {
            adj_col_ptr[col] = adj_row_idx.len() as isize;
            for idx in self.col_ptr[col]..self.col_ptr[col + 1] {
                let row = self.row_idx[idx as usize];
                if row != col && self.values[idx as usize] != T::default() {
                    adj_row_idx.push(row);
                    adj_values.push(self.values[idx as usize]); // Using the actual weight of the edge
                }
            }
        }
        adj_col_ptr[self.ncols] = adj_row_idx.len() as isize;

        SparseMatrix {
            nrows: self.nrows,
            ncols: self.ncols,
            col_ptr: adj_col_ptr,
            row_idx: adj_row_idx,
            values: adj_values,
        }
    }

    /// Sets the value at the specified row and column
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        for idx in self.col_ptr[col]..self.col_ptr[col + 1] {
            if self.row_idx[idx as usize] == row {
                self.values[idx as usize] = value;
                return;
            }
        }
    }
}

impl<T> std::ops::Mul<T> for SparseMatrix<T>
where
    T: Copy
        + Default
        + PartialEq
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
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
    T: Copy
        + Default
        + PartialEq
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
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

        let mut col_ptr: Vec<isize> = vec![0; self.ncols + 1];
        let mut row_idx = Vec::new();
        let mut values = Vec::new();

        for col in 0..self.ncols {
            let mut self_pos = self.col_ptr[col];
            let self_end = self.col_ptr[col + 1];
            let mut rhs_pos = rhs.col_ptr[col];
            let rhs_end = rhs.col_ptr[col + 1];

            while self_pos < self_end || rhs_pos < rhs_end {
                if self_pos < self_end
                    && (rhs_pos >= rhs_end
                        || self.row_idx[self_pos as usize] < rhs.row_idx[rhs_pos as usize])
                {
                    row_idx.push(self.row_idx[self_pos as usize]);
                    values.push(self.values[self_pos as usize]);
                    self_pos += 1;
                } else if rhs_pos < rhs_end
                    && (self_pos >= self_end
                        || rhs.row_idx[rhs_pos as usize] < self.row_idx[self_pos as usize])
                {
                    row_idx.push(rhs.row_idx[rhs_pos as usize]);
                    values.push(rhs.values[rhs_pos as usize]);
                    rhs_pos += 1;
                } else {
                    row_idx.push(self.row_idx[self_pos as usize]);
                    values.push(self.values[self_pos as usize] + rhs.values[rhs_pos as usize]);
                    self_pos += 1;
                    rhs_pos += 1;
                }
            }
            col_ptr[col + 1] = row_idx.len() as isize;
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
    T: Copy + std::ops::Add<Output = T>,
{
    type Output = SparseMatrix<T>;

    fn add(self, rhs: SparseMatrix<T>) -> Self::Output {
        (&self) + (&rhs)
    }
}

impl<T> std::ops::Add<&SparseMatrix<T>> for SparseMatrix<T>
where
    T: Copy + std::ops::Add<Output = T>,
{
    type Output = SparseMatrix<T>;

    fn add(self, rhs: &SparseMatrix<T>) -> Self::Output {
        &self + rhs
    }
}

impl<T> std::ops::Add<SparseMatrix<T>> for &SparseMatrix<T>
where
    T: Copy + std::ops::Add<Output = T>,
{
    type Output = SparseMatrix<T>;

    fn add(self, rhs: SparseMatrix<T>) -> Self::Output {
        self + &rhs
    }
}

impl<T> std::ops::Mul<&SparseMatrix<T>> for &SparseMatrix<T>
where
    T: Copy
        + Default
        + std::ops::AddAssign
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + PartialEq,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: &SparseMatrix<T>) -> Self::Output {
        let mut non_zero_count = 0;
        let mut row_marker = vec![0; self.nrows()];
        let mut row_values = vec![T::default(); self.nrows()];

        let required_space = 2
            * (self.col_ptr().last().copied().unwrap_or(0)
                + rhs.col_ptr().last().copied().unwrap_or(0))
            + self.nrows() as isize;

        let mut result: SparseMatrix<T> =
            SparseMatrix::zeros(self.nrows(), rhs.ncols(), required_space as usize);

        for col in 0..rhs.ncols() {
            if non_zero_count + self.nrows() > result.values().len() {
                let new_size = 2 * result.values().len() + self.nrows();
                result.row_idx_mut().resize(new_size, 0);
                result.values_mut().resize(new_size, T::default());
            }
            result.col_ptr_mut()[col] = non_zero_count as isize;
            for p in rhs.col_ptr()[col]..rhs.col_ptr()[col + 1] {
                non_zero_count = scatter(
                    self,
                    rhs.row_idx()[p as usize],
                    rhs.values()[p as usize],
                    &mut row_marker[..],
                    &mut row_values[..],
                    col + 1,
                    &mut result,
                    non_zero_count,
                );
            }
            for p in result.col_ptr()[col] as usize..non_zero_count {
                result.values_mut()[p] = row_values[result.row_idx()[p]];
            }
        }
        result.col_ptr_mut()[rhs.ncols()] = non_zero_count as isize;
        result.quick_trim();

        result
    }
}

impl<T> std::ops::Mul<SparseMatrix<T>> for SparseMatrix<T>
where
    T: Copy
        + Default
        + std::ops::AddAssign
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + PartialEq,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: SparseMatrix<T>) -> Self::Output {
        &self * &rhs
    }
}

impl<T> std::ops::Mul<&SparseMatrix<T>> for SparseMatrix<T>
where
    T: Copy
        + Default
        + std::ops::AddAssign
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + PartialEq,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: &SparseMatrix<T>) -> Self::Output {
        &self * rhs
    }
}

impl<T> std::ops::Mul<SparseMatrix<T>> for &SparseMatrix<T>
where
    T: Copy
        + Default
        + std::ops::AddAssign
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + PartialEq,
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
        let mut col_ptr: Vec<isize> = vec![0; ncols + 1];
        let mut row_idx = vec![];
        let mut values = vec![];

        for col in 0..ncols {
            for row in 0..nrows {
                if dense[row][col] != T::default() {
                    row_idx.push(row);
                    values.push(dense[row][col]);
                }
            }
            col_ptr[col + 1] = values.len() as isize;
        }

        SparseMatrix {
            nrows,
            ncols,
            col_ptr,
            row_idx,
            values,
        }
    }
}

impl From<&Triples<f64>> for SparseMatrix<f64> {
    fn from(triples: &Triples<f64>) -> Self {
        let mut col_ptr: Vec<isize> = vec![0; triples.ncols() + 1];
        let mut row_idx = vec![0; triples.values().len()];
        let mut values = vec![0.0; triples.values().len()];

        let mut count = vec![0; triples.ncols()];

        for &col in triples.column_idx() {
            count[col] += 1;
        }

        for i in 0..triples.ncols() {
            col_ptr[i + 1] = col_ptr[i] + count[i];
        }

        let mut next = col_ptr.clone();

        for i in 0..triples.values().len() {
            let col = triples.column_idx()[i];
            let dest = next[col];

            row_idx[dest as usize] = triples.row_idx()[i];
            values[dest as usize] = triples.values()[i];

            next[col] += 1;
        }

        SparseMatrix {
            nrows: triples.nrows(),
            ncols: triples.ncols(),
            col_ptr,
            row_idx,
            values,
        }
    }
}

impl<T> Default for SparseMatrix<T> {
    fn default() -> Self {
        Self {
            nrows: Default::default(),
            ncols: Default::default(),
            col_ptr: Default::default(),
            row_idx: Default::default(),
            values: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests;
