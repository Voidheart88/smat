use super::*;

/// An iterator over the columns of a sparse matrix.
///
/// This iterator yields `Option<SparseRowIter>`, where each `SparseRowIter` iterates
/// over the non-zero elements in the corresponding column. If a column has no
/// non-zero elements, `None` is returned for that column.
pub struct SparseColIter<'a, T> {
    idx: usize,
    iterable: &'a SparseMatrix<T>,
}

impl<'a,T> SparseColIter<'a,T> {
    pub fn new(idx: usize,iterable: &'a SparseMatrix<T>) -> SparseColIter<'a,T> {
        SparseColIter {
            idx,
            iterable,
        }
    }
}

impl<'a, T> Iterator for SparseColIter<'a, T>
where
    T: Copy + Default + PartialEq + PartialOrd + std::ops::Mul<Output = T>,
{
    type Item = Option<SparseRowIter<'a, T>>;

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
                    row_idx: self.iterable.row_idx[start..end].iter(),
                    values: self.iterable.values[start..end].iter(),
                }))
            }
        }
    }
}

/// An iterator over the non-zero elements in a col of a sparse matrix.
///
/// This iterator yields `(usize, f64)` tuples, where the first element is the row index
/// and the second element is the value of the non-zero element in that row.
pub struct SparseRowIter<'a, T> {
    row_idx: std::slice::Iter<'a, usize>,
    values: std::slice::Iter<'a, T>,
}

impl<'a, T> Iterator for SparseRowIter<'a, T>
where
    T: Copy + Default + PartialEq + PartialOrd + std::ops::Mul<Output = T>,
{
    type Item = (usize, T);

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