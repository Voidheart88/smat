use super::*;

pub struct SparseColIter<'a, T> {
    idx: usize,
    iterable: &'a SparseMatrix<T>,
}

impl<'a, T> SparseColIter<'a, T> {
    pub fn new(idx: usize, iterable: &'a SparseMatrix<T>) -> SparseColIter<'a, T> {
        SparseColIter { idx, iterable }
    }
}

impl<'a, T> Iterator for SparseColIter<'a, T>
where
    T: Copy + Default + PartialOrd + std::ops::Mul<Output = T>,
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

pub struct SparseRowIter<'a, T> {
    row_idx: std::slice::Iter<'a, usize>,
    values: std::slice::Iter<'a, T>,
}

impl<'a, T> Iterator for SparseRowIter<'a, T>
where
    T: Copy,
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

impl<'a, T> FromIterator<Option<SparseRowIter<'a, T>>> for SparseMatrix<T>
where
    T: Copy,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Option<SparseRowIter<'a, T>>>,
    {
        let mut col_idx = Vec::new();
        let mut row_idx = Vec::new();
        let mut values = Vec::new();

        let mut col_index = 0;

        for row_iter in iter.into_iter().flatten() {
            col_idx.push(col_index);

            for (row, value) in row_iter {
                row_idx.push(row);
                values.push(value);
            }

            col_index = values.len();
        }

        col_idx.push(col_index);

        let nrows = row_idx.iter().copied().max().map_or(0, |x| x + 1);
        let ncols = col_idx.len() - 1;

        SparseMatrix {
            nrows,
            ncols,
            col_idx,
            row_idx,
            values,
        }
    }
}
