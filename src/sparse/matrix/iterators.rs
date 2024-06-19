use super::*;

/// An Iterator over the values in a Sparse Matrix
pub struct SparseIter<'a, T> {
    matrix: &'a SparseMatrix<T>,
    current_col: usize,
    current_pos: usize,
}

impl<'a, T> SparseIter<'a, T> {
    pub fn new(matrix: &'a SparseMatrix<T>) -> Self {
        SparseIter {
            matrix,
            current_col: 0,
            current_pos: 0,
        }
    }
}

impl<'a, T> Iterator for SparseIter<'a, T>
where
    T: Copy + Default,
{
    type Item = (usize, usize, T); // (col, row, value)

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_col < self.matrix.ncols {
            let col_end = self.matrix.col_ptr[self.current_col + 1];

            if self.current_pos < col_end as usize {
                let row = self.matrix.row_idx[self.current_pos];
                let value = self.matrix.values[self.current_pos];
                let result = (self.current_col, row, value);
                self.current_pos += 1;
                return Some(result);
            }

            self.current_col += 1;
            self.current_pos = self.matrix.col_ptr[self.current_col as usize] as usize;
        }

        None
    }
}

impl<T> FromIterator<(usize, usize, T)> for SparseMatrix<T>
where
    T: Copy + Default,
{
    fn from_iter<I: IntoIterator<Item = (usize, usize, T)>>(iter: I) -> Self {
        let mut elements: Vec<(usize, usize, T)> = iter.into_iter().collect();
        elements.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1))); // Sort by (col, row)

        let nrows = elements.iter().map(|(_, row, _)| *row).max().unwrap_or(0) + 1;
        let ncols = elements.iter().map(|(col, _, _)| *col).max().unwrap_or(0) + 1;

        let mut col_ptr: Vec<isize> = vec![0; ncols + 1];
        let mut row_idx = Vec::with_capacity(elements.len());
        let mut values = Vec::with_capacity(elements.len());

        let mut current_col = 0;
        for (col, row, value) in elements {
            while current_col < col {
                col_ptr[current_col + 1] = row_idx.len() as isize;
                current_col += 1;
            }
            row_idx.push(row);
            values.push(value);
        }
        col_ptr[current_col + 1] = row_idx.len() as isize;

        SparseMatrix {
            nrows,
            ncols,
            col_ptr,
            row_idx,
            values,
        }
    }
}
