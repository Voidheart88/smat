/// A collection of iterators for sparse matrices
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

/// An Iterator over the values in the lower triangular part of a Sparse Matrix
pub struct LowerTriangularSparseIter<'a, T> {
    matrix: &'a SparseMatrix<T>,
    current_col: usize,
    current_pos: usize,
}

impl<'a, T> LowerTriangularSparseIter<'a, T> {
    pub fn new(matrix: &'a SparseMatrix<T>) -> Self {
        LowerTriangularSparseIter {
            matrix,
            current_col: 0,
            current_pos: 0,
        }
    }
}

impl<'a, T> Iterator for LowerTriangularSparseIter<'a, T>
where
    T: Copy + Default,
{
    type Item = (usize, usize, T); // (col, row, value)

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_col < self.matrix.ncols {
            let col_end = self.matrix.col_ptr[self.current_col + 1];

            while self.current_pos < col_end as usize {
                let row = self.matrix.row_idx[self.current_pos];
                let value = self.matrix.values[self.current_pos];
                self.current_pos += 1;

                if row >= self.current_col {
                    return Some((self.current_col, row, value));
                }
            }

            self.current_col += 1;
            self.current_pos = self.matrix.col_ptr[self.current_col as usize] as usize;
        }

        None
    }
}

/// An Iterator over the values in the upper triangular part of a Sparse Matrix excluding the main diagonal
pub struct UpperTriangularSparseIter<'a, T> {
    matrix: &'a SparseMatrix<T>,
    current_col: usize,
    current_pos: usize,
}

impl<'a, T> UpperTriangularSparseIter<'a, T> {
    pub fn new(matrix: &'a SparseMatrix<T>) -> Self {
        UpperTriangularSparseIter {
            matrix,
            current_col: 0,
            current_pos: 0,
        }
    }
}

impl<'a, T> Iterator for UpperTriangularSparseIter<'a, T>
where
    T: Copy + Default,
{
    type Item = (usize, usize, T); // (col, row, value)

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_col < self.matrix.ncols {
            let col_end = self.matrix.col_ptr[self.current_col + 1];

            while self.current_pos < col_end as usize {
                let row = self.matrix.row_idx[self.current_pos];
                let value = self.matrix.values[self.current_pos];
                self.current_pos += 1;

                if row < self.current_col {
                    return Some((self.current_col, row, value));
                }
            }

            self.current_col += 1;
            self.current_pos = self.matrix.col_ptr[self.current_col as usize] as usize;
        }

        None
    }
}

/// An Iterator returning the pivots
pub struct SparsePivotIter<'a, T> {
    matrix: &'a SparseMatrix<T>,
    pivot: usize
}

impl<'a, T> SparsePivotIter<'a, T> {
    pub fn new(matrix: &'a SparseMatrix<T>) -> Self {
        SparsePivotIter {
            matrix,
            pivot: 0,
        }
    }
}

impl<'a, T> Iterator for SparsePivotIter<'a, T>
where
    T: Copy + Default,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pivot >= self.matrix.ncols || self.pivot >= self.matrix.nrows {
            return None
        }
        let pivot =  self.pivot;
        self.pivot += 1;
        Some(pivot)
    }
}

impl<T> FromIterator<(usize, usize, T)> for SparseMatrix<T>
where
    T: Copy + Default,
{
    fn from_iter<I: IntoIterator<Item = (usize, usize, T)>>(iter: I) -> Self {
        let mut elements: Vec<(usize, usize, T)> = iter.into_iter().collect();
        elements.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1))); // Sort by (col, row)

        // Berechne die Anzahl der Zeilen und Spalten
        let nrows = elements.iter().map(|&(_, row, _)| row).max().unwrap_or(0) + 1;
        let ncols = elements.iter().map(|&(col, _, _)| col).max().unwrap_or(0) + 1;

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
        while current_col < ncols {
            col_ptr[current_col + 1] = row_idx.len() as isize;
            current_col += 1;
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
