use crate::SparseMatrix;

/// Adjacency Graph structures and methods

pub(crate) struct AdjGraph {}

impl AdjGraph {}

impl<'a, T> From<&'a SparseMatrix<T>> for AdjGraph {
    fn from(reference: &'a SparseMatrix<T>) -> Self {
        Self {}
    }
}
