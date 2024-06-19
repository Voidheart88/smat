use crate::SparseMatrix;

#[derive(Clone, Debug)]
pub struct LuNmrc<T> {
    pub l: SparseMatrix<T>,
    pub u: SparseMatrix<T>,
    pub pivot: Option<Vec<isize>>,
}

impl<T> LuNmrc<T> {
    /// Initializes to empty struct
    ///
    pub fn new() -> LuNmrc<T> {
        LuNmrc {
            l: SparseMatrix::default(),
            u: SparseMatrix::default(),
            pivot: None,
        }
    }
}

impl<T> Default for LuNmrc<T> {
    fn default() -> Self {
        Self::new()
    }
}
