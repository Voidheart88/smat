use crate::SparseMatrix;

/// Numeric Cholesky, LU, or QR factorization
///
#[derive(Clone, Debug)]
pub struct Numeric<T> {
    /// L for LU and Cholesky, V for QR
    pub l: SparseMatrix<T>,
    /// U for LU, R for QR, not used for Cholesky
    pub u: SparseMatrix<T>,
    /// partial pivoting for LU
    pub pinv: Option<Vec<isize>>,
    /// beta [0..n-1] for QR
    pub b: Vec<f64>,
}

impl<T> Numeric<T> {
    /// Initializes to empty struct
    ///
    pub fn new() -> Numeric<T> {
        Numeric {
            l: SparseMatrix::default(),
            u: SparseMatrix::default(),
            pinv: None,
            b: Vec::default(),
        }
    }
}

impl<T> Default for Numeric<T> {
    fn default() -> Self {
        Self::new()
    }
}
