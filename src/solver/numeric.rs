use crate::SparseMatrix;


/// Numeric Cholesky, LU, or QR factorization
///
#[derive(Clone, Debug)]
pub struct Nmrc<T> {
    /// L for LU and Cholesky, V for QR
    pub l: SparseMatrix<T>,
    /// U for LU, R for QR, not used for Cholesky
    pub u: SparseMatrix<T>,
    /// partial pivoting for LU
    pub pinv: Option<Vec<isize>>,
    /// beta [0..n-1] for QR
    pub b: Vec<f64>,
}

impl<T> Nmrc<T> {
    /// Initializes to empty struct
    ///
    pub fn new() -> Nmrc<T> {
        Nmrc {
            l: SparseMatrix::new(),
            u: SparseMatrix::new(),
            pinv: None,
            b: Vec::new(),
        }
    }
}

impl Default for Nmrc {
    fn default() -> Self {
        Self::new()
    }
}