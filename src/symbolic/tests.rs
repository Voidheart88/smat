use crate::sparse::matrix::SparseMatrix;

use super::*;

#[test]
fn test_new_symbolic() {
    let mat: SparseMatrix<f64> = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 5.0, 0.0],
        vec![0.0, 0.0, 9.0],
    ]
    .into();

    let _: Symbolic<f64> = (&mat).into();
}

#[test]
fn test_new_symbolic2() {
    let mat: SparseMatrix<i64> = vec![vec![1, 0, 0], vec![0, 5, 0], vec![0, 0, 9]].into();

    let _: Symbolic<i64> = (&mat).into();
}

#[test]
fn test_symmetric() {
    let mat: SparseMatrix<i64> = vec![vec![1, 0, 0], vec![0, 5, 0], vec![0, 0, 9]].into();

    let mut sym: Symbolic<i64> = (&mat).into();
    assert!(sym.is_symmetric())
}

#[test]
fn test_dense() {
    let mat: SparseMatrix<i64> = vec![vec![1, 0, 0], vec![0, 5, 0], vec![0, 0, 9]].into();

    let mut sym: Symbolic<i64> = (&mat).into();
    assert!(sym.is_dense())
}

#[test]
fn test_dense2() {
    let mat: SparseMatrix<i64> = vec![
        vec![1, 0, 0, 0],
        vec![0, 5, 0, 0],
        vec![0, 0, 9, 0],
        vec![0, 0, 0, 16],
    ]
    .into();

    let mut sym: Symbolic<i64> = (&mat).into();
    assert!(sym.is_dense())
}
