use super::*;

#[test]
fn test_zeros_matrix() {
    let m = 3;
    let n = 3;
    let nzmax = 9;
    let matrix: SparseMatrix<f64> = SparseMatrix::zeros(m, n, nzmax);

    assert_eq!(matrix.nrows(), m);
    assert_eq!(matrix.ncols(), n);
    assert_eq!(matrix.col_ptr, vec![0; n + 1]);
    assert_eq!(matrix.row_idx, vec![0; nzmax]);
    assert_eq!(matrix.values, vec![0.0; nzmax]);
}

#[test]
fn test_eye_matrix() {
    let n = 4;
    let matrix: SparseMatrix<f64> = SparseMatrix::eye(1.0, n);

    assert_eq!(matrix.nrows(), n);
    assert_eq!(matrix.ncols(), n);
    assert_eq!(matrix.col_ptr, vec![0, 1, 2, 3, 4]);
    assert_eq!(matrix.row_idx, vec![0, 1, 2, 3]);
    assert_eq!(matrix.values, vec![1.0; n]);
}

#[test]
fn test_get_value() {
    let col_idx = vec![0, 1, 2, 2];
    let row_idx = vec![0, 1];
    let values = vec![1.0, 2.0];
    let matrix: SparseMatrix<f64> = SparseMatrix::new(3, 3, col_idx, row_idx, values);

    assert_eq!(matrix.get(0, 0), Some(1.0));
    assert_eq!(matrix.get(1, 1), Some(2.0));
    assert_eq!(matrix.get(2, 2), None);
}

#[test]
fn test_trim() {
    let mut sparse = SparseMatrix {
        nrows: 3,
        ncols: 3,
        col_ptr: vec![0, 2, 4, 6],
        row_idx: vec![0, 2, 1, 1, 2, 2],
        values: vec![1.0, 0.0, 2.0, 0.0, 5.0, 0.0],
    };

    sparse.trim();

    let expected_sparse = SparseMatrix {
        nrows: 3,
        ncols: 3,
        col_ptr: vec![0, 1, 2, 3],
        row_idx: vec![0, 1, 2],
        values: vec![1.0, 2.0, 5.0],
    };

    assert_eq!(sparse, expected_sparse);
}

#[test]
fn test_quick_trim() {
    let mut sparse = SparseMatrix {
        nrows: 3,
        ncols: 3,
        col_ptr: vec![0, 2, 3, 5],
        row_idx: vec![0, 2, 1, 1, 2, 0, 0, 0, 0, 0],
        values: vec![1.0, 3.0, 2.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };

    sparse.quick_trim();

    let expected_sparse = SparseMatrix {
        nrows: 3,
        ncols: 3,
        col_ptr: vec![0, 2, 3, 5],
        row_idx: vec![0, 2, 1, 1, 2],
        values: vec![1.0, 3.0, 2.0, 4.0, 5.0],
    };

    assert_eq!(sparse, expected_sparse);
}

#[test]
fn test_iter() {
    let col_idx = vec![0, 1, 3, 3];
    let row_idx = vec![0, 1, 2];
    let values = vec![1.0, 2.0, 3.0];
    let matrix: SparseMatrix<f64> = SparseMatrix::new(3, 3, col_idx, row_idx, values);

    let mut col_iter = matrix.iter();
    let mut col_0 = col_iter.next().unwrap().unwrap();
    let mut col_1 = col_iter.next().unwrap().unwrap();
    let col_2 = col_iter.next().unwrap();

    assert_eq!(col_0.next(), Some((0, 1.0)));
    assert_eq!(col_1.next(), Some((1, 2.0)));
    assert_eq!(col_1.next(), Some((2, 3.0)));
    assert!(col_2.is_none());
}

#[test]
fn test_scale() {
    let mut matrix: SparseMatrix<f64> =
        SparseMatrix::new(3, 3, vec![0, 1, 3, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]);

    matrix.scale(2.0);

    assert_eq!(matrix.values, vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_from_dense() {
    let sparse: SparseMatrix<f64> = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0],
        vec![0.0, 3.0, 4.0],
    ]
    .into();

    let expected_sparse = SparseMatrix {
        nrows: 3,
        ncols: 3,
        col_ptr: vec![0, 1, 3, 4],
        row_idx: vec![0, 1, 2, 2],
        values: vec![1.0, 2.0, 3.0, 4.0],
    };

    assert_eq!(sparse, expected_sparse);
}

#[test]
fn test_from_triples() {
    let triples = Triples::new(
        3,
        3,
        vec![0, 0, 1, 2, 2],
        vec![0, 2, 1, 1, 2],
        vec![1.0, 3.0, 2.0, 4.0, 5.0],
    );

    let sparse: SparseMatrix<f64> = (&triples).into();

    let expected_sparse = SparseMatrix {
        nrows: 3,
        ncols: 3,
        col_ptr: vec![0, 2, 3, 5],
        row_idx: vec![0, 2, 1, 1, 2],
        values: vec![1.0, 3.0, 2.0, 4.0, 5.0],
    };

    assert_eq!(sparse.nrows, expected_sparse.nrows);
    assert_eq!(sparse.ncols, expected_sparse.ncols);

    assert_eq!(sparse.col_ptr, expected_sparse.col_ptr);
    assert_eq!(sparse.row_idx, expected_sparse.row_idx);
    assert_eq!(sparse.values, expected_sparse.values);
}

#[test]
fn test_from_iter_identity() {
    let matrix = SparseMatrix {
        nrows: 3,
        ncols: 3,
        col_ptr: vec![0, 2, 4, 5],
        row_idx: vec![0, 2, 1, 2, 1],
        values: vec![1.0, 3.0, 2.0, 4.0, 5.0],
    };

    let result_matrix: SparseMatrix<f64> = matrix.iter().collect();

    assert_eq!(matrix, result_matrix);
}
