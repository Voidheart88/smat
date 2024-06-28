use super::*;
use num::Complex;

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
fn test_zeros_matrix2() {
    let m = 3;
    let n = 3;
    let nzmax = 9;
    let matrix: SparseMatrix<i64> = SparseMatrix::zeros(m, n, nzmax);

    assert_eq!(matrix.nrows(), m);
    assert_eq!(matrix.ncols(), n);
    assert_eq!(matrix.col_ptr, vec![0; n + 1]);
    assert_eq!(matrix.row_idx, vec![0; nzmax]);
    assert_eq!(matrix.values, vec![0; nzmax]);
}

#[test]
fn test_zeros_matrix3() {
    let m = 3;
    let n = 3;
    let nzmax = 9;
    let matrix: SparseMatrix<Complex<f64>> = SparseMatrix::zeros(m, n, nzmax);

    assert_eq!(matrix.nrows(), m);
    assert_eq!(matrix.ncols(), n);
    assert_eq!(matrix.col_ptr, vec![0; n + 1]);
    assert_eq!(matrix.row_idx, vec![0; nzmax]);
    assert_eq!(matrix.values, vec![Complex::default(); nzmax]);
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
fn test_from_dense2() {
    let sparse: SparseMatrix<i64> = vec![vec![1, 2, 3], vec![0, 5, 6], vec![0, 0, 9]].into();

    let expected_sparse = SparseMatrix {
        nrows: 3,
        ncols: 3,
        col_ptr: vec![0, 1, 3, 6],
        row_idx: vec![0, 0, 1, 0, 1, 2],
        values: vec![1, 2, 5, 3, 6, 9],
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
fn test_add() {
    let lhs: SparseMatrix<f64> = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]
    .into();
    let rhs: SparseMatrix<f64> = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]
    .into();
    let exp: SparseMatrix<f64> = vec![
        vec![2.0, 4.0, 6.0],
        vec![8.0, 10.0, 12.0],
        vec![14.0, 16.0, 18.0],
    ]
    .into();

    let res = lhs + rhs;

    assert_eq!(res, exp)
}

#[test]
fn test_add2() {
    let lhs: SparseMatrix<f64> = vec![vec![1.0, 2.0], vec![4.0, 5.0], vec![7.0, 8.0]].into();
    let rhs: SparseMatrix<f64> = vec![vec![1.0, 2.0], vec![4.0, 5.0], vec![7.0, 8.0]].into();
    let exp: SparseMatrix<f64> = vec![vec![2.0, 4.0], vec![8.0, 10.0], vec![14.0, 16.0]].into();

    let res = lhs + rhs;
    assert_eq!(res, exp)
}

#[test]
fn test_add3() {
    let lhs: SparseMatrix<f64> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]].into();
    let rhs: SparseMatrix<f64> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]].into();
    let exp: SparseMatrix<f64> = vec![vec![2.0, 4.0, 6.0], vec![8.0, 10.0, 12.0]].into();

    let res = lhs + rhs;
    assert_eq!(res, exp)
}

#[test]
fn test_add_complex() {
    let lhs: SparseMatrix<Complex<f64>> = vec![
        vec![
            Complex::new(1.0, 1.0),
            Complex::new(0.0, 0.0),
            Complex::new(3.0, 3.0),
        ],
        vec![
            Complex::new(0.0, 0.0),
            Complex::new(5.0, 5.0),
            Complex::new(0.0, 0.0),
        ],
        vec![
            Complex::new(7.0, 7.0),
            Complex::new(0.0, 0.0),
            Complex::new(9.0, 9.0),
        ],
    ]
    .into();

    let rhs: SparseMatrix<Complex<f64>> = vec![
        vec![
            Complex::new(1.0, -1.0),
            Complex::new(2.0, 2.0),
            Complex::new(3.0, -3.0),
        ],
        vec![
            Complex::new(4.0, 4.0),
            Complex::new(0.0, 0.0),
            Complex::new(6.0, 6.0),
        ],
        vec![
            Complex::new(0.0, 0.0),
            Complex::new(8.0, 8.0),
            Complex::new(0.0, 0.0),
        ],
    ]
    .into();

    let exp: SparseMatrix<Complex<f64>> = vec![
        vec![
            Complex::new(2.0, 0.0),
            Complex::new(2.0, 2.0),
            Complex::new(6.0, 0.0),
        ],
        vec![
            Complex::new(4.0, 4.0),
            Complex::new(5.0, 5.0),
            Complex::new(6.0, 6.0),
        ],
        vec![
            Complex::new(7.0, 7.0),
            Complex::new(8.0, 8.0),
            Complex::new(9.0, 9.0),
        ],
    ]
    .into();

    let res = &lhs + &rhs;

    assert_eq!(res, exp);
}

#[test]
fn test_mul() {
    let lhs: SparseMatrix<f64> = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]
    .into();
    let rhs: SparseMatrix<f64> = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]
    .into();
    let exp: SparseMatrix<f64> = vec![
        vec![30.0, 36.0, 42.0],
        vec![66.0, 81.0, 96.0],
        vec![102.0, 126.0, 150.0],
    ]
    .into();

    let res = lhs * rhs;
    assert_eq!(res, exp);
    assert_eq!(res.ncols(), 3);
    assert_eq!(res.nrows(), 3);
}

#[test]
fn test_mul2() {
    let lhs: SparseMatrix<f64> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]].into();
    let rhs: SparseMatrix<f64> = vec![vec![1.0, 2.0], vec![4.0, 5.0], vec![7.0, 8.0]].into();
    let exp: SparseMatrix<f64> = vec![vec![30.0, 36.0], vec![66.0, 81.0]].into();

    let res = lhs * rhs;
    assert_eq!(res, exp);
    assert_eq!(res.ncols(), 2);
    assert_eq!(res.nrows(), 2);
}

#[test]
fn test_mul3() {
    let lhs: SparseMatrix<f64> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]].into();
    let rhs: SparseMatrix<f64> = vec![vec![1.0, 2.0], vec![4.0, 5.0], vec![7.0, 8.0]].into();
    let exp: SparseMatrix<f64> = vec![
        vec![9.0, 12.0, 15.0],
        vec![24.0, 33.0, 42.0],
        vec![39.0, 54.0, 69.0],
    ]
    .into();

    let res = rhs * lhs;
    assert_eq!(res, exp);
    assert_eq!(res.ncols(), 3);
    assert_eq!(res.nrows(), 3);
}

#[test]
fn test_sparse_iter() {
    let matrix: SparseMatrix<f64> = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0],
        vec![0.0, 0.0, 3.0],
    ]
    .into();

    let mut iter = SparseIter::new(&matrix);
    assert_eq!(iter.next(), Some((0, 0, 1.0)));
    assert_eq!(iter.next(), Some((1, 1, 2.0)));
    assert_eq!(iter.next(), Some((2, 2, 3.0)));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_sparse_lower_triangular_iter() {
    let matrix: SparseMatrix<u64> = vec![
        vec![1, 4, 7], 
        vec![2, 5, 8], 
        vec![3, 6, 9]
    ].into();

    let mut iter = LowerTriangularSparseIter::new(&matrix);
    assert_eq!(iter.next(), Some((0, 0, 1)));
    assert_eq!(iter.next(), Some((0, 1, 2)));
    assert_eq!(iter.next(), Some((0, 2, 3)));
    assert_eq!(iter.next(), Some((1, 1, 5)));
    assert_eq!(iter.next(), Some((1, 2, 6)));
    assert_eq!(iter.next(), Some((2, 2, 9)));
    assert_eq!(iter.next(), None);

}

#[test]
fn test_sparse_iter_complex() {
    let matrix: SparseMatrix<Complex<f64>> = vec![
        vec![
            Complex::new(1.0, 1.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ],
        vec![
            Complex::new(0.0, 0.0),
            Complex::new(2.0, 2.0),
            Complex::new(0.0, 0.0),
        ],
        vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(3.0, 3.0),
        ],
    ]
    .into();

    let mut iter = SparseIter::new(&matrix);
    assert_eq!(iter.next(), Some((0, 0, Complex::new(1.0, 1.0))));
    assert_eq!(iter.next(), Some((1, 1, Complex::new(2.0, 2.0))));
    assert_eq!(iter.next(), Some((2, 2, Complex::new(3.0, 3.0))));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_from_iterator_complex() {
    let elements = vec![
        (0, 0, Complex::new(1.0, 1.0)),
        (1, 0, Complex::new(4.0, 4.0)),
        (1, 1, Complex::new(2.0, 2.0)),
        (2, 1, Complex::new(5.0, 5.0)),
        (2, 2, Complex::new(3.0, 3.0)),
    ];
    let matrix: SparseMatrix<Complex<f64>> = elements.into_iter().collect();

    let expected: SparseMatrix<Complex<f64>> = vec![
        vec![
            Complex::new(1.0, 1.0),
            Complex::new(4.0, 4.0),
            Complex::new(0.0, 0.0),
        ],
        vec![
            Complex::new(0.0, 0.0),
            Complex::new(2.0, 2.0),
            Complex::new(5.0, 5.0),
        ],
        vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(3.0, 3.0),
        ],
    ]
    .into();

    assert_eq!(matrix, expected);
}

#[test]
fn test_from_iter_identity() {
    let matrix: SparseMatrix<f64> = vec![
        vec![1.0, 0.0, 3.0],
        vec![0.0, 5.0, 0.0],
        vec![7.0, 0.0, 9.0],
    ]
    .into();

    let result_matrix: SparseMatrix<f64> = matrix.iter().collect();

    assert_eq!(matrix, result_matrix);
}

#[test]
fn test_from_lower_triangular_iter_identity() {
    let matrix: SparseMatrix<u64> = vec![vec![1, 4, 7], vec![2, 5, 8], vec![3, 6, 9]].into();

    let exp: SparseMatrix<u64> = vec![vec![1, 0, 0], vec![2, 5, 0], vec![3, 6, 9]].into();

    let iter = LowerTriangularSparseIter::new(&matrix);
    let result_matrix: SparseMatrix<u64> = iter.collect();

    assert_eq!(result_matrix, exp);
}

#[test]
fn test_transpose() {
    let matrix: SparseMatrix<f64> = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]
    .into();

    let exp: SparseMatrix<f64> = vec![
        vec![1.0, 4.0, 7.0],
        vec![2.0, 5.0, 8.0],
        vec![3.0, 6.0, 9.0],
    ]
    .into();

    assert_eq!(matrix.transpose(), exp);
}

#[test]
fn test_to_weighted_adjacency_matrix() {
    let matrix: SparseMatrix<f64> = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]
    .into();

    let exp: SparseMatrix<f64> = vec![
        vec![0.0, 2.0, 3.0],
        vec![4.0, 0.0, 6.0],
        vec![7.0, 8.0, 0.0],
    ]
    .into();

    assert_eq!(matrix.to_weighted_adjacency_matrix(), exp);
}

#[test]
fn test_set_function() {
    let mut matrix: SparseMatrix<f64> = vec![
        vec![1.0, 0.0, 3.0],
        vec![4.0, 5.0, 0.0],
        vec![0.0, 8.0, 9.0],
    ]
    .into();

    let expected_matrix: SparseMatrix<f64> = vec![
        vec![10.0, 0.0, 3.0],
        vec![20.0, 5.0, 0.0],
        vec![0.0, 40.0, 50.0],
    ]
    .into();

    // Set new values
    matrix.set(0, 0, 10.0);
    matrix.set(1, 0, 20.0);
    matrix.set(2, 1, 40.0);
    matrix.set(2, 2, 50.0);

    assert_eq!(matrix, expected_matrix);
}

#[test]
fn test_lower_triangular() {
    let mat: SparseMatrix<i64> = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]].into();

    let expected: SparseMatrix<i64> = vec![vec![1, 0, 0], vec![4, 1, 0], vec![7, 8, 1]].into();

    let result = mat.lower_triangular();

    print!("{result:?}");

    assert_eq!(result, expected);
}

#[test]
fn test_upper_triangular() {
    let mat: SparseMatrix<i64> = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]].into();

    let expected: SparseMatrix<i64> = vec![vec![1, 2, 3], vec![0, 5, 6], vec![0, 0, 9]].into();

    let result = mat.upper_triangular();
    assert_eq!(result, expected);
}

#[test]
fn test_non_zeros() {
    let mat: SparseMatrix<i64> = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]].into();

    let expected = 9;
    assert_eq!(mat.non_zeros(), expected);
}

#[test]
fn test_non_zeros2() {
    let mat: SparseMatrix<i64> = vec![vec![1, 2, 0], vec![4, 0, 6], vec![0, 8, 9]].into();

    let expected = 6;
    assert_eq!(mat.non_zeros(), expected);
}

#[test]
fn test_nz_cols() {
    let mat: SparseMatrix<i64> = vec![vec![1, 2, 0], vec![4, 0, 6], vec![0, 8, 9]].into();

    let expected = 3;
    assert_eq!(mat.nz_columns(), expected);
}

#[test]
fn test_nz_cols2() {
    let mat: SparseMatrix<i64> = vec![vec![1, 0, 0], vec![4, 0, 6], vec![0, 0, 9]].into();

    let expected = 2;

    println!("{mat:?}");
    assert_eq!(mat.nz_columns(), expected);
}

#[test]
fn test_nz_cols3() {
    let mat: SparseMatrix<i64> = vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]].into();

    let expected = 0;

    println!("{mat:?}");
    assert_eq!(mat.nz_columns(), expected);
}

#[test]
fn test_nz_rows() {
    let mat: SparseMatrix<i64> = vec![vec![1, 4, 6], vec![2, 5, 0], vec![3, 0, 0]].into();

    let expected = 3;
    assert_eq!(mat.nz_rows(0).unwrap(), expected);
    let expected = 2;
    assert_eq!(mat.nz_rows(1).unwrap(), expected);
    let expected = 1;
    assert_eq!(mat.nz_rows(2).unwrap(), expected);
}

#[test]
fn test_nz_rows2() {
    let mat: SparseMatrix<i64> = vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]].into();

    let expected = 0;
    assert_eq!(mat.nz_rows(0).unwrap(), expected);
    let expected = 0;
    assert_eq!(mat.nz_rows(1).unwrap(), expected);
    let expected = 0;
    assert_eq!(mat.nz_rows(2).unwrap(), expected);
}

#[test]
fn test_nz_rows3() {
    let mat: SparseMatrix<i64> = vec![vec![1, 0, 0], vec![0, 2, 0], vec![0, 0, 3]].into();

    let expected = 1;
    assert_eq!(mat.nz_rows(0).unwrap(), expected);
    let expected = 1;
    assert_eq!(mat.nz_rows(1).unwrap(), expected);
    let expected = 1;
    assert_eq!(mat.nz_rows(2).unwrap(), expected);
}
