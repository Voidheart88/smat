use super::*;

#[test]
fn test_from_vec() {
    let dense = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0],
        vec![0.0, 3.0, 4.0],
    ];

    let sparse = SparseMatrix::from(&dense[..]);

    let expected_sparse = SparseMatrix {
        nrows: 3,
        ncols: 3,
        col_idx: vec![0, 1, 3, 4],
        row_idx: vec![0, 1, 2, 2],
        values: vec![1.0, 2.0, 3.0, 4.0],
    };

    println!("{sparse}");

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

    let sparse = SparseMatrix::from(&triples);

    let expected_sparse = SparseMatrix {
        nrows: 3,
        ncols: 3,
        col_idx: vec![0, 2, 3, 5],
        row_idx: vec![0, 2, 1, 1, 2],
        values: vec![1.0, 3.0, 2.0, 4.0, 5.0],
    };

    assert_eq!(sparse.nrows, expected_sparse.nrows);
    assert_eq!(sparse.ncols, expected_sparse.ncols);

    assert_eq!(sparse.col_idx, expected_sparse.col_idx);
    assert_eq!(sparse.row_idx, expected_sparse.row_idx);
    assert_eq!(sparse.values, expected_sparse.values);
}

#[test]
fn test_from_sparse_to_dense() {
    let sparse = SparseMatrix {
        nrows: 3,
        ncols: 3,
        col_idx: vec![0, 2, 3, 5],
        row_idx: vec![0, 2, 1, 1, 2],
        values: vec![1.0, 3.0, 2.0, 4.0, 5.0],
    };

    let dense: Vec<Vec<f64>> = Vec::from(&sparse);

    let expected_dense = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 2.0, 4.0],
        vec![3.0, 0.0, 5.0],
    ];

    assert_eq!(dense, expected_dense);
}

#[test]
fn test_zeros() {
    let m = 3;
    let n = 4;
    let nzmax = 5;
    let sparse = SparseMatrix::zeros(m, n, nzmax);

    assert_eq!(sparse.nrows, m);
    assert_eq!(sparse.ncols, n);
    assert_eq!(sparse.values.len(), nzmax);

    assert_eq!(sparse.col_idx, vec![0; n + 1]);
    assert_eq!(sparse.row_idx, vec![0; nzmax]);
    assert_eq!(sparse.values, vec![0.0; nzmax]);
}

#[test]
fn test_eye() {
    let n = 3;
    let sparse = SparseMatrix::eye(n);

    assert_eq!(sparse.nrows, n);
    assert_eq!(sparse.ncols, n);
    assert_eq!(sparse.values.len(), n);

    assert_eq!(sparse.col_idx, vec![0, 1, 2, 3]);
    assert_eq!(sparse.row_idx, vec![0, 1, 2]);
    assert_eq!(sparse.values, vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_get() {
    let sparse = SparseMatrix {
        nrows: 3,
        ncols: 3,
        col_idx: vec![0, 2, 3, 5],
        row_idx: vec![0, 2, 1, 1, 2],
        values: vec![1.0, 3.0, 2.0, 4.0, 5.0],
    };

    assert_eq!(sparse.get(0, 0), Some(1.0));
    assert_eq!(sparse.get(2, 0), Some(3.0));
    assert_eq!(sparse.get(1, 1), Some(2.0));
    assert_eq!(sparse.get(1, 2), Some(4.0));
    assert_eq!(sparse.get(2, 2), Some(5.0));
    assert_eq!(sparse.get(0, 1), None);
}

#[test]
fn test_trim() {
    let mut sparse = SparseMatrix {
        nrows: 3,
        ncols: 3,
        col_idx: vec![0, 2, 4, 6],
        row_idx: vec![0, 2, 1, 1, 2, 2],
        values: vec![1.0, 0.0, 2.0, 0.0, 5.0, 0.0],
    };

    sparse.trim();

    let expected_sparse = SparseMatrix {
        nrows: 3,
        ncols: 3,
        col_idx: vec![0, 1, 2, 3],
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
        col_idx: vec![0, 2, 3, 5],
        row_idx: vec![0, 2, 1, 1, 2, 0, 0, 0, 0, 0],
        values: vec![1.0, 3.0, 2.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };

    sparse.quick_trim();

    let expected_sparse = SparseMatrix {
        nrows: 3,
        ncols: 3,
        col_idx: vec![0, 2, 3, 5],
        row_idx: vec![0, 2, 1, 1, 2],
        values: vec![1.0, 3.0, 2.0, 4.0, 5.0],
    };

    assert_eq!(sparse, expected_sparse);
}

#[test]
fn test_display_sparse() {
    let mat = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    let sparse: SparseMatrix = mat.into();

    println!("{sparse}")
}

#[test]
fn test_iter() {
    let matrix = SparseMatrix {
        nrows: 4,
        ncols: 3,
        col_idx: vec![0, 2, 4, 5],
        row_idx: vec![0, 2, 1, 3, 2],
        values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
    };

    let mut col_iter = matrix.iter();
    let mut col0 = col_iter.next().unwrap().unwrap();
    assert_eq!(col0.next(), Some((0, 1.0)));
    assert_eq!(col0.next(), Some((2, 2.0)));
    assert_eq!(col0.next(), None);

    let mut col1 = col_iter.next().unwrap().unwrap();
    assert_eq!(col1.next(), Some((1, 3.0)));
    assert_eq!(col1.next(), Some((3, 4.0)));
    assert_eq!(col1.next(), None);

    let mut col2 = col_iter.next().unwrap().unwrap();
    assert_eq!(col2.next(), Some((2, 5.0)));
    assert_eq!(col2.next(), None);

    assert!(col_iter.next().is_none());
}

#[test]
fn test_iter2() {
    let matrix: SparseMatrix = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0],
        vec![0.0, 0.0, 3.0],
    ]
    .into();

    let mut col_iter = matrix.iter();
    let mut col0 = col_iter.next().unwrap().unwrap();
    assert_eq!(col0.next(), Some((0, 1.0)));
    assert_eq!(col0.next(), None);

    let mut col1 = col_iter.next().unwrap().unwrap();
    assert_eq!(col1.next(), Some((1, 2.0)));
    assert_eq!(col1.next(), None);

    let mut col2 = col_iter.next().unwrap().unwrap();
    assert_eq!(col2.next(), Some((2, 3.0)));
    assert_eq!(col2.next(), None);

    assert!(col_iter.next().is_none());
}

#[test]
fn test_iter3() {
    let matrix: SparseMatrix = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 3.0],
    ]
    .into();

    let mut col_iter = matrix.iter();
    let mut col0 = col_iter.next().unwrap().unwrap();
    assert_eq!(col0.next(), Some((0, 1.0)));
    assert_eq!(col0.next(), None);

    let mut col1 = col_iter.next().unwrap().unwrap();
    assert_eq!(col1.next(), Some((1, 2.0)));
    assert_eq!(col1.next(), None);

    let col2 = col_iter.next().unwrap();
    assert!(col2.is_none());

    let mut col3 = col_iter.next().unwrap().unwrap();
    assert_eq!(col3.next(), Some((2, 3.0)));
    assert_eq!(col3.next(), None);

    assert!(col_iter.next().is_none());
}

#[test]
fn test_mult() {
    let a: SparseMatrix = vec![vec![1.0, 0.0], vec![0.0, 2.0], vec![0.0, 0.0]].into();

    let b: SparseMatrix = vec![vec![1.0, 0.0, 0.0], vec![0.0, 2.0, 0.0]].into();

    let expected_result: SparseMatrix = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 4.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ]
    .into();

    let result = mult(&a, &b);

    assert_eq!(result, expected_result);
}

#[test]
fn test_addition0() {
    let a = SparseMatrix::zeros(3, 3, 0);
    let b = SparseMatrix::zeros(3, 3, 0);
    let result = add(&a, &b, 1.0, 1.0);
    assert_eq!(result.values.len(), 0);
}

#[test]
fn test_addition1() {
    let a_data = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0],
        vec![0.0, 0.0, 3.0],
    ];
    let b_data = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0],
        vec![0.0, 0.0, 3.0],
    ];
    let a = SparseMatrix::from(a_data);
    let b = SparseMatrix::from(b_data);
    let result = add(&a, &b, 1.0, 1.0);
    assert_eq!(result.values, vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_addition2() {
    let a_data = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0],
        vec![0.0, 0.0, 3.0],
    ];
    let b_data = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let a = SparseMatrix::from(a_data);
    let b = SparseMatrix::from(b_data);
    let result = add(&a, &b, 2.0, -1.0);
    assert_eq!(result.values, vec![1.0, 3.0, 5.0]);
}

#[test]
fn test_addition3() {
    let a = vec![
        vec![1.0, 0.0, 3.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 5.0, 0.0],
    ];

    let b = vec![
        vec![4.0, 0.0, 0.0],
        vec![0.0, 2.0, 1.0],
        vec![0.0, 3.0, 0.0],
    ];

    let a_sparse: SparseMatrix = a.into();
    let b_sparse: SparseMatrix = b.into();
    let expected_sparse = SparseMatrix {
        nrows: 3,
        ncols: 3,
        col_idx: vec![0, 1, 3, 5],
        row_idx: vec![0, 2, 1, 0, 1],
        values: vec![5.0, 8.0, 2.0, 3.0, 1.0],
    };

    let result_c = &a_sparse + &b_sparse;

    assert_eq!(result_c, expected_sparse);
}

#[test]
fn test_norm_simple_matrix() {
    let data = vec![
        vec![1.0, 0.0, 2.0],
        vec![3.0, 0.0, 4.0],
        vec![5.0, 0.0, 6.0],
    ];
    let a = SparseMatrix::from(data);

    let norm_r = a.norm();
    let expected_norm = 12.0;

    assert_eq!(norm_r, expected_norm);
}

#[test]
fn test_norm_empty_matrix() {
    let data = vec![vec![]];
    let a = SparseMatrix::from(data);

    let norm_r = a.norm();
    let expected_norm = 0.0;

    assert_eq!(norm_r, expected_norm);
}

#[test]
fn test_norm_with_negative_entries() {
    let data = vec![
        vec![-1.0, 0.0, -2.0],
        vec![3.0, 0.0, -4.0],
        vec![-5.0, 0.0, 6.0],
    ];
    let a = SparseMatrix::from(data);

    let norm_r = a.norm();
    let expected_norm = 12.0;

    assert_eq!(norm_r, expected_norm);
}

#[test]
fn test_norm2() {
    let matrix: SparseMatrix = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]
    .into();

    assert_eq!(matrix.norm(), 18.0);
}