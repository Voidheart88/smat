use std::collections::HashMap;

use graph::AdjGraph;

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

#[test]
fn test_adgraph_from_matrix_i64() {
    // 0 - 1; 1 - 2; 2 - 3;
    let mat: SparseMatrix<i64> = vec![
        vec![1, 1, 0, 0],
        vec![1, 1, 1, 0],
        vec![0, 1, 1, 1],
        vec![0, 0, 1, 1],
    ]
    .into();

    let graph: AdjGraph = (&mat).into();

    let mut expected_adjacency_list = HashMap::new();
    expected_adjacency_list.insert(0, vec![1]);
    expected_adjacency_list.insert(1, vec![0, 2]);
    expected_adjacency_list.insert(2, vec![1, 3]);
    expected_adjacency_list.insert(3, vec![2]);

    assert_eq!(graph.adjacency_list(), &expected_adjacency_list);
}

#[test]
fn test_adgraph_from_matrix_f64() {
    // 0 - 1; 1 - 2; 2 - 3;
    let mat: SparseMatrix<f64> = vec![
        vec![1.0, 1.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0, 0.0],
        vec![0.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
    ]
    .into();

    let graph: AdjGraph = (&mat).into();

    let mut expected_adjacency_list = HashMap::new();
    expected_adjacency_list.insert(0, vec![1]);
    expected_adjacency_list.insert(1, vec![0, 2]);
    expected_adjacency_list.insert(2, vec![1, 3]);
    expected_adjacency_list.insert(3, vec![2]);

    assert_eq!(graph.adjacency_list(), &expected_adjacency_list);
}

#[test]
fn test_dfs() {
    let mat: SparseMatrix<i64> = vec![
        vec![1, 1, 0, 0],
        vec![1, 1, 1, 0],
        vec![0, 1, 1, 1],
        vec![0, 0, 1, 1],
    ]
    .into();

    let graph: AdjGraph = (&mat).into();
    let result = graph.dfs(0);

    assert_eq!(result, vec![0, 1, 2, 3]);
}

#[test]
fn test_dfs2() {
    let mat: SparseMatrix<i64> = vec![
        vec![1, 0, 0, 1],
        vec![0, 1, 0, 1],
        vec![0, 0, 1, 1],
        vec![1, 1, 1, 1],
    ]
    .into();

    let graph: AdjGraph = (&mat).into();
    let result = graph.dfs(0);

    assert_eq!(result, vec![0, 3, 1, 2]);
}

#[test]
fn test_dfs3() {
    // 1 -> 2 -> 3 -> 4
    let mat: SparseMatrix<i64> = vec![
        vec![1, 0, 0, 0, 0],
        vec![1, 1, 0, 0, 0],
        vec![0, 1, 1, 0, 0],
        vec![0, 0, 1, 1, 0],
        vec![0, 0, 0, 1, 1],
    ]
    .into();

    let graph: AdjGraph = (&mat).into();
    let result = graph.dfs(0);
    assert_eq!(result, vec![0, 1, 2, 3, 4]);

    let result = graph.dfs(4);
    assert_eq!(result, vec![4]);
}

#[test]
fn test_bfs() {
    let mat: SparseMatrix<i64> = vec![
        vec![1, 1, 0, 0],
        vec![1, 1, 1, 0],
        vec![0, 1, 1, 1],
        vec![0, 0, 1, 1],
    ]
    .into();

    let graph: AdjGraph = (&mat).into();
    let result = graph.bfs(0);

    assert_eq!(result, vec![0, 1, 2, 3]);
}

#[test]
fn test_bfs2() {
    let mat: SparseMatrix<i64> = vec![
        vec![0, 1, 0, 0],
        vec![1, 0, 1, 1],
        vec![0, 1, 0, 0],
        vec![0, 1, 0, 0],
    ]
    .into();

    let graph: AdjGraph = (&mat).into();
    let result = graph.bfs(0);

    assert_eq!(result, vec![0, 1, 2, 3]);
}
