use super::*;

#[test]
fn test_new_sparse_vector() {
    let row_idx = vec![0, 2, 4];
    let values = vec![1.0, 2.0, 3.0];
    let sv = SparseVector::new(3, row_idx.clone(), values.clone());
    assert_eq!(sv.row_idx, row_idx);
    assert_eq!(sv.values, values);
    assert_eq!(sv.len(), 3)
}

#[test]
fn test_default_sparse_vector() {
    let sv: SparseVector<f64> = Default::default();
    assert!(sv.row_idx.is_empty());
    assert!(sv.values.is_empty());
}

#[test]
fn test_sparse_vector_from_vec() {
    let dense_vec = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
    let sv: SparseVector<f64> = SparseVector::from(dense_vec);
    assert_eq!(sv.row_idx, vec![1, 3, 5]);
    assert_eq!(sv.values, vec![1.0, 2.0, 3.0]);
    assert_eq!(sv.len(), 6)
}

#[test]
fn test_sparse_vector_from_vec_with_non_positive_values() {
    let dense_vec = vec![-1.0, 1.0, 0.0, 2.0, 0.0, 3.0];
    let sv: SparseVector<f64> = SparseVector::from(dense_vec);
    assert_eq!(sv.row_idx, vec![0, 1, 3, 5]);
    assert_eq!(sv.values, vec![-1.0, 1.0, 2.0, 3.0]);
    assert_eq!(sv.len(), 6)
}

#[test]
fn test_sparse_vector_clone() {
    let row_idx = vec![0, 2, 4];
    let values = vec![1.0, 2.0, 3.0];
    let sv = SparseVector::new(3, row_idx.clone(), values.clone());
    let sv_clone = sv.clone();
    assert_eq!(sv_clone.row_idx, row_idx);
    assert_eq!(sv_clone.values, values);
}

#[test]
fn test_sparse_vector_debug() {
    let row_idx = vec![0, 2, 4];
    let values = vec![1.0, 2.0, 3.0];
    let sv = SparseVector::new(3, row_idx.clone(), values.clone());
    let debug_str = format!("{:?}", sv);
    assert_eq!(
        debug_str,
        "SparseVector { len: 3, row_idx: [0, 2, 4], values: [1.0, 2.0, 3.0] }"
    );
}

#[test]
fn test_sparse_vector_into_iterator() {
    let row_idx = vec![0, 2, 4];
    let values = vec![1.0, 2.0, 3.0];
    let sv = SparseVector::new(3, row_idx, values);

    let mut iter = sv.into_iter();

    assert_eq!(iter.next(), Some((0, 1.0)));
    assert_eq!(iter.next(), Some((2, 2.0)));
    assert_eq!(iter.next(), Some((4, 3.0)));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_sparse_vector_iterator() {
    let sv = SparseVector::new(6, vec![1, 3, 5], vec![1.0, 2.0, 3.0]);

    let mut iter = sv.iter();
    assert_eq!(iter.next(), Some((1, 1.0)));
    assert_eq!(iter.next(), Some((3, 2.0)));
    assert_eq!(iter.next(), Some((5, 3.0)));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_sparse_vector_iter_with_empty_vector() {
    let row_idx: Vec<usize> = Vec::new();
    let values: Vec<f64> = Vec::new();
    let sv = SparseVector::new(0, row_idx, values);

    let mut iter = sv.into_iter();

    assert_eq!(iter.next(), None);
}

#[test]
fn test_sparse_vector_iter_with_single_element() {
    let row_idx = vec![2];
    let values = vec![3.5];
    let sv = SparseVector::new(1, row_idx, values);

    let mut iter = sv.into_iter();

    assert_eq!(iter.next(), Some((2, 3.5)));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_sparse_vector_clone_with_iter() {
    let row_idx = vec![1, 3];
    let values = vec![1.0, 2.0];
    let sv = SparseVector::new(2, row_idx.clone(), values.clone());

    let sv_clone = sv.clone();
    let mut iter = sv_clone.into_iter();

    assert_eq!(iter.next(), Some((1, 1.0)));
    assert_eq!(iter.next(), Some((3, 2.0)));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_sparse_vector_iter_with_varied_data() {
    let row_idx = vec![0, 2, 4, 6];
    let values = vec![0.1, 0.2, 0.3, 0.4];
    let sv = SparseVector::new(4, row_idx, values);

    let mut iter = sv.into_iter();

    assert_eq!(iter.next(), Some((0, 0.1)));
    assert_eq!(iter.next(), Some((2, 0.2)));
    assert_eq!(iter.next(), Some((4, 0.3)));
    assert_eq!(iter.next(), Some((6, 0.4)));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_sparse_vector_add() {
    let sv1 = SparseVector::new(5, vec![0, 2, 4], vec![1.0, 2.0, 3.0]);
    let sv2 = SparseVector::new(5, vec![1, 2, 3], vec![4.0, 5.0, 6.0]);

    let result = sv1 + sv2;

    assert_eq!(result.len, 5);
    assert_eq!(result.row_idx, vec![0, 1, 2, 3, 4]);
    assert_eq!(result.values, vec![1.0, 4.0, 7.0, 6.0, 3.0]);
}

#[test]
fn test_sparse_vector_add_with_empty() {
    let sv1 = SparseVector::new(5, vec![0, 2, 4], vec![1.0, 2.0, 3.0]);
    let sv2 = SparseVector::default();

    let result = sv1 + sv2;

    assert_eq!(result.len, 5);
    assert_eq!(result.row_idx, vec![0, 2, 4]);
    assert_eq!(result.values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_sparse_vector_add_with_overlap() {
    let sv1 = SparseVector::new(5, vec![1, 3, 5], vec![1.0, 2.0, 3.0]);
    let sv2 = SparseVector::new(5, vec![1, 3, 5], vec![4.0, 5.0, 6.0]);

    let result = sv1 + sv2;

    assert_eq!(result.len, 5);
    assert_eq!(result.row_idx, vec![1, 3, 5]);
    assert_eq!(result.values, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_sparse_vector_add_no_overlap() {
    let sv1 = SparseVector::new(4, vec![0, 2], vec![1.0, 2.0]);
    let sv2 = SparseVector::new(4, vec![1, 3], vec![3.0, 4.0]);

    let result = sv1 + sv2;

    assert_eq!(result.len, 4);
    assert_eq!(result.row_idx, vec![0, 1, 2, 3]);
    assert_eq!(result.values, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_sparse_vector_add_result_is_sorted() {
    let sv1 = SparseVector::new(6, vec![1, 3, 5], vec![1.0, 2.0, 3.0]);
    let sv2 = SparseVector::new(6, vec![0, 2, 4], vec![4.0, 5.0, 6.0]);

    let result = sv1 + sv2;

    assert_eq!(result.len, 6);
    assert_eq!(result.row_idx, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(result.values, vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]);
}

#[test]
fn test_dense_vector_iter() {
    let sparse_vector = SparseVector::new(6, vec![1, 3, 5], vec![1.0, 2.0, 3.0]);
    let mut iter = sparse_vector.dense_iter();

    assert_eq!(iter.next(), Some(0.0));
    assert_eq!(iter.next(), Some(1.0));
    assert_eq!(iter.next(), Some(0.0));
    assert_eq!(iter.next(), Some(2.0));
    assert_eq!(iter.next(), Some(0.0));
    assert_eq!(iter.next(), Some(3.0));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_dense_vector_iter_empty() {
    let sparse_vector = SparseVector::new(4, vec![], vec![]);
    let mut iter = sparse_vector.dense_iter();

    assert_eq!(iter.next(), Some(0.0));
    assert_eq!(iter.next(), Some(0.0));
    assert_eq!(iter.next(), Some(0.0));
    assert_eq!(iter.next(), Some(0.0));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_dense_vector_iter_with_default() {
    let sparse_vector = SparseVector::new(4, vec![1, 3], vec![5, 10]);
    let mut iter = sparse_vector.dense_iter();

    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), Some(5));
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), Some(10));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_sparse_vector_get() {
    let vector = SparseVector::new(5, vec![0, 2, 4], vec![1.0, 2.0, 3.0]);

    // Test für vorhandene Elemente
    assert_eq!(vector.get(0), Some(1.0));
    assert_eq!(vector.get(2), Some(2.0));
    assert_eq!(vector.get(4), Some(3.0));

    // Test für nicht vorhandene Elemente
    assert_eq!(vector.get(1), None);
    assert_eq!(vector.get(3), None);
}

#[test]
fn test_sparse_vector_get_int() {
    let vector = SparseVector::new(5, vec![0, 2, 4], vec![10, 20, 30]);

    // Test für vorhandene Elemente
    assert_eq!(vector.get(0), Some(10));
    assert_eq!(vector.get(2), Some(20));
    assert_eq!(vector.get(4), Some(30));

    // Test für nicht vorhandene Elemente
    assert_eq!(vector.get(1), None);
    assert_eq!(vector.get(3), None);
}

#[test]
fn test_sparse_vector_equality_f64() {
    let vec1 = SparseVector::from(vec![1.0, 0.0, 3.0]);
    let vec2 = SparseVector::from(vec![1.0, 0.0, 3.0]);
    let vec3 = SparseVector::from(vec![1.0, 2.0, 3.0]);

    assert_eq!(vec1, vec2);
    assert_ne!(vec1, vec3);
}

#[test]
fn test_sparse_vector_equality_i32() {
    let vec1 = SparseVector::from(vec![1, 0, 3]);
    let vec2 = SparseVector::from(vec![1, 0, 3]);
    let vec3 = SparseVector::from(vec![1, 2, 3]);

    assert_eq!(vec1, vec2);
    assert_ne!(vec1, vec3);
}

#[test]
fn test_sparse_vector_equality_empty() {
    let vec1: SparseVector<f64> = SparseVector::from(vec![]);
    let vec2: SparseVector<f64> = SparseVector::from(vec![]);

    assert_eq!(vec1, vec2);
}

#[test]
fn test_sparse_vector_equality_different_lengths() {
    let vec1 = SparseVector::from(vec![1.0, 0.0, 3.0]);
    let vec2 = SparseVector::from(vec![1.0, 0.0]);

    assert_ne!(vec1, vec2);
}