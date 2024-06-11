use super::*;

#[test]
fn test_new() {
    let m = 3;
    let n = 2;
    let p = vec![0, 1];
    let i = vec![0, 1];
    let x = vec![1.0, 2.0];

    let triples = Triples::new(m, n, p.clone(), i.clone(), x.clone());

    assert_eq!(triples.nrows(), m);
    assert_eq!(triples.ncols(), n);
    assert_eq!(triples.column_idx(), &p);
    assert_eq!(triples.row_idx(), &i);
    assert_eq!(triples.values(), &x);
}

#[test]
fn test_append() {
    let mut triples = Triples::default();
    triples.append(1, 0, 5.0);

    assert_eq!(triples.nrows(), 2);
    assert_eq!(triples.ncols(), 1);
    assert_eq!(triples.get(1, 0), Some(5.0));

    triples.append(0, 1, 3.0);

    assert_eq!(triples.nrows(), 2);
    assert_eq!(triples.ncols(), 2);
    assert_eq!(triples.get(0, 1), Some(3.0));
}

#[test]
fn test_get() {
    let m = 2;
    let n = 2;
    let p = vec![0, 1, 0];
    let i = vec![0, 0, 1];
    let x = vec![1.0, 2.0, 3.0];
    let triples = Triples::new(m, n, p, i, x);

    assert_eq!(triples.get(0, 0), Some(1.0));
    assert_eq!(triples.get(0, 1), Some(2.0));
    assert_eq!(triples.get(1, 0), Some(3.0));
    assert_eq!(triples.get(2, 0), None);
}

#[test]
fn test_get_all() {
    let m = 2;
    let n = 2;
    let p = vec![0, 0, 1];
    let i = vec![0, 1, 0];
    let x = vec![1.0, 2.0, 3.0];
    let triples = Triples::new(m, n, p, i, x);

    let (pos, val) = triples.get_all(0, 0).unwrap();
    assert_eq!(pos, vec![0]);
    assert_eq!(val, vec![1.0]);

    let (pos, val) = triples.get_all(1, 0).unwrap();
    assert_eq!(pos, vec![1]);
    assert_eq!(val, vec![2.0]);

    assert_eq!(triples.get_all(2, 0), None);
}

#[test]
fn test_sum_dupl1() {
    let mut triples = Triples {
        nrows: 3,
        ncols: 3,
        p: vec![0, 1, 2],
        i: vec![0, 1, 2],
        values: vec![5.0, 6.0, 7.0],
    };

    triples.sum_dupl();
    assert_eq!(triples.values, vec![5.0, 6.0, 7.0]);
}

#[test]
fn test_sum_dupl2() {
    let mut triples = Triples {
        nrows: 3,
        ncols: 3,
        p: vec![0, 1, 2, 2],
        i: vec![0, 1, 2, 2],
        values: vec![5.0, 6.0, 7.0, 1.0],
    };

    triples.sum_dupl();

    assert_eq!(triples.values, vec![5.0, 6.0, 0.0, 8.0]);
}
