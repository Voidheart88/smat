use super::*;

#[test]
fn test_cumsum_simple_array() {
    let n = 5;
    let mut c = vec![1, 2, 3, 4, 5];
    let mut p = vec![0; n + 1];
    let total_sum = cumsum(&mut p, &mut c, n);
    let expected_p = vec![0, 1, 3, 6, 10, 15];

    assert_eq!(p, expected_p);
    assert_eq!(total_sum, 15);
}

#[test]
fn test_cumsum_empty_array() {
    let n = 0;
    let mut c: Vec<isize> = vec![];
    let mut p = vec![0; n + 1];
    let total_sum = cumsum(&mut p, &mut c, n);
    let expected_p = vec![0];

    assert_eq!(p, expected_p);
    assert_eq!(total_sum, 0);
}
