use super::*;

#[test]
fn test_lu_decomposition() {
    let matrix = SparseMatrix::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let exp_l = SparseMatrix::from(vec![
        vec![1.0, 0.0, 0.0],
        vec![1.0 / 7.0, 1.0, 0.0],
        vec![4.0 / 7.0, 0.5, 1.0],
    ]);

    let exp_u = SparseMatrix::from(vec![
        vec![7.0, 8.0, 9.0],
        vec![0.0, 6.0 / 7.0, 12.0 / 7.0],
        vec![0.0, 0.0, 0.0],
    ]);

    let vector = vec![0.0; matrix.ncols()];

    let mut solver = LUSolver::new(&matrix, &vector);
    solver.decompose();

    assert_eq!(solver.lower, exp_l);
    assert_eq!(solver.upper, exp_u);
}

#[test]
fn test_lu_decomposition_with_unity() {
    let matrix = SparseMatrix::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let vector = vec![0.0; matrix.ncols()];

    let mut solver = LUSolver::new(&matrix, &vector);
    solver.decompose();
    println!("upper:\n{}",solver.upper);
    println!("lower:\n{}",solver.lower);

    let solution = solver.lower*solver.upper;

    println!("solution:\n{}",solution);

    assert_eq!(matrix, solution);
}

#[test]
fn test_lu_solver_f64() {
    let matrix = SparseMatrix::from(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);
    let vector = vec![1.0, 1.0, 1.0];
    let mut solver = LUSolver::new(&matrix, &vector);
    let solution = solver.solve();

    let expected_solution = vec![1.0, 1.0, 1.0];
    assert_eq!(*solution, expected_solution);
}

#[test]
fn test_lu_solver_i32() {
    let matrix = SparseMatrix::from(vec![vec![2, -1, 0], vec![-1, 2, -1], vec![0, -1, 2]]);
    let vector = vec![1, 0, 1];

    let mut solver = LUSolver::new(&matrix, &vector);

    let solution = solver.solve();
    let expected_solution = vec![1, 1, 1];

    assert_eq!(*solution, expected_solution);
}

#[test]
fn test_forward_substitution1() {
    let matrix = SparseMatrix::from(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);

    let vector = vec![1.0, 1.0, 1.0];
    let mut solver = LUSolver::new(&matrix, &vector);
    solver.lower = matrix.clone();
    solver.forward_substitution();
    let expected = vec![1.0, 1.0, 1.0];

    assert_eq!(solver.solution, expected);
}

#[test]
fn test_forward_substitution2() {
    let matrix = SparseMatrix::from(vec![
        vec![1.0, 0.0, 0.0],
        vec![1.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0],
    ]);

    let vector = vec![1.0, 1.0, 1.0];
    let mut solver = LUSolver::new(&matrix, &vector);
    solver.lower = matrix.clone();
    solver.forward_substitution();
    let expected = vec![1.0, 0.0, 0.0];

    assert_eq!(solver.solution, expected);
}

#[test]
fn test_forward_substitution3() {
    let matrix = SparseMatrix::from(vec![
        vec![1.0, 0.0, 0.0],
        vec![1.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0],
    ]);

    let vector = vec![1.0, 2.0, 3.0];
    let mut solver = LUSolver::new(&matrix, &vector);
    solver.lower = matrix.clone();
    solver.forward_substitution();
    let expected = vec![1.0, 1.0, 1.0];

    assert_eq!(solver.solution, expected);
}

#[test]
fn test_backward_substitution1() {
    let matrix = SparseMatrix::from(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);

    let vector = vec![1.0, 1.0, 1.0];
    let mut solver = LUSolver::new(&matrix, &vector);
    solver.upper = matrix.clone();
    solver.backward_substitution();
    let expected = vec![1.0, 1.0, 1.0];

    assert_eq!(solver.solution, expected);
}

#[test]
fn test_backward_substitution2() {
    let matrix = SparseMatrix::from(vec![
        vec![1.0, 1.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0],
    ]);

    let vector = vec![1.0, 1.0, 1.0];
    let mut solver = LUSolver::new(&matrix, &vector);
    solver.upper = matrix.clone();
    solver.backward_substitution();
    let expected = vec![0.0, 0.0, 1.0];

    assert_eq!(solver.solution, expected);
}

#[test]
fn test_backward_substitution3() {
    let matrix = SparseMatrix::from(vec![
        vec![1.0, 1.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0],
    ]);

    let vector = vec![3.0, 2.0, 1.0];
    let mut solver = LUSolver::new(&matrix, &vector);
    solver.upper = matrix.clone();
    solver.backward_substitution();
    let expected = vec![1.0, 1.0, 1.0];

    assert_eq!(solver.solution, expected);
}
