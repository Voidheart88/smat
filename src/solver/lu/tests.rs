use super::*;

#[test]
fn test_lu_decomposition() {
    // Definieren einer 3x3-Matrix
    let matrix = SparseMatrix::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    // Definieren einer 3x3-Matrix
    let exp_l = SparseMatrix::from(vec![
        vec![1.0, 0.0, 0.0],
        vec![1.0 / 7.0, 1.0, 0.0],
        vec![4.0 / 7.0, 0.5, 1.0],
    ]);

    // Definieren einer 3x3-Matrix
    let exp_u = SparseMatrix::from(vec![
        vec![7.0, 8.0, 9.0],
        vec![0.0, 6.0 / 7.0, 12.0 / 7.0],
        vec![0.0, 0.0, 0.0],
    ]);

    let (l, u) = lu_decompose(&matrix);

    assert_eq!(l, exp_l);
    //assert_eq!(u, exp_u);
}

#[test]
fn test_lu_solver_f64() {
    // Definieren einer 3x3-Matrix
    let matrix = SparseMatrix::from(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);

    // Definieren des Vektors
    let vector = SparseVector::from(vec![1.0, 1.0, 1.0]);

    // Erstellen des Solvers
    let solver = LUSolver {
        matrix: &matrix,
        vector: &vector,
    };

    // Lösen des Gleichungssystems
    let solution = solver.solve();

    // Erwartete Lösung
    let expected_solution = SparseVector::from(vec![1.0, 1.0, 1.0]);

    assert_eq!(solution, expected_solution);
}

#[test]
fn test_lu_solver_i32() {
    // Definieren einer 3x3-Matrix
    let matrix = SparseMatrix::from(vec![vec![2, -1, 0], vec![-1, 2, -1], vec![0, -1, 2]]);

    // Definieren des Vektors
    let vector = SparseVector::from(vec![1, 0, 1]);

    // Erstellen des Solvers
    let solver = LUSolver {
        matrix: &matrix,
        vector: &vector,
    };

    // Lösen des Gleichungssystems
    let solution = solver.solve();

    // Erwartete Lösung
    let expected_solution = SparseVector::from(vec![1, 1, 1]);

    assert_eq!(solution, expected_solution);
}
