use std::ops::Mul;

use super::matrix::SparseMatrix;

/// x = x + beta * A(:,j), where x is a dense vector and A(:,j) is sparse
///
fn scatter<T>(
    a: &SparseMatrix<T>,
    j: usize,
    beta: T,
    w: &mut [usize],
    x: &mut [T],
    mark: usize,
    c: &mut SparseMatrix<T>,
    nz: usize,
) -> usize where T: Copy + Default + Mul<Output = T> + PartialEq + PartialOrd + std::ops::AddAssign{

    // n = ncols
    // m = nrows
    // p = col_ptr
    // i = row_idx
    // x = val

    let mut i;
    let mut nzo = nz;
    for p in a.column_ptr()[j]..a.column_ptr()[j + 1]{
        i = a.row_idx()[p];
        if w[i] < mark {
            w[i] = mark;
            c.row_idx_mut()[nzo] = i;
            nzo += 1;
            x[i] = beta * a.values()[p];
        } else {
            x[i] += beta * a.values()[p];
        }
    }

    nzo
}