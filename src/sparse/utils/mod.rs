use std::ops::Mul;

use super::matrix::SparseMatrix;

pub(crate) fn scatter<T>(
    a: &SparseMatrix<T>,
    j: usize,
    beta: T,
    w: &mut [usize],
    x: &mut [T],
    mark: usize,
    c: &mut SparseMatrix<T>,
    nz: usize,
) -> usize
where
    T: Copy + Default + Mul<Output = T> + PartialEq + std::ops::AddAssign,
{
    let mut i;
    let mut nzo = nz;
    for p in a.col_ptr()[j]..a.col_ptr()[j + 1] {
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
