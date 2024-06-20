mod numeric;
use numeric::LuNmrc;

use crate::{SparseMatrix, Symbolic};

pub struct LUSolver<'a, 'b, T> {
    matrix: &'a SparseMatrix<T>,
    symb: &'b Symbolic<'a, T>,
}

impl<'a, 'b, T> LUSolver<'a, 'b, T>
where
    T: Copy
        + Default
        + PartialEq
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    fn solve(&self, tol: f64) {
        let n = self.matrix.ncols();
        let mut col;
        let mut top;
        let mut ipiv;
        let mut a_f;
        let mut t;
        let mut pivot;
        let mut x = vec![T::default(); n];
        let mut xi = vec![0; 2 * n];
        let mut n_mat = LuNmrc {
            l: SparseMatrix::zeros(n, n, n * n / 2),
            u: SparseMatrix::zeros(n, n, n * n / 2),
            pivot: Some(vec![0; n]),
        };

        x[0..n].fill(0.);
        n_mat.pivot.as_mut().unwrap()[0..n].fill(-1);

        s.lnz = 0;
        s.unz = 0;
        for k in 0..n {
            n_mat.l.col_ptr_mut()[k] = s.lnz as isize;
            n_mat.u.col_ptr_mut()[k] = s.unz as isize;

            if s.lnz + n > n_mat.l.values().len() {
                let nsz = 2 * n_mat.l.values().len() + n;
                n_mat.l.row_idx().resize(nsz, 0);
                n_mat.l.values().resize(nsz, 0.);
            }
            if s.unz + n > n_mat.u.values().len() {
                let nsz = 2 * n_mat.u.values().len() + n;
                n_mat.u.row_idx().resize(nsz, 0);
                n_mat.u.values().resize(nsz, 0.);
            }

            col = s.q.as_ref().map_or(k, |q| q[k] as usize);
            top = splsolve(&mut n_mat.l, self.matrix, col, &mut xi[..], &mut x[..], &n_mat.pivot);

            ipiv = -1;
            a_f = -1.;
            for &i in xi[top..n].iter() {
                let i = i as usize;
                if n_mat.pivot.as_ref().unwrap()[i] < 0 {
                    t = f64::abs(x[i]);
                    if t > a_f {
                        a_f = t;
                        ipiv = i as isize;
                    }
                } else {
                    n_mat.u.i[s.unz] = n_mat.pivot.as_ref().unwrap()[i] as usize;
                    n_mat.u.x[s.unz] = x[i];
                    s.unz += 1;
                }
            }
            if ipiv == -1 || a_f <= 0. {
                panic!("Could not find a pivot");
            }
            if n_mat.pivot.as_ref().unwrap()[col] < 0 && f64::abs(x[col]) >= a_f * tol {
                ipiv = col as isize;
            }

            pivot = x[ipiv as usize];
            n_mat.u.i[s.unz] = k;
            n_mat.u.x[s.unz] = pivot;
            s.unz += 1;
            n_mat.pivot.as_mut().unwrap()[ipiv as usize] = k as isize;
            n_mat.l.i[s.lnz] = ipiv as usize;
            n_mat.l.x[s.lnz] = 1.;
            s.lnz += 1;
            for &i in xi[top..n].iter() {
                let i = i as usize;
                if n_mat.pivot.as_ref().unwrap()[i] < 0 {
                    n_mat.l.i[s.lnz] = i; 
                    n_mat.l.x[s.lnz] = x[i] / pivot;
                    s.lnz += 1
                }
                x[i] = T::default();
            }
        }
        n_mat.l.p[n] = s.lnz as isize;
        n_mat.u.p[n] = s.unz as isize;
        for p in 0..s.lnz {
            n_mat.l.i[p] = n_mat.pivot.as_ref().unwrap()[n_mat.l.i[p]] as usize;
        }
        n_mat.l.quick_trim();
        n_mat.u.quick_trim();

        n_mat
    }
}

fn splsolve<T>(
    l: &mut SparseMatrix<T>,
    b: &SparseMatrix<T>,
    k: usize,
    xi: &mut [isize],
    x: &mut [T],
    pinv: &Option<Vec<isize>>,
) -> usize
where
    T: Copy + Default + PartialEq + std::ops::Mul<Output = T>,
{
    let mut jnew;
    let top = reach(l, b, k, &mut xi[..], pinv);

    for p in top..l.n {
        x[xi[p] as usize] = T::default();
    }
    for p in b.p[k] as usize..b.p[k + 1] as usize {
        x[b.i[p]] = b.x[p];
    }
    for j in xi.iter().take(l.n).skip(top) {
        let j_u = *j as usize;
        jnew = match pinv {
            Some(pinv) => pinv[j_u],
            None => *j,
        };
        if jnew < 0 {
            continue;
        }
        for p in (l.p[jnew as usize] + 1) as usize..l.p[jnew as usize + 1] as usize {
            x[l.i[p]] -= l.x[p] * x[j_u];
        }
    }

    top
}

fn reach<T>(
    l: &mut SparseMatrix<T>,
    b: &SparseMatrix<T>,
    k: usize,
    xi: &mut [isize],
    pinv: &Option<Vec<isize>>,
) -> usize
where
    T: Copy + Default + PartialEq + std::ops::Mul<Output = T>,
{
    let mut top = l.ncols();

    for p in b.col_ptr()[k] as usize..b.col_ptr()[k + 1] as usize {
        if !marked(&l.col_ptr_mut()[..], b.row_idx()[p]) {
            let n = l.ncols();
            top = dfs(b.row_idx()[p], l, top, &mut xi[..], &n, pinv);
        }
    }
    for i in xi.iter().take(l.ncols()).skip(top) {
        mark(&mut l.col_ptr_mut()[..], *i as usize);
    }

    top
}

fn dfs<T>(
    j: usize,
    l: &mut SparseMatrix<T>,
    top: usize,
    xi: &mut [isize],
    pstack_i: &usize,
    pinv: &Option<Vec<isize>>,
) -> usize
where
    T: Copy + Default + PartialEq + std::ops::Mul<Output = T>,
{
    let mut i;
    let mut j = j;
    let mut jnew;
    let mut head = 0;
    let mut done;
    let mut p2;
    let mut top = top;

    xi[0] = j as isize;
    while head >= 0 {
        j = xi[head as usize] as usize;
        if pinv.is_some() {
            jnew = pinv.as_ref().unwrap()[j];
        } else {
            jnew = j as isize;
        }
        if !marked(&l.col_ptr()[..], j) {
            mark(&mut l.col_ptr_mut()[..], j);
            if jnew < 0 {
                xi[pstack_i + head as usize] = 0;
            } else {
                xi[pstack_i + head as usize] = unflip(l.col_ptr()[jnew as usize]);
            }
        }
        done = true;
        if jnew < 0 {
            p2 = 0;
        } else {
            p2 = unflip(l.col_ptr()[(jnew + 1) as usize]);
        }
        for p in xi[pstack_i + head as usize]..p2 {
            i = l.row_idx()[p as usize];
            if marked(&l.col_ptr()[..], i) {
                continue;
            }
            xi[pstack_i + head as usize] = p;
            head += 1;
            xi[head as usize] = i as isize;
            done = false;
            break;
        }
        if done {
            head -= 1;
            top -= 1;
            xi[top] = j as isize;
        }
    }

    top
}

#[inline]
fn marked(ap: &[isize], j: usize) -> bool {
    ap[j] < 0
}

#[inline]
fn mark(ap: &mut [isize], j: usize) {
    ap[j] = flip(ap[j])
}

#[inline]
fn flip(i: isize) -> isize {
    -(i) - 2
}

#[inline]
fn unflip(i: isize) -> isize {
    if i < 0 {
        flip(i)
    } else {
        i
    }
}
