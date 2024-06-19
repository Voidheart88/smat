mod numeric;
use numeric::Nmrc;

use crate::{SparseMatrix, Symbolic};

pub struct LUSolver<'a,'b,T> {
    matrix: &'a SparseMatrix<T>,
    symb: &'b Symbolic<'a,T>,
}

impl<'a,'b,T> LUSolver<'a,'b,T> 
where
    T: Copy + Default + PartialEq + std::ops::Mul<Output = T>,
{
    fn solve(&self,tol:f64) {

    let n = self.matrix.ncols();
    let mut col;
    let mut top;
    let mut ipiv;
    let mut a_f;
    let mut t;
    let mut pivot;
    let mut x = vec![0.; n];
    let mut xi = vec![0; 2 * n];
    let mut n_mat = Nmrc {
        l: SparseMatrix::zeros(n, n, n*n/2), // initial L and U
        u: SparseMatrix::zeros(n, n, n*n/2),
        pinv: Some(vec![0; n]),
        b: Vec::new(),
    };

    x[0..n].fill(0.); // clear workspace
    n_mat.pinv.as_mut().unwrap()[0..n].fill(-1); // no rows pivotal yet
    n_mat.l.p[0..n + 1].fill(0); // no cols of L yet

    s.lnz = 0;
    s.unz = 0;
    for k in 0..n {
        // compute L(:,k) and U(:,k)
        // --- Triangular solve ---------------------------------------------
        n_mat.l.p[k] = s.lnz as isize; // L(:,k) starts here
        n_mat.u.p[k] = s.unz as isize; // L(:,k) starts here

        // Resize L and U
        if s.lnz + n > n_mat.l.nzmax {
            let nsz = 2 * n_mat.l.nzmax + n;
            n_mat.l.nzmax = nsz;
            n_mat.l.i.resize(nsz, 0);
            n_mat.l.x.resize(nsz, 0.);
        }
        if s.unz + n > n_mat.u.nzmax {
            let nsz = 2 * n_mat.u.nzmax + n;
            n_mat.u.nzmax = nsz;
            n_mat.u.i.resize(nsz, 0);
            n_mat.u.x.resize(nsz, 0.);
        }

        col = s.q.as_ref().map_or(k, |q| q[k] as usize);
        top = splsolve(&mut n_mat.l, a, col, &mut xi[..], &mut x[..], &n_mat.pinv); // x = L\A(:,col)

        // --- Find pivot ---------------------------------------------------
        ipiv = -1;
        a_f = -1.;
        for &i in xi[top..n].iter() {
            let i = i as usize;
            if n_mat.pinv.as_ref().unwrap()[i] < 0 {
                // row i is not pivotal
                t = f64::abs(x[i]);
                if t > a_f {
                    a_f = t; // largest pivot candidate so far
                    ipiv = i as isize;
                }
            } else {
                // x(i) is the entry U(Pinv[i],k)
                n_mat.u.i[s.unz] = n_mat.pinv.as_ref().unwrap()[i] as usize;
                n_mat.u.x[s.unz] = x[i];
                s.unz += 1;
            }
        }
        if ipiv == -1 || a_f <= 0. {
            panic!("Could not find a pivot");
        }
        if n_mat.pinv.as_ref().unwrap()[col] < 0 && f64::abs(x[col]) >= a_f * tol {
            ipiv = col as isize;
        }

        // --- Divide by pivot ----------------------------------------------
        pivot = x[ipiv as usize]; // the chosen pivot
        n_mat.u.i[s.unz] = k; // last entry in U(:,k) is U(k,k)
        n_mat.u.x[s.unz] = pivot;
        s.unz += 1;
        n_mat.pinv.as_mut().unwrap()[ipiv as usize] = k as isize; // ipiv is the kth pivot row
        n_mat.l.i[s.lnz] = ipiv as usize; // first entry in L(:,k) is L(k,k) = 1
        n_mat.l.x[s.lnz] = 1.;
        s.lnz += 1;
        for &i in xi[top..n].iter() {
            let i = i as usize;
            if n_mat.pinv.as_ref().unwrap()[i] < 0 {
                // x(i) is an entry in L(:,k)
                n_mat.l.i[s.lnz] = i; // save unpermuted row in L
                n_mat.l.x[s.lnz] = x[i] / pivot; // scale pivot column
                s.lnz += 1
            }
            x[i] = 0.; // x [0..n-1] = 0 for next k
        }
    }
    // --- Finalize L and U -------------------------------------------------
    n_mat.l.p[n] = s.lnz as isize;
    n_mat.u.p[n] = s.unz as isize;
    // fix row indices of L for final Pinv
    for p in 0..s.lnz {
        n_mat.l.i[p] = n_mat.pinv.as_ref().unwrap()[n_mat.l.i[p]] as usize;
    }
    n_mat.l.quick_trim();
    n_mat.u.quick_trim();

    n_mat
    }
}


/// Solve Lx=b(:,k), leaving pattern in xi[top..n-1], values scattered in x.
///
fn splsolve<T>(
    l: &mut SparseMatrix<T>,
    b: &SparseMatrix<T>,
    k: usize,
    xi: &mut [isize],
    x: &mut [f64],
    pinv: &Option<Vec<isize>>,
) -> usize {
    let mut jnew;
    let top = reach(l, b, k, &mut xi[..], pinv); // xi[top..n-1]=Reach(B(:,k))

    for p in top..l.n {
        x[xi[p] as usize] = 0.; // clear x
    }
    for p in b.p[k] as usize..b.p[k + 1] as usize {
        x[b.i[p]] = b.x[p]; // scatter B
    }
    for j in xi.iter().take(l.n).skip(top) {
        let j_u = *j as usize; // x(j) is nonzero
        jnew = match pinv {
            Some(pinv) => pinv[j_u], // j is column jnew of L
            None => *j,              // j is column jnew of L
        };
        if jnew < 0 {
            continue; // column jnew is empty
        }
        for p in (l.p[jnew as usize] + 1) as usize..l.p[jnew as usize + 1] as usize {
            x[l.i[p]] -= l.x[p] * x[j_u]; // x(i) -= L(i,j) * x(j)
        }
    }

    top // return top of stack
}

/// xi [top...n-1] = nodes reachable from graph of L*P' via nodes in B(:,k).
/// xi [n...2n-1] used as workspace.
///
fn reach(l: &mut Sprs, b: &Sprs, k: usize, xi: &mut [isize], pinv: &Option<Vec<isize>>) -> usize {
    let mut top = l.n;

    for p in b.p[k] as usize..b.p[k + 1] as usize {
        if !marked(&l.p[..], b.i[p]) {
            // start a dfs at unmarked node i
            let n = l.n;
            top = dfs(b.i[p], l, top, &mut xi[..], &n, pinv);
        }
    }
    for i in xi.iter().take(l.n).skip(top) {
        mark(&mut l.p[..], *i as usize); // restore L
    }

    top
}

/// depth-first-search of the graph of a matrix, starting at node j
/// if pstack_i is used for pstack=xi[pstack_i]
///
fn dfs<T>(
    j: usize,
    l: &mut SparseMatrix<T>,
    top: usize,
    xi: &mut [isize],
    pstack_i: &usize,
    pinv: &Option<Vec<isize>>,
) -> usize {
    let mut i;
    let mut j = j;
    let mut jnew;
    let mut head = 0;
    let mut done;
    let mut p2;
    let mut top = top;

    xi[0] = j as isize; // initialize the recursion stack
    while head >= 0 {
        j = xi[head as usize] as usize; // get j from the top of the recursion stack
        if pinv.is_some() {
            jnew = pinv.as_ref().unwrap()[j];
        } else {
            jnew = j as isize;
        }
        if !marked(&l.col_ptr()[..], j) {
            mark(&mut l.col_ptr()[..], j); // mark node j as visited
            if jnew < 0 {
                xi[pstack_i + head as usize] = 0;
            } else {
                xi[pstack_i + head as usize] = unflip(l.col_ptr()[jnew as usize]);
            }
        }
        done = true; // node j done if no unvisited neighbors
        if jnew < 0 {
            p2 = 0;
        } else {
            p2 = unflip(l.p[(jnew + 1) as usize]);
        }
        for p in xi[pstack_i + head as usize]..p2 {
            // examine all neighbors of j
            i = l.i[p as usize]; // consider neighbor node i
            if marked(&l.p[..], i) {
                continue; // skip visited node i
            }
            xi[pstack_i + head as usize] = p; // pause depth-first search of node j
            head += 1;
            xi[head as usize] = i as isize; // start dfs at node i
            done = false; // node j is not done
            break; // break, to start dfs (i)
        }
        if done {
            // depth-first search at node j is done
            head -= 1; // remove j from the recursion stack
            top -= 1;
            xi[top] = j as isize; // and place in the output stack
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