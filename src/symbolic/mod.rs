use crate::SparseMatrix;

/// Symbolic analysis for sparse matrices
/// This Module provides methods for symbolic analysis of sparse matrices.

/// The Order for the approximate minimum degree (amd) algorithm.
pub enum Order {
    Natural,  // No reordering
    Cholesky, // Cholesky
    Lu,       // LU decomposition
    Qr,       // QR decomposition
}

/// The structure of the symbolic analysis. It holds the results of the Symbolic
/// analysis.
/// The analysis is lazy calculated. Every entry will be calculated as needed
pub struct Symbolic<'a, T> {
    matrix: &'a SparseMatrix<T>,

    lu_perm: Option<Vec<isize>>, // Fill reducing permutation

    is_symmetric: Option<bool>, // Check if the matrix is symmetric ( Mat = Mat' )
    is_dense: Option<bool>,     // Check if the matrix is dense
}

/// Construct a Symbolic analysis from a reference to a Sparse Matrix
impl<'a, T> Symbolic<'a, T>
where
    T: Copy
        + Default
        + PartialEq
        + std::ops::AddAssign
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    fn is_symmetric(&mut self) -> bool {
        if self.is_symmetric.is_some() {
            return self.is_symmetric.unwrap();
        } else {
            self.is_symmetric = Some(self.check_symmetric());
            return self.is_symmetric.unwrap();
        }
    }

    fn check_symmetric(&self) -> bool {
        if self.matrix.nrows() != self.matrix.ncols() {
            return false; // A non-square matrix cannot be symmetric
        }

        for col in 0..self.matrix.ncols() {
            let start = self.matrix.col_ptr()[col];
            let end = self.matrix.col_ptr()[col + 1];

            for idx in start..end {
                let row = self.matrix.row_idx()[idx as usize];
                let value = self.matrix.values()[idx as usize];

                // Find the corresponding element in the transposed position
                let transposed_start = self.matrix.col_ptr()[row];
                let transposed_end = self.matrix.col_ptr()[row + 1];

                let mut symmetric = false;
                for t_idx in transposed_start..transposed_end {
                    if self.matrix.row_idx()[t_idx as usize] == col {
                        if self.matrix.values()[t_idx as usize] == value {
                            symmetric = true;
                            break;
                        } else {
                            return false;
                        }
                    }
                }

                if !symmetric {
                    return false;
                }
            }
        }
        true
    }

    fn is_dense(&mut self) -> bool {
        if self.is_dense.is_some() {
            return self.is_dense.unwrap();
        } else {
            self.is_dense = Some(self.check_dense());
            return self.is_dense.unwrap();
        }
    }

    // The Sparse/Dense Threshold
    // If nnz/(m*n) > 10%, the matrix is considered as Dense (FIXME = provide a better threshold)
    fn check_dense(&self) -> bool {
        let size = self.matrix.ncols() * self.matrix.nrows();
        let nnz = self.matrix.row_idx().len();

        if (nnz * 10) / size > 1 {
            return true;
        }
        false
    }

    // Returns the LU permutation vector after AMD
    fn lu_perm(&mut self) -> Vec<isize> {
        if self.lu_perm.is_some() {
            return self.lu_perm.clone().unwrap();
        } else {
            self.lu_perm = Some(self.lu_amd());
            return self.lu_perm.clone().unwrap();
        }
    }

    /// calculates the LU permutation vector with the amd algorithm.
    /// Implementation from csparse/rsparse
    fn lu_amd(&mut self) -> Vec<isize> {
        // calculate the transposed
        let at = if self.is_symmetric() {
            self.matrix.clone()
        } else {
            self.matrix.transpose()
        };

        let c = &at * self.matrix; // C=A'*A
        let mut c = c.to_weighted_adjacency_matrix();

        let n = self.matrix.ncols(); // number of columns
        let mut cnz = c.col_ptr()[n]; //Fixme - why not c.x.len() ?
        let nsz = cnz as usize + cnz as usize / 5 + 2 * n;
        c.row_idx_mut().resize(nsz, 0);
        c.values_mut().resize(nsz, T::default());

        // Fixme -> put in an extra fkt
        // --- Initialize quotient graph ----------------------------------------
        let mut ww: Vec<isize> = vec![0; 8 * (n + 1)];
        let len = 0; // of ww
        for k in 0..n {
            ww[len + k] = (c.col_ptr()[k + 1] - c.col_ptr()[k]) as isize; //Fixme - why is len here used since it is 0
        }

        ww[len + n] = 0;
        // offsets of ww (pointers in csparse)
        let mut p_v = vec![0; n + 1]; // allocate result
        let nv = n + 1; // of ww
        let next = 2 * (n + 1); // of ww
        let head = 3 * (n + 1); // of ww
        let elen = 4 * (n + 1); // of ww
        let degree = 5 * (n + 1); // of ww
        let w = 6 * (n + 1); // of ww
        let hhead = 7 * (n + 1); // of ww
        let last = 0; // of p_v // use P as workspace for last

        for i in 0..=n {
            ww[head + i] = -1; // degree list i is empty
            p_v[last + i] = -1;
            ww[next + i] = -1;
            ww[hhead + i] = -1; // hash list i is empty
            ww[nv + i] = 1; // node i is just one node
            ww[w + i] = 1; // node i is alive
            ww[elen + i] = 0; // Ek of node i is empty
            ww[degree + i] = ww[len + i]; // degree of node i
        }

        // clear w ALERT!!! C implementation passes w (pointer to ww)
        let mut mark_v = wclear(0, 0, &mut ww[..], w, n);
        ww[elen + n] = -2; // n is a dead element
        c.col_ptr_mut()[n] = c.col_ptr_mut()[n] as isize - 1; // n is a root of assembly tree
        ww[w + n] = 0; // n is a dead element

        // --- Initialize degree lists ------------------------------------------
        let mut nel = 0;
        let dense = std::cmp::max(16, (10. * f32::sqrt(n as f32)) as isize);
        let dense = std::cmp::min((n - 2) as isize, dense);
        for i in 0..n {
            let d = ww[degree + i];
            if d == 0 {
                // node i is empty
                ww[elen + i] = -2; // element i is dead
                nel += 1;
                c.col_ptr_mut()[n] = c.col_ptr_mut()[n] as isize - 1; // i is a root of assembly tree
                ww[w + i] = 0;
            } else if d > dense {
                // node i is dense
                ww[nv + i] = 0; // absorb i into element n
                ww[elen + i] = -1; // node i is dead
                nel += 1;
                c.col_ptr_mut()[i] = flip(n as isize);
                ww[nv + n] += 1;
            } else {
                if ww[(head as isize + d) as usize] != -1 {
                    let wt = ww[(head as isize + d) as usize];
                    p_v[(last as isize + wt) as usize] = i as isize;
                }
                ww[next + i] = ww[(head as isize + d) as usize]; // put node i in degree list d
                ww[(head as isize + d) as usize] = i as isize;
            }
        }

        let mut mindeg = 0;
        let mut elenk;
        let mut nvk;
        let mut p;
        let mut p1;
        let mut p2;
        let mut pk1;
        let mut pk2;
        let mut e;
        let mut pj;
        let mut ln;
        let mut nvi;
        let mut i;
        let mut lemax = 0;
        let mut eln;
        let mut wnvi;
        let mut pn;
        let mut h;
        let mut d;
        let mut dext;
        let mut p3;
        let mut p4;
        let mut j;
        let mut nvj;
        let mut jlast;

        // Fixme: Refactor this while loop!!!
        while nel < n {
            // while (selecting pivots) do
            // --- Select node of minimum approximate degree --------------------
            let mut k;
            loop {
                k = ww[head + mindeg];
                if !(mindeg < n && k == -1) {
                    break;
                }
                mindeg += 1;
            }

            if ww[(next as isize + k) as usize] != -1 {
                let wt = ww[(next as isize + k) as usize];
                p_v[(last as isize + wt) as usize] = -1;
            }
            ww[head + mindeg] = ww[(next as isize + k) as usize]; // remove k from degree list
            elenk = ww[(elen as isize + k) as usize]; // elenk = |Ek|
            nvk = ww[(nv as isize + k) as usize]; // # of nodes k represents
            nel += nvk as usize; // nv[k] nodes of A eliminated

            // --- Garbage collection -------------------------------------------
            perform_garbage_collection(&mut c, &mut cnz, mindeg, elenk as usize, &ww, len, n);
            // --- Construct new element ----------------------------------------
            let mut dk = 0;

            ww[(nv as isize + k) as usize] = -nvk; // flag k as in Lk
            p = c.col_ptr()[k as usize];
            if elenk == 0 {
                // do in place if elen[k] == 0
                pk1 = p;
            } else {
                pk1 = cnz;
            }
            pk2 = pk1;
            for k1 in 1..=(elenk + 1) as usize {
                if k1 > elenk as usize {
                    e = k; // search the nodes in k
                    pj = p; // list of nodes starts at Ci[pj]
                    ln = ww[(len as isize + k) as usize] - elenk; // length of list of nodes in k
                } else {
                    e = c.row_idx()[p as usize] as isize; // search the nodes in e
                    p += 1;
                    pj = c.col_ptr()[e as usize];
                    ln = ww[(len as isize + e) as usize]; // length of list of nodes in e
                }
                for _ in 1..=ln {
                    i = c.row_idx()[pj as usize] as isize;
                    pj += 1;
                    nvi = ww[(nv as isize + i) as usize];
                    if nvi <= 0 {
                        continue; // node i dead, or seen
                    }
                    dk += nvi; // degree[Lk] += size of node i
                    ww[(nv as isize + i) as usize] = -nvi; // negate nv[i] to denote i in Lk
                    c.row_idx_mut()[pk2 as usize] = i as usize; // place i in Lk
                    pk2 += 1;
                    if ww[(next as isize + i) as usize] != -1 {
                        let wt = ww[(next as isize + i) as usize];
                        p_v[(last as isize + wt) as usize] = p_v[last + i as usize];
                    }
                    if p_v[(last as isize + i) as usize] != -1 {
                        // remove i from degree list
                        let wt = p_v[(last as isize + i) as usize];
                        ww[(next as isize + wt) as usize] = ww[(next as isize + i) as usize];
                    } else {
                        let wt = ww[degree + i as usize];
                        ww[(head as isize + wt) as usize] = ww[next + i as usize];
                    }
                }
                if e != k {
                    c.col_ptr_mut()[e as usize] = flip(k); // absorb e into k
                    ww[(w as isize + e) as usize] = 0; // e is now a dead element
                }
            }
            if elenk != 0 {
                cnz = pk2; // Ci [cnz...nzmax] is free
            }
            ww[(degree as isize + k) as usize] = dk; // external degree of k - |Lk\i|
            c.col_ptr_mut()[k as usize] = pk1; // element k is in Ci[pk1..pk2-1]
            ww[(len as isize + k) as usize] = (pk2 - pk1) as isize;
            ww[(elen as isize + k) as usize] = -2; // k is now an element

            // --- Find set differences -----------------------------------------
            mark_v = wclear(mark_v, lemax, &mut ww[..], w, n); // clear w if necessary
            for pk in pk1..pk2 {
                // scan1: find |Le\Lk|
                i = c.row_idx()[pk as usize] as isize;
                eln = ww[(elen as isize + i) as usize];
                if eln <= 0 {
                    continue; // skip if elen[i] empty
                }
                nvi = -ww[(nv as isize + i) as usize]; // nv [i] was negated
                wnvi = mark_v - nvi;
                for p in c.col_ptr()[i as usize] as usize
                    ..=(c.col_ptr()[i as usize] as isize + eln - 1) as usize
                {
                    // scan Ei
                    e = c.row_idx()[p] as isize;
                    if ww[(w as isize + e) as usize] >= mark_v {
                        ww[(w as isize + e) as usize] -= nvi; // decrement |Le\Lk|
                    } else if ww[(w as isize + e) as usize] != 0 {
                        // ensure e is a live element
                        ww[(w as isize + e) as usize] = ww[(degree as isize + e) as usize] + wnvi;
                        // 1st time e seen in scan 1
                    }
                }
            }

            // --- Degree update ------------------------------------------------
            for pk in pk1..pk2 {
                // scan2: degree update
                i = c.row_idx()[pk as usize] as isize; // consider node i in Lk
                p1 = c.col_ptr()[i as usize] as isize;
                p2 = p1 + ww[(elen as isize + i) as usize] - 1;
                pn = p1;
                h = 0;
                d = 0;
                for p in p1..=p2 {
                    // scan Ei
                    e = c.row_idx()[p as usize] as isize;
                    if ww[(w as isize + e) as usize] != 0 {
                        // e is an unabsorbed element
                        dext = ww[(w as isize + e) as usize] - mark_v; // dext = |Le\Lk|
                        if dext > 0 {
                            d += dext; // sum up the set differences
                            c.row_idx_mut()[pn as usize] = e as usize; // keep e in Ei
                            pn += 1;
                            h += e as usize; // compute the hash of node i
                        } else {
                            c.col_ptr_mut()[e as usize] = flip(k); // aggressive absorb. e->k
                            ww[(w as isize + e) as usize] = 0; // e is a dead element
                        }
                    }
                }
                ww[(elen as isize + i) as usize] = pn - p1 + 1; // elen[i] = |Ei|
                p3 = pn;
                p4 = p1 + ww[(len as isize + i) as usize];
                for p in p2 + 1..p4 {
                    // prune edges in Ai
                    j = c.row_idx()[p as usize] as isize;
                    nvj = ww[(nv as isize + j) as usize];
                    if nvj <= 0 {
                        continue; // node j dead or in Lk
                    }
                    d += nvj; // degree(i) += |j|
                    c.row_idx_mut()[pn as usize] = j as usize; // place j in node list of i
                    pn += 1;
                    h += j as usize; // compute hash for node i
                }
                if d == 0 {
                    // check for mass elimination
                    c.col_ptr_mut()[i as usize] = flip(k); // absorb i into k
                    nvi = -ww[(nv as isize + i) as usize];
                    dk -= nvi; // |Lk| -= |i|
                    nvk += nvi; // |k| += nv[i]
                    nel += nvi as usize;
                    ww[(nv as isize + i) as usize] = 0;
                    ww[(elen as isize + i) as usize] = -1; // node i is dead
                } else {
                    ww[(degree as isize + i) as usize] =
                        std::cmp::min(ww[(degree as isize + i) as usize], d); // update degree(i)
                    c.row_idx_mut()[pn as usize] = c.row_idx()[p3 as usize]; // move first node to end
                    c.row_idx_mut()[p3 as usize] = c.row_idx()[p1 as usize]; // move 1st el. to end of Ei
                    c.row_idx_mut()[p1 as usize] = k as usize; // add k as 1st element in of Ei
                    ww[(len as isize + i) as usize] = pn - p1 + 1; // new len of adj. list of node i
                    h %= n; // finalize hash of i
                    ww[(next as isize + i) as usize] = ww[hhead + h]; // place i in hash bucket
                    ww[hhead + h] = i;
                    p_v[(last as isize + i) as usize] = h as isize; // save hash of i in last[i]
                }
            } // scan2 is done
            ww[(degree as isize + k) as usize] = dk; // finalize |Lk|
            lemax = std::cmp::max(lemax, dk);
            mark_v = wclear(mark_v + lemax, lemax, &mut ww[..], w, n); // clear w

            // --- Supernode detection ------------------------------------------
            for pk in pk1..pk2 {
                i = c.row_idx()[pk as usize] as isize;
                if ww[(nv as isize + i) as usize] >= 0 {
                    continue; // skip if i is dead
                }
                h = p_v[(last as isize + i) as usize] as usize; // scan hash bucket of node i
                i = ww[hhead + h];
                ww[hhead + h] = -1; // hash bucket will be empty

                while i != -1 && ww[(next as isize + i) as usize] != -1 {
                    ln = ww[(len as isize + i) as usize];
                    eln = ww[(elen as isize + i) as usize];
                    for p in c.col_ptr()[i as usize] + 1..=c.col_ptr()[i as usize] + ln - 1 {
                        ww[w + c.row_idx()[p as usize]] = mark_v;
                    }
                    jlast = i;

                    let mut ok;
                    j = ww[(next as isize + i) as usize];
                    while j != -1 {
                        // compare i with all j
                        ok = ww[(len as isize + j) as usize] == ln
                            && ww[(elen as isize + j) as usize] == eln;

                        p = c.col_ptr()[j as usize] + 1;
                        while ok && p < c.col_ptr()[j as usize] + ln {
                            if ww[w + c.row_idx()[p as usize]] != mark_v {
                                // compare i and j
                                ok = false;
                            }

                            p += 1;
                        }
                        if ok {
                            // i and j are identical
                            c.col_ptr_mut()[j as usize] = flip(i); // absorb j into i
                            ww[(nv as isize + i) as usize] += ww[(nv as isize + j) as usize];
                            ww[(nv as isize + j) as usize] = 0;
                            ww[(elen as isize + j) as usize] = -1; // node j is dead
                            j = ww[(next as isize + j) as usize]; // delete j from hash bucket
                            ww[(next as isize + jlast) as usize] = j;
                        } else {
                            jlast = j; // j and i are different
                            j = ww[(next as isize + j) as usize];
                        }
                    }

                    // increment while loop
                    i = ww[(next as isize + i) as usize];
                    mark_v += 1;
                }
            }

            // --- Finalize new element------------------------------------------
            p = pk1;
            for pk in pk1..pk2 {
                // finalize Lk
                i = c.row_idx()[pk as usize] as isize;
                nvi = -ww[(nv as isize + i) as usize];
                if nvi <= 0 {
                    continue; // skip if i is dead
                }
                ww[(nv as isize + i) as usize] = nvi; // restore nv[i]
                d = ww[(degree as isize + i) as usize] + dk - nvi; // compute external degree(i)
                d = std::cmp::min(d, n as isize - nel as isize - nvi);
                if ww[(head as isize + d) as usize] != -1 {
                    let wt = ww[(head as isize + d) as usize];
                    p_v[(last as isize + wt) as usize] = i;
                }
                ww[(next as isize + i) as usize] = ww[(head as isize + d) as usize]; // put i back in degree list
                p_v[(last as isize + i) as usize] = -1;
                ww[(head as isize + d) as usize] = i;
                mindeg = std::cmp::min(mindeg, d as usize); // find new minimum degree
                ww[(degree as isize + i) as usize] = d;
                c.row_idx_mut()[p as usize] = i as usize; // place i in Lk
                p += 1;
            }
            ww[(nv as isize + k) as usize] = nvk; // # nodes absorbed into k
            ww[(len as isize + k) as usize] = (p - pk1) as isize;
            if ww[(len as isize + k) as usize] == 0 {
                // length of adj list of element k
                c.col_ptr_mut()[k as usize] = c.col_ptr_mut()[k as usize] - 1; // k is a root of the tree
                ww[(w as isize + k) as usize] = 0; // k is now a dead element
            }
            if elenk != 0 {
                cnz = p; // free unused space in Lk
            }
        }
        // --- Post-ordering ----------------------------------------------------
        for i in 0..n {
            c.col_ptr_mut()[i] = flip(c.col_ptr()[i] as isize); // fix assembly tree
        }
        for j in 0..=n {
            ww[head + j] = -1;
        }
        for j in (0..=n).rev() {
            // place unordered nodes in lists
            if ww[nv + j] > 0 {
                continue; // skip if j is an element
            }
            ww[next + j] = ww[(head as isize + c.col_ptr()[j] as isize) as usize]; // place j in list of its parent
            ww[(head as isize + c.col_ptr()[j] as isize) as usize] = j as isize;
        }
        for e in (0..=n).rev() {
            // place elements in lists
            if ww[nv + e] <= 0 {
                continue; // skip unless e is an element
            }
            if c.col_ptr_mut()[e] != c.col_ptr_mut()[e] - 1 {
                ww[next + e] = ww[(head as isize + c.col_ptr_mut()[e] as isize) as usize]; // place e in list of its parent
                ww[(head as isize + c.col_ptr_mut()[e] as isize) as usize] = e as isize;
            }
        }
        let mut k = 0;
        for i in 0..=n {
            // postorder the assembly tree
            if c.col_ptr_mut()[i] as isize == -1 {
                k = tdfs(i as isize, k, &mut ww[..], head, next, &mut p_v[..], w);
                // Note that CSparse passes the pointers of ww
            }
        }

        p_v
    }
}

impl<'a, T> From<&'a SparseMatrix<T>> for Symbolic<'a, T> {
    fn from(value: &'a SparseMatrix<T>) -> Self {
        Self {
            matrix: value,
            lu_perm: None,
            is_symmetric: None,
            is_dense: None,
        }
    }
}

/// clears Workspace
fn wclear(mark_v: isize, lemax: isize, ww: &mut [isize], w: usize, n: usize) -> isize {
    let mut mark = mark_v;
    if mark < 2 || (mark + lemax < 0) {
        for k in 0..n {
            if ww[w + k] != 0 {
                ww[w + k] = 1;
            }
        }
        mark = 2;
    }
    mark // at this point, w [0..n-1] < mark holds
}

fn perform_garbage_collection<T>(
    c: &mut SparseMatrix<T>,
    cnz: &mut isize,
    mindeg: usize,
    elenk: usize,
    ww: &[isize],
    len: usize,
    n: usize,
) where
    T: Copy
        + Default
        + PartialEq
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    if elenk > 0 && (*cnz + mindeg as isize) as usize >= c.values().len() {
        for j in 0..n {
            let p = c.col_ptr()[j];
            if p >= 0 {
                // j is a live node or element
                c.col_ptr_mut()[j] = c.row_idx_mut()[p as usize] as isize; // save first entry of object
                c.row_idx_mut()[p as usize] = flip(j as isize) as usize; // first entry is now FLIP(j)
            }
        }

        let mut q = 0;
        let mut p = 0;
        while p < *cnz {
            // scan all of memory
            let j = flip(c.row_idx()[p as usize] as isize);
            p += 1;
            if j >= 0 {
                // found object j
                c.row_idx_mut()[q] = c.col_ptr()[j as usize] as usize; // restore first entry of object
                c.col_ptr_mut()[j as usize] = q as isize; // new pointer to object j
                q += 1;
                for _ in 0..ww[(len as isize + j) as usize] - 1 {
                    c.row_idx_mut()[q] = c.row_idx()[p as usize];
                    q += 1;
                    p += 1;
                }
            }
        }
        *cnz = q as isize; // Ci [cnz...nzmax-1] now free
    }
}

/// depth-first search and postorder of a tree rooted at node j (for fn amd())
///
fn tdfs(
    j: isize,
    k: isize,
    ww: &mut [isize],
    head: usize,
    next: usize,
    post: &mut [isize],
    stack: usize,
) -> isize {
    let mut i;
    let mut p;
    let mut top = 0;
    let mut k = k;

    ww[stack] = j; // place j on the stack
    while top >= 0 {
        // while (stack is not empty)
        p = ww[(stack as isize + top) as usize]; // p = top of stack
        i = ww[(head as isize + p) as usize]; // i = youngest child of p
        match i {
            -1 => {
                top -= 1; // p has no unordered children left
                post[k as usize] = p; // node p is the kth post-ordered node
                k += 1;
            }
            _ => {
                ww[(head as isize + p) as usize] = ww[(next as isize + i) as usize]; // remove i from children of p
                top += 1;
                ww[(stack as isize + top) as usize] = i; // start dfs on child node i
            }
        }
    }

    k
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

#[cfg(test)]
mod tests;
