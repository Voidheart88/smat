use std::marker::PhantomData;

/// Sparse Vector in compressed format
#[derive(Clone, Debug)]
pub struct SparseVector<T> {
    len: usize,
    row_idx: Vec<usize>,
    values: Vec<T>,
}

impl<T> SparseVector<T> {
    pub fn new(len: usize, row_idx: Vec<usize>, values: Vec<T>) -> Self {
        SparseVector {
            len,
            row_idx,
            values,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn dense_iter(&self) -> DenseVectorIter<T> {
        DenseVectorIter {
            idx: 0,
            iterable: self,
            phantom: PhantomData,
        }
    }

    pub fn iter(&self) -> SparseVectorIter<'_, T> {
        SparseVectorIter {
            idx: 0,
            iterable: self,
        }
    }
}

impl<T> Default for SparseVector<T> {
    fn default() -> Self {
        Self {
            len: 0,
            row_idx: Default::default(),
            values: Default::default(),
        }
    }
}

impl<T> From<Vec<T>> for SparseVector<T>
where
    T: PartialOrd<f64>,
{
    fn from(value: Vec<T>) -> Self {
        let len = value.len();
        let (row_idx, values): (Vec<usize>, Vec<T>) = value
            .into_iter()
            .enumerate()
            .filter(|(_, val)| *val != 0.0)
            .unzip();
        Self {
            len,
            row_idx,
            values,
        }
    }
}

impl<T> std::ops::Add for SparseVector<T>
where
    T: Copy + std::ops::Add<Output = T> + PartialOrd + Default,
{
    type Output = SparseVector<T>;

    fn add(self, rhs: SparseVector<T>) -> Self::Output {
        let len = self.len().max(rhs.len());

        let mut lhs_iter = self.into_iter().peekable();
        let mut rhs_iter = rhs.into_iter().peekable();

        let mut row_idx = Vec::new();
        let mut values = Vec::new();

        while lhs_iter.peek().is_some() || rhs_iter.peek().is_some() {
            match (lhs_iter.peek(), rhs_iter.peek()) {
                (Some(&(lhs_idx, lhs_val)), Some(&(rhs_idx, rhs_val))) => {
                    if lhs_idx == rhs_idx {
                        let sum = lhs_val + rhs_val;
                        if sum != T::default() {
                            row_idx.push(lhs_idx);
                            values.push(sum);
                        }
                        lhs_iter.next();
                        rhs_iter.next();
                    } else if lhs_idx < rhs_idx {
                        if lhs_val != T::default() {
                            row_idx.push(lhs_idx);
                            values.push(lhs_val);
                        }
                        lhs_iter.next();
                    } else {
                        if rhs_val != T::default() {
                            row_idx.push(rhs_idx);
                            values.push(rhs_val);
                        }
                        rhs_iter.next();
                    }
                }
                (Some(&(lhs_idx, lhs_val)), None) => {
                    if lhs_val != T::default() {
                        row_idx.push(lhs_idx);
                        values.push(lhs_val);
                    }
                    lhs_iter.next();
                }
                (None, Some(&(rhs_idx, rhs_val))) => {
                    if rhs_val != T::default() {
                        row_idx.push(rhs_idx);
                        values.push(rhs_val);
                    }
                    rhs_iter.next();
                }
                (None, None) => break,
            }
        }

        SparseVector {
            len,
            row_idx,
            values,
        }
    }
}

impl<'a, T> IntoIterator for &'a SparseVector<T>
where
    T: Copy,
{
    type Item = (usize, T);
    type IntoIter = SparseVectorIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        SparseVectorIter {
            idx: 0,
            iterable: &self,
        }
    }
}

pub struct SparseVectorIter<'a, T> {
    idx: usize,
    iterable: &'a SparseVector<T>,
}

impl<'a, T> Iterator for SparseVectorIter<'a, T>
where
    T: Copy,
{
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.iterable.values.len() {
            let idx = self.idx;
            self.idx += 1;
            Some((self.iterable.row_idx[idx], self.iterable.values[idx]))
        } else {
            None
        }
    }
}

pub struct DenseVectorIter<'a, T> {
    idx: usize,
    iterable: &'a SparseVector<T>,
    phantom: PhantomData<&'a T>,
}

impl<'a, T> Iterator for DenseVectorIter<'a, T>
where
    T: Copy + Default,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.iterable.len {
            let result =
                if let Some(pos) = self.iterable.row_idx.iter().position(|&r| r == self.idx) {
                    self.iterable.values[pos]
                } else {
                    T::default()
                };
            self.idx += 1;
            Some(result)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests;
