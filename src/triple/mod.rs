#[derive(Clone, Debug)]
pub struct Triples {
    nrows: usize,
    ncols: usize,
    p: Vec<isize>,
    i: Vec<usize>,
    values: Vec<f64>,
}

impl Triples {
    #[inline]
    pub fn rows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn columns(&self) -> usize {
        self.ncols
    }

    #[inline]
    pub fn column_idx(&self) -> &Vec<isize> {
        &self.p
    }

    #[inline]
    pub fn row_idx(&self) -> &Vec<usize> {
        &self.i
    }

    #[inline]
    pub fn values(&self) -> &Vec<f64> {
        &self.values
    }

    pub fn new(
        nrows: usize,
        ncols: usize,
        p: Vec<isize>,
        i: Vec<usize>,
        values: Vec<f64>,
    ) -> Triples {
        Triples {
            nrows,
            ncols,
            p,
            i,
            values,
        }
    }

    pub fn append(&mut self, row: usize, column: usize, value: f64) {
        if row + 1 > self.nrows {
            self.nrows = row + 1;
        }
        if column + 1 > self.ncols {
            self.ncols = column + 1;
        }

        self.p.push(column as isize);
        self.i.push(row);
        self.values.push(value);
    }

    fn process_indices(&mut self, i: usize, j: isize) {
        if let Some((pos, val)) = self.get_all(i, j as usize) {
            pos.iter().for_each(|&i| self.values[i] = 0.0);
            self.values[pos[pos.len() - 1]] = val.iter().sum();
        }
    }

    pub fn sum_dupl(&mut self) {
        let indices: Vec<(usize, isize)> = self
            .i
            .iter()
            .flat_map(|&i| self.p.iter().map(move |&j| (i, j)))
            .collect();

        indices.into_iter().for_each(|(i, j)| {
            self.process_indices(i, j);
        });
    }

    pub fn get(&self, row: usize, column: usize) -> Option<f64> {
        self.i
            .iter()
            .zip(self.p.iter())
            .zip(self.values.iter())
            .find(|((i, &p), _)| **i == row && p as usize == column)
            .map(|((_, _), &val)| val)
    }

    pub fn get_all(&self, row: usize, column: usize) -> Option<(Vec<usize>, Vec<f64>)> {
        let (pos, r): (Vec<_>, Vec<_>) = self
            .i
            .iter()
            .zip(self.p.iter())
            .zip(self.values.iter())
            .enumerate()
            .filter(|(_, ((&i, &p), _))| i == row && p as usize == column)
            .map(|(idx, ((_, _), &value))| (idx, value))
            .unzip();

        if !r.is_empty() {
            Some((pos, r))
        } else {
            None
        }
    }
}

impl Default for Triples {
    fn default() -> Self {
        Self::new(0, 0, Vec::new(), Vec::new(), Vec::new())
    }
}

#[cfg(test)]
mod tests;
