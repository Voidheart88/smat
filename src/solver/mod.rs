use crate::SparseVector;

mod lu;

pub(crate) trait Solver<T> {
    fn solve(&self) -> SparseVector<T>;
}
