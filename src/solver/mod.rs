mod lu;

pub(crate) trait Solver<T> {
    fn solve(&self) -> Vec<T>;
}
