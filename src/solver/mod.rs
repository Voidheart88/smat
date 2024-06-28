mod lu;

pub(crate) trait Solver<T> {
    fn solve(&mut self) -> &Vec<T>;
}
