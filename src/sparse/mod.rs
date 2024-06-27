pub mod matrix;
pub(crate) mod utils;
pub mod vector;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SparseError {
    ColumnOutOfBounds,
}
