#![allow(dead_code)]
mod solver;
mod sparse;
mod symbolic;
mod triple;

pub use sparse::matrix::SparseMatrix;
pub use sparse::vector::SparseVector;
pub use symbolic::Symbolic;

#[allow(unused)]
#[macro_use]
extern crate assert_float_eq;
