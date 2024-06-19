#![allow(dead_code)]
mod sparse;
mod symbolic;
//mod solver;
mod triple;

pub use sparse::matrix::SparseMatrix;
pub use sparse::vector::SparseVector;
pub use symbolic::Symbolic;

#[allow(unused)]
#[macro_use]
extern crate assert_float_eq;
