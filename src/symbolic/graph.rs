use std::collections::{HashMap, HashSet, VecDeque};

use num::One;

use crate::SparseMatrix;

/// Adjacency Graph structures and methods

pub(crate) struct AdjGraph {
    adjacency_list: HashMap<usize, Vec<usize>>,
}

impl AdjGraph {
    /// Create a new empty adjacency graph
    pub fn new() -> Self {
        AdjGraph {
            adjacency_list: HashMap::new(),
        }
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.adjacency_list
            .entry(from)
            .or_insert_with(Vec::new)
            .push(to);
    }

    /// Get the adjacency list of the graph
    pub fn adjacency_list(&self) -> &HashMap<usize, Vec<usize>> {
        &self.adjacency_list
    }

    /// Depth-First Search (DFS)
    pub fn dfs(&self, start: usize) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut stack = vec![start];
        let mut result = vec![];

        while let Some(node) = stack.pop() {
            if !visited.contains(&node) {
                visited.insert(node);
                result.push(node);

                if let Some(neighbors) = self.adjacency_list.get(&node) {
                    for &neighbor in neighbors.iter().rev() {
                        stack.push(neighbor);
                    }
                }
            }
        }

        result
    }

    /// Breadth-First Search (BFS)
    pub fn bfs(&self, start: usize) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = vec![];

        visited.insert(start);
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            result.push(node);

            if let Some(neighbors) = self.adjacency_list.get(&node) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        result
    }
}

impl<'a, T> From<&'a SparseMatrix<T>> for AdjGraph
where
    T: PartialEq
        + Default
        + Copy
        + One
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    fn from(reference: &'a SparseMatrix<T>) -> Self {
        let mut graph = AdjGraph::new();

        for col in 0..reference.ncols() {
            for idx in reference.col_ptr()[col] as usize..reference.col_ptr()[col + 1] as usize {
                let row = reference.row_idx()[idx];
                if reference.values()[idx] != T::default() && col != row {
                    graph.add_edge(col, row);
                }
            }
        }

        graph
    }
}
