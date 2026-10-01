use crate::graph::Graph;
use crate::kernel::peo::PeoFailure;
use crate::witness::CycleWitness;

use super::AnalysisError;

pub const DEFAULT_CYCLE_SEARCH_BUDGET: u64 = 1_000_000;

pub fn find_cycle_from_peo_failure(
    graph: &Graph,
    failure: &PeoFailure,
    budget: u64,
) -> Result<Option<CycleWitness>, AnalysisError> {
    let n = graph.vertex_count();
    let (start, target) = failure.missing_edges[0];
    let mut blocked = vec![false; n];
    blocked[failure.vertex] = true;
    blocked[start] = false;
    blocked[target] = false;
    for &neighbor in &failure.later_neighbors {
        if neighbor != start && neighbor != target {
            blocked[neighbor] = true;
        }
    }

    let mut queue = vec![start];
    let mut visited = vec![false; n];
    let mut parent = vec![usize::MAX; n];
    visited[start] = true;
    let mut head = 0usize;
    let mut operations = 0u64;

    while head < queue.len() {
        let current = queue[head];
        head += 1;
        operations += 1;
        if operations > budget {
            return Err(AnalysisError::ResourceExhausted(format!(
                "chordless cycle search exceeded {budget} edge-relaxation steps"
            )));
        }

        if current == target {
            let mut path = Vec::new();
            let mut cursor = target;
            while cursor != usize::MAX {
                path.push(cursor);
                if cursor == start {
                    break;
                }
                cursor = parent[cursor];
            }
            path.reverse();
            if path.first() != Some(&start) || path.last() != Some(&target) {
                return Err(AnalysisError::ComputationFailure(
                    "BFS parent chain could not reconstruct witness path".to_owned(),
                ));
            }
            path.push(failure.vertex);
            return Ok(Some(CycleWitness { cycle: path }));
        }

        for &neighbor in graph.neighbors(current) {
            if blocked[neighbor] || visited[neighbor] {
                continue;
            }
            visited[neighbor] = true;
            parent[neighbor] = current;
            queue.push(neighbor);
        }
    }

    let _ = graph;
    Ok(None)
}
