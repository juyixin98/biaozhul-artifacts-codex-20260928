use crate::model::Graph;
use crate::topology::{find_cycle, CycleEvidence};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchingResult {
    pub pairs: Vec<(usize, usize)>,
    pub size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathCover {
    pub paths: Vec<Vec<usize>>,
    pub matching: MatchingResult,
    pub minimum_path_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathCoverError {
    Cycle(CycleEvidence),
}

pub fn maximum_bipartite_matching(graph: &Graph) -> MatchingResult {
    let mut right_owner = vec![None; graph.vertex_count];

    fn augment(
        graph: &Graph,
        left: usize,
        right_owner: &mut [Option<usize>],
        seen_right: &mut [bool],
    ) -> bool {
        for &right in &graph.adjacency[left] {
            if seen_right[right] {
                continue;
            }
            seen_right[right] = true;
            match right_owner[right] {
                None => {
                    right_owner[right] = Some(left);
                    return true;
                }
                Some(owner) => {
                    if augment(graph, owner, right_owner, seen_right) {
                        right_owner[right] = Some(left);
                        return true;
                    }
                }
            }
        }
        false
    }

    for left in 0..graph.vertex_count {
        let mut seen_right = vec![false; graph.vertex_count];
        augment(graph, left, &mut right_owner, &mut seen_right);
    }

    let mut pairs: Vec<(usize, usize)> = right_owner
        .into_iter()
        .enumerate()
        .filter_map(|(right, owner)| owner.map(|left| (left, right)))
        .collect();
    pairs.sort_unstable();

    MatchingResult {
        size: pairs.len(),
        pairs,
    }
}

pub fn minimum_path_cover(graph: &Graph) -> Result<PathCover, PathCoverError> {
    if let Some(cycle) = find_cycle(graph) {
        return Err(PathCoverError::Cycle(cycle));
    }

    let matching = maximum_bipartite_matching(graph);
    let minimum_path_count = graph.vertex_count - matching.size;

    let mut successor = vec![None; graph.vertex_count];
    let mut has_predecessor = vec![false; graph.vertex_count];
    for &(from, to) in &matching.pairs {
        successor[from] = Some(to);
        has_predecessor[to] = true;
    }

    let mut paths = Vec::with_capacity(minimum_path_count);
    for start in (0..graph.vertex_count).filter(|v| !has_predecessor[*v]) {
        let mut path = vec![start];
        let mut current = start;
        while let Some(next) = successor[current] {
            path.push(next);
            current = next;
        }
        paths.push(path);
    }

    paths.sort_by_cached_key(|path| *path.first().expect("paths are non-empty"));

    Ok(PathCover {
        paths,
        matching,
        minimum_path_count,
    })
}
