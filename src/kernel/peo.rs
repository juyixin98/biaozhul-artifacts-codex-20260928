use crate::graph::Graph;

#[derive(Debug, Clone)]
pub struct McsStep {
    pub step: usize,
    pub selected_vertex: usize,
    pub selected_weight: usize,
    pub weights_before: Vec<usize>,
    pub earlier_neighbors: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct PeoFailure {
    pub position: usize,
    pub vertex: usize,
    pub later_neighbors: Vec<usize>,
    pub missing_edges: Vec<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub struct PeoReport {
    pub valid: bool,
    pub first_failure: Option<PeoFailure>,
    pub reason: String,
}

pub fn maximum_cardinality_order(graph: &Graph) -> (Vec<usize>, Vec<McsStep>) {
    let n = graph.vertex_count();
    let mut chosen = vec![false; n];
    let mut position = vec![usize::MAX; n];
    let mut weights = vec![0usize; n];
    let mut order = Vec::with_capacity(n);
    let mut steps = Vec::with_capacity(n);

    for step in 0..n {
        let mut selected = usize::MAX;
        let mut best_weight = usize::MAX;
        for vertex in 0..n {
            if !chosen[vertex]
                && (selected == usize::MAX
                    || weights[vertex] > best_weight
                    || (weights[vertex] == best_weight && vertex < selected))
            {
                selected = vertex;
                best_weight = weights[vertex];
            }
        }

        let earlier_neighbors = graph
            .neighbors(selected)
            .iter()
            .copied()
            .filter(|vertex| chosen[*vertex])
            .collect();

        chosen[selected] = true;
        position[selected] = step;
        order.push(selected);
        steps.push(McsStep {
            step,
            selected_vertex: selected,
            selected_weight: best_weight,
            weights_before: weights.clone(),
            earlier_neighbors,
        });

        for &neighbor in graph.neighbors(selected) {
            if !chosen[neighbor] {
                weights[neighbor] += 1;
            }
        }
    }

    let _ = position;
    (order, steps)
}

pub fn check_perfect_elimination_order(graph: &Graph, order: &[usize]) -> PeoReport {
    if order.len() != graph.vertex_count() {
        return PeoReport {
            valid: false,
            first_failure: None,
            reason: format!(
                "order length {} does not match vertex count {}",
                order.len(),
                graph.vertex_count()
            ),
        };
    }

    let mut position = vec![usize::MAX; graph.vertex_count()];
    for (index, vertex) in order.iter().enumerate() {
        if *vertex >= graph.vertex_count() || position[*vertex] != usize::MAX {
            return PeoReport {
                valid: false,
                first_failure: None,
                reason: format!("order is not a permutation of graph vertices at index {index}"),
            };
        }
        position[*vertex] = index;
    }

    for (index, &vertex) in order.iter().enumerate() {
        let later_neighbors: Vec<usize> = graph
            .neighbors(vertex)
            .iter()
            .copied()
            .filter(|neighbor| position[*neighbor] > index)
            .collect();

        let mut missing_edges = Vec::new();
        for (i, &left) in later_neighbors.iter().enumerate() {
            for &right in later_neighbors.iter().skip(i + 1) {
                if !graph.has_edge(left, right) {
                    missing_edges.push((left, right));
                }
            }
        }

        if !missing_edges.is_empty() {
            return PeoReport {
                valid: false,
                first_failure: Some(PeoFailure {
                    position: index,
                    vertex,
                    later_neighbors,
                    missing_edges,
                }),
                reason: format!(
                    "successors of vertex {} in the elimination order are not a clique",
                    graph.label(vertex)
                ),
            };
        }
    }

    PeoReport {
        valid: true,
        first_failure: None,
        reason: "for every vertex, all neighbors later in the order are pairwise adjacent"
            .to_owned(),
    }
}
