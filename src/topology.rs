use crate::model::Graph;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleEvidence {
    pub vertices: Vec<usize>,
    pub edges: Vec<(usize, usize)>,
}

pub fn find_cycle(graph: &Graph) -> Option<CycleEvidence> {
    #[derive(Clone, Copy)]
    enum Mark {
        Active,
        Done,
    }

    let mut marks: Vec<Option<Mark>> = vec![None; graph.vertex_count];
    let mut stack = Vec::new();
    let mut stack_positions = vec![None; graph.vertex_count];

    fn visit(
        graph: &Graph,
        vertex: usize,
        marks: &mut [Option<Mark>],
        stack: &mut Vec<usize>,
        stack_positions: &mut [Option<usize>],
    ) -> Option<CycleEvidence> {
        marks[vertex] = Some(Mark::Active);
        stack_positions[vertex] = Some(stack.len());
        stack.push(vertex);

        for &next in &graph.adjacency[vertex] {
            match &marks[next] {
                None => {
                    if let Some(cycle) = visit(graph, next, marks, stack, stack_positions) {
                        return Some(cycle);
                    }
                }
                Some(Mark::Active) => {
                    let start = stack_positions[next].expect("active vertex has a stack position");
                    let cycle_vertices = stack[start..].to_vec();
                    let mut cycle_edges: Vec<(usize, usize)> = cycle_vertices
                        .windows(2)
                        .map(|pair| (pair[0], pair[1]))
                        .collect();
                    cycle_edges.push((*cycle_vertices.last().unwrap(), next));
                    return Some(CycleEvidence {
                        vertices: cycle_vertices,
                        edges: cycle_edges,
                    });
                }
                Some(Mark::Done) => {}
            }
        }

        stack.pop();
        stack_positions[vertex] = None;
        marks[vertex] = Some(Mark::Done);
        None
    }

    for vertex in 0..graph.vertex_count {
        if marks[vertex].is_none() {
            if let Some(evidence) =
                visit(graph, vertex, &mut marks, &mut stack, &mut stack_positions)
            {
                return Some(evidence);
            }
        }
    }
    None
}

pub fn is_acyclic(graph: &Graph) -> bool {
    find_cycle(graph).is_none()
}
