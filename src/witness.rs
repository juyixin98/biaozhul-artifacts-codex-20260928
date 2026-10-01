use crate::graph::Graph;

#[derive(Debug, Clone)]
pub struct CycleWitness {
    pub cycle: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct WitnessCheck {
    pub valid: bool,
    pub reason: String,
}

impl CycleWitness {
    pub fn cycle_length(&self) -> usize {
        self.cycle.len()
    }

    pub fn edges(&self) -> Vec<(usize, usize)> {
        self.cycle
            .iter()
            .zip(self.cycle.iter().cycle().skip(1))
            .map(|(u, v)| if u <= v { (*u, *v) } else { (*v, *u) })
            .take(self.cycle.len())
            .collect()
    }
}

pub fn check_induced_cycle(graph: &Graph, witness: &CycleWitness) -> WitnessCheck {
    let n = graph.vertex_count();
    let cycle = &witness.cycle;
    if cycle.len() < 4 {
        return invalid(format!(
            "cycle length {} is too short; chordless cycle witness requires length at least 4",
            cycle.len()
        ));
    }

    let mut seen = vec![false; n];
    for &vertex in cycle {
        if vertex >= n {
            return invalid(format!("cycle contains unknown vertex index {vertex}"));
        }
        if seen[vertex] {
            return invalid(format!("cycle repeats vertex {vertex}"));
        }
        seen[vertex] = true;
    }

    let length = cycle.len();
    for position in 0..length {
        let u = cycle[position];
        let v = cycle[(position + 1) % length];
        if !graph.has_edge(u, v) {
            return invalid(format!(
                "cycle boundary edge ({}, {}) is missing",
                graph.label(u),
                graph.label(v)
            ));
        }
    }

    for i in 0..length {
        for j in (i + 2)..length {
            if i == 0 && j == length - 1 {
                continue;
            }
            let u = cycle[i];
            let v = cycle[j];
            if graph.has_edge(u, v) {
                return invalid(format!(
                    "nonconsecutive cycle vertices {} and {} have a chord",
                    graph.label(u),
                    graph.label(v)
                ));
            }
        }
    }

    WitnessCheck {
        valid: true,
        reason: format!("{length} vertices form a simple induced cycle with no chord"),
    }
}

fn invalid(reason: String) -> WitnessCheck {
    WitnessCheck {
        valid: false,
        reason,
    }
}
