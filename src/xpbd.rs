//! Implement xpbd on a tetrahedral mesh.

use std::ops::IndexMut;

use raylib::math::Vector3;

use crate::{
    constraint::{Constraint, ValueGrad},
    mesh::{Tetrahedral, Vertex, VertexId},
};

/// State for Extended Position Based Dynamics simulation.
pub struct XpbdState {
    /// Velocities of each particle.
    velocities: Vec<Vector3>,
    /// Number of substeps per simulation step.
    n_substeps: usize,
    /// Time step for each simulation substep.
    time_substep: f32,
}

impl XpbdState {
    /// Initialize the XPBD state with given number of vertices, substeps, and time step.
    pub fn new(n_vertices: usize, n_substeps: usize, time_step: f32) -> Self {
        Self {
            velocities: vec![Vector3::zero(); n_vertices],
            n_substeps,
            time_substep: time_step / (n_substeps as f32),
        }
    }
}

pub struct TetConstraintValues {
    lengths: Vec<f32>,
    volumes: Vec<f32>,
}

pub fn evaluate_tet_constraints(mesh: &Tetrahedral) -> TetConstraintValues {
    let lengths = mesh.edges.iter().map(|e| e.value(&mesh.vertices)).collect();
    let volumes = mesh
        .tetrahedra
        .iter()
        .map(|t| t.value(&mesh.vertices))
        .collect();
    TetConstraintValues { lengths, volumes }
}

/// Apply a constraint correction with uniform inverse mass to all participants.
fn apply_constraint_uniform<const N: usize, V>(
    vag: ValueGrad<N>,
    reference_value: f32,
    vertices: &mut V,
) where
    V: IndexMut<VertexId, Output = Vertex>,
{
    let lambda =
        (reference_value - vag.value) / vag.grad.into_iter().map(|g| g.dot(g)).sum::<f32>();
    for (i, vertex_id) in vag.participants.into_iter().enumerate() {
        let grad = vag.grad[i];
        let vertex = &mut vertices[vertex_id];
        vertex.position += grad * lambda;
    }
}

pub fn step_basic(
    state: XpbdState,
    mesh: &mut Tetrahedral,
    initial_value: &TetConstraintValues,
) -> XpbdState {
    let XpbdState {
        mut velocities,
        n_substeps,
        time_substep,
    } = state;
    for _ in 0..n_substeps {
        // copy old positions each time.
        let old_positions = mesh.vertices.clone();
        let gravity = Vector3::new(0.0, -0.1, 0.0);

        for (i, vertex) in mesh.vertices.iter_mut().enumerate() {
            velocities[i] += gravity * time_substep; // unit mass for now
            vertex.position += velocities[i] * time_substep;
            if vertex.position.y < 0.0 {
                vertex.position.y = 0.0;
            }
        }

        for (i, edge) in mesh.edges.iter().enumerate() {
            let ref_length = *initial_value
                .lengths
                .get(i)
                .expect("Edge should have an initial length.");
            let result = edge.value_and_grad(&mesh.vertices);
            apply_constraint_uniform(result, ref_length, &mut mesh.vertices);
        }

        for (i, tetrahedron) in mesh.tetrahedra.iter().enumerate() {
            let ref_volume = *initial_value
                .volumes
                .get(i)
                .expect("Tetrahedron should have an initial volume.");
            let result = tetrahedron.value_and_grad(&mesh.vertices);
            apply_constraint_uniform(result, ref_volume, &mut mesh.vertices);
        }

        // Update velocities based on position changes
        for (i, vertex) in mesh.vertices.iter().enumerate() {
            velocities[i] = (vertex.position - old_positions[i].position) / time_substep;
        }
    }
    XpbdState {
        velocities,
        n_substeps,
        time_substep,
    }
}
