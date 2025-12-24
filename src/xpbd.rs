//! Implement xpbd on a tetrahedral mesh.

use std::ops::IndexMut;

use bitvec::vec::BitVec;
use raylib::math::Vector3;

use crate::{
    constraint::{Constraint, apply_constraint},
    mesh::{Tetrahedral, Vertex, VertexId},
};

/// State for Extended Position Based Dynamics simulation.
pub struct XpbdState {
    /// Velocities of each particle.
    velocities: Vec<Vector3>,
    /// Boolean vector indicating inactive constraints by index.
    inactive_constraints: BitVec,
}

/// Immutable parameters for the XPBD simulation.
#[derive(Clone, Debug)]
pub struct XpbdParams {
    /// Stiffness for edge length constraints.
    stiffness_volume: f32,
    /// Stiffness for tetrahedral volume constraints.
    stiffness_length: f32,
    /// Number of substeps per simulation step.
    n_substeps: usize,
    /// Time step for each simulation substep.
    time_substep: f32,
    /// Length constraint force-threshold for deactivation.
    l_threshold_length: f32,
    /// Volume constraint force-threshold for deactivation.
    l_threshold_volume: f32,
}

impl XpbdParams {
    /// Create new XPBD parameters.
    pub fn new(
        n_substeps: usize,
        time_step: f32,
        stiffness_length: f32,
        stiffness_volume: f32,
        l_threshold_length: f32,
        l_threshold_volume: f32,
    ) -> Self {
        Self {
            stiffness_length,
            stiffness_volume,
            n_substeps,
            time_substep: time_step / n_substeps as f32,
            l_threshold_length,
            l_threshold_volume,
        }
    }
}

impl XpbdState {
    /// Initialize the XPBD state with given number of vertices, substeps, and time step.
    pub fn new(n_vertices: usize, n_constraints: usize) -> Self {
        Self {
            velocities: vec![Vector3::zero(); n_vertices],
            inactive_constraints: BitVec::repeat(false, n_constraints),
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

/// Helper function to process constraints and deactivate them if necessary.
fn process_constraints<'a, V, I, C, const N: usize>(
    base_index: usize,
    iter: I,
    vertices: &mut V,
    inactive_constraints: &mut BitVec,
    l_threshold: f32,
    alpha: f32,
) -> usize
where
    I: Iterator<Item = (&'a C, f32)>,
    C: Constraint<N> + 'a,
    V: IndexMut<VertexId, Output = Vertex>,
{
    let mut constraint_index = base_index;
    for (constraint, ref_value) in iter {
        if !inactive_constraints[constraint_index] {
            let result = constraint.value_and_grad(vertices);
            if apply_constraint(result, ref_value, alpha, vertices) > l_threshold {
                inactive_constraints.set(constraint_index, true);
            }
        }
        constraint_index += 1;
    }
    constraint_index
}

// TODO: Implement more generic Xpbd function.
pub fn step_basic(
    params: &XpbdParams,
    state: XpbdState,
    mesh: &mut Tetrahedral,
    initial_value: &TetConstraintValues,
) -> XpbdState {
    let XpbdState {
        mut velocities,
        mut inactive_constraints,
    } = state;
    let XpbdParams {
        stiffness_volume,
        stiffness_length,
        n_substeps,
        time_substep,
        l_threshold_length,
        l_threshold_volume,
    } = params.clone();
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

        let mut constraint_index = 0;
        constraint_index += process_constraints(
            constraint_index,
            mesh.edges.iter().zip(initial_value.lengths.iter().copied()),
            &mut mesh.vertices,
            &mut inactive_constraints,
            l_threshold_length,
            stiffness_length / (time_substep * time_substep),
        );

        process_constraints(
            constraint_index,
            mesh.tetrahedra
                .iter()
                .zip(initial_value.volumes.iter().copied()),
            &mut mesh.vertices,
            &mut inactive_constraints,
            l_threshold_volume,
            stiffness_volume / (time_substep * time_substep),
        );

        // Update velocities based on position changes
        for (i, vertex) in mesh.vertices.iter().enumerate() {
            velocities[i] = (vertex.position - old_positions[i].position) / time_substep;
        }
    }
    XpbdState {
        velocities,
        inactive_constraints,
    }
}
