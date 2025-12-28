//! Implement xpbd on a tetrahedral mesh.

use std::ops::IndexMut;

use bitvec::vec::BitVec;
use rand::seq::SliceRandom;
use raylib::math::Vector3;

use crate::{
    constraint::{Constraint, TetConstraintValues, apply_constraint},
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
    pub stiffness_volume: f32,
    /// Stiffness for tetrahedral volume constraints.
    pub stiffness_length: f32,
    /// Number of substeps per simulation step.
    pub n_substeps: usize,
    /// Time step for each simulation substep.
    pub time_substep: f32,
    /// Length constraint force-threshold for deactivation.
    pub l_threshold_length: f32,
    /// Volume constraint force-threshold for deactivation.
    pub l_threshold_volume: f32,
    /// A constant acceleration applied to all vertices (e.g., gravity).
    pub constant_field: Vector3,
    /// Whether to shuffle constraint order for better convergence.
    pub shuffle_constraints: bool,
    /// Velocity damping factor (0.0 = no damping, 1.0 = full damping).
    pub damping: f32,
}

impl Default for XpbdParams {
    fn default() -> Self {
        Self {
            stiffness_length: 0.0,
            stiffness_volume: 0.0,
            n_substeps: 10,
            time_substep: 0.016 / 10.0,
            l_threshold_length: f32::INFINITY,
            l_threshold_volume: f32::INFINITY,
            constant_field: Vector3::new(0.0, -0.981, 0.0),
            shuffle_constraints: true,
            damping: 0.0,
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

/// Helper struct to solve constraints
pub struct ConstraintProcessor<'solver, V: IndexMut<VertexId, Output = Vertex>> {
    inactive_constraints: &'solver mut BitVec,
    vertices: &'solver mut V,
    constraint_index: usize,
}

impl<V: IndexMut<VertexId, Output = Vertex>> ConstraintProcessor<'_, V> {
    /// Process constraints from an iterator, applying them to the vertices and deactivating those that exceed the threshold.
    /// Internally, the index of each constraint in the iterator is used to track constraint active status.
    /// Thus, it is imperative that `process` is called each time with constraints in the same order.
    pub fn process<'a, I, C, const N: usize>(
        mut self,
        iter: I,
        l_threshold: f32,
        alpha: f32,
    ) -> Self
    where
        I: Iterator<Item = (&'a C, f32)>,
        C: Constraint<N> + 'a,
    {
        let base_index = self.constraint_index;
        let n = iter
            .enumerate() // <-- `i` used to identify each constraint and track for deactivation.
            // TODO: shuffle the iteration order for better convergence
            .fold(base_index, |counter, (i, (constraint, ref_value))| {
                let current_index = base_index + i;
                if !self.inactive_constraints[current_index] {
                    let result = constraint.value_and_grad(self.vertices);
                    if apply_constraint(result, ref_value, alpha, self.vertices) > l_threshold {
                        self.inactive_constraints.set(current_index, true);
                    }
                }
                counter + 1
            });
        self.constraint_index = n;
        self
    }

    /// Process constraints with shuffled iteration order for better convergence.
    /// Collects constraints into a vec and shuffles before processing.
    pub fn process_shuffled<'a, I, C, const N: usize>(
        mut self,
        iter: I,
        l_threshold: f32,
        alpha: f32,
    ) -> Self
    where
        I: Iterator<Item = (&'a C, f32)>,
        C: Constraint<N> + 'a,
    {
        let base_index = self.constraint_index;
        let mut constraints: Vec<(usize, (&'a C, f32))> = iter.enumerate().collect();
        
        // Shuffle the constraint order for better Gauss-Seidel convergence
        let mut rng = rand::thread_rng();
        constraints.shuffle(&mut rng);

        let n = constraints
            .into_iter()
            .fold(base_index, |counter, (i, (constraint, ref_value))| {
                let current_index = base_index + i;
                if !self.inactive_constraints[current_index] {
                    let result = constraint.value_and_grad(self.vertices);
                    if apply_constraint(result, ref_value, alpha, self.vertices) > l_threshold {
                        self.inactive_constraints.set(current_index, true);
                    }
                }
                counter + 1
            });
        self.constraint_index = n;
        self
    }
}

// TODO: Implement more generic Xpbd function.
/// Basic XPBD step function for tetrahedral meshes.
/// Additionally accepts a vertex correction function to handle collisions and other vertex corrections that need to be applied after the kinematic update.
pub fn step_basic<F>(
    params: &XpbdParams,
    state: XpbdState,
    mesh: &mut Tetrahedral,
    initial_value: &TetConstraintValues,
    vertex_correction: F,
) -> XpbdState
where
    F: FnMut(&mut Vertex),
{
    // TODO: Call `step` with appropriate arguments.
    step(
        params,
        state,
        &mut mesh.vertices,
        &mesh.constraints,
        initial_value,
        vertex_correction,
    )
}

pub trait ConstraintSet<V: IndexMut<VertexId, Output = Vertex>, I> {
    fn evaluate(&self, on: &V) -> I;
    fn solve(&self, processor: ConstraintProcessor<V>, params: &XpbdParams, reference: &I);
    /// Solve constraints with shuffled iteration order for better convergence.
    fn solve_shuffled(&self, processor: ConstraintProcessor<V>, params: &XpbdParams, reference: &I);
}

pub fn step<'v, I, F, C>(
    // generic over V
    params: &XpbdParams,
    mut state: XpbdState,
    vertices: &mut Vec<Vertex>,
    constraint_set: &C,
    initial_value: &I,
    mut post_kinematic_correction: F,
) -> XpbdState
where
    C: ConstraintSet<Vec<Vertex>, I>, // TODO: change constraint set to generic V
    // V: IndexMut<VertexId, Output = Vertex>, // TODO: Add appropriate trait bounds
    F: FnMut(&mut Vertex),
{
    for _ in 0..params.n_substeps {
        // copy old positions each time.
        let mut old_positions: Vec<Vector3> = vec![Vector3::zero(); state.velocities.len()];

        for (i, vertex) in vertices.into_iter().enumerate() {
            // save old position
            old_positions[i] = vertex.position;

            // unit mass for now
            vertex.position += params.constant_field * params.time_substep * params.time_substep
                + state.velocities[i] * params.time_substep;

            post_kinematic_correction(vertex);
        }

        let processor = ConstraintProcessor {
            inactive_constraints: &mut state.inactive_constraints,
            vertices,
            constraint_index: 0,
        };
        // Use shuffled or sequential constraint solving based on params
        if params.shuffle_constraints {
            constraint_set.solve_shuffled(processor, params, initial_value);
        } else {
            constraint_set.solve(processor, params, initial_value);
        }

        // Update velocities based on position changes
        for (i, vertex) in vertices.into_iter().enumerate() {
            let new_velocity = (vertex.position - old_positions[i]) / params.time_substep;
            // Apply damping to velocity
            state.velocities[i] = new_velocity * (1.0 - params.damping);
        }
    }
    state
}
