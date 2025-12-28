# XPBDRS

A Rust implementation of Extended Position Based Dynamics (XPBD) for physics-based simulation of deformable bodies using tetrahedral and triangulated surface meshes.

## Overview

This library provides a constraint-based physics engine implementing the Extended Position Based Dynamics algorithm. It supports multiple mesh representations and constraint types for simulating deformable objects with real-time visualization capabilities.

## Features

- **XPBD Physics Engine**: Stable constraint-based dynamics with configurable compliance parameters
- **Multi-Mesh Support**: Tetrahedral volumetric meshes and triangulated surface meshes
- **Constraint System**: Edge length, tetrahedral volume, and weak bending constraints
- **Adaptive Constraint Deactivation**: Force-threshold based constraint pruning for performance optimization
- **Real-time Visualization**: Interactive 3D rendering with wireframe and surface display modes
- **Mesh I/O**: TetGen format (.node, .edge, .face, .ele) and binary serialization support
- **Ground Collision**: Basic collision detection and response

## Algorithm

The XPBD method extends Position Based Dynamics (PBD) by incorporating material compliance parameters, enabling more physically accurate simulation of soft bodies. The implementation uses:

- Substep integration for numerical stability
- Gradient-based constraint projection
- Lagrange multiplier accumulation for proper material response

## Dependencies

- **Raylib**: 3D graphics rendering and interaction
- **Clap**: Command-line interface
- **Serde/Bincode**: Mesh serialization
- **Tracing**: Structured logging and performance profiling

## Usage

### Library

```rust
use xpbdrs::{mesh::Tetrahedral, xpbd::{XpbdParams, XpbdState, step_basic}};

// Load tetrahedral mesh
let mut mesh = Tetrahedral::from_files("mesh_prefix")?;
let initial_values = mesh.constraints.evaluate(&mesh.vertices);

// Configure simulation parameters
let params = XpbdParams {
    length_compliance: 0.001,
    volume_compliance: 0.001,
    n_substeps: 10,
    time_substep: 0.016 / 10.0,
    ..Default::default()
};

// Initialize simulation state
let mut state = XpbdState::new(mesh.vertices.len(), 
    mesh.constraints.edges.len() + mesh.constraints.tetrahedra.len());

// Simulation loop
loop {
    state = step_basic(&params, state, &mut mesh, &initial_values, 
        |v| v.position.y = v.position.y.max(0.0)); // ground collision
}
```

### Command Line

```bash
# Export TetGen files to binary format
cargo run -- export -i mesh_prefix -o mesh.bin

# Run interactive simulation
cargo run -- demo mesh.bin
```
