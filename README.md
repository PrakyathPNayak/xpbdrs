# XPBD Cloth Simulation

A Rust implementation of Extended Position Based Dynamics (XPBD) for cloth simulation using tetrahedral meshes with real-time 3D visualization.

## Overview

This project demonstrates physics-based cloth simulation using the Extended Position Based Dynamics (XPBD) algorithm. It can load tetrahedral mesh data from TetGen format files or a custom bincode-based format, and provides an interactive 3D visualization using Raylib.

## Features

- **XPBD Physics Engine**: Implementation of Extended Position Based Dynamics for stable cloth simulation
- **Tetrahedral Mesh Support**: Load and simulate cloth using tetrahedral mesh representations
- **Real-time Visualization**: Interactive 3D rendering with wireframe and face display modes
- **Multiple Input Formats**: Support for TetGen format (.node, .edge, .face, .ele) and binary format
- **Constraint-based Physics**: Edge length and tetrahedral volume constraints
- **Ground Collision**: Basic collision detection with ground plane

## Dependencies

- **Raylib**: 3D graphics and window management
- **Clap**: Command-line argument parsing
- **Serde/Bincode**: Mesh serialization and deserialization
- **Tracing**: Structured logging
