//! Module to handle tetrahedral meshes.

use raylib::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::{Index, IndexMut};
use std::path::Path;

use crate::constraint::TetConstraints;
use crate::xpbd::XpbdState;

fn default_inv_mass() -> f32 {
    1.0
}

/// A vertex in 3D space with position and inverse mass.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Vertex {
    /// 3D position of the vertex.
    pub position: Vector3,
    /// Inverse mass (1/mass) of the vertex.
    #[serde(default = "default_inv_mass")]
    pub inv_mass: f32,
}

/// Unique identifier for a vertex.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct VertexId(pub u32);

/// A tetrahedron defined by four vertex indices.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Tetrahedron {
    /// Four vertex indices forming the tetrahedron.
    pub indices: [VertexId; 4],
}

/// Unique identifier for a tetrahedron.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TetrahedronId(pub u32);

/// An edge connecting two vertices.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Edge(pub VertexId, pub VertexId);

/// Unique identifier for an edge.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct EdgeId(pub u32);

/// A triangular face defined by three vertex indices and their edges.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Triangle {
    /// Three vertex indices forming the triangle.
    pub verts: [VertexId; 3],
    /// Three edge indices connecting the vertices (None if no constraint exists).
    pub edges: [Option<EdgeId>; 3],
}

/// Result type for mesh operations.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Struct to contain data of a delanuay tetrahedralized mesh.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Tetrahedral {
    /// Vertices of the tetrahedral mesh.
    pub vertices: Vec<Vertex>,
    /// Constraints for physics simulation.
    pub constraints: TetConstraints,
    /// Triangular faces of the mesh.
    pub faces: Vec<Triangle>,
}

impl Tetrahedral {
    /// Parse a generic tetgen file format.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or parsed.
    fn parse_file<T>(filename: &str, processor: impl Fn(&[&str]) -> Result<T>) -> Result<Vec<T>> {
        let file = File::open(filename)?;

        let mut lines = BufReader::new(file).lines();
        let count: usize = lines
            .next()
            .ok_or("Empty file")??
            .split_whitespace()
            .next()
            .ok_or("Invalid first line")?
            .parse()?;

        lines
            .map_while(std::result::Result::ok)
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| {
                let tokens: Vec<&str> = line.split_whitespace().collect();
                processor(&tokens)
            })
            .take(count)
            .collect()
    }

    fn parse_indices(tokens: &[&str], start: usize, count: usize) -> Result<Vec<u32>> {
        tokens[start..start + count]
            .iter()
            .map(|&t| t.parse().map_err(Into::into))
            .collect()
    }

    /// Translate the entire mesh by a vector.
    pub fn translate(&mut self, by: Vector3) {
        for vertex in &mut self.vertices {
            vertex.position += by;
        }
    }

    /// Load tetrahedral mesh from tetgen files.
    ///
    /// # Errors
    /// Returns an error if files cannot be read or parsed.
    pub fn from_files(prefix: &str) -> Result<Self> {
        // TODO: Clean up dedup and hashmap logic using generics.
        //  Preferably, Tetrahedron and Edge should manually implement Hash and Eq traits emphasizing unordered equality.
        //  Then any generic dedup + map can work.
        let vertices = Self::load_vertices(prefix)?;
        let edges = Self::load_and_dedup_edges(prefix)?;
        let face_triangles = Self::load_face_vertices(prefix)?;
        let (edges, faces) = Self::build_faces_with_edges(edges, face_triangles);
        let tetrahedra = Self::load_and_dedup_tetrahedra(prefix)?;

        Ok(Self {
            vertices,
            constraints: TetConstraints { edges, tetrahedra },
            faces,
        })
    }

    fn load_vertices(prefix: &str) -> Result<Vec<Vertex>> {
        Self::parse_file(&format!("{prefix}.node"), |tokens| {
            let coords: Vec<f32> = tokens[1..4]
                .iter()
                .map(|&t| t.parse().map_err(Into::into))
                .collect::<Result<_>>()?;
            Ok(Vertex {
                position: Vector3::new(coords[0], coords[1], coords[2]),
                inv_mass: 1.0,
            })
        })
    }

    fn load_and_dedup_edges(prefix: &str) -> Result<Vec<Edge>> {
        use std::collections::HashSet;
        use tracing::warn;

        let edges = if Path::new(&format!("{prefix}.edge")).exists() {
            Self::parse_file(&format!("{prefix}.edge"), |tokens| {
                let ids = Self::parse_indices(tokens, 1, 2)?;
                Ok(Edge(VertexId(ids[0]), VertexId(ids[1])))
            })?
        } else {
            Vec::new()
        };

        // Dedup edges
        let mut edge_set = HashSet::new();
        let mut dedup_edges = Vec::new();
        for edge in edges {
            // Normalize edge to ensure consistent ordering
            let normalized = if edge.0.0 <= edge.1.0 {
                (edge.0, edge.1)
            } else {
                (edge.1, edge.0)
            };

            if edge_set.insert(normalized) {
                dedup_edges.push(Edge(normalized.0, normalized.1));
            } else {
                warn!(
                    "Duplicate edge constraint found: {:?} - {:?}",
                    normalized.0, normalized.1
                );
            }
        }

        Ok(dedup_edges)
    }

    fn load_face_vertices(prefix: &str) -> Result<Vec<[VertexId; 3]>> {
        if Path::new(&format!("{prefix}.face")).exists() {
            Self::parse_file(&format!("{prefix}.face"), |tokens| {
                let ids = Self::parse_indices(tokens, 1, 3)?;
                Ok([VertexId(ids[0]), VertexId(ids[1]), VertexId(ids[2])])
            })
        } else {
            Ok(Vec::new())
        }
    }

    fn build_faces_with_edges(
        edges: Vec<Edge>,
        face_triangles: Vec<[VertexId; 3]>,
    ) -> (Vec<Edge>, Vec<Triangle>) {
        use std::collections::HashMap;

        let edge_map: HashMap<Edge, EdgeId> = edges
            .iter()
            .enumerate()
            .map(|(i, edge)| {
                (
                    edge.clone(),
                    EdgeId(
                        i.try_into()
                            .expect("Edge count should not be more than u32::MAX+1"),
                    ),
                )
            })
            .collect();

        let faces = face_triangles
            .into_iter()
            .map(|verts| {
                let edge_ids = [
                    Self::find_existing_edge_id(&edge_map, verts[0], verts[1]),
                    Self::find_existing_edge_id(&edge_map, verts[1], verts[2]),
                    Self::find_existing_edge_id(&edge_map, verts[2], verts[0]),
                ];

                Triangle {
                    verts,
                    edges: edge_ids,
                }
            })
            .collect();

        (edges, faces)
    }

    fn find_existing_edge_id(
        edge_map: &std::collections::HashMap<Edge, EdgeId>,
        v1: VertexId,
        v2: VertexId,
    ) -> Option<EdgeId> {
        let normalized_edge = if v1.0 <= v2.0 {
            Edge(v1, v2)
        } else {
            Edge(v2, v1)
        };
        // TODO: Maybe consider inferring missing edges?
        edge_map.get(&normalized_edge).copied()
    }

    fn load_and_dedup_tetrahedra(prefix: &str) -> Result<Vec<Tetrahedron>> {
        use std::collections::HashSet;
        use tracing::warn;

        let tetrahedra = if Path::new(&format!("{prefix}.ele")).exists() {
            Self::parse_file(&format!("{prefix}.ele"), |tokens| {
                let ids = Self::parse_indices(tokens, 1, 4)?;
                Ok(Tetrahedron {
                    indices: [
                        VertexId(ids[0]),
                        VertexId(ids[1]),
                        VertexId(ids[2]),
                        VertexId(ids[3]),
                    ],
                })
            })?
        } else {
            Vec::new()
        };

        // Dedup tetrahedra
        let mut tetra_set = HashSet::new();
        let dedup_tetrahedra = tetrahedra
            .into_iter()
            .filter(|tetra| {
                // Sort indices for consistent comparison
                let mut sorted_indices = tetra.indices;
                sorted_indices.sort_by_key(|id| id.0);

                if tetra_set.insert(sorted_indices) {
                    true
                } else {
                    warn!(
                        "Duplicate tetrahedron constraint found: {:?}",
                        sorted_indices
                    );
                    false
                }
            })
            .collect();

        Ok(dedup_tetrahedra)
    }

    /// Load tetrahedral mesh from bincode file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or deserialized.
    pub fn from_bincode(filename: &str) -> Result<Self> {
        let data = std::fs::read(filename)?;
        let mesh: Self = bincode::deserialize(&data)?;
        Ok(mesh)
    }

    /// Export mesh to bincode format.
    ///
    /// # Errors
    /// Returns an error if serialization fails or file cannot be written.
    pub fn export_to_bincode(&self, output_path: &str) -> Result<()> {
        use std::io::Write;
        use tracing::{debug, info};

        info!("Serializing to binary format");
        let encoded = bincode::serialize(self)?;

        let mut file = std::fs::File::create(output_path)?;
        file.write_all(&encoded)?;

        info!(
            output_path,
            size_bytes = encoded.len(),
            "Successfully exported mesh"
        );

        // Verify deserialization works
        debug!("Verifying serialized data");
        let _: Self = bincode::deserialize(&encoded)?;
        debug!("Verification successful");

        Ok(())
    }

    /// Load mesh with automatic format detection.
    ///
    /// # Errors
    /// Returns an error if the file format is unsupported or loading fails.
    pub fn load_mesh(mesh_path: &str) -> Result<Self> {
        use tracing::{debug, error, info};

        info!(mesh_path, "Attempting to load mesh");

        let mesh = if std::path::Path::new(mesh_path)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
        {
            debug!("Loading as bincode file");
            Self::from_bincode(mesh_path)
        } else {
            debug!("Loading as tetgen files");
            Self::from_files(mesh_path)
        };

        match &mesh {
            Ok(m) => {
                info!(
                    vertices = m.vertices.len(),
                    edges = m.constraints.edges.len(),
                    faces = m.faces.len(),
                    tetrahedra = m.constraints.tetrahedra.len(),
                    "Mesh loaded successfully"
                );
            }
            Err(e) => {
                error!(mesh_path, error = %e, "Failed to load mesh");
            }
        }

        mesh
    }

    /// Draw wireframe of the mesh
    pub fn draw_wireframe(&self, d3: &mut RaylibMode3D<RaylibDrawHandle>, color: Color) {
        // Draw explicit edges if available
        for edge in &self.constraints.edges {
            if let (Some(v1), Some(v2)) = (
                self.vertices.get((edge.0.0 - 1) as usize),
                self.vertices.get((edge.1.0 - 1) as usize),
            ) {
                let start = v1.position;
                let end = v2.position;
                d3.draw_line_3D(start, end, color);
            }
        }
    }

    /// Get bounding box of the mesh
    #[must_use]
    pub fn bounding_box(&self) -> (Vector3, Vector3) {
        if self.vertices.is_empty() {
            return (Vector3::zero(), Vector3::zero());
        }

        let mut min = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        for vertex in &self.vertices {
            min.x = min.x.min(vertex.position.x);
            min.y = min.y.min(vertex.position.y);
            min.z = min.z.min(vertex.position.z);
            max.x = max.x.max(vertex.position.x);
            max.y = max.y.max(vertex.position.y);
            max.z = max.z.max(vertex.position.z);
        }

        (min, max)
    }

    /// Draw filled faces
    pub fn draw_faces(
        &self,
        d3: &mut RaylibMode3D<RaylibDrawHandle>,
        state: &XpbdState,
        color: Color,
    ) {
        for face in &self.faces {
            let verts = [
                self.vertices[(face.verts[0].0 - 1) as usize],
                self.vertices[(face.verts[1].0 - 1) as usize],
                self.vertices[(face.verts[2].0 - 1) as usize],
            ];

            // A triangle is "torn" if any of its corresponding edge constraints are inactive.
            let torn = face
                .edges
                .iter()
                .filter_map(|e| e.as_ref()) // Only check edges that have constraints
                .any(|e| state.constraint_inactive(e.0 as usize)); // in this constaint set, edges are solved first, so base index is 0.

            if !torn {
                d3.draw_triangle3D(
                    verts[0].position,
                    verts[1].position,
                    verts[2].position,
                    color,
                );
            }
        }
    }
}

impl Index<VertexId> for Vec<Vertex> {
    type Output = Vertex;

    fn index(&self, index: VertexId) -> &Self::Output {
        self.get((index.0 - 1) as usize).unwrap_or_else(|| {
            panic!(
                "Invalid vertex id: {}, only {} available.",
                index.0,
                self.len()
            )
        })
    }
}

impl IndexMut<VertexId> for Vec<Vertex> {
    fn index_mut(&mut self, index: VertexId) -> &mut Self::Output {
        let len = self.len();
        self.get_mut((index.0 - 1) as usize)
            .unwrap_or_else(|| panic!("Invalid vertex id: {}, only {} available.", index.0, len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn create_test_files(prefix: &str) {
        fs::write(
            format!("{prefix}.node"),
            "2 3 0 0\n1 0.0 0.0 0.0\n2 1.0 1.0 1.0\n",
        )
        .unwrap();
        fs::write(format!("{prefix}.edge"), "1 0\n1 1 2\n").unwrap();
        fs::write(format!("{prefix}.face"), "1 0\n1 1 2 1\n").unwrap();
        fs::write(format!("{prefix}.ele"), "1 4 0\n1 1 2 1 2\n").unwrap();
    }

    #[test]
    fn test_parse() {
        let prefix = "test";
        create_test_files(prefix);

        let mesh = Tetrahedral::from_files(prefix).unwrap();
        assert_eq!(mesh.vertices.len(), 2);
        assert_eq!(mesh.constraints.edges.len(), 1);
        assert_eq!(mesh.faces.len(), 1);
        assert_eq!(mesh.constraints.tetrahedra.len(), 1);

        // Cleanup
        for ext in &["node", "edge", "face", "ele"] {
            let _ = fs::remove_file(format!("{prefix}.{ext}"));
        }
    }
}
