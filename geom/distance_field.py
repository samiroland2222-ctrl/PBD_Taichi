
import gmsh
import numpy as np


class Mesh:
    def __init__(self, verts: np.ndarray, faces: np.ndarray):
        self.verts = verts
        self.faces = faces


class TetMesh:
    def __init__(self, verts: np.ndarray, tets: np.ndarray):
        self.verts = verts  # (N, 3) float64
        self.tets = tets    # (M, 4) int32, 0-indexed


def boolean_merge_meshes(meshes: list[Mesh]) -> Mesh:

    gmsh.initialize()
    gmsh.model.add("boolean_merge")

    surf_tags = []
    for i, mesh in enumerate(meshes):
        # Add points
        point_tags = []
        for v in mesh.verts:
            tag = gmsh.model.occ.addPoint(float(v[0]), float(v[1]), float(v[2]))
            point_tags.append(tag)
        # Add triangles as surfaces
        surf_tags_mesh = []
        for f in mesh.faces:
            l1 = gmsh.model.occ.addLine(point_tags[f[0]], point_tags[f[1]])
            l2 = gmsh.model.occ.addLine(point_tags[f[1]], point_tags[f[2]])
            l3 = gmsh.model.occ.addLine(point_tags[f[2]], point_tags[f[0]])
            cl = gmsh.model.occ.addCurveLoop([l1, l2, l3])
            surf = gmsh.model.occ.addPlaneSurface([cl])
            surf_tags_mesh.append((2, surf))
        surf_tags.extend(surf_tags_mesh)

    gmsh.model.occ.synchronize()

    # Boolean union (fuse) all surfaces
    if len(surf_tags) > 1:
        out = gmsh.model.occ.fuse(surf_tags[:1], surf_tags[1:])
        merged_tags = out[0]
    else:
        merged_tags = surf_tags

    gmsh.model.occ.synchronize()

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Extract mesh nodes and elements
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    verts = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    faces = np.array(elem_node_tags[0], dtype=np.int32).reshape(-1, 3) - 1  # gmsh is 1-based

    gmsh.finalize()

    return Mesh(verts=verts, faces=faces)


def build_boundary_layer(surface_mesh: Mesh, layer_thickness: float, debug_save_path=None) -> TetMesh:
    """Build a tetrahedral boundary layer that skins the provided surface mesh.

    Uses gmsh's built-in ``extrudeBoundaryLayer`` to extrude the surface
    outward along vertex normals, creating N prism layers with a
    geometric-progression thickness distribution.  Prisms are split into
    tetrahedra before returning.

    Parameters
    ----------
    surface_mesh : Mesh
        Triangulated surface mesh.
    layer_thickness : float
        Total thickness of the boundary layer.
    debug_save_path : str
        If provided, saves the resulting model to the given path.

    Returns
    -------
    TetMesh
        Tetrahedral mesh of the boundary-layer region.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # suppress console output
    gmsh.model.add("boundary_layer")

    # ── 1. Register the surface mesh as a discrete entity ─────────────────────
    n_verts = len(surface_mesh.verts)
    surf_entity = gmsh.model.addDiscreteEntity(2)
    node_tags = list(range(1, n_verts + 1))
    coords_flat = surface_mesh.verts.flatten().tolist()
    gmsh.model.mesh.addNodes(2, surf_entity, node_tags, coords_flat)

    n_faces = len(surface_mesh.faces)
    face_tags = list(range(1, n_faces + 1))
    conn_flat = (surface_mesh.faces + 1).flatten().tolist()   # 1-indexed
    gmsh.model.mesh.addElements(2, surf_entity, [2], [face_tags], [conn_flat])

    # Classify the mesh topology and build discrete CAD entities so that
    # extrudeBoundaryLayer can find proper surface/curve/point entities.
    gmsh.model.mesh.classifySurfaces(np.pi, True, True, np.pi)
    gmsh.model.mesh.createGeometry()
    gmsh.model.geo.synchronize()

    # ── 2. Build layer heights (geometric progression, sum = layer_thickness) ─
    N = 1        # number of prism layers
    r = 1.5      # growth ratio between successive layers
    # d_i = d0 * r^i;  sum_{i=0}^{N-1} d_i = d0*(r^N-1)/(r-1) = layer_thickness
    d0 = layer_thickness * (r - 1.0) / (r ** N - 1.0)
    heights = []
    h = 0.0
    for i in range(N):
        h += d0 * r ** i
        heights.append(h)

    # ── 3. Extrude boundary layer ──────────────────────────────────────────────
    surfaces = gmsh.model.getEntities(2)
    gmsh.model.geo.extrudeBoundaryLayer(surfaces, [1] * N, heights, True)
    gmsh.model.geo.synchronize()

    # ── 4. Generate 3D mesh ────────────────────────────────────────────────────
    gmsh.model.mesh.generate(3)

    # ── 5. Extract nodes ───────────────────────────────────────────────────────
    node_tags_out, node_coords_out, _ = gmsh.model.mesh.getNodes()
    verts_out = np.array(node_coords_out, dtype=np.float64).reshape(-1, 3)
    max_tag = int(max(node_tags_out))
    tag_to_idx = np.full(max_tag + 1, -1, dtype=np.int32)
    for local_idx, t in enumerate(node_tags_out):
        tag_to_idx[int(t)] = local_idx

    # ── 6. Extract 3D elements; split prisms → tetrahedra ─────────────────────
    # Type 4 = 4-node tetrahedron, Type 6 = 6-node triangular prism (wedge).
    # extrudeBoundaryLayer produces prisms; split each into 3 tets.
    elem_types, _, elem_node_tags_out = gmsh.model.mesh.getElements(dim=3)
    all_tets: list[np.ndarray] = []
    for etype, enodes in zip(elem_types, elem_node_tags_out):
        raw = np.array(enodes, dtype=np.int32)
        if etype == 4:                          # 4-node tetrahedron
            all_tets.append(tag_to_idx[raw.reshape(-1, 4)])
        elif etype == 6:                        # 6-node prism → 3 tets
            v = tag_to_idx[raw.reshape(-1, 6)] # shape (M, 6)
            # bottom triangle: v[:,0:3], top triangle: v[:,3:6]
            # lateral edges: v0-v3, v1-v4, v2-v5
            all_tets.append(np.stack([v[:, 0], v[:, 1], v[:, 2], v[:, 5]], axis=1))
            all_tets.append(np.stack([v[:, 0], v[:, 1], v[:, 5], v[:, 4]], axis=1))
            all_tets.append(np.stack([v[:, 0], v[:, 4], v[:, 5], v[:, 3]], axis=1))

    tets_out = (np.concatenate(all_tets, axis=0)
                if all_tets else np.zeros((0, 4), dtype=np.int32))

    if debug_save_path:
        gmsh.write(debug_save_path)

    gmsh.finalize()

    return TetMesh(verts=verts_out, tets=tets_out)

