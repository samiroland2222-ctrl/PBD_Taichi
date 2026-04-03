import gmsh
import numpy as np


class BasicTriMesh:
    def __init__(self, verts: np.ndarray, faces: np.ndarray):
        self.verts = verts
        self.faces = faces


class BasicTetMesh:
    def __init__(self, verts: np.ndarray, tets: np.ndarray):
        self.verts = verts  # (N, 3) float64
        self.tets = tets    # (M, 4) int32, 0-indexed


# ── private helpers ────────────────────────────────────────────────────────────

def _surface_area(mesh: BasicTriMesh) -> float:
    """Total surface area of a triangle mesh."""
    a = mesh.verts[mesh.faces[:, 0]]
    b = mesh.verts[mesh.faces[:, 1]]
    c = mesh.verts[mesh.faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()


def _target_edge_len(mesh: BasicTriMesh, target_tet_count: int, n_layers: int) -> float:
    """Characteristic edge length that yields ~``target_tet_count`` tetrahedra.

    Equilateral-triangle packing:
      A_tri = (√3/4)·e²  →  n_tri ≈ 4·A / (√3·e²)
      n_tet = 3·N·n_tri  →  e = sqrt(12·N·A / (√3·target))
    """
    area = _surface_area(mesh)
    return float(np.sqrt(12.0 * n_layers * area / (np.sqrt(3.0) * target_tet_count)))


def _remesh_surface(mesh: BasicTriMesh, target_edge_len: float) -> BasicTriMesh:
    """Isotropic remesh to approximately ``target_edge_len`` using pymeshlab.

    Uses two passes with a correction step:
      1. Remesh with the requested edge length.
      2. Measure the actual mean edge length of the result.
      3. Scale the target by ``sqrt(actual_n_tri / target_n_tri)`` and remesh
         once more so the output is within ~5 % of the target triangle count.

    ``targetlen`` requires a ``pymeshlab.PureValue`` (world-space units).
    """
    import pymeshlab

    def _do_remesh(verts: np.ndarray, faces: np.ndarray, e: float):
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces))
        ms.meshing_isotropic_explicit_remeshing(
            targetlen=pymeshlab.PureValue(float(e)),
            iterations=20,
        )
        m = ms.current_mesh()
        return (
            np.asarray(m.vertex_matrix(), dtype=np.float64),
            np.asarray(m.face_matrix(),   dtype=np.int32),
        )

    # ── pass 1 ────────────────────────────────────────────────────────────────
    v1, f1 = _do_remesh(mesh.verts, mesh.faces, target_edge_len)

    # ── correction: scale e so actual n_tri → target n_tri ────────────────────
    # n_tri ∝ 1/e²  →  e_new = e_old * sqrt(n_tri_actual / n_tri_target)
    n_tri_target = _surface_area(mesh) * 4.0 / (np.sqrt(3.0) * target_edge_len ** 2)
    n_tri_actual = len(f1)
    if n_tri_actual > 0:
        e_corrected = target_edge_len * np.sqrt(n_tri_actual / n_tri_target)
        v1, f1 = _do_remesh(v1, f1, float(e_corrected))

    return BasicTriMesh(
        verts=np.asarray(v1, dtype=np.float64),
        faces=np.asarray(f1, dtype=np.int32),
    )


def _vertex_normals(mesh: BasicTriMesh) -> np.ndarray:
    """Per-vertex outward normals: area-weighted average, unit-normalised."""
    verts, faces = mesh.verts, mesh.faces
    n = np.zeros_like(verts)
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    fn = np.cross(b - a, c - a)        # 2×area-weighted face normals
    np.add.at(n, faces[:, 0], fn)
    np.add.at(n, faces[:, 1], fn)
    np.add.at(n, faces[:, 2], fn)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(norms > 0, n / norms, 0.0)


def _extrude_to_tets(mesh: BasicTriMesh, heights: list[float], debug_save_path: str | None=None) -> BasicTetMesh:
    """Pure-numpy boundary layer extrusion along vertex normals.

    Vertex layout: [base, layer_0, layer_1, ..., layer_{N-1}]
    Each face × each layer → 1 prism → 3 tetrahedra.
    Exact output: ``3 · N · n_faces`` tets.
    """
    verts, faces = mesh.verts, mesh.faces
    normals = _vertex_normals(mesh)
    n_v = len(verts)

    rings = [verts] + [verts + normals * h for h in heights]
    all_verts = np.concatenate(rings, axis=0)   # (N+1)·n_v × 3

    all_tets: list[np.ndarray] = []
    for layer in range(len(heights)):
        b_off = layer * n_v           # bottom ring offset
        t_off = (layer + 1) * n_v    # top    ring offset
        v0 = faces[:, 0] + b_off;  v1 = faces[:, 1] + b_off;  v2 = faces[:, 2] + b_off
        v3 = faces[:, 0] + t_off;  v4 = faces[:, 1] + t_off;  v5 = faces[:, 2] + t_off
        # canonical 3-tet split of prism (v0,v1,v2 | v3,v4,v5)
        all_tets.append(np.stack([v0, v1, v2, v5], axis=1))
        all_tets.append(np.stack([v0, v1, v5, v4], axis=1))
        all_tets.append(np.stack([v0, v4, v5, v3], axis=1))

    mesh = BasicTetMesh(
        verts=all_verts.astype(np.float64),
        tets=np.concatenate(all_tets, axis=0).astype(np.int32),
    )

    # save to mesh file (msh, vtk, ply, etc) using meshio or gmsh_41_writer
    if debug_save_path:
        if debug_save_path.endswith(".msh"):
            from PBD_Taichi.utils import gmsh_41_writer
            coords = mesh.verts
            node_tags = np.arange(1, len(coords)+1, dtype=np.int32)
            # tets are 0-based, must convert to 1-based node tags
            tets = mesh.tets + 1
            tet_tags = np.arange(1, len(tets)+1, dtype=np.int32)
            gmsh_41_writer.write_msh41_single_block(coords, node_tags, tets, tet_tags, debug_save_path)
        else:
            import meshio
            cells = [("tetra", mesh.tets)]
            meshio.write_points_cells(
                debug_save_path,
                mesh.verts,
                cells,
            )

    return mesh


# ── public API ─────────────────────────────────────────────────────────────────

def boolean_merge_meshes(meshes: list[BasicTriMesh]) -> BasicTriMesh:

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

    return BasicTriMesh(verts=verts, faces=faces)


def build_boundary_layer(
    surface_mesh: BasicTriMesh,
    layer_thickness: float,
    reparamterize_target_tet_count: int | None = None,
    remesh_surface: bool = False,
    debug_save_path: str | None = None,
) -> BasicTetMesh:
    """Build a tetrahedral boundary layer that skins the provided surface mesh.

    Uses gmsh's built-in ``extrudeBoundaryLayer`` to extrude the surface
    outward along vertex normals, creating N prism layers with a
    geometric-progression thickness distribution.  Prisms are split into
    tetrahedra before returning.

    Parameters
    ----------
    surface_mesh : BasicTriMesh
        Triangulated surface mesh.
    layer_thickness : float
        Total thickness of the boundary layer.
    reparamterize_target_tet_count : int, optional
        Desired approximate output tet count.  Two strategies are available,
        selected by ``remesh_surface``:

        * ``remesh_surface=False`` (**Option A** — gmsh mesh-size field):
          Computes a characteristic length ``e`` from the equilateral-triangle
          packing formula and installs it as a uniform ``MathEval`` background
          mesh field before ``generate(3)``.  Single gmsh pass; accuracy ~±20 %.

        * ``remesh_surface=True`` (**Option C** — pymeshlab + numpy extrusion):
          Isotropically remeshes the input surface to the target edge length
          with pymeshlab, then extrudes the remeshed surface in pure numpy
          (bypassing ``generate(3)``).  Output tet count is **exact**:
          ``3 · N · n_faces_remeshed``.  ``debug_save_path`` is ignored in
          this mode.
    remesh_surface : bool
        Select Option C when ``True``, Option A when ``False``.
        Ignored if ``reparamterize_target_tet_count`` is ``None``.
    debug_save_path : str, optional
        If provided (and ``remesh_surface=False``), saves the gmsh model.

    Returns
    -------
    BasicTetMesh
        Tetrahedral mesh of the boundary-layer region.
    """
    # ── layer height schedule (geometric progression) ─────────────────────────
    N = 1
    r = 1.5
    d0 = layer_thickness * (r - 1.0) / (r ** N - 1.0)
    heights: list[float] = []
    h = 0.0
    for i in range(N):
        h += d0 * r ** i
        heights.append(h)

    # ── Option C: pymeshlab remesh → pure-numpy extrusion (exact count) ───────
    if reparamterize_target_tet_count is not None and remesh_surface:
        e = _target_edge_len(surface_mesh, reparamterize_target_tet_count, N)
        surface_mesh = _remesh_surface(surface_mesh, e)
        return _extrude_to_tets(surface_mesh, heights, debug_save_path=debug_save_path)

    # ── gmsh pipeline (Option A or no reparameterisation) ─────────────────────
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

    # ── Option A: uniform MathEval background field ────────────────────────────
    # Uses the same pattern as breast_mesh_generator.py. Installing the field
    # before extrudeBoundaryLayer means generate(3) picks it up for both the
    # surface re-mesh and the prism extrusion density.
    if reparamterize_target_tet_count is not None:
        lc = _target_edge_len(surface_mesh, reparamterize_target_tet_count, N)
        f = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(f, "F", repr(lc))
        gmsh.model.mesh.field.setAsBackgroundMesh(f)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints",           0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",        0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary",   0)

    # ── 2. Extrude boundary layer ──────────────────────────────────────────────
    surfaces = gmsh.model.getEntities(2)
    gmsh.model.geo.extrudeBoundaryLayer(surfaces, [1] * N, heights, True)
    gmsh.model.geo.synchronize()

    # ── 3. Generate 3D mesh ────────────────────────────────────────────────────
    gmsh.model.mesh.generate(3)

    # ── 4. Extract nodes ───────────────────────────────────────────────────────
    node_tags_out, node_coords_out, _ = gmsh.model.mesh.getNodes()
    verts_out = np.array(node_coords_out, dtype=np.float64).reshape(-1, 3)
    max_tag = int(max(node_tags_out))
    tag_to_idx = np.full(max_tag + 1, -1, dtype=np.int32)
    for local_idx, t in enumerate(node_tags_out):
        tag_to_idx[int(t)] = local_idx

    # ── 5. Extract 3D elements; split prisms → tetrahedra ─────────────────────
    elem_types, _, elem_node_tags_out = gmsh.model.mesh.getElements(dim=3)
    all_tets: list[np.ndarray] = []
    for etype, enodes in zip(elem_types, elem_node_tags_out):
        raw = np.array(enodes, dtype=np.int32)
        if etype == 4:                          # 4-node tetrahedron
            all_tets.append(tag_to_idx[raw.reshape(-1, 4)])
        elif etype == 6:                        # 6-node prism → 3 tets
            v = tag_to_idx[raw.reshape(-1, 6)]
            all_tets.append(np.stack([v[:, 0], v[:, 1], v[:, 2], v[:, 5]], axis=1))
            all_tets.append(np.stack([v[:, 0], v[:, 1], v[:, 5], v[:, 4]], axis=1))
            all_tets.append(np.stack([v[:, 0], v[:, 4], v[:, 5], v[:, 3]], axis=1))

    tets_out = (np.concatenate(all_tets, axis=0)
                if all_tets else np.zeros((0, 4), dtype=np.int32))

    if debug_save_path:
        gmsh.write(debug_save_path)

    gmsh.finalize()
    return BasicTetMesh(verts=verts_out, tets=tets_out)

