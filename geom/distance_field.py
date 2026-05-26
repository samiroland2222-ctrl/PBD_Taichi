from collections import Counter

import gmsh
import numpy as np
from meshlib import mrmeshnumpy, mrmeshpy

class BasicTriMesh:
    def __init__(self, verts: np.ndarray, faces: np.ndarray, label: str = None):
        self.verts = verts # [N_v, 3]
        self.faces = faces # [N_f, 3]
        self.label = label

    def deduplicated(self, tol: float = 1e-8) -> 'BasicTriMesh':
        """Return a new BasicTriMesh with duplicate vertices removed and faces remapped.
        Vertices within `tol` are considered identical.
        """
        # Round vertices to avoid floating-point noise
        verts_rounded = np.round(self.verts / tol) * tol
        # Find unique vertices and mapping from old to new
        verts_unique, inverse = np.unique(verts_rounded, axis=0, return_inverse=True)
        # Remap faces
        faces_dedup = inverse[self.faces]
        return BasicTriMesh(verts_unique, faces_dedup)

    def mirrored_x(self) -> 'BasicTriMesh':
        mirrored_verts = self.verts.copy()
        mirrored_faces = self.faces.copy()
        # bilateral mirror around Y-Z plane
        mirrored_verts[:, 0] *= -1
        return BasicTriMesh(mirrored_verts, mirrored_faces).swapped_winding_order()

    def swapped_winding_order(self) -> 'BasicTriMesh':
        swapped_faces = self.faces.copy()
        # swap the first two vertices of each face to flip winding order
        swapped_faces[:, 0] = self.faces[:, 1]
        swapped_faces[:, 1] = self.faces[:, 0]
        return BasicTriMesh(self.verts, swapped_faces)

    def save(self, path: str):
        import meshio
        meshio.write_points_cells(
            path,
            self.verts,
            [("triangle", self.faces)],
        )


def extract_surface_triangles(verts: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Extract the boundary (surface) triangles of a tetrahedral mesh.

    Surface faces are those belonging to exactly one tetrahedron.  Winding
    is corrected so each triangle's outward normal points away from the
    opposite interior vertex of its owning tet.

    Parameters
    ----------
    verts : (N, 3) float array
    tets  : (M, 4) int array, 0-indexed

    Returns
    -------
    faces : (F, 3) int32 – surface triangles with consistent outward winding.
    """
    # Each tet contributes 4 faces; (face_verts, opposite_vertex_local_idx)
    tet_face_defs = [
        ([0, 1, 2], 3),
        ([0, 1, 3], 2),
        ([0, 2, 3], 1),
        ([1, 2, 3], 0),
    ]

    # Map sorted-vertex-key → (a, b, c, opposite) or None (shared → interior)
    face_map: dict = {}
    for tet in tets:
        for face_verts, opp_local in tet_face_defs:
            key = tuple(sorted(tet[i] for i in face_verts))
            if key in face_map:
                face_map[key] = None          # shared by two tets → interior
            else:
                face_map[key] = (int(tet[face_verts[0]]),
                                 int(tet[face_verts[1]]),
                                 int(tet[face_verts[2]]),
                                 int(tet[opp_local]))

    surface = []
    for val in face_map.values():
        if val is None:
            continue
        a_i, b_i, c_i, opp_i = val
        a, b, c, opp = verts[a_i], verts[b_i], verts[c_i], verts[opp_i]
        # Flip winding if normal points toward the opposite (interior) vertex
        if np.dot(np.cross(b - a, c - a), a - opp) < 0:
            surface.append([a_i, c_i, b_i])
        else:
            surface.append([a_i, b_i, c_i])

    return np.array(surface, dtype=np.int32) if surface else np.zeros((0, 3), dtype=np.int32)


class BasicTetMesh:
    def __init__(self, verts: np.ndarray, tets: np.ndarray):
        self.verts = verts  # (N, 3) float64
        self.tets = tets    # (M, 4) int32, 0-indexed

    def surface_faces(self) -> np.ndarray:
        """Return the surface (boundary) triangles of this tet mesh.

        Surface triangles are those belonging to exactly one tetrahedron.
        Winding is corrected to be consistently outward-facing.

        Returns
        -------
        faces : (F, 3) int32 – vertex-index triples of each surface triangle.
        """
        return extract_surface_triangles(self.verts, self.tets)


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

    def _do_remesh(verts: np.ndarray, faces: np.ndarray, e_pct: float):
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces))
        ms.meshing_isotropic_explicit_remeshing(
            #featuredeg=90,
            adaptive=True,
            targetlen=pymeshlab.PureValue(float(e_pct)),
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
    from PBD_Taichi.geom import geom3d
    return geom3d.vertex_normals_trimesh(mesh.verts, mesh.faces)


def _bary_interp_normals(
    query_pts: np.ndarray,    # (Q, 3)
    ref_verts: np.ndarray,    # (V, 3)
    ref_faces: np.ndarray,    # (F, 3) int
    ref_vnorms: np.ndarray,   # (V, 3) unit normals
    k_faces: int = 8,
) -> np.ndarray:
    """Smooth normals for each query point by projecting onto the nearest
    reference triangles and blending vertex normals at the closest point.

    For each query point:
      1. Find ``k_faces`` nearest face centroids via KDTree.
      2. Compute exact closest point on each candidate triangle (Ericson
         algorithm, fully vectorised over (Q, K)).
      3. Pick the closest triangle, compute barycentric coordinates, and
         interpolate the three face-vertex normals.

    Returns unit normals, shape (Q, 3).
    """
    from scipy.spatial import KDTree

    centroids = (ref_verts[ref_faces[:, 0]] +
                 ref_verts[ref_faces[:, 1]] +
                 ref_verts[ref_faces[:, 2]]) / 3.0          # (F, 3)
    k_eff = min(k_faces, len(centroids))
    tree = KDTree(centroids)
    _, near_fi = tree.query(query_pts, k=k_eff)             # (Q,) or (Q, k)
    if near_fi.ndim == 1:
        near_fi = near_fi[:, np.newaxis]

    Q, K = near_fi.shape

    a_idx = ref_faces[near_fi, 0]                           # (Q, K)
    b_idx = ref_faces[near_fi, 1]
    c_idx = ref_faces[near_fi, 2]

    A = ref_verts[a_idx]                                    # (Q, K, 3)
    B = ref_verts[b_idx]
    C = ref_verts[c_idx]
    P = query_pts[:, np.newaxis, :]                         # (Q, 1, 3) → broadcasts

    # ── Ericson closest-point-on-triangle (vectorised) ────────────────────────
    AB = B - A;  AC = C - A
    AP = P - A;  BP = P - B;  CP = P - C

    d1 = (AB * AP).sum(-1);  d2 = (AC * AP).sum(-1)
    d3 = (AB * BP).sum(-1);  d4 = (AC * BP).sum(-1)
    d5 = (AB * CP).sum(-1);  d6 = (AC * CP).sum(-1)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    # Barycentric coords (u, v, w) → closest = u·A + v·B + w·C
    u = np.ones((Q, K), dtype=np.float64)
    v = np.zeros((Q, K), dtype=np.float64)
    w = np.zeros((Q, K), dtype=np.float64)

    # Vertex A region
    m_a = (d1 <= 0) & (d2 <= 0)

    # Vertex B region
    m_b = (~m_a) & (d3 >= 0) & (d4 <= d3)
    u[m_b] = 0.0;  v[m_b] = 1.0

    # Vertex C region
    m_c = (~m_a) & (~m_b) & (d6 >= 0) & (d5 <= d6)
    u[m_c] = 0.0;  w[m_c] = 1.0

    # Edge AB  (vc ≤ 0, d1 ≥ 0, d3 ≤ 0)
    m_ab = (~m_a) & (~m_b) & (~m_c) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    den_ab = d1 - d3
    t_ab = np.where(np.abs(den_ab) > 1e-30, d1 / den_ab, 0.5)
    u[m_ab] = 1.0 - t_ab[m_ab];  v[m_ab] = t_ab[m_ab];  w[m_ab] = 0.0

    # Edge AC  (vb ≤ 0, d2 ≥ 0, d6 ≤ 0)
    m_ac = (~m_a) & (~m_b) & (~m_c) & (~m_ab) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    den_ac = d2 - d6
    t_ac = np.where(np.abs(den_ac) > 1e-30, d2 / den_ac, 0.5)
    u[m_ac] = 1.0 - t_ac[m_ac];  v[m_ac] = 0.0;  w[m_ac] = t_ac[m_ac]

    # Edge BC  (va ≤ 0, d4−d3 ≥ 0, d5−d6 ≥ 0)
    d43 = d4 - d3;  d56 = d5 - d6
    m_bc = (~m_a) & (~m_b) & (~m_c) & (~m_ab) & (~m_ac) & (va <= 0) & (d43 >= 0) & (d56 >= 0)
    den_bc = d43 + d56
    t_bc = np.where(np.abs(den_bc) > 1e-30, d43 / den_bc, 0.5)
    u[m_bc] = 0.0;  v[m_bc] = 1.0 - t_bc[m_bc];  w[m_bc] = t_bc[m_bc]

    # Interior
    m_int = (~m_a) & (~m_b) & (~m_c) & (~m_ab) & (~m_ac) & (~m_bc)
    den_int = va + vb + vc
    safe = np.abs(den_int) > 1e-30
    v_int = np.where(safe, vb / den_int, 1.0 / 3.0)
    w_int = np.where(safe, vc / den_int, 1.0 / 3.0)
    v[m_int] = v_int[m_int]
    w[m_int] = w_int[m_int]
    u[m_int] = np.clip(1.0 - v[m_int] - w[m_int], 0.0, 1.0)

    # ── Pick the closest triangle per query ───────────────────────────────────
    closest = (A * u[:, :, np.newaxis] +
               B * v[:, :, np.newaxis] +
               C * w[:, :, np.newaxis])                     # (Q, K, 3)
    sq_dists = ((closest - P) ** 2).sum(-1)                 # (Q, K)
    best_k = np.argmin(sq_dists, axis=1)                    # (Q,)
    qi = np.arange(Q)
    u_b = u[qi, best_k];  v_b = v[qi, best_k];  w_b = w[qi, best_k]
    ai  = a_idx[qi, best_k]
    bi  = b_idx[qi, best_k]
    ci  = c_idx[qi, best_k]

    # ── Barycentric normal interpolation ─────────────────────────────────────
    n_interp = (ref_vnorms[ai] * u_b[:, np.newaxis] +
                ref_vnorms[bi] * v_b[:, np.newaxis] +
                ref_vnorms[ci] * w_b[:, np.newaxis])        # (Q, 3)
    norms = np.linalg.norm(n_interp, axis=1, keepdims=True)
    return np.where(norms > 1e-12, n_interp / norms, ref_vnorms[ai])


def _smooth_skin_to_reference(
    skin: BasicTriMesh,
    reference: BasicTriMesh,
    iterations: int = 10,
    step: float = 0.5,
) -> BasicTriMesh:
    """Remove voxel-grid staircase artefacts from a marching-cubes mesh using
    tangential Laplacian smoothing guided by smooth normals from ``reference``.

    Each iteration:
      1. For every skin vertex find its smooth target normal by projecting onto
         the nearest face in ``reference`` and barycentric-interpolating the
         reference vertex normals (see ``_bary_interp_normals``).
      2. Compute the Laplacian displacement (ring-mean − vertex).
      3. Remove the component along the smooth normal (project onto the
         tangent plane) so movement stays *on* the surface.
      4. Apply ``step × tangential_laplacian`` to each vertex.

    Because movement is restricted to the tangent plane, the surface does not
    shrink and the overall shape defined by ``reference`` is preserved.
    """
    ref_vnorms = _vertex_normals(reference)

    # ── Build directed edge table for vectorised Laplacian ────────────────────
    edges: set[tuple[int, int]] = set()
    for f in skin.faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        edges.update(((a, b), (b, a), (b, c), (c, b), (a, c), (c, a)))
    src = np.array([e[0] for e in edges], dtype=np.int32)
    dst = np.array([e[1] for e in edges], dtype=np.int32)
    n_v = len(skin.verts)
    # Degree of each vertex (number of ring neighbours)
    deg = np.bincount(src, minlength=n_v).astype(np.float64)  # (V,)
    deg_safe = np.maximum(deg, 1.0)[:, np.newaxis]            # (V, 1)

    verts = skin.verts.copy()

    for _ in range(iterations):
        # Smooth target normal at each current vertex position
        target_n = _bary_interp_normals(
            verts, reference.verts, reference.faces, ref_vnorms
        )                                                     # (V, 3)

        # Vectorised Laplacian: mean of ring neighbours − vertex
        nbr_sum = np.zeros_like(verts)
        np.add.at(nbr_sum, src, verts[dst])
        lap = nbr_sum / deg_safe - verts                      # (V, 3)

        # Project out normal component → tangential displacement only
        dot = (lap * target_n).sum(axis=1, keepdims=True)    # (V, 1)
        lap_t = lap - dot * target_n                          # (V, 3)

        verts = verts + step * lap_t

    return BasicTriMesh(verts=verts, faces=skin.faces.copy())


def _project_normals_from_surface(
    query_verts: np.ndarray,
    source: BasicTriMesh,
    k: int = 4,
    use_kdtree: bool = False,
) -> np.ndarray:
    """KDTree-based (or brute-force) inverse-distance-weighted normal transfer.

    For each vertex in *query_verts*, finds the ``k`` nearest vertices of
    *source* and blends their normals weighted by 1/distance.  Falls back to
    the closest normal when a query point coincides exactly with a source
    vertex (distance == 0).

    Parameters
    ----------
    query_verts : (Q, 3) float array
        Positions whose normals are to be estimated.
    source : BasicTriMesh
        The reference surface that supplies smooth normals.
    k : int
        Number of nearest neighbours to blend (default 4).
    use_kdtree : bool
        Use scipy cKDTree (default False).  When False (default) a pure-numpy
        brute-force cdist search is used instead — slower for large meshes but
        immune to cKDTree SIGSEGV on degenerate/small point clouds.
        Set to True only once the KDTree crash is resolved.
    """
    src_normals = _vertex_normals(source)          # (S, 3)

    _v  = np.ascontiguousarray(source.verts,  dtype=np.float64)
    _qv = np.ascontiguousarray(query_verts,   dtype=np.float64)

    if use_kdtree:
        from scipy.spatial import KDTree
        tree = KDTree(_v, leafsize=max(10, len(_v) // 10))
        dists, idxs = tree.query(_qv, k=k)
    else:
        # Brute-force: O(S·Q) but only ~1 M ops for typical mesh sizes here.
        # cdist returns (Q, S); argpartition gives the k smallest per row.
        from scipy.spatial.distance import cdist
        D     = cdist(_qv, _v)                                      # (Q, S)
        part  = np.argpartition(D, k, axis=1)[:, :k]               # (Q, k) – unordered
        idxs  = part[np.arange(len(_qv))[:, None],
                     np.argsort(D[np.arange(len(_qv))[:, None], part], axis=1)]
        dists = D[np.arange(len(_qv))[:, None], idxs]              # (Q, k) – sorted

    # Handle exact coincidences: replace zero distance with a large weight.
    zero_mask = dists == 0.0
    weights = np.where(zero_mask, 1e12, 1.0 / np.maximum(dists, 1e-30))  # (Q, k)

    # When any neighbour is exactly coincident, zero-out all other weights.
    has_exact = zero_mask.any(axis=1, keepdims=True)
    weights = np.where(has_exact, np.where(zero_mask, weights, 0.0), weights)

    weights /= weights.sum(axis=1, keepdims=True)  # normalise → sum to 1

    blended = (weights[:, :, np.newaxis] * src_normals[idxs]).sum(axis=1)  # (Q, 3)

    norms = np.linalg.norm(blended, axis=1, keepdims=True)
    return np.where(norms > 0, blended / norms, 0.0)


def _extrude_to_tets(
    mesh: BasicTriMesh,
    heights: list[float],
    normals: np.ndarray | None = None,
    debug_save_path: str | None = None,
) -> BasicTetMesh:
    """Pure-numpy boundary layer extrusion along vertex normals.

    Vertex layout: [base, layer_0, layer_1, ..., layer_{N-1}]
    Each face × each layer → 1 prism → 3 tetrahedra.
    Exact output: ``3 · N · n_faces`` tets.

    Parameters
    ----------
    mesh : BasicTriMesh
        Surface mesh to extrude.
    heights : list[float]
        Cumulative extrusion heights for each layer.
    normals : (N, 3) float array, optional
        Per-vertex outward normals to use for extrusion.  When *None*, the
        normals are computed from *mesh* itself via ``_vertex_normals``.
        Pass pre-projected smooth normals here to avoid artefacts from
        noisy marching-cubes geometry.
    debug_save_path : str, optional
        If provided, save the resulting tet mesh to this path.
    """
    verts, faces = mesh.verts, mesh.faces
    if normals is None:
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

def _is_watertight(m: BasicTriMesh, repair=True):
    """
    Check whether m is a watertight triangle mesh.
    """
    edge_count = Counter()
    unused_vertices = set(range(len(m.verts)))
    for face in m.faces:
        edges = [(face[i], face[(i + 1) % 3]) for i in range(3)]
        for a, b in edges:
            key = tuple(sorted((a, b)))
            edge_count[key] += 1
        for vi in face:
            unused_vertices.discard(vi)

    broken = []
    for edge, count in edge_count.items():
        if count != 2:
            print(f"edge {edge} has count {count}")
            broken.append(edge)

    if broken:
        return False

    #if unused_vertices:
    #    return False

    return True


def boolean_merge_meshes(meshes: list[BasicTriMesh], debug_save_path=None) -> BasicTriMesh:
    """Union all meshes into a single watertight surface mesh using manifold3d.

    Parameters
    ----------
    meshes : list[BasicTriMesh]
        Input surface meshes to union together.
    debug_save_path : str, optional
        If provided, save the resulting mesh to this path via meshio.

    Returns
    -------
    BasicTriMesh
        The boolean union of all input meshes.
    """

    if not meshes:
        return BasicTriMesh(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32))

    for i, m in enumerate(meshes):
        if not _is_watertight(m):
            #raise ValueError(f"Input meshes must be watertight - mesh {i} ({m.label}) failed the watertight test")
            print(f"Input meshes must be watertight - mesh {i} ({m.label}) failed the watertight test")

    def _to_meshlib(m: BasicTriMesh):
        #m = m.swapped_winding_order()
        mesh = mrmeshnumpy.meshFromFacesVerts(faces=m.faces, verts=m.verts)
        return mesh

    def _from_meshlib(m) -> BasicTriMesh:
        verts = mrmeshnumpy.getNumpyVerts(m)
        faces = mrmeshnumpy.getNumpyFaces(m.topology)
        tri_mesh = BasicTriMesh(verts=verts, faces=faces)
        #return tri_mesh.swapped_winding_order()
        return tri_mesh

    #meshes = [meshes[0], meshes[2], meshes[4], meshes[5]]

    result = None
    for i, mesh in enumerate(meshes):
        if result is None:
            result = _to_meshlib(mesh)
        else:
            result = mrmeshpy.boolean(result, _to_meshlib(mesh), mrmeshpy.BooleanOperation.Union).mesh
            print("merged", i, "-> ", len(mrmeshnumpy.getNumpyVerts(result)), "verts, ", mrmeshnumpy.getNumpyFaces(result.topology).shape[0], "faces")
            if debug_save_path is not None:
                _from_meshlib(result).save(debug_save_path + '.pt_' + str(i) + '.ply')

    result_trimesh = _from_meshlib(result)
    if debug_save_path:
        result_trimesh.save(debug_save_path)
    return result_trimesh

def build_boundary_layer_sdf(
    surface_mesh: BasicTriMesh,
    layer_thickness: float,
    target_tet_count: int | None = None,
    smooth_normals: bool = False,
    debug_save_path: str | None = None,
):
    # Capture the original surface *before* the SDF/marching-cubes pipeline
    # replaces surface_mesh.  Force an owned copy so that the arrays survive
    # any meshlib GC that may have occurred before smooth_normals is used.
    original_surface = BasicTriMesh(
        verts=np.array(surface_mesh.verts, dtype=np.float64, order='C'),
        faces=np.array(surface_mesh.faces, dtype=np.int32,   order='C'),
    )
    # build a (SDF) signed distance field by voxelizing surface_mesh
    resolution = 0.005
    padding = layer_thickness + 20 * resolution  # ensure the iso-surface fits inside the grid

    ml_mesh = mrmeshnumpy.meshFromFacesVerts(faces=surface_mesh.faces, verts=surface_mesh.verts)
    bb = ml_mesh.computeBoundingBox()

    origin = mrmeshpy.Vector3f(
        bb.min.x - padding,
        bb.min.y - padding,
        bb.min.z - padding,
    )
    dims = mrmeshpy.Vector3i(
        int(np.ceil((bb.max.x - bb.min.x + 2 * padding) / resolution)) + 1,
        int(np.ceil((bb.max.y - bb.min.y + 2 * padding) / resolution)) + 1,
        int(np.ceil((bb.max.z - bb.min.z + 2 * padding) / resolution)) + 1,
    )

    vol_params = mrmeshpy.DistanceVolumeParams()
    vol_params.origin = origin
    vol_params.voxelSize = mrmeshpy.Vector3f(resolution, resolution, resolution)
    vol_params.dimensions = dims

    dist_opts = mrmeshpy.SignedDistanceToMeshOptions()
    dist_opts.signMode = mrmeshpy.SignDetectionMode.HoleWindingRule
    # compute distances precisely up to 3× layer thickness; approximate beyond
    dist_opts.maxDistSq = float((layer_thickness * 3) ** 2)
    dist_opts.nullOutsideMinMax = False  # keep approximate values so marching cubes finds a closed surface

    sdf_params = mrmeshpy.MeshToDistanceVolumeParams()
    sdf_params.vol = vol_params
    sdf_params.dist = dist_opts

    sdf_volume = mrmeshpy.meshToDistanceVolume(ml_mesh, sdf_params)

    # smooth the sdf
    # SimpleVolumeMinMax → VdbVolume → Gaussian filter (width=3 voxels) → SimpleVolumeMinMax
    _vdb = mrmeshpy.simpleVolumeToVdbVolume(sdf_volume)
    _vdb = mrmeshpy.voxelFilter(_vdb, mrmeshpy.VoxelFilterType.Gaussian, 3)
    sdf_volume = mrmeshpy.vdbVolumeToSimpleVolume(_vdb)

    # find the surface of the SDF -> skin mesh
    # Extract the iso-surface at level 0: a clean, watertight reconstruction
    # of the original surface mesh, suitable for remeshing and extrusion.
    mc_params = mrmeshpy.MarchingCubesParams()
    mc_params.iso = 0.0
    mc_params.lessInside = True  # required for signed distance volumes
    mc_params.origin = origin    # must match the SDF grid origin

    skin_ml = mrmeshpy.marchingCubes(sdf_volume, mc_params)
    # getNumpyVerts/getNumpyFaces return non-owning views; copy immediately
    # before skin_ml can be GC'd or mutated.
    skin_verts = np.array(mrmeshnumpy.getNumpyVerts(skin_ml), dtype=np.float64, order='C')
    skin_faces = np.array(mrmeshnumpy.getNumpyFaces(skin_ml.topology), dtype=np.int32, order='C')

    # The Gaussian blur blends SDF values with implicit zeros outside the grid
    # boundary, which can push edge-voxel values through zero and create a
    # spurious closed shell at the volume boundary.  Remove any face whose
    # vertices fall within `margin` of the grid edge — the real iso-surface is
    # always deeper inside the grid thanks to `padding`.
    vol_min = np.array([origin.x, origin.y, origin.z])
    vol_max = vol_min + np.array([dims.x, dims.y, dims.z]) * resolution
    margin = 4 * resolution          # 4-voxel safety strip around the grid edge
    inner_min = vol_min + margin
    inner_max = vol_max - margin
    face_verts_3d = skin_verts[skin_faces]           # (F, 3, 3)
    keep = np.all(
        (face_verts_3d >= inner_min) & (face_verts_3d <= inner_max),
        axis=(1, 2),
    )
    skin_faces = skin_faces[keep]
    if skin_faces.size > 0:
        used = np.unique(skin_faces)
        remap = np.full(len(skin_verts), -1, dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        skin_verts = skin_verts[used]
        skin_faces = remap[skin_faces]

    surface_mesh = BasicTriMesh(verts=skin_verts, faces=skin_faces)

    # ── Smooth the marching-cubes surface before remeshing ────────────────────
    # The iso-surface recovered from a voxel SDF inherits the grid structure,
    # producing a lumpy/staircase outer surface even after Gaussian-smoothing
    # the SDF.  Tangential Laplacian smoothing guided by smooth normals from
    # the original surface removes these artefacts *before* remeshing, so the
    # subsequent isotropic remesh operates on a clean, smooth input.
    if smooth_normals:
        surface_mesh = _smooth_skin_to_reference(
            surface_mesh, original_surface,
            iterations=10,
            step=0.5,
        )

    if debug_save_path:
        surface_mesh.save(debug_save_path + ".surface.ply")

    # remesh the skin mesh and extrude to tets
    heights = [layer_thickness]
    e = _target_edge_len(surface_mesh, target_tet_count, n_layers=1)
    surface_mesh = _remesh_surface(surface_mesh, e)

    # Also use smooth normals for the extrusion direction so the outer shell
    # of the tet layer reflects the true surface curvature rather than the
    # marching-cubes reconstruction.
    extrude_normals = None
    if smooth_normals:
        extrude_normals = _bary_interp_normals(
            surface_mesh.verts,
            original_surface.verts,
            original_surface.faces,
            _vertex_normals(original_surface),
        )

    return _extrude_to_tets(surface_mesh, heights, normals=extrude_normals, debug_save_path=debug_save_path)

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