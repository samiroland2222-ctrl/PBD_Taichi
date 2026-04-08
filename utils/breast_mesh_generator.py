import argparse
import gmsh
import math
import numpy as np

from PBD_Taichi.utils.gmsh_41_writer import write_msh41_single_block


def generate_breast_msh(radius=0.07, height=0.06, k=0.7, k2=0.2,
                         axillary_extension=0.04,
                         axillary_height_fraction=0.2,
                         target_tets=500, flange_depth=0.02, flange_radius_fraction=1.0, ribcage=None,
                        debug_save_path=None):
    """
    Generate an anatomical breast mesh via a single loft through N cross-section wires.

    The section wires interpolate from a teardrop-shaped (axillary-extended) ring
    at the posterior face (z = -flange_depth) to a small circle at the nipple apex
    (z = height).  No revolve or boolean union is required.

    Parameters
    ----------
    radius                   : float  Breast radius at z = height/3 (m)
    height                   : float  Height from chest-wall base (z=0) to nipple (m)
    k                        : float  Mound curvature: 0=cone, 1=hemisphere, >1=bulge outward
    k2                       : float  Flange curvature: 0=straight sided, 1=hemisphere-like
                                      roll; also scales flange-base width
    axillary_extension       : float  Maximum extra extent in +y (axilla side) at the base (m);
                                      0 = fully symmetric/round
    axillary_height_fraction : float  z/height fraction above which axillary extension is zero;
                                      below this the extension blends smoothly to its maximum
    target_tets              : int    Approximate target tetrahedron count
    flange_depth             : float  Posterior depth into chest wall (m)
    flange_radius_fraction   : float  Flange radius as a fraction of radius (1.0 = cylinndrical flange, >1 = flared/conical outwards)
    ribcage                  : mesh   Optional ribcage mesh for boolean subtraction
                                      (needs .vertices or .verts and .faces attributes)

    Returns
    -------
    coords        : (N, 3) float array — node positions
    node_tags     : (N,)   int array  — 1-indexed gmsh node tags
    tet_node_tags : (M, 4) int array  — node tags per tet (1-indexed)
    tet_tags      : (M,)   int array  — element tags
    base_idx      : (K,)   int array  — 0-indexed node indices on the posterior face
    """
    gmsh.initialize()
    gmsh.model.add("AnatomicalBreast")
    occ = gmsh.model.occ

    # ------------------------------------------------------------------ #
    # Profile parameters                                                   #
    # ------------------------------------------------------------------ #
    # Exponent maps curvature semantics:
    #   n = 1   → straight cone
    #   n = 0.5 → hemisphere-like profile
    #   n < 0.5 → bulges outward beyond a hemisphere
    n  = 1.0 / (1.0 + k)
    n2 = 1.0 / (1.0 + k2)

    # Base radius at z=0 derived from the constraint r(height/3) = radius.
    # r(z) = R_base * ((height-z)/height)^n  →  radius = R_base*(2/3)^n
    R_base   = radius * (1.5 ** n)

    # Flange radius at z=-flange_depth: k2 governs both roll curvature and width
    R_flange = radius * flange_radius_fraction

    # Height below which axillary extension is blended in
    z_ax = axillary_height_fraction * height

    # L (length scale for meshing) estimated from target tet count and rough breast volume:
    mound_vol_est  = math.pi * R_base**2 * height / 3
    flange_vol_est = math.pi * R_flange**2 * flange_depth
    total_vol      = mound_vol_est + flange_vol_est
    L = (total_vol / (target_tets * 0.117)) ** (1/3)

    # ------------------------------------------------------------------ #
    # Helper functions                                                     #
    # ------------------------------------------------------------------ #
    def profile_r(z):
        """Radius of the symmetric (circular) profile at height z."""
        if z >= 0.0:
            t = max((height - z) / height, L/3)   # 0 at apex, 1 at z=0
            R_apex = L * num_ang / (6.0 * math.pi)
            return R_apex + (R_base - R_apex) * (t ** n)
        else:
            t = -z / flange_depth                  # 0 at z=0, 1 at z=-flange_depth
            return R_base + (R_flange - R_base) * (t ** n2)

    def ax_blend(z):
        """Smoothstep axillary weight: 1 at the posterior face, 0 above z_ax."""
        if z >= z_ax:
            return 0.0
        raw = (z_ax - z) / (z_ax + flange_depth)
        raw = max(0.0, min(1.0, raw))
        return raw * raw * (3.0 - 2.0 * raw)   # smoothstep

    # ------------------------------------------------------------------ #
    # 1.  SECTION WIRES — from posterior teardrop to apex circle          #
    # ------------------------------------------------------------------ #
    num_z_sections = 4   # loft slices along z
    num_ang        = 12   # angular samples per ring

    z_values = np.linspace(-flange_depth, height, num_z_sections)

    section_wires = []
    for zi in z_values:
        ri    = profile_r(zi)
        assert ri > 0
        blend = ax_blend(zi)

        pt_tags = []
        for j in range(num_ang):
            theta = 2.0 * math.pi * j / num_ang
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            x = ri * cos_t
            y = ri * sin_t
            # Axillary tail: only on the +y (axilla) side, weighted by sin²
            # so it peaks in the +y direction and tapers naturally to zero at ±x
            if sin_t > 0.0 and blend > 0.0:
                y += blend * (sin_t ** 2) * axillary_extension

            pt_tags.append(occ.addPoint(x, y, zi))

        # Closed periodic spline — duplicate first tag to close the curve
        spline = occ.addSpline(pt_tags + [pt_tags[0]])
        section_wires.append(occ.addWire([spline]))

    # ------------------------------------------------------------------ #
    # 2.  loft and fuse                        #
    # ------------------------------------------------------------------ #
    # Loft the body (posterior face -> z_values[2])
    loft_result     = occ.addThruSections(section_wires, makeSolid=True, makeRuled=False)
    breast_vol = loft_result[0][1]

    # ------------------------------------------------------------------ #
    # 3.  OPTIONAL RIBCAGE BOOLEAN SUBTRACTION                            #
    # ------------------------------------------------------------------ #
    if ribcage is not None:
        verts = getattr(ribcage, 'vertices', None) or ribcage.verts
        rib_pt_tags = [occ.addPoint(*v) for v in verts]
        rib_surfs   = []
        for face in ribcage.faces:
            pts   = [rib_pt_tags[n] for n in face]
            lines = [occ.addLine(pts[0], pts[1]),
                     occ.addLine(pts[1], pts[2]),
                     occ.addLine(pts[2], pts[0])]
            loop  = occ.addCurveLoop(lines)
            rib_surfs.append(occ.addPlaneSurface([loop]))
        rib_shell = occ.addSurfaceLoop(rib_surfs)
        rib_vol   = occ.addVolume([rib_shell])
        cut_result, _ = occ.cut([(3, breast_vol)], [(3, rib_vol)])
        breast_vol = cut_result[0][1]

    occ.synchronize()

    vols = gmsh.model.getEntities(dim=3)
    print(f"Volumes after construction: {len(vols)}")  # should be 1

    # ------------------------------------------------------------------ #
    # 4.  MESHING                                                         #
    # ------------------------------------------------------------------ #

    f = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f, "F",
        f"{L}")#{L * 0.5} * Sqrt(x*x + y*y) + {L:.4f}")
    gmsh.model.mesh.field.setAsBackgroundMesh(f)

    gmsh.option.setNumber("Mesh.Algorithm3D", 4)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = node_coords.reshape(-1, 3)

    _, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    tet_tags      = elem_tags[0]
    tet_node_tags = elem_node_tags[0].reshape(-1, 4)

    n_tets = len(tet_tags)
    print(f"k={k:.2f}, k2={k2:.2f}, ax={axillary_extension:.3f}, L={L:.4f} → {n_tets} tetrahedra")

    # ------------------------------------------------------------------ #
    # 5.  IDENTIFY POSTERIOR (BASE) NODES                                 #
    # ------------------------------------------------------------------ #
    if ribcage is not None:
        from scipy.spatial import cKDTree
        verts    = getattr(ribcage, 'vertices', None) or ribcage.verts
        tree     = cKDTree(np.array(verts, dtype=float))
        dist, _  = tree.query(coords)
        base_idx = np.where(dist < 0.001)[0]
    else:
        # Nodes on/near the posterior face (z ≈ −flange_depth)
        base_idx = np.where(coords[:, 2] < -flange_depth + 0.002)[0]

    if debug_save_path:
        gmsh.write(debug_save_path)
    gmsh.finalize()

    return coords, node_tags, tet_node_tags, tet_tags, base_idx

def generate_breast_msh_and_write(radius=0.07, height=0.06, k=0.7, k2=0.2,
                                   axillary_extension=0.04,
                                   axillary_height_fraction=0.4,
                                   target_tets=500, flange_depth=0.02,
                                   filename="breast.msh", ribcage=None):
    coords, node_tags, tet_node_tags, tet_tags, base_idx = generate_breast_msh(
        radius, height, k, k2,
        axillary_extension=axillary_extension,
        axillary_height_fraction=axillary_height_fraction,
        target_tets=target_tets, flange_depth=flange_depth, ribcage=ribcage)
    write_msh41_single_block(coords, node_tags, tet_node_tags, tet_tags, filename)
    return coords, node_tags, tet_node_tags, tet_tags, base_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a breast mesh using Gmsh API.')
    parser.add_argument('--radius',       type=float, default=0.07,  help='Radius at z=height/3 (m)')
    parser.add_argument('--height',       type=float, default=0.06,  help='Height from chest wall to nipple (m)')
    parser.add_argument('--alpha',        type=float, default=0.7,   help='Mound curvature k (0=cone, 1=hemisphere, >1=bulge)')
    parser.add_argument('--k2',           type=float, default=0.2,   help='Flange curvature (0=straight, 1=hemisphere-like roll)')
    parser.add_argument('--axillary_extension',       type=float, default=0.04,
                        help='Axillary (+y) extension at the base, m (0=round)')
    parser.add_argument('--axillary_height_fraction', type=float, default=0.4,
                        help='z/height fraction above which axillary extension is zero')
    parser.add_argument('--flange_depth', type=float, default=0.02,  help='Posterior depth into chest wall (m)')
    parser.add_argument('--target_tet_count', type=int, default=500, help='Target tetrahedron count')
    parser.add_argument('--out',          type=str,   default='breast.msh', help='Output filename')

    args = parser.parse_args()

    generate_breast_msh_and_write(
        radius=args.radius, height=args.height,
        k=args.alpha, k2=args.k2,
        axillary_extension=args.axillary_extension,
        axillary_height_fraction=args.axillary_height_fraction,
        flange_depth=args.flange_depth,
        target_tets=args.target_tet_count,
        filename=args.out)
