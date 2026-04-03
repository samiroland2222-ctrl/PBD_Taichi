import argparse
import gmsh
import math

from PBD_Taichi.utils.gmsh_41_writer import write_msh41_single_block


def generate_breast_msh(radius=0.07, height=0.06, k=0.7, target_tets=300, filename="breast.msh"):
    """
    k=0: sharp cone (straight profile)
    k=1: hemisphere (circular arc profile)
    returns a tuple of (coords, node_tags, tet_node_tags, tet_tags) where:
    coords    : (N, 3) float array
    node_tags : (N,)   int array, 1-indexed
    tets      : (M, 4) int array, node tags (1-indexed)
    tet_tags  : (M,)   int array
    """
    gmsh.initialize()
    gmsh.model.add("cone")

    volume = math.pi * radius**2 * height / 3
    L = (volume / (target_tets * 0.117)) ** (1/3)

    # Control point
    smid = (radius * 0.5, 0, height * 0.5)
    ang  = math.pi / 4
    Rv   = math.sqrt(radius**2 + height**2)
    amid = (Rv * math.sin(ang), 0, height - Rv * (1 - math.cos(ang)))
    cp_coords = (
        smid[0]*(1-k) + amid[0]*k,
        0,
        smid[2]*(1-k) + amid[2]*k,
    )

    def make_half_profile():
        apex_pt        = gmsh.model.occ.addPoint(0, 0, height)
        base_pt        = gmsh.model.occ.addPoint(radius, 0, 0)
        center_base_pt = gmsh.model.occ.addPoint(0, 0, 0)
        cp             = gmsh.model.occ.addPoint(*cp_coords)

        profile  = gmsh.model.occ.addBezier([apex_pt, cp, base_pt])
        center   = gmsh.model.occ.addLine(base_pt, center_base_pt)
        axis_seg = gmsh.model.occ.addLine(center_base_pt, apex_pt)
        loop     = gmsh.model.occ.addCurveLoop([profile, center, axis_seg])
        face     = gmsh.model.occ.addPlaneSurface([loop])
        return face

    face1 = make_half_profile()
    face2 = make_half_profile()

    # Revolve each half by π in opposite directions
    rev1 = gmsh.model.occ.revolve([(2, face1)], 0,0,0, 0,0,1,  math.pi)
    rev2 = gmsh.model.occ.revolve([(2, face2)], 0,0,0, 0,0,1, -math.pi)

    vols1 = [v for v in rev1 if v[0] == 3]
    vols2 = [v for v in rev2 if v[0] == 3]

    # Fuse removes the shared internal cut plane
    gmsh.model.occ.fuse(vols1, vols2, removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()

    # Verify we have exactly one volume
    vols = gmsh.model.getEntities(dim=3)
    print(f"Volumes after fuse: {len(vols)}")  # should be 1

    f = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f, "F",
        f"{L * 0.5} * Sqrt(x*x + y*y) + {L:.4f}"
    )
    gmsh.model.mesh.field.setAsBackgroundMesh(f)

    gmsh.option.setNumber("Mesh.Algorithm3D", 4)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    #gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.write(filename)

    types, tags, _ = gmsh.model.mesh.getElements(dim=3)
    n_tets = len(tags[0]) if tags else 0
    print(f"k={k:.2f}, L={L:.3f} → {n_tets} tetrahedra")

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = node_coords.reshape(-1, 3)

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    # elem_tags[0] and elem_node_tags[0] are the tet data
    # (index 0 because there should be exactly one element type in dim=3)
    tet_tags      = elem_tags[0]
    tet_node_tags = elem_node_tags[0].reshape(-1, 4)

    gmsh.finalize()

    return coords, node_tags, tet_node_tags, tet_tags

def generate_breast_msh_and_write(radius=0.07, height=0.06, k=0.7, target_tets=300, filename="breast.msh"):
    coords, node_tags, tet_node_tags, tet_tags = generate_breast_msh(radius, height, k, target_tets, filename)
    write_msh41_single_block(coords, node_tags, tet_node_tags, tet_tags, filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a breast mesh using Gmsh API.')
    parser.add_argument('--radius', type=float, default=0.07, help='Base radius (m)')
    parser.add_argument('--height', type=float, default=0.06, help='Height (m)')
    parser.add_argument('--alpha', type=float, default=0.7, help='Shape factor')
    parser.add_argument('--target_tet_count', type=int, default=300, help='Number of tetrahedra to target')
    parser.add_argument('--out', type=str, default='breast.msh', help='Output filename')
    
    args = parser.parse_args()
    
    generate_breast_msh(
        radius=args.radius, height=args.height, k=args.alpha,
        target_tets=args.target_tet_count, filename=args.out)
