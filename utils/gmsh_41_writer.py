
def write_msh41_single_block(coords, node_tags, tets, tet_tags, filename):
    """
    Write MSH4.1 with all nodes in a single block and all tets in a single block.
    coords    : (N, 3) float array
    node_tags : (N,)   int array, 1-indexed
    tets      : (M, 4) int array, node tags (1-indexed)
    tet_tags  : (M,)   int array
    """
    n_nodes = len(coords)
    n_tets  = len(tets)

    with open(filename, "w") as f:
        # Header
        f.write("$MeshFormat\n")
        f.write("4.1 0 8\n")   # version, ASCII, data-size
        f.write("$EndMeshFormat\n")

        # Single node block
        # $Nodes
        #   numEntityBlocks  numNodes  minNodeTag  maxNodeTag
        #   entityDim  entityTag  parametric  numNodesInBlock
        #   node tags...
        #   x y z...
        f.write("$Nodes\n")
        f.write(f"1 {n_nodes} 1 {n_nodes}\n")   # 1 block
        f.write(f"3 1 0 {n_nodes}\n")            # dim=3, entity=1, no params
        for tag in node_tags:
            f.write(f"{tag}\n")
        for x, y, z in coords:
            f.write(f"{x:.10g} {y:.10g} {z:.10g}\n")
        f.write("$EndNodes\n")

        # Single element block
        # element type 4 = 4-node tetrahedron
        f.write("$Elements\n")
        f.write(f"1 {n_tets} 1 {n_tets}\n")     # 1 block
        f.write(f"3 1 4 {n_tets}\n")             # dim=3, entity=1, type=4
        for tag, (a, b, c, d) in zip(tet_tags, tets):
            f.write(f"{tag} {a} {b} {c} {d}\n")
        f.write("$EndElements\n")
