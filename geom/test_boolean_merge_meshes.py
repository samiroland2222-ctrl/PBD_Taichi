import numpy as np
from PBD_Taichi.geom.distance_field import Mesh, boolean_merge_meshes

def test_boolean_merge_meshes_triangle():
    # Two triangles sharing an edge, should merge into a quad
    verts1 = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float64)
    faces1 = np.array([[0,1,2]], dtype=np.int32)
    verts2 = np.array([[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
    faces2 = np.array([[0,1,2]], dtype=np.int32)
    # Shift verts2 indices for global mesh
    verts = np.vstack([verts1, verts2])
    faces = np.vstack([faces1, faces2+3])
    mesh1 = Mesh(verts1, faces1)
    mesh2 = Mesh(verts2, faces2)
    merged = boolean_merge_meshes([mesh1, mesh2])
    assert merged.verts.shape[1] == 3
    assert merged.faces.shape[1] == 3
    assert merged.verts.shape[0] >= 3
    assert merged.faces.shape[0] >= 1
    print("Merged verts:\n", merged.verts)
    print("Merged faces:\n", merged.faces)

def test_boolean_merge_meshes_disjoint():
    # Two disjoint triangles
    verts1 = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float64)
    faces1 = np.array([[0,1,2]], dtype=np.int32)
    verts2 = np.array([[2,0,0],[3,0,0],[2,1,0]], dtype=np.float64)
    faces2 = np.array([[0,1,2]], dtype=np.int32)
    mesh1 = Mesh(verts1, faces1)
    mesh2 = Mesh(verts2, faces2)
    merged = boolean_merge_meshes([mesh1, mesh2])
    assert merged.verts.shape[1] == 3
    assert merged.faces.shape[1] == 3
    assert merged.verts.shape[0] >= 6
    assert merged.faces.shape[0] >= 2
    print("Merged disjoint verts:\n", merged.verts)
    print("Merged disjoint faces:\n", merged.faces)

if __name__ == "__main__":
    test_boolean_merge_meshes_triangle()
    test_boolean_merge_meshes_disjoint()

