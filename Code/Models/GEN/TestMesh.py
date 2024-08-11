import numpy as np
from CreateMesh import IcosphereMesh

def test_create_icosahedron():
    """
    Test create_icosahedron function
    """
    icosphere = IcosphereMesh(0)
    vertices = icosphere.vertices
    faces = icosphere.faces
    assert len(vertices) == 12, f"Expected 12 vertices, got {len(vertices)}"
    assert len(faces) == 20, f"Expected 20 faces, got {len(faces)}"

def test_refinement():
    """
    Check that icosphere refinement works as expected 
    """
    refinement_level = 0
    icosphere = IcosphereMesh(refinement_level)
    vertices = icosphere.vertices
    faces = icosphere.faces

    correct_num_vertices = 10 * 4**refinement_level + 2
    correct_num_faces = 20 * 4**refinement_level    
    print(f"Refinement level {refinement_level}: {len(vertices)} vertices, {len(faces)} faces")
    assert len(vertices) == correct_num_vertices, f"Expected {correct_num_vertices} vertices, got {len(vertices)}"
    assert len(faces) == correct_num_faces, f"Expected {correct_num_faces} faces, got {len(faces)}"
    
def main():
    test_refinement()
    print('All tests passed')

if __name__ == "__main__":
    main()