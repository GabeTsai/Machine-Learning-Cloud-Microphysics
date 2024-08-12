import numpy as np
from CreateMesh import IcosphereMesh
import plotly.graph_objects as go


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

def test_icosphere_shape():
    """
    Check that icosphere visually looks like a sphere
    """
    refinement_level = 1
    icosphere = IcosphereMesh(refinement_level)
    vertices = icosphere.vertices
    faces = icosphere.faces
    x = [v[0] for v in vertices]
    y = [v[1] for v in vertices]
    z = [v[2] for v in vertices]
    i = [face[0] for face in faces]
    j = [face[1] for face in faces]
    k = [face[2] for face in faces]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(size=5)
        ),
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            opacity=0.5,
            color='blue',
        )
    ])
    fig.show()

def main():
    test_create_icosahedron()
    test_refinement()
    test_icosphere_shape()
    print('All tests passed')

if __name__ == "__main__":
    main()