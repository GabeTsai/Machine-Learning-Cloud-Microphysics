import numpy as np
from Icosphere import IcosphereMesh, IcosphereTetrahedron
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    refinement_level = 1
    icosphere = IcosphereMesh(refinement_level)
    vertices = icosphere.vertices
    faces = icosphere.faces

    correct_num_vertices = 10 * 4**refinement_level + 2
    correct_num_faces = 20 * 4**refinement_level    
    print(f"Refinement level {refinement_level}: {len(vertices)} vertices, {len(faces)} faces")
    assert len(vertices) == correct_num_vertices, f"Expected {correct_num_vertices} vertices, got {len(vertices)}"
    assert len(faces) == correct_num_faces, f"Expected {correct_num_faces} faces, got {len(faces)}"


def plot_icosphere(vertices, faces, edges):
    """
    Plot icosphere using plotly
    """
    x = [v[0] for v in vertices]
    y = [v[1] for v in vertices]
    z = [v[2] for v in vertices]

    i = [face[0] for face in faces]
    j = [face[1] for face in faces]
    k = [face[2] for face in faces]

    edge_x = []
    edge_y = []
    edge_z = []

    for edge in edges.T:
        edge_x.extend([vertices[edge[0]][0], vertices[edge[1]][0], None])
        edge_y.extend([vertices[edge[0]][1], vertices[edge[1]][1], None])
        edge_z.extend([vertices[edge[0]][2], vertices[edge[1]][2], None])

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(size=3)
        ),
        # go.Mesh3d(
        #     x=x,
        #     y=y,
        #     z=z,
        #     i=i,
        #     j=j,
        #     k=k,
        #     opacity=0.01,
        #     color='blue',
        # ),
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='black', width=1)
        )
    ])
    fig.show()

def test_icosphere_shape():
    """
    Check that icosphere visually looks like a sphere
    """
    refinement_level = 3
    icosphere = IcosphereMesh(refinement_level)
    vertices = icosphere.vertices
    faces = icosphere.faces
    edges = icosphere.edges

    plot_icosphere(vertices, faces, edges)
    print(f'Number of vertices: {len(vertices)}')
    print(f'Number of faces: {len(faces)}')
    print(f'Number of edges: {len(edges[0]/2)}')

def test_tetrahedron():
    """
    Test that tetrahedralization works
    """
    n_refine = 3
    icosphere_tetrahedron = IcosphereTetrahedron(n_refine)
    vertices = icosphere_tetrahedron.vertices
    faces = icosphere_tetrahedron.faces 
    edges = icosphere_tetrahedron.edges
    plot_icosphere(vertices, faces, edges)
    print(f'Number of vertices: {len(vertices)}')
    print(f'Number of faces: {len(faces)}') 
    print(f'Number of edges: {len(edges[0]/2)}')

def test_edge_weights():
    n_refine = 3
    icosphere_tetrahedron = IcosphereTetrahedron(n_refine)
    vertices = icosphere_tetrahedron.vertices
    faces = icosphere_tetrahedron.faces 
    edges = icosphere_tetrahedron.edges
    print(icosphere_tetrahedron.edge_feat)
    print(edges[0].shape)
    plt.hist(icosphere_tetrahedron.edge_feat, bins = 200, alpha = 0.5, label = 'Edge weights')
    
def main():
    # test_create_icosahedron()
    # test_refinement()
    # test_icosphere_shape()
    # test_tetrahedron()
    test_edge_weights()
    print('All tests passed')

if __name__ == "__main__":
    main()