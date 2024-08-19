import numpy as np
import tetgen 

class IcosphereMesh: 
    """
    Class to create a icosphere mesh and refine it.
    Help from https://en.wikipedia.org/wiki/Regular_icosahedron and http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html

    Args:
        n_refine (int): Number of times to refine the mesh
    
    Attributes:
        vertices (np.array): Vertices of the icosphere
        faces (np.array): Faces of the icosphere
        midpoint_cache (dict): Cache to store midpoints of vertices to avoid recomputation
    """
    def __init__(self, n_refine):
        self.vertices, self.faces = self.create_icosahedron()
        self.midpoint_cache = {}
        
        for _ in range(n_refine):
            self.refine_mesh()   
        self.vertices = np.array(self.vertices).astype(np.float32)
        self.faces = np.array(self.faces).astype(np.int32)   
        self.edges = np.array(self.create_edges(self.faces)).astype(np.int32)     
    
    def create_icosahedron(self):
        """
        Generate vertices and faces, each as list of tuples for an icosahedron.
        Faces are generated in counter-clockwise order using golden ratio.
        Reference: https://en.wikipedia.org/wiki/Regular_icosahedron 

        Returns:
            tuple: Tuple containing vertices and faces of the icosahedron
        """
        phi = (1 + np.sqrt(5)) / 2
        vertices = []
        for c1 in [-1, 1]:
            for c2 in [-phi, phi]:
                vertices.append(self.add_to_unit_sphere((0, c1, c2)))
                vertices.append(self.add_to_unit_sphere((c1, c2, 0)))
                vertices.append(self.add_to_unit_sphere((c2, 0, c1)))
        
        # Manually create 20 faces, a tuples of indices into vertifces
        faces = [(0, 1, 2), 
                (7, 3, 1), 
                (7, 1, 0),
                (5, 7, 0), 
                (1, 3, 8), 
                (11, 3, 7), 
                (11, 9, 3),
                (3, 9, 8), #DJI
                (1, 8, 2), #BIC
                (5, 11, 7), #FLH
                (2, 4, 6), #CEG
                (6, 10, 5), #GKF
                (4, 10, 6), #EKG
                (4, 9, 10), #EJK
                (2, 8, 4), #CIE
                (10, 11, 5),#KLF
                (0, 2, 6), #ACG
                (6, 5, 0), #GFA
                (4, 8, 9), #EIJ
                (9, 11, 10) #JLK
                ]
        return vertices, faces

    def add_to_unit_sphere(self, vertex):
        """
        Add a vertex to the unit sphere
        Args:
            vertex (tuple): Vertex to add to the unit sphere
        Returns:
            tuple: Vertex on the unit sphere
        """
        x, y, z = vertex
        length = np.sqrt(x**2 + y**2 + z**2)
        return x / length, y / length, z / length

    def get_midpoint(self, vertex1, vertex2):
        """
        Get the midpoint between two vertices
        Args:
            vertex1 (tuple): First vertex
            vertex2 (tuple): Second vertex
        Returns:
            tuple: Midpoint between the two vertices
        """
        if vertex1 < vertex2:
            key = (vertex1, vertex2) # Use smaller index first
        else:
            key = (vertex2, vertex1)
        if key in self.midpoint_cache: # Check if midpoint already computed
            return self.midpoint_cache[key]
        v1 = self.vertices[vertex1]
        v2 = self.vertices[vertex2]
        midpoint = ((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2, (v1[2] + v2[2]) / 2)
        midpoint = self.add_to_unit_sphere(midpoint)
        self.vertices.append(midpoint)
        index = len(self.vertices) - 1
        self.midpoint_cache[key] = index
        return index

    def refine_face(self, face):
        """
        Refine a face by adding new vertices and faces
        Args:
            face (tuple): Face to refine
        Returns:
            list: List of new faces
        """
        v1, v2, v3 = face
        v4 = self.get_midpoint(v1, v2)
        v5 = self.get_midpoint(v2, v3)
        v6 = self.get_midpoint(v3, v1)
        return [(v1, v4, v6), (v2, v5, v4), (v3, v6, v5), (v4, v5, v6)]
    
    def refine_mesh(self):
        """
        Refine the mesh once by adding new vertices midway between existing vertices and corresponding faces
        """
        new_faces = []
        for face in self.faces:
            new_faces.extend(self.refine_face(face))
        self.faces = new_faces

    def create_edges(self, faces):
        """
        Create edges from faces
        Returns:
            list: list of edges in COO format for PyG
        """
        edge_list1, edge_list2 = [], []
        for face in faces:
            v1, v2, v3 = face
            edge_list1 += [v1, v2, v3, v2, v3, v1]
            edge_list2 += [v2, v3, v1, v3, v1, v2]
        edge_index = [edge_list1, edge_list2]
        return edge_index

class IcosphereTetrahedron(IcosphereMesh):
    """
    Create tetrahedron from icosphere mesh.

    Args:
        icosphere (IcosphereMesh): Icosphere mesh to create tetrahedron from.
    """
    
    def __init__(self, n_refine, order = 1, minratio = 1.1, verbosity=0):
        """
        Initialize the IcosphereTetrahedron.

        Args:
            icosphere (IcosphereMesh): Icosphere mesh to create tetrahedron from.
            verbosity (int, optional): Verbosity level for tetgen. Defaults to 1.
            order (int, optional): Order of tetrahedron. Defaults to 2 (allow mid-edge nodes)
            minratio (float, optional): Minimum circumradius to shortest edge length ratio. Defaults to 1.0.
        """
        super().__init__(n_refine)
        self.verbosity = verbosity
        self.order = order
        self.minratio = minratio
        self.vertices, self.faces = self.create_tetrahedron()
        self.edges = np.array(self.create_edges(self.faces)).astype(np.int32)
        self.edge_feat = self.create_edge_feat()
        
    def create_tetrahedron(self):
        """
        Create tetrahedron from icosphere mesh.

        Returns:
            tuple: Tuple containing vertices and faces of the tetrahedron
        """
        # Initialize TetGen
        tgen = tetgen.TetGen(self.vertices, self.faces)        
        # Generate tetrahedral mesh
        nodes, tetrahedra = tgen.tetrahedralize(order = self.order, minratio = self.minratio, verbose = self.verbosity)
        
        #Extract re-indexed faces from tetrahedra
        faces = []
        for tet in tetrahedra:
            faces.append([tet[0], tet[1], tet[2]])
            faces.append([tet[0], tet[1], tet[3]])
            faces.append([tet[0], tet[2], tet[3]])
            faces.append([tet[1], tet[2], tet[3]])
        
        return np.array(nodes).astype(np.float32), np.array(faces).astype(np.int32)

    def create_edge_feat(self):
        #vertices[edges[0]] -> positions of sender nodes
        #vertices[edges[1]] -> positions of receiver nodes
        dist_pos = self.vertices[self.edges[0]] - self.vertices[self.edges[1]] #(num_edges, 3)
        x = dist_pos[:, 0]
        y = dist_pos[:, 1]
        z = dist_pos[:, 2]
        distances = np.sqrt(x**2 + y**2 + z**2)
        max_dist = np.max(distances)
        min_dist = np.min(distances)
        edge_weights = (distances - min_dist)/(max_dist - min_dist) #(num_edges, 1)
        edge_weights = np.expand_dims(edge_weights, 1)
        edge_feat = np.concatenate((dist_pos, edge_weights), axis = 1) #(num_edges, 4)
        return edge_feat
    
    