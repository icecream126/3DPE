
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse import coo_matrix, diags


# https://stackoverflow.com/questions/66607716/how-to-extract-surface-triangles-from-a-tetrahedral-mesh
def list_faces(t):
  t.sort(axis=1)
  n_t, m_t= t.shape 
  f = np.empty((4*n_t, 3) , dtype=int)
  i = 0
  for j in range(4):
    f[i:i+n_t,0:j] = t[:,0:j]
    f[i:i+n_t,j:3] = t[:,j+1:4]
    i=i+n_t
  return f

def extract_unique_triangles(t):
  _, indxs, count  = np.unique(t, axis=0, return_index=True, return_counts=True)
  return t[indxs[count==1]]

def extract_surface(t):
  f=list_faces(t)
  f=extract_unique_triangles(f)
  return f

class TriMesh(object):
    def __init__(self, verts, faces):
        self.verts = verts
        self.faces = faces
        self.normals = None
        self.face_areas = None
        self.per_vert_areas = None
        self.area = None

        self.eigen_vals = None
        self.eigen_vecs = None
        self.mass = None

        self.feats = None

    def LB_decomposition(self, k=None):
        # stiffness matrix
        self.stiffness = self.compute_stiffness_matrix()
        # mass matrix
        self.mass = self.compute_fem_mass_matrix()

        if k is None:
            k = self.verts.shape[0] - 1
        elif k <= 0:
            return

        # compute Laplace-Beltrami basis (eigen-vectors are stored column-wise)
        print('self.stiffness : ',self.stiffness.todense())
        print('self.mass : ',self.mass.todense())
        self.eigen_vals, self.eigen_vecs = eigsh(A=self.stiffness, k=k, M=self.mass, sigma=-0.01)

        self.eigen_vals[0] = 0
        
        return self.eigen_vals, self.eigen_vecs

    def compute_stiffness_matrix(self):
        verts = self.verts
        faces = self.faces
        print('compute stiffness matrix on test')
        print('verts  : ',verts.shape)
        print(verts)
        print()
        print('faces : ',faces.shape)
        print(faces)
        print()
        
        v1 = verts[faces[:, 0]]
        v2 = verts[faces[:, 1]]
        v3 = verts[faces[:, 2]]

        e1 = v3 - v2
        e2 = v1 - v3
        e3 = v2 - v1

        # compute cosine alpha/beta
        L1 = np.linalg.norm(e1, axis=1)
        L2 = np.linalg.norm(e2, axis=1)
        L3 = np.linalg.norm(e3, axis=1)
        cos1 = np.einsum('ij,ij->i', -e2, e3) / (L2 * L3)
        cos2 = np.einsum('ij,ij->i', e1, -e3) / (L1 * L3)
        cos3 = np.einsum('ij,ij->i', -e1, e2) / (L1 * L2)

        # cot(arccos(x)) = x/sqrt(1-x^2)
        I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        S = np.concatenate([cos3, cos1, cos2])
        S = 0.5 * S / np.sqrt(1 - S**2)

        In = np.concatenate([I, J, I, J]) 
        Jn = np.concatenate([J, I, I, J])
        Sn = np.concatenate([-S, -S, S, S])

        N = verts.shape[0]
        stiffness = coo_matrix((Sn, (In, Jn)), shape=(N, N)).tocsc()

        return stiffness

    def compute_fem_mass_matrix(self):
        verts = self.verts
        faces = self.faces
        # compute face areas
        v1 = verts[faces[:, 0]]
        v2 = verts[faces[:, 1]]
        v3 = verts[faces[:, 2]]
        face_areas = 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1), axis=1)

        I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        S = np.concatenate([face_areas, face_areas, face_areas])

        In = np.concatenate([I, J, I])
        Jn = np.concatenate([J, I, I])
        Sn = 1. / 12. * np.concatenate([S, S, 2*S])

        N = verts.shape[0]
        mass = coo_matrix((Sn, (In, Jn)), shape=(N, N)).tocsc()

        return mass
        
    def compute_normals(self):
        v1 = self.verts[self.faces[:, 0]]
        v2 = self.verts[self.faces[:, 1]]
        v3 = self.verts[self.faces[:, 2]]

        normals = np.cross(v2-v1, v3-v1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        return normals

    def grad(self, f, normalize=False):
        v1 = self.verts[self.faces[:,0]]  # (m,3)
        v2 = self.verts[self.faces[:,1]]  # (m,3)
        v3 = self.verts[self.faces[:,2]]  # (m,3)

        f1 = f[self.faces[:,0]]  # (m,p) or (m,)
        f2 = f[self.faces[:,1]]  # (m,p) or (m,)
        f3 = f[self.faces[:,2]]  # (m,p) or (m,)

        if self.face_areas is None:
            self.face_areas = self.compute_face_areas()

        if self.normals is None:
            self.normals = self.compute_normals()

        grad2 = np.cross(self.normals, v1-v3)/(2*self.face_areas[:,None])  # (m,3)
        grad3 = np.cross(self.normals, v2-v1)/(2*self.face_areas[:,None])  # (m,3)

        if f.ndim == 1:
            gradf = (f2-f1)[:,None] * grad2 + (f3-f1)[:,None] * grad3  # (m,3)
        else:
            gradf = (f2-f1)[:,:,None] * grad2[:,None,:] + (f3-f1)[:,:,None] * grad3[:,None,:]  # (m,3)

        if normalize:
            gradf /= np.linalg.norm(gradf,axis=1,keepdims=True)

        return gradf

    def compute_face_areas(self):
        v1 = self.verts[self.faces[:, 0]]
        v2 = self.verts[self.faces[:, 1]]
        v3 = self.verts[self.faces[:, 2]]

        face_areas = 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1),axis=1)
        return face_areas

    def compute_per_vert_areas(self):
        n_vertices = self.verts.shape[0]

        if self.face_areas is None:
            self.face_areas = self.compute_face_areas()

        I = np.concatenate([self.faces[:,0], self.faces[:,1], self.faces[:,2]])
        J = np.zeros_like(I)
        
        V = np.tile(self.face_areas / 3, 3)

        per_vert_areas = np.array(coo_matrix((V, (I, J)), shape=(n_vertices, 1)).todense()).flatten()

        return per_vert_areas

    def orientation_op(self, gradf, normalize=False):
        if normalize:
            gradf /= np.linalg.norm(gradf, axis=1, keepdims=True)

        n_vertices = self.verts.shape[0]
        
        v1 = self.verts[self.faces[:,0]]  # (n_f,3)
        v2 = self.verts[self.faces[:,1]]  # (n_f,3)
        v3 = self.verts[self.faces[:,2]]  # (n_f,3)

        # compute normals
        if self.normals is None:
            self.normals = self.compute_normals()
        # computer per vertex area
        if self.per_vert_areas is None:
            self.per_vert_area = self.compute_per_vert_areas()

        # Define (normalized) gradient directions for each barycentric coordinate on each face
        # Remove normalization since it will disappear later on after multiplcation
        Jc1 = np.cross(self.normals, v3-v2)/2
        Jc2 = np.cross(self.normals, v1-v3)/2
        Jc3 = np.cross(self.normals, v2-v1)/2

        # Rotate the gradient field
        rot_field = np.cross(self.normals, gradf)  # (n_f,3)

        I = np.concatenate([self.faces[:,0], self.faces[:,1], self.faces[:,2]])
        J = np.concatenate([self.faces[:,1], self.faces[:,2], self.faces[:,0]])

        # Compute pairwise dot products between the gradient directions
        # and the gradient field
        Sij = 1/3*np.concatenate([np.einsum('ij,ij->i', Jc2, rot_field),
                                np.einsum('ij,ij->i', Jc3, rot_field),
                                np.einsum('ij,ij->i', Jc1, rot_field)])

        Sji = 1/3*np.concatenate([np.einsum('ij,ij->i', Jc1, rot_field),
                                np.einsum('ij,ij->i', Jc2, rot_field),
                                np.einsum('ij,ij->i', Jc3, rot_field)])

        In = np.concatenate([I, J, I, J])
        Jn = np.concatenate([J, I, I, J])
        Sn = np.concatenate([Sij, Sji, -Sij, -Sji])

        W = coo_matrix((Sn, (In, Jn)), shape=(n_vertices, n_vertices)).tocsc()
        inv_area = diags(1/self.per_vert_area, shape=(n_vertices, n_vertices), format='csc')

        return inv_area @ W

# Generate random 3D point cloud
# point_cloud = np.random.rand(9, 3)
point_cloud = np.array([[-1.2698136e-02,  1.0858041e+00,  8.0009960e-03],
 [ 2.1504159e-03, -6.0313176e-03,  1.9761203e-03],
 [ 1.0117308e+00,  1.4637512e+00,  2.7657481e-04],
 [-5.4081506e-01,  1.4475266e+00, -8.7664372e-01],
 [-5.2381361e-01,  1.4379326e+00,  9.0639728e-01]])
print('point cloud shape : ',point_cloud.shape) # (5,3)
print('point cloud : ',point_cloud)

# Compute Delaunay triangulation
tri = Delaunay(point_cloud)

# Get vertices and tetrahedron surface
print('tri.vertices : ',tri.vertices.shape) # vertices info which are in same tetrahedron (4,4)
print(tri.vertices)
print()
vertices = point_cloud[tri.vertices]
T = tri.simplices #  tetrahedron

# F_all = list_faces(T) # 4C3 combination from tetrahedron

# Get triangle list from tetrahedron
F_surf = extract_surface(T)

trimesh = TriMesh(point_cloud, F_surf) 

eig_val, eig_vec = trimesh.LB_decomposition(k=2)
print('eig_val')
print(eig_val)
print('eig-vec')
print(eig_vec)