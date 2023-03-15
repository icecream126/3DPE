
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.sparse.linalg import eigsh, eigs, svds
from scipy.linalg import eig
from scipy.sparse import coo_matrix, diags
import pyvista as pv 



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

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)
    
class HMRPE(object):
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

    # def LB_decomposition(self, k=None):
    def __call__(self, k=None):
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
        np_stiffness = self.stiffness.todense()
        np_mass = self.mass.todense()
        
        # print('stifness nan any : ',np.isnan(np_stiffness))
        # print('mass nan any : ',np.isnan(np_mass))

        # print('stifness inf any : ',np.isinf(np_stiffness))
        # print('mass inf any : ',np.isinf(np_mass))

        print('is invertible stiffness : ',is_invertible(np_stiffness))
        print('is invertible mass : ',is_invertible(np_mass)) # if mass becomes singular matrix -> fails to eigendecomposition..


        print('is psd stiffness : ',is_pos_def(np_stiffness))
        print('is psd mass : ',is_pos_def(np_mass)) # if mass becomes singular matrix -> fails to eigendecomposition..

        self.eigen_vals, self.eigen_vecs = eigsh(A=self.stiffness.todense(), k=k,  M=self.mass.todense(), sigma=-0.01)
        # self.eigen_vals, self.eigen_vecs = svds(A=self.stiffness, k=k, which='SM')

        self.eigen_vals[0] = 0
        
        # return self.eigen_vals, self.eigen_vecs
        return self.eigen_vecs

    def compute_stiffness_matrix(self):
        verts = self.verts
        faces = self.faces
        
        # print('verts  : ',verts.shape)
        # print('faces : ',faces.shape)
        
        v1 = verts[faces[:, 0]]
        v2 = verts[faces[:, 1]]
        v3 = verts[faces[:, 2]]

        e1 = v3 - v2
        e2 = v1 - v3
        e3 = v2 - v1

        # print('e1 : ',e1)
        # print('e2 : ',e2)
        # print('e3 : ',e3)

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
        print('verts ; ',verts.shape)
        print('faces ; ',faces.shape)
        # compute face areas
        v1 = verts[faces[:, 0]]
        v2 = verts[faces[:, 1]]
        v3 = verts[faces[:, 2]]
        print('v2-v1 : ',(v2-v1).shape)
        print('v3-v1 : ',(v3-v1).shape)
        face_areas = 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1), axis=1)
        print('face_areas : ',face_areas)

        I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        S = np.concatenate([face_areas, face_areas, face_areas])
        print('I :',I.shape)
        print('J : ',J.shape)
        print('S : ',S.shape)

        In = np.concatenate([I, J, I])
        Jn = np.concatenate([J, I, I])
        Sn = 1. / 12. * np.concatenate([S, S, 2*S])

        N = verts.shape[0]
        print('In : ',In.shape)
        print('Jn : ',Jn.shape)
        print('Sn : ',Sn.shape)
        

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


if __name__=="__main__":
    # Generate random 3D point cloud
    point_cloud = np.random.rand(5, 3)
    cloud = pv.PolyData(point_cloud)
    # cloud.plot(screenshot='cloud.png')
    
    alpha=5
    mesh = cloud.delaunay_3d(alpha=alpha)
    surface = mesh.extract_surface()
    print('mesh : ',mesh)
    print('surface : ',surface)
    # surface.plot(screenshot='surface.png')

    # Compute Delaunay triangulation
    # tri = Delaunay(point_cloud)

    # Get vertices and tetrahedron surface
    # vertices = point_cloud[tri.vertices]
    # T = tri.simplices #  tetrahedron

    # F_all = list_faces(T) # 4C3 combination from tetrahedron

    # Get triangle list from tetrahedron
    # F_surf = extract_surface(T)
    h = ConvexHull(point_cloud)
    # print(h.simplices)

    trimesh = HMRPE(point_cloud, h.simplices)

    eig_vec = trimesh(k=2)
    print(eig_vec)