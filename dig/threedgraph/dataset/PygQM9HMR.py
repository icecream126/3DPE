
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix, diags
import sys
# sys.path.append('/home/hsjang/hmkim/3DPE/dig/threedgraph/positional_encoding')
# sys.path.append('/home/hsjang/hmkim/3DPE/dig/threedgraph/dataset')

sys.path.append('/home/guest_khm/hyomin/3DPE/dig/threedgraph/positional_encoding')
sys.path.append('/home/guest_khm/hyomin/3DPE/dig/threedgraph/dataset')


from hmrpe import HMRPE
from PygQM93D import QM93D


import torch
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import radius_graph

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mayavi import mlab

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
'''
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
        self.eigen_vals, self.eigen_vecs = eigsh(A=self.stiffness, k=k, M=self.mass, sigma=-0.01)

        self.eigen_vals[0] = 0
        
        return self.eigen_vals, self.eigen_vecs

    def compute_stiffness_matrix(self):
        verts = self.verts
        faces = self.faces
        
        print('compute stiffness matrix')
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
'''
'''
# Generate random 3D point cloud
point_cloud = np.random.rand(5, 3)

# Compute Delaunay triangulation
tri = Delaunay(point_cloud)

# Get vertices and tetrahedron surface
vertices = point_cloud[tri.vertices]
T = tri.simplices #  tetrahedron

# F_all = list_faces(T) # 4C3 combination from tetrahedron

# Get triangle list from tetrahedron
F_surf = extract_surface(T)

trimesh = TriMesh(point_cloud, F_surf)

eig_val, eig_vec = trimesh.LB_decomposition(k=2)
print(eig_val)
print(eig_vec)
'''


class QM9HMRPE(InMemoryDataset):
    
    # def __init__(self,k, cutoff): 
    def __init__(self, data_list, k, cutoff) : # Before first processing
    
    # After first processing
    # def __init__(self,k, cutoff): 
        self.k = k
        self.data_list = data_list # Before first processing
        self.orig_qm9 = QM93D(root='dataset/')
        self.cutoff = cutoff
        super(QM9HMRPE, self).__init__('./dataset/qm9/lappe/')
        # self.data, self.slices = torch.load(self.processed_paths[0]) # After first processing
        self.data, self.slices = self.collate(data_list) # Before first processing
        
        
    def __str__(self):
        return f"k is {self.k}, cutoff is {self.cutoff}"

    @property
    def raw_file_names(self):
        return 'qm9_hmrpe_k_'+str(self.k)+'_cutoff_'+str(self.cutoff)+'.npz'

    @property
    def processed_file_names(self):
        return 'qm9_hmrpe_k_'+str(self.k)+'_cutoff_'+str(self.cutoff)+'.pt'
        

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

    
    def process(self):
        data, slices = self.collate(data_list)

        print('Saving hmrpe with k = '+str(self.k)+' and cutoff = '+str(self.cutoff)+'...')
        torch.save((data, slices), self.processed_paths[0])
    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial import Delaunay

def plot_tri_simple(ax, points, tri):
    for tr in tri:
        pts = points[tr, :]
        ax.plot3D(pts[[0,1],0], pts[[0,1],1], pts[[0,1],2], color='g', lw='0.1')
        ax.plot3D(pts[[0,2],0], pts[[0,2],1], pts[[0,2],2], color='g', lw='0.1')
        ax.plot3D(pts[[0,3],0], pts[[0,3],1], pts[[0,3],2], color='g', lw='0.1')
        ax.plot3D(pts[[1,2],0], pts[[1,2],1], pts[[1,2],2], color='g', lw='0.1')
        ax.plot3D(pts[[1,3],0], pts[[1,3],1], pts[[1,3],2], color='g', lw='0.1')
        ax.plot3D(pts[[2,3],0], pts[[2,3],1], pts[[2,3],2], color='g', lw='0.1')

    ax.scatter(points[:,0], points[:,1], points[:,2], color='b')

if __name__=="__main__":
    cutoff=10.0
    k=2
    origdataset = QM93D()
    data_list = []
    # hmrpe = HMRPE(walk_length=k)
    cnt=1
    for data in origdataset:
        # edge_index = radius_graph(data.pos, r=cutoff)
        # print('data.pos shape : ',data.pos.shape)
        point_cloud = data.pos.numpy()
        point_cloud -= point_cloud.mean(axis=0)
        # point_cloud = np.random.rand(5, 3)
        print('point cloud shape : ',point_cloud.shape)
        print('point cloud : ',point_cloud)
        tri = Delaunay(point_cloud)

        # Get vertices and tetrahedron surface
        # print('tri.vertices : ',tri.vertices.shape)
        # print(tri.vertices)
        # print()
        vertices = point_cloud[tri.vertices]
        faces = tri.simplices #  tetrahedron
        # print('Tetrahedron : ',T.shape)
        # print(T)
        # print()

        # Get triangle list from tetrahedron
        # F_surf = extract_surface(T)
        # print('Unique triangles : ',F_surf.shape)
        # print(F_surf)
        h = ConvexHull(point_cloud,incremental=True,qhull_options="Qs")
        print('h.simplices : ',h.simplices)
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # plot_tri_simple(ax, vertices,tri.simplices)
        # mlab.triangular_mesh(vertices[0,:], vertices[1,:], vertices[2,:], faces.T)
        # mlab.show()
        # plt.plot(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], 'o')
        # for simplex in h.simplices:
        #     plt.plot(point_cloud[simplex,0], point_cloud[simplex,1],point_cloud[simplex,2], 'k-')
        # plt.show()
        
        
        hmrpe = HMRPE(point_cloud, h.simplices)
        data.pe = hmrpe(k=2)
        data_list.append(data)
        print('Processed # of data : ',cnt,' / ',len(origdataset))
        cnt+=1

    dataset = QM9HMRPE(data_list = data_list, k=k, cutoff=cutoff) # Change these parameters as you want
    print(dataset.data)