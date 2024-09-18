import sys
import numpy as np

sys.path.extend(['../'])
import numpy as np

def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def get_ins_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(np.ones_like(I)-I)
    Out = In.T
    A = np.stack((I, In, Out))
    return A

def get_spatial_graphnextv2(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    #In = normalize_digraph(edge2mat(inward, num_node))
    #Out = normalize_digraph(edge2mat(outward, num_node))
    #SELF = np.eye(num_node)
    A = np.stack((I, I, I, I))
    return A

def get_spatial_graphnext(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    SELF = np.eye(num_node)
    A = np.stack((I, In, Out, SELF))
    return A

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    A1 = edge2mat(inward, num_node)
    A2 = edge2mat(outward, num_node)
    A3 = k_adjacency(A1, 2)
    A4 = k_adjacency(A2, 2)
    A1 = normalize_digraph(A1)
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A = np.stack((I, A1, A2, A3, A4))
    return A



def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A

COCO = [0,5,6,7,8,9,10,11,12,13,14,15,16]
COCO_sub = [[0,1,2,3,4,6,5],[5,12,11],[6,11,12],7,8,9,10,[11,6,5],[12,5,6],13,14,15,16]
NTU =  [4,5,9,6,10,7,11,13,17,14,18,15,19]
NTU_sub = [[4,3,21,9,5],[5,2,17,1,13],[9,2,13,1,17],[6,6,6,6,6],[10,10,10,10,10],[7,7,8,22,23],[11,11,12,24,25],[13,2,9,21,5],[17,2,5,21,9],[14,14,14,14,14],[18,18,18,18,18],[15,15,15,16,16,],[19,19,19,20,20]]
NTU = [i-1 for i in NTU]
for i,si in enumerate(NTU_sub):
    NTU_sub[i] = [i-1 for i in NTU_sub[i]]


num_node = 13
self_link = [(i, i) for i in range(num_node)]
inward = [
    [0,1],[0,2],[1,3],[3,5],[2,4],[4,6],[0,7],[0,8],[7,9],[9,11],[8,10],[10,12]
]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

        self.A_binary = edge2mat(neighbor, num_node)
        self.A_norm = normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = get_k_scale_graph(scale, self.A_binary)

        #self.A_A1 = ((self.A_binary + np.eye(num_node)) / np.sum(self.A_binary + np.eye(self.A_binary.shape[0]), axis=1, keepdims=True))[indices_1]
        #self.A1_A2 = tools.edge2mat(neighbor_1, num_node_1) + np.eye(num_node_1)
        #self.A1_A2 = (self.A1_A2 / np.sum(self.A1_A2, axis=1, keepdims=True))[indices_2]


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'spatialnext':
            A = get_spatial_graphnext(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
