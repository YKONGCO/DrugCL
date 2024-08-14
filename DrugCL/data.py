import os
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from sklearn.model_selection import KFold
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data,HeteroData
import numpy as np
import torch as th 

_paths = {
    'Gdataset': './raw_data/drug_data/Gdataset/Gdataset.mat',
    'Cdataset': './raw_data/drug_data/Cdataset/Cdataset.mat',
    'Ldataset': './raw_data/drug_data/Ldataset/lagcn',
    'lrssl': './raw_data/drug_data/lrssl',
    'covid': './raw_data/drug_data/covid'

}

def normalize(mx):
    """Row-normalize sparse matrix 稀疏矩阵归一化函数""" 
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten() #结果展平
    r_inv[np.isinf(r_inv)] = 0. #inf（正负无穷大）替换为0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sys_normalized_adjacency(adj):
   row_sum = np.array(adj.sum(1))
   row_sum = (row_sum == 0)*1 + row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def generate_adj_graph(association_matrix):
    adj_matrix = association_matrix
    top_row = th.cat((th.zeros((adj_matrix.size(0), adj_matrix.size(0)), dtype=th.float32), adj_matrix), dim=1)
    bottom_row = th.cat((adj_matrix.T, th.zeros((adj_matrix.T.size(0), adj_matrix.T.size(0)), dtype=th.float32)), dim=1)
    adj_matrix  = th.cat((top_row, bottom_row), dim=0)
    return adj_matrix  

def read_sparse_matrix(filename,shape):
    import numpy as np
    from scipy.sparse import coo_matrix
    rows, cols, data = [], [], []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()  
            rows.append(int(parts[0]))
            cols.append(int(parts[1])) 
            data.append(int(parts[2]))  
    sparse_matrix = coo_matrix((data, (rows, cols)), shape=shape)
    dense_matrix = sparse_matrix.toarray()
    return dense_matrix





class DrugDataLoader(object):
    def __init__(self,
                 name,
                 device,
                 symm=True,
                 k=2):
        self._name = name
        self.device = device
        self._symm = symm
        self.num_neighbor = k
        print(f"Starting processing {self._name} ...")
        self._dir = os.path.join(_paths[self._name])
        self._load_drug_data(self._dir, self._name)
        self.drug_graph, self.disease_graph = self._generate_feat_graph()
        self.k_data=self._generate_k_hetero_graph()
        self.set_device()
        self.print_info()
        print('Load data successfully!')

    def _load_drug_data(self, file_path, data_name):
        if data_name in ['Gdataset', 'Cdataset']:
            data = sio.loadmat(file_path)
            self.association_matrix = data['didr'].T
            self.disease_sim_features = data['disease']
            self.drug_sim_features = data['drug']
        elif data_name in ['Ldataset']:
            self.association_matrix = np.loadtxt(os.path.join(file_path, 'drug_dis.csv'), delimiter=",")
            self.disease_sim_features = np.loadtxt(os.path.join(file_path, 'dis_sim.csv'), delimiter=",")
            self.drug_sim_features = np.loadtxt(os.path.join(file_path, 'drug_sim.csv'), delimiter=",")
        elif data_name in ['lrssl']:
            data = pd.read_csv(os.path.join(file_path, 'drug_dis.txt'), index_col=0, delimiter='\t')
            self.association_matrix = data.values
            self.disease_sim_features = pd.read_csv(
                os.path.join(file_path, 'dis_sim.txt'), index_col=0, delimiter='\t').values
            self.drug_sim_features = pd.read_csv(
                os.path.join(file_path, 'drug_sim.txt'), index_col=0, delimiter='\t').values
        self._num_drug = self.association_matrix.shape[0]
        self._num_disease = self.association_matrix.shape[1]

    



    def _generate_feat_graph(self):
        def knn_graph(disMat, k):
            k_neighbor = np.argpartition(-disMat, kth=k, axis=1)[:, :k]
            row_index = np.arange(k_neighbor.shape[0]).repeat(k_neighbor.shape[1])
            col_index = k_neighbor.reshape(-1)
            edges = np.array([row_index, col_index]).astype(int).T
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                        shape=(disMat.shape[0], disMat.shape[0]),
                                        dtype=np.float32)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            return adj

        # drug feature graph
        drug_sim = self.drug_sim_features
        drug_num_neighbor = self.num_neighbor
        if drug_num_neighbor > drug_sim.shape[0] or drug_num_neighbor < 0:
            drug_num_neighbor = drug_sim.shape[0]

        drug_adj = knn_graph(drug_sim, drug_num_neighbor)
        drug_graph = normalize(drug_adj + sp.eye(drug_adj.shape[0]))
        drug_edge_index,drug_edge_weight=from_scipy_sparse_matrix(drug_graph)
        drug_graph = Data(x= th.tensor(self.drug_sim_features,dtype=th.float32),edge_index=drug_edge_index)
        # disease feature graph
        disease_sim = self.disease_sim_features
        disease_num_neighbor = self.num_neighbor
        if disease_num_neighbor > disease_sim.shape[0] or disease_num_neighbor < 0:
            disease_num_neighbor = disease_sim.shape[0]

        disease_adj = knn_graph(disease_sim, disease_num_neighbor)
        disease_graph = normalize(disease_adj + sp.eye(disease_adj.shape[0]))
        dis_edge_index,dis_edge_weight=from_scipy_sparse_matrix(disease_graph)
        disease_graph = Data(x= th.tensor(self.disease_sim_features,dtype=th.float32),edge_index=dis_edge_index)

        return drug_graph, disease_graph #drug 与 disease 的knn特征图
    def set_device(self):
        self.drug_graph.to(self.device)
        self.disease_graph.to(self.device) 
    






    def _generate_k_hetero_graph(self,k=10):  
        hetero_train=HeteroData()
        hetero_train['drug'].x=th.tensor(self.drug_sim_features,dtype=th.float32)
        hetero_train['disease'].x=th.tensor(self.disease_sim_features,dtype=th.float32)
        
        hetero_test=HeteroData()
        hetero_test['drug'].x=th.tensor(self.drug_sim_features,dtype=th.float32)
        hetero_test['disease'].x=th.tensor(self.disease_sim_features,dtype=th.float32)
           
        kfold = KFold(n_splits=k, shuffle=True, random_state=1024)
        association_matrix=th.tensor(self.association_matrix)
        pos_row, pos_col = th.nonzero(association_matrix).t()
        neg_row, neg_col = th.nonzero(1 - association_matrix).t()
        
        
        k_data = {}
        i=0
        for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                                kfold.split(neg_row)):


            # train_neg_idx=th.tensor( random.sample(train_neg_idx.tolist(), 0*len(train_pos_idx)))
            train_pos_edge =th.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
            train_pos_values = th.tensor([1] * len(train_pos_edge[0]))
            train_neg_edge =th.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
            train_neg_values = th.tensor([0] * len(train_neg_edge[0]))

            test_pos_edge = th.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
            test_pos_values = th.tensor([1] * len(test_pos_edge[0]))

            test_neg_edge = th.stack([neg_row[test_neg_idx], neg_col[test_neg_idx]])
            test_neg_values =th.tensor( [0] * len(test_neg_edge[0]))

            train_edge = th.cat([train_pos_edge, train_neg_edge], axis=1)
            train_values = th.cat([train_pos_values, train_neg_values])
       
            test_edge = th.cat([test_pos_edge, test_neg_edge], axis=1)
            test_values = th.cat([test_pos_values, test_neg_values])
            
            

            hetero_train['drug', '_1', 'disease'].edge_index = train_pos_edge
            hetero_train['drug', '_0', 'disease'].edge_index = train_neg_edge
            hetero_train['disease', 'rev1', 'drug'].edge_index = train_pos_edge[[1,0]]
            hetero_train['disease', 'rev0', 'drug'].edge_index = train_neg_edge[[1,0]]

            train_data={
                "drug_dis":train_edge,
                "graph":hetero_train,
                "value":train_values
            }
            
            test_data={
                "drug_dis":test_edge,
                "graph":hetero_train,
                "value":test_values
            }
            
            k_data[i]=[train_data,test_data]
            i+=1
        return k_data
        
    
    def getindex(self,disease_index):
        disease_tensor=th.full((self.num_drug,), disease_index)
        drug_tensor=th.arange(self.num_drug)
        edge_index=th.stack([drug_tensor,disease_tensor])
        return edge_index
    
        

    def print_info(self):
        print("-----------------------------------------------------------")
        print(f"Name: {self._name}")
        print(f"Device: {self.device}")
        print(f"Symmetry: {self._symm}")
        print(f"Number of Neighbors: {self.num_neighbor}")
        print(f"Directory: {self._dir}")
        print(f"Drug Graph:{self.drug_graph}")
        print(f"Disease Graph:{self.disease_graph}")
        print(f"Number of Drugs: {self._num_drug}")
        print(f"Number of Diseases: {self._num_disease}")
        print("-----------------------------------------------------------")


    
    def get_empty_hetero_graph(self):
        hetero_train=HeteroData()
        hetero_train['drug'].x=th.tensor(self.drug_sim_features,dtype=th.float32)
        hetero_train['disease'].x=th.tensor(self.disease_sim_features,dtype=th.float32)
        association_matrix=th.tensor(self.association_matrix)
        pos_row, pos_col = th.nonzero(association_matrix).t()
        neg_row, neg_col = th.nonzero(1 - association_matrix).t() 
        pos_edge =th.stack([pos_row, pos_col])
        pos_values = th.tensor([1] * len(pos_edge[0]))
        neg_edge =th.stack([neg_row, neg_col])
        neg_values = th.tensor([0] * len(neg_edge[0]))
        
        hetero_train['drug', '_1', 'disease'].edge_index = pos_edge
        hetero_train['drug', '_0', 'disease'].edge_index = th.tensor([[],[]])
        hetero_train['disease', 'rev1', 'drug'].edge_index = pos_edge[[1,0]]
        hetero_train['disease', 'rev0', 'drug'].edge_index = th.tensor([[],[]])
        truth_values = th.tensor([1] * len(pos_edge[0]))
        edge = th.cat([pos_edge, neg_edge], axis=1)
        values = th.cat([pos_values, neg_values])
        return hetero_train,neg_edge,pos_edge,truth_values
    
    
    
    


    
    
    @property
    def num_disease(self):
        return self._num_disease

    @property
    def num_drug(self):
        return self._num_drug


if __name__ == '__main__':
    DrugDataLoader("lrssl", device=th.device('cuda'), symm=True)
 
