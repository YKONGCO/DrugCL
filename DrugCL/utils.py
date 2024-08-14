import random
import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy import sparse as sp
import time
import os,json
import GCL.augmentors as A
from torch_geometric.data import HeteroData
from torch_geometric.data import Data



def get_activation(act):
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError
    else:
        return act


def get_optimizer(opt,):
    if opt == 'sgd':
        return optim.SGD
    elif opt == 'adam':
        return optim.Adam
    else:
        raise NotImplementedError



def common_loss(emb1, emb2):
    emb1 = emb1 - th.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - th.mean(emb2, dim=0, keepdim=True)
    emb1 = th.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = th.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = th.matmul(emb1, emb1.t())
    cov2 = th.matmul(emb2, emb2.t())
    cost = th.mean((cov1 - cov2) ** 2)
    return cost


def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True



def knn_graph(disMat, k):
    k_neighbor = np.argpartition(-disMat, kth=k, axis=1)[:, :k]
    row_index = np.arange(k_neighbor.shape[0]).repeat(k_neighbor.shape[1])
    col_index = k_neighbor.reshape(-1)
    edges = np.array([row_index, col_index]).astype(int).T
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(disMat.shape[0], disMat.shape[0]),
                                dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    return adj




def time_log(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print("running time", time.strftime("%H:%M:%S", time.gmtime(round(end_time - start_time))),"\n")

    return wrapper



def save_model(model, save_path):
    th.save(model.state_dict(), save_path)
def save_log(args, auc_list, aupr_list):
    log_data = {
        "args": {k: v for k, v in vars(args).items()},
        "metrics": {
            "auc": auc_list,
            "aupr": aupr_list,
            "average_auc": sum(auc_list) / len(auc_list) if auc_list else None,
            "average_aupr": sum(aupr_list) / len(aupr_list) if aupr_list else None
        }
    }
    filename = f"{args.dataset}_{args.optimizer}_{args.epoch}_{args.log_id}.json"
    filepath = os.path.join(args.log_save_dir, args.dataset, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)
        
        
        


def hetero_augmentor(hetero_graph, node_augmentor=[A.FeatureMasking(pf=0.1)], edge_augmentor=[A.EdgeRemoving(pe=0.5)]):
    hetero_graph1=HeteroData()
    for key, x in hetero_graph.x_dict.items():
        for aug in node_augmentor:
            edge_index = th.tensor([])
            x, edge_index, _ = aug(x, edge_index)
        hetero_graph1[key].x = x
    
    
    for key, edge_index in hetero_graph.edge_index_dict.items():
        for aug in edge_augmentor:
            x = th.tensor([])
            x, edge_index, _ = aug(x, edge_index)
        hetero_graph1[key].edge_index = edge_index
   
    
        
    return hetero_graph1

def graph_augmentor(graph1,aug1,aug2):
    x1,edge_index1,_=aug1(graph1.x,graph1.edge_index)
    x2,edge_index2,_=aug2(graph1.x,graph1.edge_index)
    graph1=Data(x=x1,edge_index=edge_index1)
    graph2=Data(x=x2,edge_index=edge_index2)
    return graph1,graph2


import sys


class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag
    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass