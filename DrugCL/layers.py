import torch as th
import torch.nn as nn
from torch_geometric.nn import GCN 
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import HeteroConv
from typing import List, Optional, Tuple, Union
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, Size
from typing import Optional
from torch_geometric.utils import degree



class FGCN(nn.Module): 
    def __init__(self, fdim_drug, fdim_disease, nhid1, nhid2, dropout,num_layers=1):
        super(FGCN, self).__init__()


        self.FGCN1 =GCN(fdim_drug, nhid1,num_layers,nhid2, dropout,normalize=False,bias=True)
        self.FGCN2 = GCN(fdim_disease, nhid1,num_layers,nhid2, dropout,normalize=False,bias=True)
        self.dropout = dropout

    def forward(self, drug_graph, dis_graph):
        emb1 = self.FGCN1(drug_graph.x,drug_graph.edge_index)
        emb2 = self.FGCN2(dis_graph.x,dis_graph.edge_index)
        return emb1,emb2

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
    def forward(self, z):
        w = self.project(z)
        beta = th.softmax(w, dim=1)
        return  (beta * z).sum(1)

class MLP(nn.Module):
    def __init__(self,
                 in_units,
                 dropout_rate=0.2):
        super(MLP, self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(2*in_units, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64,1),
        )

       
    def forward(self,drug_feat, dis_feat,edge_index):
        edge_feat=th.cat((drug_feat[edge_index[0]],dis_feat[edge_index[1]]),dim=1)
        pred_ratings=self.layer(edge_feat)
        return pred_ratings




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





class TCNConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        dropout = 0.0,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
            
        super().__init__(aggr, **kwargs)
        self.weight = nn.Parameter(th.Tensor(in_channels[0], out_channels))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()



    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
               nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
        weight = None
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if size is None:
            size=(x[0].size(0), x[1].size(0))
            self.size=size
        if weight is None:
            weight = self.weight
       
        ci,cj=self.get_cij(edge_index)
        x=list(x)
        x[0]=th.mm(x[0]*cj,weight)
        x[1]=x[1]*ci
        # propagate_type: (x: OptPairTensor)
  
        out = self.propagate(edge_index,x=x,size=size)
        return out


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def get_cij(self,edge_index):
        x_j = degree(edge_index[0],self.size[0],dtype=th.float32)   
        x_i = degree(edge_index[1],self.size[1],dtype=th.float32)  
        def norm(x: Tensor):
            x[x == 0.] = float('inf') 
            return x.pow_(-0.5).unsqueeze(-1)
        
        x_j=norm(x_j)
        x_i=norm(x_i)
        return x_i,self.dropout(x_j)


class TCNlayer(nn.Module):
    def __init__(self,
                in_channels, 
                msg_units, 
                out_units, 
                dropout=0.0, 
                agg='sum', 
                agg_act=None,  
                share_param=False,  
                **kwargs): 

        super().__init__(**kwargs)
        self.agg = agg  # sum
        self.share_param = share_param 
        self.ufc = nn.Linear(msg_units, out_units) 
        self.msg_units = msg_units  
        if share_param:
            self.ifc = self.ufc
        else:
            self.ifc = nn.Linear(msg_units, out_units)
            
        self.msg_units = msg_units
        self.dropout = nn.Dropout(dropout) 
        self.agg_act = get_activation(agg_act)
        
        self.conv = HeteroConv({
            ('drug', '_1', 'disease'):TCNConv(in_channels, msg_units,dropout=dropout) ,
            ('drug', '_0', 'disease'): TCNConv(in_channels, msg_units, dropout=dropout),
            ('disease', 'rev1', 'drug'):TCNConv((in_channels[1],in_channels[0]), msg_units, dropout=dropout),
            ('disease', 'rev0', 'drug'):TCNConv((in_channels[1],in_channels[0]), msg_units, dropout=dropout)
            }, aggr=agg)
        self.reset_parameters()


    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    
    def forward(self,x_dict,edge_index_dict):
        out=self.conv(x_dict,edge_index_dict)
        out["drug"]=self.dropout(self.agg_act(self.ufc(out["drug"])))
        out["disease"]=self.dropout(self.agg_act(self.ifc(out["disease"])))
        return out
        
        
        

    
