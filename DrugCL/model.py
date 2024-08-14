from layers import *
import GCL.augmentors as A
from utils import *
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.device = args.device    
        self.FGCN = FGCN(args.drug_fnum, #ndrug
                         args.disease_fnum, #ndisease
                         args.nhid1,
                         args.out_channels,
                         args.dropout,
                         args.layers
                         ).to(self.device)
        
        self.attention = Attention(args.out_channels,args.attention_nhid)
        self.decoder = MLP(in_units=args.out_channels,dropout_rate=args.dropout).to(self.device)
        self.TCN=TCNlayer( (args.drug_fnum,args.disease_fnum),args.nhid2,args.out_channels,agg_act=args.agg_act,share_param=args.share_param,dropout=args.dropout)
        
    
    def forward(self,drug_graph,
                dis_graph, drug_dis_graph,edge_indexs):
        
        # Feature convolution operation
        drug_sim_out, dis_sim_out = self.FGCN(drug_graph, dis_graph)


        # GCN operation
        out=self.TCN(drug_dis_graph.x_dict,drug_dis_graph.edge_index_dict)
        drug_out=out['drug']
        dis_out=out['disease']


        # Attention operation 
        drug_feats = th.stack([drug_out, drug_sim_out], dim=1)      
        drug_feats= self.attention(drug_feats)
        dis_feats = th.stack([dis_out, dis_sim_out], dim=1)
        dis_feats= self.attention(dis_feats)



        pred_ratings = self.decoder( drug_feats, dis_feats,edge_indexs).to(self.device)
        
        return pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out




class Encoder(nn.Module):
    def __init__(self, model, augmentor,augmentor_dict={"node":[],"edge":[]}):
        super(Encoder, self).__init__()
        self.model = model
        self.augmentor = augmentor
        self.augmentor_dict=augmentor_dict
    

    def forward(self,drug_graph,
                dis_graph, drug_dis_graph,edge_indexs):
        aug1,aug2=self.augmentor
        drug_graph1,drug_graph2=graph_augmentor(drug_graph,aug1,aug2)
        dis_graph1,dis_graph2=graph_augmentor(dis_graph,aug1,aug2)

        pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out=self.model(drug_graph,dis_graph, drug_dis_graph,edge_indexs)
        
        drug_sim_out1,dis_sim_out1=self.model.FGCN(drug_graph1,dis_graph1)
        drug_sim_out2,dis_sim_out2=self.model.FGCN(drug_graph2,dis_graph2)
        
        return  pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out, drug_sim_out1,dis_sim_out1, drug_sim_out2,dis_sim_out2

