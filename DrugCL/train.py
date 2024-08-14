from model import *
from data import *
import torch
import torch.nn as nn
from utils import * 
from config import *
import time
from tensorboardX import SummaryWriter
from evaluate import *
import torch
import GCL.losses as L
import torch_geometric.transforms as T
from GCL.models.contrast_model import WithinEmbedContrast
import warnings


warnings.filterwarnings("ignore", message="'dropout_adj' is deprecated")




def train(args,encoder,drug_graph,disease_graph,train_data,test_data,k,save=False):
    torch.cuda.empty_cache()
    loss_fn = nn.BCEWithLogitsLoss().to(args.device)
    optimizer = get_optimizer(args.optimizer)(encoder.parameters(), lr=args.lr)
    contrast_model = WithinEmbedContrast(loss=L.BarlowTwins(lambda_=args.lambda_)).to(args.device)
    
    #data
    drug_dis_graph=train_data["graph"]
    edge_index=train_data["drug_dis"]
    truth_value=train_data["value"]
    
        
    start_time = time.perf_counter()
    # writer = SummaryWriter(log_dir=f"./runs/{args.dataset}")
    best_auc=0
    best_aupr=0
    best_wt=None
    
    
    for epoch in range(args.epoch):    
        encoder.train()
        optimizer.zero_grad()
        
        pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out, drug_sim_out1,dis_sim_out1, drug_sim_out2,dis_sim_out2 = encoder(drug_graph, disease_graph,drug_dis_graph,edge_index)
        
        loss_drug=contrast_model(drug_sim_out1,drug_sim_out2)
        loss_disease=contrast_model(dis_sim_out1,dis_sim_out2)
        
        loss_com_drug = common_loss(drug_out, drug_sim_out)
        loss_com_dis = common_loss(dis_out, dis_sim_out)
        loss =  args.mu * loss_com_dis + args.mu * loss_com_drug + \
     loss_fn(pred_ratings.float().squeeze(-1), truth_value.float()) + \
                args.varphi*(loss_drug+loss_disease)
        
        
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), args.train_grad_clip)
        optimizer.step()


        if epoch % args.eval_step == 0 and epoch != 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, args.epoch, loss.item()))
            auc, aupr, _, _=evaluate(encoder.model, drug_graph, disease_graph,test_data)
            print('AUC: {:.4f}, AUPR: {:.4f}'.format(auc, aupr))
            if auc*aupr > best_auc*best_aupr:
                best_auc = auc
                best_aupr = aupr
                if(save):
                    best_wt=encoder.model.state_dict() #is need to save
            
            
            
            # writer.close()
        # writer.add_scalar(f'Loss dataset:{args.dataset} lr:{args.lr} drop:{args.dropout} mu:{args.mu} k:{k} time:{start_time}', loss.item(), epoch)  # 使用epoch作为x轴
       
    end_time = time.perf_counter()
    print("running time", time.strftime("%H:%M:%S", time.gmtime(round(end_time - start_time))),"\n")
    if(save):
        th.save(best_wt, f'./models/{args.dataset}/{args.log_id}_{k}_{best_auc}_{best_aupr}.pth')
    return best_auc, best_aupr
    
    
 

       
def main(args):
    setup_seed(args.seed)
    dataset = DrugDataLoader(args.dataset, args.device,k=args.k_neighbor)
    k_data=dataset.k_data
    drug_graph=dataset.drug_graph
    dis_graph=dataset.disease_graph
    args.drug_fnum=dataset.num_drug
    args.disease_fnum=dataset.num_disease
    args.adj_fnum=dataset.num_drug+dataset.num_disease



    aug1 = A.Compose([A.EdgeRemoving(pe=args.pe), A.FeatureMasking(pf=args.pf)])
    aug2 = A.Compose([A.EdgeRemoving(pe=args.pe), A.FeatureMasking(pf=args.pf)])
    
    augmentor=(aug1,aug2)
    augmentor_dict={"node":[A.FeatureMasking(pf=0.1)],"edge":[A.EdgeRemoving(pe=0.5)]}
    
    auc_list=[]
    aupr_list=[]
    for i in range(10):
        print("-----------------",'fold:',i,"-----------------")
        model = Net(args)
        encoder=Encoder(model,augmentor,augmentor_dict)
        encoder.to(args.device)

        auc,aupr=train(args, encoder,drug_graph,dis_graph,k_data[i][0],k_data[i][1],i)
        auc_list.append(auc)
        aupr_list.append(aupr)
        print('BEST AUC: {:.4f}, AUPR: {:.4f}'.format(auc, aupr))
        save_log(args,auc_list,aupr_list)
        
        
        
        
