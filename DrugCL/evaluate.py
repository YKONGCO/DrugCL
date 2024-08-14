from utils import * 
import torch as th
from sklearn import metrics
def evaluate(model, 
             drug_graph,  
             dis_graph, 
             test_data):
    
    adj=test_data["graph"]
    edge_index=test_data["drug_dis"]
    values=test_data["value"]
    model.eval()
    with th.no_grad():
        pred_ratings, _, _, _, _ =  model(drug_graph, dis_graph,adj,edge_index)
    y_score = pred_ratings.squeeze(1).cpu().tolist()
    y_true = values.cpu().tolist()
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall, precision)
    return auc, aupr, y_true, y_score