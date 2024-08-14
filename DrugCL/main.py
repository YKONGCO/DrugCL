from train import * 
import sys
from  utils import *
if __name__ == '__main__':
    for datasetname in ["lrssl"]:
        config1={
                'seed': 2024,
                'device': "cuda:0",
                'log_save_dir': "./log/",
                'log_id':f"test",
                'dropout': 0.1,
                'out_channels': 75,
                'epoch': 4000,
                'nhid1': 500, #feature
                'nhid2': 128, #t
                'lr': 0.001,
                'dataset':  datasetname,  #lrssl Ldataset Gdataset Cdataset
                'mu': 0.001,  # feature T
                'varphi':0.7, # CL
                'model_save_dir': "./models/",
                'optimizer': "adam",
                'agg_act':'relu',
                'share_param': True,
                'attention_nhid': 16,
                'train_grad_clip':1.0,
                "eval_step":250,
                "k_neighbor":1,
                "layers":1,
                "pf":0.1,
                "pe":0.5,
                "lambda_":None,
                "console_output_dir":"./console_output/",
                "tag":f"tes"
            }

        args = config(config1)
        sys.stdout = Logger(args.console_output_dir+args.dataset+"/"+args.log_id+".txt", sys.stdout)
        main(args)