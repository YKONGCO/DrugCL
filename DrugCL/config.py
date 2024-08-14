import torch as th
import argparse
import os
# def config():
#     parser = argparse.ArgumentParser(description='AdaDR')
#     parser.add_argument('--seed', default=2024, type=int)
#     parser.add_argument('--device', default="cuda:0", type=str, help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
#     parser.add_argument('--log_save_dir', type=str,default="./log/" ,help='The saving directory')
#     parser.add_argument('--dropout', type=float, default=0.3)
#     parser.add_argument('--out_channels', type=int, default=75)
#     parser.add_argument('--epoch', type=int, default=1000) # 4000
#     parser.add_argument('--nhid1', type=int, default=500)
#     parser.add_argument('--nhid2', type=int, default=500)
#     parser.add_argument('--lr', type=float, default=0.001) 
#     parser.add_argument('--layers', type=int, default=2) #之后弄
#     parser.add_argument('--dataset', default='lrssl', type=str,help="'lrssl' 'Ldataset' 'Gdataset' 'Cdataset'")
#     parser.add_argument('--mu', type=float, default=0)  # 0.1
#     parser.add_argument('--model_save_dir', type=str, default="./models/")
#     parser.add_argument('--optimizer', type=str, default="adam")
#     parser.add_argument('--attention_nhid', type=int , default=16)
#     args = parser.parse_args()
#     th.set_default_device(args.device)
#     if not os.path.isdir(args.log_save_dir+args.dataset):
#         os.makedirs(args.log_save_dir+args.dataset)

#     if not os.path.isdir(args.model_save_dir+args.dataset):
#         os.makedirs(args.model_save_dir+args.dataset)

#     print(args)
#     return args



def config(config_dict=None):
    default_config = {
        'seed': 2024,
        'device': "cuda:0",
        'log_save_dir': "./log/",
        'log_id':"2",
        'dropout': 0.1,
        'out_channels': 75,
        'epoch': 1000,
        'nhid1': 500,
        'nhid2': 500,
        'lr': 0.001,
        'dataset': 'lrssl',
        'mu': 0.01,
        'varphi':0.7,
        'model_save_dir': "./models/",
        'optimizer': "adam",
        'agg_act':'relu',
        'share_param': True,
        'train_grad_clip':1.0,
        'attention_nhid': 16,
        "eval_step":500,
        "console_output_dir":"./console_output/",
        "tag":"注释性的文字"
    }

    if config_dict is None:
        config_dict = default_config
        
    parser = argparse.ArgumentParser(description='AdaDR')
    for key in config_dict:
        parser.add_argument(f'--{key}', type=type(config_dict[key]))
        
    args = parser.parse_args([])
    for key, value in config_dict.items():
        setattr(args, key, value)
        
    th.set_default_device(args.device)
    for directory in [args.log_save_dir, args.model_save_dir,args.console_output_dir]:
        dir_path = os.path.join(directory, args.dataset) if args.dataset else directory
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    print(args)
    return args