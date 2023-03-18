import os
import sys
import time
import argparse
import copy
import configparser
import numpy as np
from tqdm import tqdm

# Torch Library.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Relative package.
sys.path.append('../../code/')
from utils.dataloader import DatasetLoader
from utils.metrics import RegressionLoss, RegressionMetrics
from utils.metrics import ClassificationLoss, ClassificationMetrics
from utils.utils import compute_sampling_threshold, print_model_parameters
from model.AGLSTAN import AGLSTAN


# Testing print info.
from icecream import ic

# two city.
# chi : Chicago
# ny : New York
CITY = 'chi'
# CITY = 'ny'

############################InitSeed###################################
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

############################Arguments###################################
# Read configure file
config = configparser.ConfigParser()
config.read(f'../data/{CITY}/config.conf')

# set arguments
args = argparse.ArgumentParser(description='arguments')

# data
args.add_argument('--adj_filename', default=config['data']['adj_filename'], type=str)
args.add_argument('--node_features_filename', default=config['data']['node_features_filename'], type=str)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--window', default=config['data']['window'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=str)
# model
args.add_argument('--model_name', default=config['model']['model_name'], type=str)
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--cheb_k', default=config['model']['cheb_k'], type=int)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--filter_type', default=config['model']['filter_type'], type=str)
args.add_argument('--activation_func', default=config['model']['activation_func'], type=str)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--filter_size', default=config['model']['filter_size'], type=int)
# train
args.add_argument('--epoch', default=config['train']['epoch'], type=int)
args.add_argument('--lr', default=config['train']['lr'], type=float)
args.add_argument('--factor', default=config['train']['factor'], type=float)
args.add_argument('--patience', default=config['train']['patience'], type=float)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=int)
args.add_argument('--train_loss_filename', default=config['train']['train_loss_filename'], type=str)
args.add_argument('--val_loss_filename', default=config['train']['val_loss_filename'], type=str)
args.add_argument('--binary', default=config['train']['binary'], type=str)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--loss_function', default=config['train']['loss_function'], type=str)
args.add_argument('--cf_decay_steps', default=config['train']['cf_decay_steps'], type=int)
args.add_argument('--teacher_forcing', default=config['train']['teacher_forcing'], type=str)
# other
args.add_argument('--device', default=config['other']['device'], type=str)
args.add_argument('--data_path', default=config['other']['data_path'], type=str)
args.add_argument('--res_path', default=config['other']['res_path'], type=str)
args = args.parse_args()

##############################DataLoder#################################
loader = DatasetLoader(args)
data, adj, scaler, pos_weight, threshold = loader.get_dataset()
train_loader, val_loader, test_loaders = data
# with pre-defined or not
if args.default_graph == 'true':
    adj = None

##############################ModelLoder#################################
device = torch.device(args.device)
model = AGLSTAN(args)
model.to(device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)

print("#" * 40 + "Model Info" + "#" * 40)
print_model_parameters(model, only_num=False)

##############################Optm&Loss#################################
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
lr_decay_steps = [15, 25, 35, 55, 75]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=0.5)
if args.binary == 'false':
    criterion = RegressionLoss(scaler, mask_value=args.mask_value, loss_type=args.loss_function)
    metrics = RegressionMetrics(scaler, mask_value=args.mask_value)
else:
    criterion = ClassificationLoss(pos_weight=pos_weight.to(device), loss_type=args.loss_function, lambda_value=0.1 , device=device)
    # criterion = ClassificationLoss(loss_type=args.loss_function, lambda_value=0.5, device=device)
    metrics = ClassificationMetrics(threshold)

##############################Training#################################
print("#" * 40 + "Training" + "#" * 40)
init_time = time.time()

train_loss_list = []
val_loss_list = []
not_improved = 0

for epoch in range(args.epoch):
    # for training
    model.train()
    train_loss = 0

    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if args.teacher_forcing == 'true':
            global_step = epoch * len(train_loader) + idx
            teacher_forcing_ratio = compute_sampling_threshold(global_step, args.cf_decay_steps)
        else:
            teacher_forcing_ratio = 1.

        preds = model(inputs, labels, teacher_forcing_ratio) # teacher forcing 
        loss = criterion(preds, labels)
        train_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    train_loss = train_loss / len(train_loader)
    train_loss_list.append(train_loss)

    model.eval()
    val_loss = 0
    val_metrics = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs, labels, teacher_forcing_ratio=0.) # not teacher forcing 

            val_loss += criterion(preds, labels)
            val_metrics.append(metrics(preds, labels))

        val_loss = val_loss / len(val_loader)
        val_loss_list.append(val_loss)

    ic(optimizer.param_groups[0]['lr'])
    
    # for regression task.
    # scheduler.step(val_loss)
    
    # for binary classification task.
    scheduler.step()
    # scheduler.step(val_loss)

    # print the training and validation info for each epoch.
    if args.binary == 'false':
        val_rmse = torch.mean(torch.stack(list(map(lambda x : x[0], val_metrics))))
        val_mape = torch.mean(torch.stack(list(map(lambda x : x[1], val_metrics))))

        print("epoch : {}, train_{}: {:.4f}, val_{}: {:.4f}, val_rmse: {:.4f}, val_mape: {:.4f}, duration: {:.2f}s".format(
            epoch,
            args.loss_function,
            train_loss.cpu().item(),
            args.loss_function,
            val_loss.cpu().item(),
            val_rmse.cpu().item(),
            val_mape.cpu().item(),        
            time.time() - init_time
        ))
    else:
        val_micro_f1 = np.mean(list(map(lambda x : x[0], val_metrics)))
        val_macro_f1 = np.mean(list(map(lambda x : x[1], val_metrics)))
        val_F1_lst = np.mean(list(map(lambda x : x[2], val_metrics)), axis=0)
        
        print("epoch : {}, train_{}: {:.4f}, val_{}: {:.4f}, val_micro_f1: {:.4f}, val_macro_f1: {:.4f}, val_F1_lst: {}, duration: {:.2f}s".format(
            epoch,
            args.loss_function,
            train_loss.cpu().item(),
            args.loss_function,
            val_loss.cpu().item(),
            val_micro_f1,
            val_macro_f1,
            np.round(val_F1_lst, 4),        
            time.time() - init_time
        ))
    # end print.

    # early stop condition and saving the best model 
    if min(val_loss_list) == val_loss_list[-1]:
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': args
        }
        torch.save(state, os.path.join(args.res_path, args.model_name))
        not_imporved = 0
        print("The Best Model has been saved!")

    elif not_imporved == args.early_stop:
        print("Validation performance didn\'t improve for {} epochs. Training stops.".format(args.early_stop))
        print("The best val_loss : {:.4f}".format(min(val_loss_list)))
        break
    
    else:
        not_imporved += 1        

##############################Testing#################################
print("#" * 40 + "Testing" + "#" * 40)
best_model = torch.load(os.path.join(args.res_path, args.model_name))
state_dict = best_model['state_dict']
args = best_model['config']
model.load_state_dict(state_dict)
model.to(args.device)
model.eval()

test_dataset_name = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for month, test_loader in enumerate(test_loaders):
    test_loss = 0
    test_metrics = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds = model(inputs, labels, teacher_forcing_ratio=0.) # not teacher forcing 
                
            test_loss += criterion(preds, labels)
            test_metrics.append(metrics(preds, labels))
            
        test_loss = test_loss / len(test_loader)

    print("Test on {} dataset".format(test_dataset_name[month]))
    if args.binary == 'false':
        test_rmse = torch.mean(torch.stack(list(map(lambda x : x[0], test_metrics))))
        test_mape = torch.mean(torch.stack(list(map(lambda x : x[1], test_metrics))))

        print("test_{}: {:.4f}, test_rmse: {:.4f}, test_mape: {:.4f}".format(
            args.loss_function,
            test_loss.cpu().item(),
            test_rmse.cpu().item(),
            test_mape.cpu().item()
        ))
    else:
        test_micro_f1 = np.mean(list(map(lambda x : x[0], test_metrics)))
        test_macro_f1 = np.mean(list(map(lambda x : x[1], test_metrics)))
        test_F1_lst = np.mean(list(map(lambda x : x[2], test_metrics)), axis=0)
        
        print("test_{}: {:.4f}, test_micro_f1: {:.4f}, test_macro_f1: {:.4f}, test_F1_lst: {}".format(
            args.loss_function,
            train_loss.cpu().item(),
            test_micro_f1,
            test_macro_f1,
            np.round(test_F1_lst, 4)
        ))

############################Save Results###############################
# save training and validation loss
np.save(os.path.join(args.res_path, args.train_loss_filename), np.array([each.cpu().detach().numpy() for each in train_loss_list]))
np.save(os.path.join(args.res_path, args.val_loss_filename), np.array([each.cpu().detach().numpy() for each in val_loss_list]))

# save the trained node embedding.
params = dict()
for name, param in list(model.named_parameters()):
    params[name] = param
node_embeddings = params['node_embeddings']
supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
support_set = [torch.eye(args.num_nodes).to(supports.device), supports]
# supports = torch.stack(support_set, dim=0)
np.save(os.path.join(args.res_path, 'adj.npy'), supports.cpu().detach().numpy())