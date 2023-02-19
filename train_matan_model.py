import torch
from torch import nn
from ordinaloss.trainers.trainers import SingleGPUTrainer, SingleGPUTrainerMatan
from ordinaloss.utils.pretrained_models import classification_model_vgg, DummyModel,classification_model_resnet
from ordinaloss.utils.data_utils import create_dsets_melanoma, load_multi_gpu, load_single_gpu
from torch.optim import SGD, Adam, RMSprop
from torch.distributed import destroy_process_group, init_process_group
from ordinaloss.utils.basic_utils import satisfy_constraints, modify_lambdas,get_only_metrics
from ordinaloss.utils.loss_utils import CostSensitiveLoss
import os
from ordinaloss.utils.loss_utils import create_ordinal_cost_matrix
import argparse
import torch.multiprocessing as mp
import numpy as np
import sys
import mlflow
import pandas as pd
from ordinaloss.utils.basic_utils import create_mock_dsets, create_mock_model
from pathlib import Path

from  mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "Maruloss_two_factor"

def run_experiment(
    n_epochs:int,
    n_procs:int,
    batch_size:int,
    device_id:int,
    optim:str,
    data_path: str,
    model_architecture:str,
    lamda:float,
    gama:float,
    research_type:str,
    is_mock:int,
    momentum:float,
    patience:int,
    min_delta:float,
    sch_gamma:float,
    sch_step_size:int,
    weight_decay:float,
    lr:float,
    run_id:str,
    opt_metric:str
    ):

    mlflow.end_run()

    
    existing_exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not existing_exp:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        #Logging params
        mlflow.log_params(args)
        # mlflow.log_params({'TEST':1})

        cost_matrix_mapper = {
            "ce": np.array([[1,1],[1,1]]),
            "wce": np.array([[lamda,lamda],[1-lamda, 1-lamda]]),
            "csce": np.array([[lamda,1- lamda],[1- lamda, lamda]]),
            "csce_2": np.array([[lamda,1- lamda],[1- gama, gama]]),
                            }
        
        cost_matrix = cost_matrix_mapper[research_type]
        
        #Setting the datasets.
        if is_mock:
            print("====RUNNING MOCK VERSION=====")
            model = create_mock_model(num_classes=2)
            dsets = create_mock_dsets(num_classes=2)
        
        if 'resnet' in model_architecture:
            print('res')
            model = classification_model_resnet(model_architecture, num_classes=2)
            dsets = create_dsets_melanoma(data_path) 
                      
        if 'vgg' in model_architecture:
            print('vgg')
            model = classification_model_vgg(model_architecture, num_classes=2)
            dsets = create_dsets_melanoma(data_path)

        loaders = load_single_gpu(dsets, batch_size)

        #Setting the optimizer
        if optim=="Adam":
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim=="SGD":
            optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        elif optim=="RMSProp":
            optimizer = RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)


        #Setting the trainer
        trainer = SingleGPUTrainerMatan(
            model=model, 
            loaders=loaders,
            optimizer=optimizer, 
            gpu_id=device_id, 
            num_classes=2, 
            save_every=2,
            opt_metric = opt_metric
            )

        #Setting the loss
        loss_fn = CostSensitiveLoss(weight= 10000, cost_matrix = cost_matrix)
        trainer.set_loss_fn(loss_fn)

        # Start training
        train_results,val_results,test_results = trainer.train_until_converge(
                                                            n_epochs=n_epochs, 
                                                            patience=patience, 
                                                            min_delta=min_delta, 
                                                            sch_gamma=sch_gamma, 
                                                            sch_stepsize=sch_step_size)
        
        # trainer.train(
        #                                                     n_epochs=n_epochs, 
        #                                                     patience=patience, 
        #                                                     min_delta=min_delta, 
        #                                                     sch_gamma=sch_gamma, 
        #                                                     sch_stepsize=sch_step_size)
        
        #Save results:
        
        curr_exp_path = Path("results",opt_metric, str(run_id))
        
        if not os.path.exists(curr_exp_path):
            os.makedirs(curr_exp_path)
            
        if research_type =='csce':
            pd.DataFrame ({'test_y_pred' : test_results['test_y_pred'],
                       'test_y_true': test_results['test_y_true'] }).to_csv(str(curr_exp_path)+f'/test_predict_{lamda}.csv',index=False)
            
        elif research_type =='csce_2':
            pd.DataFrame ({'test_y_pred' : test_results['test_y_pred'],
                       'test_y_true': test_results['test_y_true'] }).to_csv(str(curr_exp_path)+f'/test_predict_{lamda}_{gama}.csv',index=False)
        else:
            pd.DataFrame ({'test_y_pred' : test_results['test_y_pred'],
                       'test_y_true': test_results['test_y_true'] }).to_csv(str(curr_exp_path)+f'/test_predict_{research_type}.csv',index=False)
        
        print('prediciton saved ')

        train_results['phase'] = 'train'
        val_results['phase'] = 'val'
        test_results['phase'] = 'test'

        train_results['train_lamda'] = lamda
        val_results['val_lamda'] = lamda
        test_results['test_lamda'] = lamda
        
        ds = [train_results, val_results,test_results]
        d = {}
        for k in train_results.keys():
            for phas in ['test','val','test']:
                if k =='phase':
                    d['phase'] = tuple(d['phase'] for d in ds)

                else:
                    new_key = '_'+'_'.join(k.split('_')[1:])
                    d[new_key] = tuple(d[d['phase']+new_key] for d in ds)
                    
        if research_type =='csce':
            pd.DataFrame(d).to_csv(str(curr_exp_path)+f'/metrics_{lamda}.csv',index=False)
        elif research_type =='csce_2':
            pd.DataFrame(d).to_csv(str(curr_exp_path)+f'/metrics_{lamda}_{gama}.csv',index=False)
        else:
            pd.DataFrame(d).to_csv(str(curr_exp_path)+f'/metrics_{research_type}.csv',index=False)

        print('metrics results saved')
    
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    
    parser.add_argument('--n_epochs', default=16, type=int, help='Total maximum epochs to train the model')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device')
    parser.add_argument('--n_procs', default=1, type=int, help="Total number of processes (GPUs used)")
    parser.add_argument('--device_id', default=0, type=int, help="The device, used only if n_procs is 1")
    parser.add_argument('--min_delta', default=1.0, type=float, help="The minimum delta for early stopping")
    parser.add_argument('--lr', default=1.0e-3, type=float, help="ordindary learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="SGD momentum")
    parser.add_argument('--weight_decay', default=5.0e-2, type=float, help="weight decay")
    parser.add_argument('--is_mock', default=0, type=int, help="Use 1 to dry run on random data")
    parser.add_argument('--optim', default="SGD", type=str, help="What is the optimizer")
    parser.add_argument('--sch_step_size', default=5, type=int, help="After how many epochs do we change lr?")
    parser.add_argument('--sch_gamma', default=0.9, type=float, help="What is the gamma to change it?")
    parser.add_argument("--data_path", default="../datasets/DermMel/", type=str, help="asdasd")
    parser.add_argument("--model_architecture", default="vgg19", type=str, help="One of vgg19 or vgg16")
    parser.add_argument("--lamda", default=0.5, type=float, help= "asdasd")
    parser.add_argument("--gama", default=0.5, type=float, help= "asdasd")
    parser.add_argument("--research_type", default="ce", type=str, help= "One of {ce, wce, csce}")
    parser.add_argument('--patience', default=5, type=int, help="The patience of the model for raise in loss")
    parser.add_argument('--opt_metric', default='val_loss', type=str, help="loss to opt the model by val_..")
    parser.add_argument('--run_id', default=1, type=str, help="run id to save results")

    args = vars(parser.parse_args())
    run_experiment(**args)



