import torch
from torch import nn
from ordinaloss.trainers.trainers import SingleGPUTrainer, SingleGPUTrainerMatan
from ordinaloss.utils.pretrained_models import classification_model_vgg, DummyModel
from ordinaloss.utils.data_utils import create_datasets, load_multi_gpu, load_single_gpu
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

from ordinaloss.utils.basic_utils import create_mock_dsets, create_mock_model
from ordinaloss.utils.data_utils import data_load

from  mlflow.tracking import MlflowClient

def run_experiment(
    n_epochs:int,
    n_procs:int,
    batch_size:int,
    device_id:int,
    optim:str,
    data_path: str,
    model_architecture:str,
    lamda:float,
    research_type:str,
    is_mock:int,
    momentum:float,
    patience:int,
    min_delta:float,
    sch_gamma:float,
    sch_step_size:int,
    weight_decay:float,
    lr:float
    ):

    mlflow.end_run()
    mlflow.create_experiment("Maruloss")
    mlflow.set_experiment("Maruloss")

    with mlflow.start_run():
        mlflow.log_params(args)

        cost_matrix_mapper = {
            "ce": np.array([[1,1],[1,1]]),
            "wce": np.array([[lamda,lamda],[1-lamda, 1-lamda]]),
            "csce": np.array([[lamda,1- lamda],[1- lamda, lamda]]),
                        }
        
        cost_matrix = cost_matrix_mapper[research_type]

        if is_mock:
            print("====RUNNING MOCK VERSION=====")
            model = create_mock_model(num_classes=2)
            dsets = create_mock_dsets(num_classes=2)
            loaders = load_single_gpu(dsets, batch_size)

        else:
            model = classification_model_vgg(model_architecture, num_classes=2)
            #dsets = create_datasets(data_path)
            loaders = data_load(data_path, batch_size, db="Melanoma")

        if optim=="Adam":
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim=="SGD":
            optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        elif optim=="RMSProp":
            optimizer = RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

        trainer = SingleGPUTrainerMatan(
            model=model, 
            loaders=loaders,
            optimizer=optimizer, 
            gpu_id=device_id, 
            num_classes=2, 
            save_every=2
            )

        loss_fn = CostSensitiveLoss(weight= 10000, cost_matrix = cost_matrix)
        trainer.set_loss_fn(loss_fn)
        trainer.train_until_converge(
            n_epochs=n_epochs, 
            patience=patience, 
            min_delta=min_delta, 
            sch_gamma=sch_gamma, 
            sch_stepsize=sch_step_size)

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
    parser.add_argument("--research_type", default="ce", type=str, help= "One of {ce, wce, csce}")
    parser.add_argument('--patience', default=16, type=int, help="The patience of the model for raise in loss")

    args = vars(parser.parse_args())
    run_experiment(**args)



