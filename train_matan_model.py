import torch
from torch import nn
from ordinaloss.trainers.trainers import SingleGPUTrainerMatan
from ordinaloss.utils.pretrained_models import classification_model_vgg
from ordinaloss.utils.data_utils import create_dsets_melanoma, load_single_gpu
from torch.optim import SGD, Adam, RMSprop
from ordinaloss.utils.loss_utils import CostSensitiveLoss
import argparse
import numpy as np
import mlflow
import pandas as pd
from ordinaloss.utils.basic_utils import create_mock_dsets, create_mock_model
import random
print(f"loaded {__name__}")


EXPERIMENT_NAME = "Maruloss_two_factor"

class Defaults:
    N_EPOCHS = 16
    BATCH_SIZE = 32
    DEVICE_ID = 0
    MIN_DELTA = 1.0
    LR = 1.0e-3
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5.0e-2
    IS_MOCK
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



def run_experiment(
    n_epochs:int = Defaults.N_EPOCHS,
    batch_size:int= Defaults.N_EPOCHS,
    device_id:int= Defaults.N_EPOCHS,
    optim:str= Defaults.N_EPOCHS,
    data_path: str= Defaults.N_EPOCHS,
    model_architecture:str= Defaults.N_EPOCHS,
    lamda:float= Defaults.N_EPOCHS,
    gama:float= Defaults.N_EPOCHS,
    research_type:str= Defaults.N_EPOCHS,
    is_mock:int= Defaults.N_EPOCHS,
    momentum:float= Defaults.N_EPOCHS,
    patience:int= Defaults.N_EPOCHS,
    min_delta:float= Defaults.N_EPOCHS,
    sch_gamma:float= Defaults.N_EPOCHS,
    sch_step_size:int= Defaults.N_EPOCHS,
    weight_decay:float= Defaults.N_EPOCHS,
    lr:float= Defaults.N_EPOCHS,
    opt_metric:str= Defaults.N_EPOCHS
    ):
    """
    One funciton to run an experiment. 
    Including logging, setting deterministic, parameters and metrics.
    """
    args = {
             'n_epochs': n_epochs, 
             'batch_size': batch_size,
             'optim':optim,
             'data_path': data_path,
             'model_architecture':model_architecture,
             'lamda':lamda,
             'gama':gama,
             'research_type':research_type,
             'is_mock':is_mock,
             'momentum':momentum,
             'patience':patience,
             'min_delta':min_delta,
             'sch_gamma':sch_gamma,
             'sch_step_size':sch_step_size,
             'weight_decay':weight_decay,
             'lr':lr,
             'opt_metric':opt_metric}
    
    print(args)
    return 

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.seed_all()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mlflow.end_run()
    
    existing_exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not existing_exp:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        #Logging params
        mlflow.log_params(args)

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
            
        else:
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
            opt_metric = opt_metric
            )

        #Setting the loss
        loss_fn = CostSensitiveLoss(weight= 10000, cost_matrix = cost_matrix)
        trainer.set_loss_fn(loss_fn)

        # Start training
        trainer.train_until_converge(
            n_epochs=n_epochs, 
            patience=patience, 
            min_delta=min_delta, 
            sch_gamma=sch_gamma, 
            sch_stepsize=sch_step_size)

        trainer._eval_epoch("test")
  
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

    args = vars(parser.parse_args())
    run_experiment(**args)



