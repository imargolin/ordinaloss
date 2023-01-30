import torch
from torch import nn
#from ordinaloss_distributed.engine.trainer import Trainer
from ordinaloss.trainers.trainers import SingleGPUTrainer, MultiGPUTrainer
from ordinaloss.utils.pretrained_models import classification_model_vgg, DummyModel
from ordinaloss.utils.data_utils import create_dsets_knee, load_multi_gpu, load_single_gpu
from ordinaloss.utils.basic_utils import create_mock_dsets, create_mock_model
from torch.optim import SGD, Adam, RMSprop
from torch.distributed import destroy_process_group, init_process_group
from ordinaloss.utils.basic_utils import satisfy_constraints, modify_lambdas,get_only_metrics
from ordinaloss.utils.loss_utils import CSCELoss, PredictionLoss, CombinedLoss, SinimLoss
import os
from ordinaloss.utils.loss_utils import create_ordinal_cost_matrix
import argparse
import torch.multiprocessing as mp
import numpy as np
import sys
import mlflow
import json

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def train_single_gpu(
                    device_id:int, 
                    batch_size:int, 
                    lr:float, 
                    momentum:float,
                    meta_lr:float, 
                    weight_decay:float, 
                    cost_distance:float, 
                    diagonal_value:float, 
                    patience:int,
                    n_epochs:int,
                    min_delta:float,
                    num_classes:int,
                    save_every:int,
                    is_mock:bool,
                    model_architecture:str,
                    optim:str,
                    loss_type:str,
                    sch_gamma:float,
                    sch_step_size:int,
                    constraints_path:str,
                    **kwargs
                    ):


    if is_mock:
        print("====RUNNING MOCK VERSION=====")
        model = create_mock_model(num_classes=num_classes)
        dsets = create_mock_dsets(num_classes=num_classes)

    else:
        model = classification_model_vgg(model_architecture, num_classes=num_classes)
        dsets = create_dsets_knee("../datasets/kneeKL224/")
    
    if optim=="Adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim=="SGD":
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optim=="RMSProp":
        optimizer = RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    loaders = load_single_gpu(dsets, batch_size)

    trainer = SingleGPUTrainer(
        model, 
        loaders=loaders,
        optimizer=optimizer, 
        gpu_id=device_id, 
        save_every=save_every,
        num_classes=num_classes
        )


    print(f"Loading constraints from path: {constraints_path}")
    with open(constraints_path) as f:
        constraints = json.load(f)
        constraints = torch.tensor(constraints, device= device_id)

    print(f"Setting initial lambdas:")
    current_lambdas = torch.tensor([0.00, 0.00, 0.00, 0.00, 0.00], device= device_id)

    mlflow.end_run()
    mlflow.log_params(args)

    while True:
        mlflow.log_metrics({f"lambda_{k}": v for k,v in enumerate(current_lambdas.tolist())}, step = trainer.epochs_trained)

        if loss_type=="Sinim":
            cls_weights = np.array([[1, 3, 5, 7, 9],
                                    [3, 1, 3, 5, 7],
                                    [5, 3, 1, 3, 5],
                                    [7, 5, 3, 1, 3],
                                    [9, 7, 5, 3, 1]], dtype=np.float32)
            cls_weights = torch.tensor(cls_weights)
            data_loss = SinimLoss(cls_weights)
            
        elif loss_type =="CSCE":
            cost_matrix = create_ordinal_cost_matrix(num_classes, cost_distance=cost_distance, diagonal_value=diagonal_value, normalize=False)
            data_loss = CSCELoss(cost_matrix)

        
        prediction_loss = PredictionLoss(lambdas = current_lambdas)
        loss_fn = CombinedLoss(data_loss, prediction_loss)

        trainer.set_loss_fn(loss_fn)
        
        trainer.train_until_converge(
            n_epochs=n_epochs, 
            patience=patience, 
            min_delta=min_delta, 
            sch_gamma=sch_gamma, 
            sch_stepsize=sch_step_size)

        test_results = trainer._eval_epoch("test")

        test_distribution = torch.tensor(test_results["test_distribution"], device=device_id)
        print(f"Done training! info: {test_results}")
        
        #Logging the data of the test, obviously we can't tell what the right performances are.
        mlflow.log_metrics({f"test_dist_{k}": v for k,v in enumerate(test_distribution.tolist())}, step = trainer.epochs_trained)
        mlflow.log_metrics(get_only_metrics(test_results), step = trainer.epochs_trained)


        if not satisfy_constraints(test_distribution, constraints):
            #modify loss function
            current_lambdas = modify_lambdas(constraints, test_distribution, current_lambdas, meta_lr).to(device=device_id)

        else:
            print("Done!")
            break

def train_multi_gpu(device:int, world_size:int, n_epochs:int, batch_size:int):
    ddp_setup(device, world_size=world_size)

    print(f"Starting main on device {device}")
    META_LEARNING_RATE = 10
    LEARNING_RATE = 1.0e-3
    WEIGHT_DECAY = 5.0e-2
    COST_DISTANCE = 3
    DIAGONAL_VALUE = 20
    PATIENCE = 3
    MIN_DELTA = 0.99

    model = classification_model_vgg("vgg19", num_classes=5)
    dsets = create_datasets("../datasets/kneeKL224/")
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loaders = load_multi_gpu(dsets, batch_size)

    cost_matrix = create_ordinal_cost_matrix(5, cost_distance=COST_DISTANCE, diagonal_value=DIAGONAL_VALUE)
    csce_loss = CSCELoss(cost_matrix)
    
    trainer = MultiGPUTrainer(model, train_data=loaders["train"], validation_data=loaders["val"], optimizer = optimizer, gpu_id=device, save_every=3)
    trainer.set_loss_fn(csce_loss) #The loss is assigned again

    trainer._train_epoch(epoch=0)
    print(trainer._eval_epoch())
    destroy_process_group()
    return model

if __name__ == "__main__":
    #print(torch.cuda.device_count())
    
    parser = argparse.ArgumentParser(description='simple distributed training job')
    
    parser.add_argument('--constraints_path', type=str, help="The path for the constraints")
    parser.add_argument('--n_epochs', default=16, type=int, help='Total maximum epochs to train the model')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device')
    parser.add_argument('--n_procs', default=1, type=int, help="Total number of processes (GPUs used)")
    parser.add_argument('--device_id', default=0, type=int, help="The device, used only if n_procs is 1")
    parser.add_argument('--meta_lr', default=10, type=float, help="The meta learning rate, how does the loss change")
    parser.add_argument('--patience', default=16, type=int, help="The patience of the model for raise in loss")
    parser.add_argument('--cost_distance', default=3.0, type=float, help="The cost distance between 2 cosecutive orders")
    parser.add_argument('--diagonal_value', default=5.0, type=float, help="The diagonal value of the loss function")
    parser.add_argument('--min_delta', default=1.0, type=float, help="The minimum delta for early stopping")
    parser.add_argument('--lr', default=1.0e-3, type=float, help="ordindary learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="SGD momentum")
    parser.add_argument('--weight_decay', default=5.0e-2, type=float, help="weight decay")
    parser.add_argument('--num_classes', default=5, type=int, help="Total number of classes")
    parser.add_argument('--save_every', default=3, type=int, help="How many epochs between each save")
    parser.add_argument('--is_mock', default=0, type=int, help="Use 1 to dry run on random data")
    parser.add_argument('--model_architecture', default="vgg19", type=str, help="One of vgg19 or vgg16")
    parser.add_argument('--optim', default="SGD", type=str, help="What is the optimizer")
    parser.add_argument('--loss_type', default="CSCE", type=str, help="Can be one of CSCE or Sinim")
    parser.add_argument('--sch_step_size', default=5, type=int, help="After how many epochs do we change lr?")
    parser.add_argument('--sch_gamma', default=0.9, type=float, help="What is the gamma to change it?")
    
    args = vars(parser.parse_args()) #transform args to dictionary.
    print(args)

    torch.manual_seed(0)

    if args["n_procs"]==-1 or args["n_procs"]>1:
        print(f"Training in a distributed mode, device id is {args['device_id']}")
        if args["n_procs"]==-1:
            world_size = torch.cuda.device_count()
        else:
            world_size = args["n_procs"]

        mp.spawn(train_multi_gpu, args=(world_size, args.total_epochs, args.batch_size), nprocs=world_size)

    else:
        print(f"Training on a single mode, device id is {args['device_id']}")
        train_single_gpu(**args)