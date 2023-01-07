import torch
from ordinaloss_distributed.engine.trainer import Trainer
from ordinaloss.utils.pretrained_models import classification_model_vgg, DummyModel
from ordinaloss_distributed.utils import data_utils
from torch.optim import SGD
from torch.distributed import destroy_process_group, init_process_group

from ordinaloss.utils.loss_utils import CSCELoss
import os
from ordinaloss.utils.loss_utils import create_ordinal_cost_matrix
import argparse
import torch.multiprocessing as mp
import sys


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(device:int, world_size:int, n_epochs:int, batch_size:int):
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
    dsets = data_utils.create_datasets("../datasets/kneeKL224/")
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loaders = data_utils.prepare_data_loader(dsets, batch_size)

    cost_matrix = create_ordinal_cost_matrix(5, cost_distance=COST_DISTANCE, diagonal_value=DIAGONAL_VALUE)
    csce_loss = CSCELoss(cost_matrix)
    
    trainer = Trainer(model, train_data=loaders["train"], validation_data=loaders["val"], optimizer = optimizer, gpu_id=device, save_every=3)
    trainer.set_loss_fn(csce_loss) #The loss is assigned again

    trainer._train_epoch(epoch=0)
    print(trainer._eval_epoch())
    destroy_process_group()

if __name__ == "__main__":
    #print(torch.cuda.device_count())
    
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=20, type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    #world_size = 3

    mp.spawn(main, args=(world_size, args.total_epochs, args.batch_size), nprocs=world_size)    
