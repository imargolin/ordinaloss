import subprocess
import numpy as np
import os 
import time

start_time = time.time()
exp_range = list(np.linspace(0,1,10))
count_majorty = len(os.listdir('../datasets/DermMel/train/0'))
count_mainor = len(os.listdir('../datasets/DermMel/train/1'))
exp_range.append(count_mainor / count_majorty)

for lamda in exp_range:
    bashCommand = f"python train_matan_model.py --research_type csce --n_epochs 16 --n_procs 1 --batch_size 32 --device_id 0 --optim SGD --data_path ../datasets/DermMel/ --model_architecture vgg19 --lamda {lamda} --is_mock 0 --momentum 0.9 --patience 16 --min_delta 1.0 --sch_gamma 0.9 --sch_step_size 5 --weight_decay 5.0e-2 --lr 1.0e-3 --run_id 1"
    print(bashCommand)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    
bashCommand = f"python train_matan_model.py --research_type wce --n_epochs 16 --n_procs 1 --batch_size 32 --device_id 0 --optim SGD --data_path ../datasets/DermMel/ --model_architecture vgg19 --lamda {count_mainor / count_majorty} --is_mock 0 --momentum 0.9 --patience 16 --min_delta 1.0 --sch_gamma 0.9 --sch_step_size 5 --weight_decay 5.0e-2 --lr 1.0e-3 --run_id 1"
print(bashCommand)
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

bashCommand = f"python train_matan_model.py --research_type ce --n_epochs 16 --n_procs 1 --batch_size 32 --device_id 0 --optim SGD --data_path ../datasets/DermMel/ --model_architecture vgg19 --lamda {lamda} --is_mock 0 --momentum 0.9 --patience 16 --min_delta 1.0 --sch_gamma 0.9 --sch_step_size 5 --weight_decay 5.0e-2 --lr 1.0e-3 --run_id 1"
print(bashCommand)
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

print("--- %s seconds ---" % (time.time() - start_time))
