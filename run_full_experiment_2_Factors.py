import subprocess
import numpy as np
import os 
import time
import itertools
start_time = time.time()
exp_range = [0.9,0.8,0.85,0.83,0.87]
# exp_range = list(np.linspace(0.75,0.95,20).round(2))
# exp_range_2 = list(itertools.product(exp_range, exp_range))
count_majorty = len(os.listdir('../datasets/DermMel/train/0'))
count_mainor = len(os.listdir('../datasets/DermMel/train/1'))
exp_range.append(count_mainor / count_majorty)

# opt_metric = 'loss_val'
run_id = 'lamda_gama_fix_gama_04'
gama = 0.1
# for opt_metric in ["val_loss"``
#                 ,"val_accuracy"
#                 ,"val_f1_score"
#                 ,"val_precision_score"
#                 ,"val_recall_score"
#                 ,"val_roc_auc_score"]:

for opt_metric in ["val_f1_score",'val_loss']:

    # for lamda,gama in exp_range_2:
    for lamda in exp_range:
        bashCommand = f"python train_matan_model.py --research_type csce_2 --n_epochs 20 --n_procs 1 --batch_size 32 --device_id 0 --optim SGD --data_path ../datasets/DermMel/ --model_architecture vgg19 --lamda {lamda} --gama {gama} --is_mock 0 --momentum 0.9 --patience 5 --min_delta 1.0 --sch_gamma 1.0 --sch_step_size 5 --weight_decay 5.0e-2 --lr 1.0e-3 --opt_metric {opt_metric} --run_id {run_id}"
        print(bashCommand)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        
        
    bashCommand = f"python train_matan_model.py --research_type wce --n_epochs 20 --n_procs 1 --batch_size 32 --device_id 0 --optim SGD --data_path ../datasets/DermMel/ --model_architecture vgg19 --lamda {count_mainor / count_majorty} --is_mock 0 --momentum 0.9 --patience 5 --min_delta 1.0 --sch_gamma 1.0 --sch_step_size 5 --weight_decay 5.0e-2 --lr 1.0e-4 --opt_metric {opt_metric} --run_id {run_id}"
    print(bashCommand)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    bashCommand = f"python train_matan_model.py --research_type ce --n_epochs 20 --n_procs 1 --batch_size 32 --device_id 0 --optim SGD --data_path ../datasets/DermMel/ --model_architecture vgg19 --lamda {lamda} --is_mock 0 --momentum 0.9 --patience 5 --min_delta 1.0 --sch_gamma 1.0 --sch_step_size 5 --weight_decay 5.0e-2 --lr 1.0e-4 --opt_metric {opt_metric} --run_id {run_id}"
    print(bashCommand)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

print("--- %s seconds ---" % (time.time() - start_time))
