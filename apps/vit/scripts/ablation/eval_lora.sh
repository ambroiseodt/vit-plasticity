#!/usr/bin/bash

# Launch evaluation using apps/vit/eval.py. It will create a dedicated
# tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/eval_lora.sh
# ```

# CIFAR100
session="eval_lora_cifar100"
dataset_name="cifar100"
device="cuda:2"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 
do
    for lr in \
        1e-1 \
        3e-1 \
        6e-1
    do
        # Rescale lr following Kumar et al., 2023
        adam_lr=$(awk "BEGIN {printf \"%.2e\", $lr / 100}")
        log_dir="ablation/vit_${dataset_name}_lora_seed_${seed}_lr_${adam_lr}_comp_mha"
        run="log_dir=${log_dir} device=${device}"
        command="python -m apps.vit.eval_lora config=apps/vit/configs/eval.yaml ${run}"
        echo "Running command: ${command}"
        tmux send-keys -t ${session} "${command}" C-m    
    done
done


# DOMAINNET
dataset_name="domainnet"
domain="clipart"
session="eval_lora_${dataset_name}_${domain}"
device="cuda:2"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 
do
    for lr in \
        3e-0 \
        6e-0 \
        1e1 
    do
        # Rescale lr following Kumar et al., 2023
        adam_lr=$(awk "BEGIN {printf \"%.2e\", $lr / 100}")
        log_dir="ablation/vit_${dataset_name}_${domain}_lora_seed_${seed}_lr_${adam_lr}_comp_mha"
        run="log_dir=${log_dir} device=${device}"
        command="python -m apps.vit.eval_lora config=apps/vit/configs/eval.yaml ${run}"
        echo "Running command: ${command}"
        tmux send-keys -t ${session} "${command}" C-m  
    done
done