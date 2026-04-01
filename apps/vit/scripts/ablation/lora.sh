#!/usr/bin/bash

# Launch finetuning runs using apps/vit/train.py. It will create a dedicated
# tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/lora.sh
# ```

# Define components to freeze
comps="components=["emb","attn_norm","ffn_norm","ffn_fc1","ffn_fc2"]"

# Optimization setup
optimizer="adamw"
momentum=0.0

# CIFAR100
session="fin_lora_cifar100"
dataset_name="cifar100"
device="cuda:5"

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
        run="log_dir=${log_dir} optimizer=${optimizer} momentum=${momentum} seed=${seed} lr=${adam_lr} ${comps} device=${device}"
        command="python -m apps.vit.train_lora config=apps/vit/configs/${dataset_name}.yaml ${run}"
        echo "Running command: ${command}"
        tmux send-keys -t ${session} "${command}" C-m
    done
done

# DOMAINNET
session="fin_lora_domainnet_clipart"
dataset_name="domainnet"
domain="clipart"
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
        run="log_dir=${log_dir} dataset_name=${dataset_name}-${domain} seed=${seed} lr=${adam_lr} ${comps} device=${device}"
        command="python -m apps.vit.train_lora config=apps/vit/configs/${dataset_name}.yaml ${run}"
        echo "Running command: ${command}"
        tmux send-keys -t ${session} "${command}" C-m
    done
done
