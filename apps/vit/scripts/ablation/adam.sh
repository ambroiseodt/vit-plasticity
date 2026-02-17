#!/usr/bin/bash

# Launch finetuning runs with Adam using apps/vit/train.py. It will create a dedicated
# tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/adam.sh
# ```

# Define components to freeze
declare -a comps=("components"=[]
                 "components=["attn_norm","mha","ffn_norm","ffn_fc1","ffn_fc2"]"
                 "components=["emb","mha","ffn_norm","ffn_fc1","ffn_fc2"]"
                 "components=["emb","attn_norm","ffn_norm","ffn_fc1","ffn_fc2"]"
                 "components=["emb","attn_norm","mha","ffn_fc1","ffn_fc2"]"
                 "components=["emb","attn_norm","mha","ffn_norm","ffn_fc2"]"
                 "components=["emb","attn_norm","mha","ffn_norm","ffn_fc1"]"
                )

# Optimization setup
optimizer="adamw"
momentum=0.0

# CIFAR100
dataset_name="cifar100"
session="rebuttal_adam_${dataset_name}"
device="cuda:0"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 \
    42 \
    3407
do
    for lr in \
        1e-3 \
        3e-3 \
        1e-2 \
        3e-2
    do
        for i in "${!comps[@]}"
        do
            # Skip the emb finetuning
            if [[ "$i" == 1 ]]; then
                continue
            fi
            # Rescale lr following Kumar et al., 2023
            adam_lr=$(awk "BEGIN {printf \"%.2e\", $lr / 100}")
            log_dir="rebuttal/vit_${dataset_name}_adamw_seed_${seed}_lr_${adam_lr}_comp_${i}"
            run="log_dir=${log_dir} optimizer=${optimizer} momentum=${momentum} seed=${seed} lr=${adam_lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done

# CIFAR10-C
dataset_name="cifar10_c"
corruption="gaussian_noise"
severity=5
session="rebuttal_adam_${dataset_name}_${corruption}"
device="cuda:1"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 \
    42 \
    3407
do
    for lr in \
        1e-3 \
        3e-3 \
        1e-2 \
        3e-2
    do
        for i in "${!comps[@]}"
        do
            # Skip the emb finetuning
            if [[ "$i" == 1 ]]; then
                continue
            fi
            # Rescale lr following Kumar et al., 2023
            adam_lr=$(awk "BEGIN {printf \"%.2e\", $lr / 100}")
            log_dir="rebuttal/vit_${dataset_name}_${corruption}_${severity}_seed_${seed}_lr_${adam_lr}_comp_${i}"
            run="log_dir=${log_dir} optimizer=${optimizer} momentum=${momentum} dataset_name=${dataset_name}-corruption-${corruption}-severity-${severity} seed=${seed} lr=${adam_lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done

# DOMAINNET
dataset_name="domainnet"
domain="clipart"
session="rebuttal_adam_${dataset_name}_${domain}"
device="cuda:2"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 \
    42 \
    3407
do
    for lr in \
        3e-3 \
        1e-2 \
        3e-2 \
        6e-2
    do
        for i in "${!comps[@]}"
        do
            # Skip the emb finetuning
            if [[ "$i" == 1 ]]; then
                continue
            fi
            # Rescale lr following Kumar et al., 2023
            adam_lr=$(awk "BEGIN {printf \"%.2e\", $lr / 100}")
            log_dir="rebuttal/vit_${dataset_name}_${domain}_seed_${seed}_lr_${adam_lr}_comp_${i}"
            run="log_dir=${log_dir} optimizer=${optimizer} momentum=${momentum} dataset_name=${dataset_name}-${domain} seed=${seed} lr=${adam_lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done

# DOMAINNET
dataset_name="domainnet"
domain="sketch"
session="rebuttal_adam_${dataset_name}_${domain}"
device="cuda:3"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 \
    42 \
    3407
do
    for lr in \
        3e-3 \
        1e-2 \
        3e-2 \
        6e-2
    do
        for i in "${!comps[@]}"
        do
            # Skip the emb finetuning
            if [[ "$i" == 1 ]]; then
                continue
            fi
            # Rescale lr following Kumar et al., 2023
            adam_lr=$(awk "BEGIN {printf \"%.2e\", $lr / 100}")
            log_dir="rebuttal/vit_${dataset_name}_${domain}_seed_${seed}_lr_${adam_lr}_comp_${i}"
            run="log_dir=${log_dir} optimizer=${optimizer} momentum=${momentum} dataset_name=${dataset_name}-${domain} seed=${seed} lr=${adam_lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done