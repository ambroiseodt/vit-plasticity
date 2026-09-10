#!/usr/bin/bash

# Launch finetuning with ViT-Large and ViT-Huge. It will create a dedicated
# tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/model_size.sh
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

# DOMAINNET
dataset_name="domainnet"
domain="clipart"

# LARGE
model_name="large"
device="cuda:2"
batch_size=128
val_batch_size=128
grad_acc_steps=4

# Runs
session="ablation_${model_name}_${dataset_name}_${domain}"
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
            log_dir="ablation/vit_${dataset_name}_${domain}_${model_name}_seed_${seed}_lr_${lr}_comp_${i}"
            ablation="model_name=${model_name} batch_size=${batch_size} val_batch_size=${val_batch_size} grad_acc_steps=${grad_acc_steps}"
            run="log_dir=${log_dir} ${ablation} dataset_name=${dataset_name}-${domain} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done


# HUGE
model_name="huge"
patch_size=14
device="cuda:3"
batch_size=64
val_batch_size=64
grad_acc_steps=8

# Runs
session="ablation_${model_name}_${dataset_name}_${domain}"
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
            log_dir="ablation/vit_${dataset_name}_${domain}_${model_name}_seed_${seed}_lr_${lr}_comp_${i}"
            ablation="model_name=${model_name} patch_size=${patch_size} batch_size=${batch_size} val_batch_size=${val_batch_size} grad_acc_steps=${grad_acc_steps}"
            run="log_dir=${log_dir} ${ablation} dataset_name=${dataset_name}-${domain} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done
