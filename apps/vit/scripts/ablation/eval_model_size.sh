#!/usr/bin/bash

# Launch evaluation using apps/vit/eval.py. It will create a dedicated
# tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/eval_model_size.sh
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

# LARGE
model_name="large"
dataset_name="domainnet"
domain="clipart"
session="eval_model_size_${model_name}_${dataset_name}_${domain}"
device="cuda:4"
batch_size=512

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 
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
            run="log_dir=${log_dir} batch_size=${batch_size} device=${device}"
            command="python -m apps.vit.eval config=apps/vit/configs/eval.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done        
    done
done