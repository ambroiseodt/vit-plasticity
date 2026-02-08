#!/usr/bin/bash

# Launch plasticity analysis using apps/vit/analysis.py. It will create a dedicated
# tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/analysis.sh
# ```

# BASE
batch_size=64
n_steps=200
device="cuda:0"

session="base_cifar"
tmux new-session -d -s ${session}
for dataset_name in \
            "cifar10" \
            "cifar100" 
do
    command="python -m apps.vit.analysis run --dataset_name ${dataset_name} --batch_size ${batch_size} --n_steps ${n_steps} --device ${device}"
    tmux send-keys -t ${session} "${command}" C-m
done


session="base_cifar10c"
tmux new-session -d -s ${session}
for dataset_name in \
            "cifar10_c-corruption-contrast-severity-5" \
            "cifar10_c-corruption-gaussian_noise-severity-5" \
            "cifar10_c-corruption-motion_blur-severity-5" \
            "cifar10_c-corruption-snow-severity-5" \
            "cifar10_c-corruption-speckle_noise-severity-5"  
do
    command="python -m apps.vit.analysis run --dataset_name ${dataset_name} --batch_size ${batch_size} --n_steps ${n_steps} --device ${device}"
    tmux send-keys -t ${session} "${command}" C-m
done

session="base_others"
tmux new-session -d -s ${session}
for dataset_name in \
            "domainnet-clipart" \
            "domainnet-sketch" \
            "pet" \
            "flowers102"
do
    command="python -m apps.vit.analysis run --dataset_name ${dataset_name} --batch_size ${batch_size} --n_steps ${n_steps} --device ${device}"
    tmux send-keys -t ${session} "${command}" C-m
done

# LARGE
batch_size=32
n_steps=400
device="cuda:0"
model_name="large"

session="large_cifar"
tmux new-session -d -s ${session}
for dataset_name in \
            "cifar10" \
            "cifar100" 
do
    command="python -m apps.vit.analysis run --model_name ${model_name} --dataset_name ${dataset_name} --batch_size ${batch_size} --n_steps ${n_steps} --device ${device}"
    tmux send-keys -t ${session} "${command}" C-m
done


session="large_cifar10c"
tmux new-session -d -s ${session}
for dataset_name in \
            "cifar10_c-corruption-contrast-severity-5" \
            "cifar10_c-corruption-gaussian_noise-severity-5" \
            "cifar10_c-corruption-motion_blur-severity-5" \
            "cifar10_c-corruption-snow-severity-5" \
            "cifar10_c-corruption-speckle_noise-severity-5"  
do
    command="python -m apps.vit.analysis run --model_name ${model_name} --dataset_name ${dataset_name} --batch_size ${batch_size} --n_steps ${n_steps} --device ${device}"
    tmux send-keys -t ${session} "${command}" C-m
done

session="large_others"
tmux new-session -d -s ${session}
for dataset_name in \
            "domainnet-clipart" \
            "domainnet-sketch" \
            "pet" \
            "flowers102"
do
    command="python -m apps.vit.analysis run --model_name ${model_name} --dataset_name ${dataset_name} --batch_size ${batch_size} --n_steps ${n_steps} --device ${device}"
    tmux send-keys -t ${session} "${command}" C-m
done


# HUGE
batch_size=32
n_steps=400
device="cuda:0"
model_name="huge"
patch_size=14

session="huge_cifar"
tmux new-session -d -s ${session}
for dataset_name in \
            "cifar10" \
            "cifar100" 
do
    command="python -m apps.vit.analysis run --model_name ${model_name} --patch_size ${patch_size} --dataset_name ${dataset_name} --batch_size ${batch_size} --n_steps ${n_steps} --device ${device}"
    tmux send-keys -t ${session} "${command}" C-m
done


session="huge_cifar10c"
tmux new-session -d -s ${session}
for dataset_name in \
            "cifar10_c-corruption-contrast-severity-5" \
            "cifar10_c-corruption-gaussian_noise-severity-5" \
            "cifar10_c-corruption-motion_blur-severity-5" \
            "cifar10_c-corruption-snow-severity-5" \
            "cifar10_c-corruption-speckle_noise-severity-5"  
do
    command="python -m apps.vit.analysis run --model_name ${model_name} --patch_size ${patch_size} --dataset_name ${dataset_name} --batch_size ${batch_size} --n_steps ${n_steps} --device ${device}"
    tmux send-keys -t ${session} "${command}" C-m
done

session="huge_others"
tmux new-session -d -s ${session}
for dataset_name in \
            "domainnet-clipart" \
            "domainnet-sketch" \
            "pet" \
            "flowers102"
do
    command="python -m apps.vit.analysis run --model_name ${model_name} --patch_size ${patch_size} --dataset_name ${dataset_name} --batch_size ${batch_size} --n_steps ${n_steps} --device ${device}"
    tmux send-keys -t ${session} "${command}" C-m
done
