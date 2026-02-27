# Vision Transformer Plasticity
[![arXiv](https://img.shields.io/badge/arXiv-2602.06883-b31b1b.svg)](https://arxiv.org/abs/2602.06883)
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2602.06883)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Official implementation of the paper [Vision Transformer Finetuning Benefits from Non-Smooth Components](https://arxiv.org/pdf/2602.06883). <br>
**Goal**: Investigate the plasticity of the vision transformer components by analyzing their average rates of change. <br>
**Findings**: Finetuning non-smooth components (with high plasticity) yields better and more stable performance. <br>
**Illustration**: Non-smooth components allow larger gradient norms and faster descent towards (local) minima.

<img width="945" height="855" alt="loss_landscape" src="https://github.com/user-attachments/assets/1888835f-e58c-4659-8a3e-3a3dd702632e" />

## Abstract
> The smoothness of the transformer architecture has been extensively studied in the context of generalization, training stability, and adversarial robustness. However, its role in transfer learning remains poorly understood. In this paper, we analyze the ability of vision transformer components to adapt their outputs to changes in inputs, or, in other words, their *plasticity*. Defined as an average rate of change, it captures the sensitivity to input perturbation; in particular, a high plasticity implies low smoothness. We demonstrate through theoretical analysis and comprehensive experiments that this perspective provides principled guidance in choosing the components to prioritize during adaptation. A key takeaway for practitioners is that the high plasticity of the attention modules and feedforward layers consistently leads to better finetuning performance. Our findings depart from the prevailing assumption that smoothness is desirable, offering a novel perspective on the functional properties of transformers.
<img width="1891" height="667" alt="intro" src="https://github.com/user-attachments/assets/1b28656c-a84f-4577-bfcb-41cc2cf81139" />

**Illustration**: The high plasticity of non-smooth components leads to greater finetuning benefits (relative gain).

## Overview
Our codebase was tailored to study transformers finetuning; we highly encourage you to use that as a template and modify it however you please to suit your experiments. We tried to make the code as easily modular as possible, so feel free to branch out or fork and play with it. Our codebase is structured as follows:

```
🛠️ vit-plasticity
┣ 📂apps 
┃ ┣ 📂vit # ViT finetuning and plasticity 
┃ ┃ ┣ 📂configs
┃ ┃ ┣ 📂scripts
┃ ┃ ┣ 📄analysis.py
┃ ┃ ┣ 📄eval.py
┃ ┃ ┣ 📄linear_probing.py
┃ ┃ ┣ 📄train.py
┃ ┃ ┗ 📄utils.py
┃ ┣ 📂plots # Figures
┗ 📂src 
  ┗ 📂vitef # Core library
    ┣ 📂data
    ┣ 📂model
    ┣ 📂monitor
    ┣ 📄__init__.py
    ┣ 📄config.py
    ┣ 📄distributed.py
    ┣ 📄optim.py
    ┗ 📄utils.py
```
The ```vitef``` folder contains essential and generic components related to vision transformers, which can be put together in the ```apps``` folder. In particular, ```apps/vit``` can be used to reproduce the experiments of our [paper](https://arxiv.org/pdf/2602.06883). 

## Getting started
The code runs Python 3.10+. Here are some installation instructions:
Install [miniforge](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). Follow the instruction online, most likely you will execute the following commands:
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash ~/Miniforge3-latest-Linux-x86_64.sh
source ~/.bashrc
```
Install Python in a new conda environment (be mindful to install a Python version compatible with Pytorch):
```bash
conda create -n myenv python==3.10
conda activate myenv
```
Install the repository (the vit dependencies are optional but allow for a faster download of pretrained weights):
```bash
git clone <repo url>
cd <repo path>
pip install -e ".[vit]"
```
To install the development and visualization dependencies, you can swap the previous command for the following one:
```bash
pip install -e ".[vit,dev,visu]"
```

#### Accelerate specific instructions
To load models from HuggingFace Transformers library, the accelerate package is needed. After installing it, one needs to configure it. Follow the instruction online [configure-accelerate](https://huggingface.co/docs/accelerate/en/basic_tutorials/install), most likely you will execute the following command and answer the questions prompted to you:
```bash
accelerate config
```

## Launching jobs
We provide below the commands useful to conduct experiments. They must be run from the root of the repository.  

### Configuration
Most experiments need a configuration file interfaced with the command line. Configuration objects are represented as dataclass objetc. 
For example, the file ```your_config.yaml``` looks like:
```yaml
log_dir: your_launch
model_name: base
patch_size: 16
dataset_name: cifar10
batch_size: 512
device: cuda:0
seed: 42
```
It can be used to initialize a dataclass that looks like
```python
@dataclass
class YourConfig:
  log_dir: str = "your_launch"
  model_name: str = "base"
  patch_size: int = 16
  dataset_name: str = "cifar10"
  batch_size: int = 512
  device: str = "cuda:0"
  seed: int = 42
```
In most scripts (```train.py```, ```eval.py```, ```linear_probing.py```), we use [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments). The behavior is as follows:
1. YourConfig is instantiated with its default values,
2. Those default values are overridden with the ones in your_config.yaml,
3. We override the result with the additional arguments provided through command line.
    
### Vision transformer plasticity
To compute the plasticity of ViT components on cifar10, run:
```bash
python -m apps.vit.analysis run --dataset_name cifar10
```
### Linear probing
To launch a linear probing job according to linear_probing.yaml, run:
```bash
python -m apps.vit.linear_probing config=apps/vit/configs/linear_probing.yaml
```
### Finetuning
To launch a finetuning job on Cifar10, run:
```bash
python -m apps.vit.train config=apps/vit/configs/cifar10.yaml
```
### Evaluation
To launch an evaluation job according to eval.yaml, run:
```bash
python -m apps.vit.eval config=apps/vit/configs/eval.yaml
```

## Reproducibility
The experiments of our [paper](https://arxiv.org/pdf/2602.06883) can be reproduced using the scripts in ```apps/vit/scripts```. Launching them will automatically create dedicated ```tmux``` sessions for each group of experiments. After launching those scripts, the linear probing and finetuning performance can be recovered in a folder ```results/``` by running the following command from the root of the repository:
```bash
python -m apps.plots.finetuning csv
```
The figures of our paper can then be reproduced using the files in ```apps/plots```.

## Acknowledgements
Our codebase is designed to study the finetuning dynamics and generalization properties of transformers. It draws inspiration from librairies like [itl](https://github.com/ambroiseodt/itl), [lingua](https://github.com/facebookresearch/lingua) and [pal](https://github.com/facebookresearch/pal).

## Contact
If you have any questions, feel free to reach out at [```ambroiseodonnattechnologie@gmail.com```](mailto:ambroiseodonnattechnologie@gmail.com).

## Citation
If you find our work useful, please consider giving a star ⭐, and citing us as:
```
@misc{odonnat2026vitplasticity,
      title={Vision Transformer Finetuning Benefits from Non-Smooth Components}, 
      author={Ambroise Odonnat and Laetitia Chapel and Romain Tavenard and Ievgen Redko},
      year={2026},
      eprint={2602.06883},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.06883}, 
}
```
