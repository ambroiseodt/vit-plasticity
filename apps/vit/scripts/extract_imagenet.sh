#!/bin/bash

# To extract imagenet training and validation data, run this file from the root of the repository:
# ```bash
# $ bash <path_to_file_folder>/extract_imagenet.sh
# ```

# Adapted from: https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh
# Requirement: ILSVRC2012_img_train.tar (about 138 GB) & ILSVRC2012_img_val.tar (about 6.3 GB) in the current directory
# If only one of those files is available, comment the commands not related to it (e.g., the training part).

# Make imagenet directory
mkdir datasets/imagenet

#########################################################
# Extract training data
#########################################################

# Create train directory; move .tar file; change directory
mkdir datasets/imagenet/train && mv ILSVRC2012_img_train.tar datasets/imagenet/train/ && cd datasets/imagenet/train

# Extract training set; remove compressed file
tar -xvf ILSVRC2012_img_train.tar

# At this stage imagenet/train will contain 1000 compressed .tar files, one for each category
# Use the following command to apply the 3 following steps for each .tar file:
#   1. create directory with same name as .tar file
#   2. extract and copy contents of .tar file into directory
#   3. remove .tar file
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done

# This results in a training directory like so:
#  imagenet/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......

# Change back to original directory
cd ../../..

# Check that all the training files were extracted; it should output 1281167 
find datasets/imagenet/train/ -name "*.JPEG" | wc -l

###########################################################
# Extract validation data
###########################################################

# Create validation directory; move .tar file; change directory; extract validation .tar; remove compressed file
mkdir datasets/imagenet/val && mv ILSVRC2012_img_val.tar datasets/imagenet/val/ && cd datasets/imagenet/val && tar -xvf ILSVRC2012_img_val.tar

# get script from soumith and run; this script creates all class directories and moves images into corresponding directories
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

# Change back to original directory
cd ../../..

# Check that all the validation files were extracted; it should output 50000 
find datasets/imagenet/val/ -name "*.JPEG" | wc -l

# This results in a validation directory like so:
#  imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
