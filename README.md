# Fingerprints of Super Resolution Networks

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is the official implementation of our paper *fingerprints of super resolution networks*

## Installation

### Requirements
Requriements are the same as for BasicSR:

 * Python >= 3.7
 * PyTorch >= 1.7
 * NVIDIA GPU + CUDA
 * Linux (we have not tested on Windows)

### Setup

Clone this Github repo, and clone [my fork of BasicSR](https://github.com/xinntao/BasicSR) inside of it. Install the Python package requirements for both projects.
    
```bash
git clone git@github.com:JeremyIV/SISR-fingerprints.git
cd SISR-fingerprints
pip install -r requirements.txt
git clone git@github.com:JeremyIV/BasicSR.git
cd BasicSR
pip install -r requirements.txt 
```

## How to Reproduce

This repo reproduces all experiments and results presented in the paper, from training the SISR models, to generating (almost) all figures and values presented in the paper, to generating the paper itself. This is a complex, many-step process. Reproducing some steps will require considerable effort. To make things easier, you can quickly start anywhere in the process by downloading the prerequisite data from the previous steps.

TODO: create links to each section. Also add links whever sections reference previous sections for prerequisites

1. Download the SISR trainining datasets
2. Train the custom-trained SISR models
3. Download the Flickr1k dataset
4. Generate the super-resolved image dataset
5. Train the model attribution/parsing classifiers
6. Generate the figures and numerical values used in the paper.
7. Generate the paper itself.

### 1. Download the SISR training datasets

[Google Drive](TODO) (TODO GB). Extract contents to `BasicSR/datasets/`.

### 2. Train the custom-trained SISR models

To skip this step and download the results:
[Google Drive](TODO) (60 GB). Extract contents to `BasicSR/finished_models/`.

Otherwise, first complete step 1

### 3. Download the Flickr1k dataset
[Google Drive](TODO) (1.3  GB). Extract contents to `BasicSR/datasets/`.


### 4. Generate the super-resolved image dataset
This step takes a significant amount of time and effort. To make things easier, we provide the super-resolved images from the 25 pretrained models on [Google Drive](TODO) (20 GB). Extract contents to `classification/datasets/data/SISR/`.
You can also generate this data yourself following the instructions below, but it is a lot of work. 

You will have to generate the super-resolved images for the 180 custom-trained models yourself, following the instructions below:

First complete steps 2 and 3. Then:

If you want to generate the data for the pretrained models yourself, follow the instructions from their respective GitHub repos:

| Model         | Github Link |
|---------------|-------------|
|EDSR, ESRGAN, RDN, RCAN, SRGAN, SwinIR| https://github.com/XPixelGroup/BasicSR |
|EnhanceNet      | https://github.com/msmsajjadi/EnhanceNet-Code |
|ProSR, ProGanSR | https://github.com/fperazzi/proSR |
|SAN            | https://github.com/daitao/SAN |
|SRFBN          | https://github.com/Paper99/SRFBN_CVPR19 |
|SPSR           | https://github.com/Maclory/SPSR |
|DRN            | https://github.com/guoyongcs/DRN |
|NCSR           | https://github.com/younggeun-kim/NCSR |
|LIIF           | https://github.com/yinboc/liif |
|Real-ESRGAN    | https://github.com/xinntao/Real-ESRGAN |
|NLSN           | https://github.com/HarukiYqM/Non-Local-Sparse-Attention |

Use each of these repositories to super-resolve the Flickr1K dataset. Place the resulting images in `classification/datasets/data/SISR/{pretrained model name}` where `{pretrained model name}` in directories named:

| Model |
|--------|
| EDSR-div2k-x2-L1-NA-pretrained |
| LIIF-div2k-x2-L1-NA-pretrained |
| NLSN-div2k-x2-L1-NA-pretrained |
| RCAN-div2k-x2-L1-NA-pretrained |
| RDN-div2k-x2-L1-NA-pretrained |
| SRFBN-NA-x2-L1-NA-pretrained |
| SwinIR-div2k-x2-L1-NA-pretrained |
| Real_ESRGAN-div2k-x2-GAN-NA-pretrained |
| DRN-div2k-x4-L1-NA-pretrained |
| EDSR-div2k-x4-L1-NA-pretrained |
| LIIF-div2k-x4-L1-NA-pretrained |
| NLSN-div2k-x4-L1-NA-pretrained |
| RCAN-div2k-x4-L1-NA-pretrained |
| RDN-div2k-x4-L1-NA-pretrained |
| SAN-div2k-x4-L1-NA-pretrained |
| SRFBN-NA-x4-L1-NA-pretrained |
| SwinIR-div2k-x4-L1-NA-pretrained |
| proSR-div2k-x4-L1-NA-pretrained |
| ESRGAN-NA-x4-ESRGAN-NA-pretrained |
| EnhanceNet-NA-x4-EnhanceNet-NA-pretrained |
| Real_ESRGAN-div2k-x4-GAN-NA-pretrained |
| SwinIR-div2k-x4-GAN-NA-pretrained |
| NCSR-div2k-x4-NCSR_GAN-NA-pretrained |
| proSR-div2k-x4-ProSRGAN-NA-pretrained |
| SPSR-div2k-x4-SPSR_GAN-NA-pretrained |

To generate the super-resolved images for the custom-trained models, run:

```bash
cd BasicSR
python scripts/custom_sisr_dataset/make_train_test_configs.py --generate_SISR_dataset
```

Then run 

```bash
python basicsr/test.py -opt options/test/generate_sisr_dataset/<config.yml>
```

for each config in `options/test/generate_sisr_dataset/`.

Then cd back into the main SISR-fingerprints directory.
Then run

```bash
python classification/datasets/data/move_sisr_datasets.py
``` 

To move the datasets where they need to be.

### 5. Train and evaluate the model attribution/parsing classifiers
To skip training, you can get the pretrained classifiers from [Google Drive](TODO) (25 GB). Extract contents to `classification/classifiers/`. You will still need to evaluate the classifiers.

Otherwise, first complete Step 4.

To train all the classifiers with a single command, run

```bash
bash classification/scripts/train_all_classifiers.sh
```

Be warned that while this approach is simple, it will take a very long time to finish: about 90 GPU-days using Titan X GPUs. We reccommend distributing the work across a large number of GPUs. 

### 8. Generate the figures and numerical values used in the paper.

To generate the numerical values used in the paper, run

```bash
python results/values/generate_values.py
```

Expect run time around twenty minutes. This will generate `paper/computed_values.tex`.

To generate the figures used in this paper, run

```bash
bash results/figures/generate_all.sh
```
Expect run time around twenty minutes. This will generate almost all figures used in the paper and save them to `paper/figures`.

### 9. Generate the paper itself.
Finally, to generate a copy of our published paper, run

```
cd paper
latex # TODO...
```