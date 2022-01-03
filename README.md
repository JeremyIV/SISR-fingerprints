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

This repo reproduces all experiments and results presented in the paper, from training the SISR models, to generating (almost) all figures and values presented in the paper, to generating the paper itself. This is a complex multi-step process. You can start anywhere in the process by downloading the data produced in the previous step.

1. Download the SISR trainining datasets
2. Train the custom-trained SISR models
3. Download the pretrained SISR models
4. Download the Flickr1k dataset
5. Generate the super-resolved image dataset
6. Train the model attribution/parsing classifiers
8. Generate the figures and numerical values used in the paper.
9. Generate the paper itself.

### 1. Download the SISR training datasets

### 6. Train the model attribution/parsing classifiers

To train all the classifiers with a single command, run

```bash
bash classification/scripts/train_all_classifiers.sh
```

Be warned that this will take a very long time (about 1 week) to finish.

### 8. Generate the figures and numerical values used in the paper.

To generate the numerical values used in the paper, run

```bash
python results/values/generate_values.py
```

Expect run time around an hour. This will generate `paper/computed_values.tex`.

To generate the figures used in this paper, run

```bash
bash results/figures/generate_all.sh
```
Expect run time around an hour. This will generate almost all figures used in the paper and save them to `paper/figures`.

### 9. Generate the paper itself.
Finally, to generate a copy of our published paper, run

```
cd paper
latex # TODO...
```