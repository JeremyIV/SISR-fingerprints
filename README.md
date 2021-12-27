# Fingerprints of Super Resolution Networks

This is the official implementation of our paper *fingerprints of super resolution networks*

## Installation

### Requirements
Requriements are the same as for BasicSR:

 * Python >= 3.7
 * PyTorch >= 1.7
 * NVIDIA GPU + CUDA
 * Linux (we have not tested on Windows)

### Instructions

Clone this Github repo, and clone my fork of [BasicSR](https://github.com/xinntao/BasicSR) inside of it. Install the python package requirements for both projects.
    ```bash
    git clone TODO
    cd SISR_fingerprints
    pip install -r requirements.txt
    git clone git@github.com:JeremyIV/BasicSR.git
    cd BasicSR
    pip install -r requirements.txt 
    ```

## How to Reproduce

This repo reproduces all experiments and results presented in the paper, from training the SISR models to generating the figures and tables. This is a complex multi-step process:

1. Download the SISR trainining datasets
2. Train the custom-trained SISR models
3. Download the pretrained SISR models
4. Download the Flickr1k dataset
5. Generate the super-resolved image dataset
6. Train the model attribution/parsing classifiers
7. Run analysis of the attribution/parsing classifiers
8. Generate the figures and numerical values used in the paper.

### 1. Download the SISR training datasets