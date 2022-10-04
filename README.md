# Fingerprints of Super Resolution Networks
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

👷 **These instructions are under construction. They have not been tested. Check back soon for more thorough and reliable instructions.** 

This is the official implementation of our TMLR paper [fingerprints of super resolution networks](https://openreview.net/forum?id=Jj0qSbtwdb).

## Installation

### Requirements
Requriements are the same as for [BasicSR](https://github.com/XPixelGroup/BasicSR):

 * Python >= 3.7
 * PyTorch >= 1.7
 * NVIDIA GPU + CUDA
 * Linux (we have not tested on Windows)

### Setup

Clone this Github repo and install the Python package requirements.
    
```bash
git clone git@github.com:JeremyIV/SISR-fingerprints.git
cd SISR-fingerprints
pip install -r requirements.txt
```

## How to Reproduce

### 1. Download our dataset.

Our dataset of 205,000 super-resolved images can be downloaded from [Google Drive](https://drive.google.com/drive/folders/174guowoIpE07MNLB5noWbkZAL9NJRH59?usp=sharing) (164 Gb). 

We provide a script to perform this download, although google drive's limiting of command-line downloads may [cause issues](https://github.com/wkentaro/gdown/issues/43). To try the script:

```bash
cd classification/datasets/data/SISR
bash download_SISR_data.sh
```

If that stops working, you can download the dataset through a browser. Download all of the zip files into the `classification/datasets/data/SISR` folder, then, same as in the script, run:

```bash
for folder in $(ls *.zip); do unzip $folder && rm $folder; done
mv SISR/* ./
rmdir SISR
```

### 2. Create the evaluation results database.

All of the model attribution and parsing results are recorded in an sqlite database. To initialize that databse, run

```bash
python database/create_database.py
```

### 3. Train/evaluate the attribution and parsing classifiers.

The `classification/train_classifier.py` script can both train and evaluate model attribution and parsing classifiers. The simplest way to train and evaluate these classifiers is to run:

```bash
python classification/train_classifier.py -opt classification/options/path/to/options_file.yaml
```
 However, while training can be performed in parallel across many GPUs or machines, only one process may record results to the sqlite database at a time. Since training all of these classifiers takes almost 90 GPU days on Titan X gpus, we highly recommend training in parallel, and then evaluating serially afterwards. To train a model without evaluating it, run:

 ```bash
 python classification/train_classifier.py -opt classification/options/path/to/classifier_config.yaml --mode train
 ```

 Then to evaluate that model, run:

  ```bash
 python classification/train_classifier.py -opt classification/options/path/to/classifier_config.yaml --mode test
 ```

### 4. Analyze the evaluation results.

Almost all figures and numerical values presented in the paper can be automatically generated from scripts in the `analysis/` directory. 

#### Numerical Values
To generate the numerical values that appear in our paper, run:

```bash
python analysis/values/generate_values.py
```
These values will appear in `paper/computed_values.tex`.

#### Figures
Each figure in our paper (except for Figures 1 and 2) can be generated by a script in the `analysis/figures/` directory. For example, to generate Figure 3, run:

```bash
python analysis/figures/custom_model_tsne.py
```

These figures will appear in `paper/figures/`.

### 5. Render the paper.
coming soon...
