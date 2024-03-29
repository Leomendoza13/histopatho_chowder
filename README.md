# Detection of PIK3CA Mutation Using Chowder Model

This project uses a model called Chowder to predict the probability of a PIK3CA mutation based on features generated by a pre-trained ResNet50 model. When provided with these features, the system compute the probability of a PIK3CA mutation, assissting in the detection and analysis of mutations.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Report](#report)

## Installation

To set up the project environment, follow these steps:

- Install Python3.8

```
    sudo add-apt-repository ppa:deadsnakes/ppa
```
During installation, the system will ask you to press “Enter” from your keyboard to continue and complete the process.
Then put the command
```
    sudo apt install python3.8 -y
```

- Pull submodule

```
    git submodule update --init
```

- Install virtualenv

```
    python -m pip install --user virtualenv
```

- Create environnement

```
    virtualenv -p python3.8 env
    source env/bin/activate
```

- Install requirements.txt

```
    pip install -r requirements.txt
```

- Download datas

By default, the data folder must be at the root of the project as below:

    data/
    ├── test_input
    │   └── moco_features
    │       ├── ID_003.npy
    │       │   .........
    │       └── ID_493.npy
    ├── test_metadata.csv
    └── train_input
        ├── moco_features
        │   ├── ID_001.npy
        │   │   .........
        │   └── ID_491.npy
        └── train_output_76GDcgx.csv

## Usage

### Training

- To train Chowder and predict on the test data, use this command:

```
    python train.py
```

The train_output.csv, containing the predictions of each slides, is then created at the root of the project by default (can be changed with the output_path argument).

- These arguments can be used:

| Arguments              | Type     | Default   | Description                                                  |
|------------------------|----------|-----------|--------------------------------------------------------------|
| --train_feature_dir    | str      | 'data/train_input/moco_features' | Directory containing training features                  |
| --test_feature_dir     | str      | 'data/test_input/moco_features'  | Directory containing testing features                   |
| --labels_path          | str      | 'data/train_input/train_output_76GDcgx.csv' | Path to the training labels file           |
| --train_val_split_ratio| float    | 80        | Ratio of training data to validation data                     |
| --n_top                | int      | 5         | Number of top features for Chowder model                      |
| --n_bottom             | int      | 5         | Number of bottom features for Chowder model                   |
| --mlp_hidden           | list[int]| 200 100 | List of integers representing the hidden layers of MLP        |
| --batch_size           | int      | 16        | Batch size for training                                       |
| --num_epochs           | int      | 15        | Number of epochs for training                                 |
| --lr                   | float    | 0.001      | Learning rate for training                                    |
| --weight_decay         | float    | 0.0       | Weight decay for training                                     |
| --bias                 | str      | "True"    | Whether to add bias for layers of the tiles MLP               |
| --test_metadata_path   | str      | 'data/test_metadata.csv'        | Path to test_metadata.csv to build output                    |
| --output_path          | str      | 'train_output.csv'               | Path to output CSV file with predictions                     |


- Example

```
    python train.py --batch_size 15 --num_epochs 30
```

### Ensemble Prediction

You first have to train multiple Chowder and save the weights. To do so, use this command:

```
python train.py --save <filename>.pth
```
The weights will be save at the weights directory at the root of the project.
Do this above command as many as Chowder you want for the ensemble prediction.

**BE CAREFUL: THE HYPERPARAMETERS OF EACH CHOWDER MUST BE THE SAME**

 - To ensemble predict on many Chowder, use this command:

```
    python ensemble.py
```
The script will loads the weights of each model in the weights directory.

The ensemble_output.csv, containing the ensemble prediction (average predictions of every CHOWDER in weights directory) of each slides, is then created at the root of the project by default (can be changed with the output_path argument).

**DON'T FORGET TO USE THE SAME PARAMETERS ARGUMENTS ON ENSEMBLE.PY AS TRAIN.PY**

- Example:

```
python train.py --n_top 2 --n_bottom 2 --save weights1.pth
python train.py --n_top 2 --n_bottom 2 --save weights2.pth
python ensemble.py --n_top 2 --n_bottom 2
```

## Report

Here's is the [report](docs/report.md).