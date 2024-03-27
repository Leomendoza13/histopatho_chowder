# Histography Chowder

## Setup

To set up the project environment, follow these steps:

- Install Python3.8

$sudo add-apt-repository ppa:deadsnakes/ppa

During installation, the system will ask you to press “Enter” from your keyboard to continue and complete the process.
Then put the command

$sudo apt install python3.8 -y

- Pull submodule

$git submodule update --init

- Install virtualenv

$python -m pip install --user virtualenv

- Create environnement

$virtualenv-p python3.8 env

$source env/bin/activate

- Install requirements.txt

$pip install -r requirements.txt

- Download datas

Put the data folder at the root of the project as below:

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

- How to run the train.py script?

    $python train.py

- Example

## Report

Here's is the [report](docs/report.md).