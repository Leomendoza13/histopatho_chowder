"""Define Script"""

import os
import torch

from histopatho.metric import auc
from histopatho.trainer import (
    slide_level_train_step,
    slide_level_val_step,
)

from histopatho.prediction import predict
from histopatho.utils import generate_output_csv
from histopatho.utils import (
    load_npy_from_dir,
    load_data_from_csv,
    shuffle_data,
    split_dataset_in_subset,
)

from HistoSSLscaling.rl_benchmarks.trainers import TorchTrainer
from HistoSSLscaling.rl_benchmarks.datasets import SlideFeaturesDataset
from HistoSSLscaling.rl_benchmarks.models import Chowder

if __name__ == '__main__':

    # Define paths for training and testing data
    train_path = 'data/train_input/moco_features'
    train_labels_path = 'data/train_input/train_output_76GDcgx.csv'
    test_path = 'data/test_input/moco_features'

    # Load training and testing data
    train_values = load_npy_from_dir(train_path)
    train_labels = load_data_from_csv(train_labels_path)
    test_values = load_npy_from_dir(test_path)

    # Shuffle training data and labels
    train_values, train_labels = shuffle_data(train_values, train_labels)

    # Create dataset objects for training
    train_dataset = SlideFeaturesDataset(
        features=train_values,
        labels=train_labels,
    )

    # Split training dataset into training and validation subsets
    train_subset, val_subset = split_dataset_in_subset(train_dataset)

    # Define Chowder model architecture
    chowder = Chowder(
        in_features=2048,
        out_features=1,
        n_top=5,
        n_bottom=5,
        mlp_hidden=[200, 100],
        mlp_activation=torch.nn.Sigmoid(),
        bias=True,
    )

    # Define loss function, optimizer, and evaluation metrics
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam
    metrics = {"auc": auc}

    # Choose device for training (cuda if available, otherwise cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize trainer for model training
    trainer = TorchTrainer(
        model=chowder,
        criterion=criterion,
        metrics=metrics,
        device=device,
        optimizer=optimizer,
        batch_size=3,
        num_epochs=3,
        learning_rate=1e-3,
        weight_decay=0.0,
        train_step=slide_level_train_step,
        val_step=slide_level_val_step,
    )

    # Train the model and obtain training and validation metrics
    train_metrics, val_metrics = trainer.train(train_set=train_subset, val_set=val_subset)

    # Move model to device
    chowder = chowder.to(device)

    # Move testing data to device
    test_values_tensor = torch.tensor(test_values)
    test_values_tensor = test_values_tensor.to(device)

    # Make predictions on testing data
    prediction = predict(chowder, test_values_tensor)

    # Generate output CSV file with predictions
    generate_output_csv('data/test_metadata.csv', prediction, os.path.join('.', 'train_output.csv'))
