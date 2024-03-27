"""Define Script"""

import argparse
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

    parser = argparse.ArgumentParser(description='Chowder Model Training and Prediction on PIK3CA')

    # Define command line arguments
    parser.add_argument(
        '--train_feature_dir',
        type=str,
        default='data/train_input/moco_features',
        help='Directory containing training features',
    )
    parser.add_argument(
        '--test_feature_dir',
        type=str,
        default='data/test_input/moco_features',
        help='Directory containing testing features',
    )
    parser.add_argument(
        '--labels_path',
        type=str,
        default='data/train_input/train_output_76GDcgx.csv',
        help='Path to the training labels file',
    )
    parser.add_argument(
        '--train_val_split_ratio',
        type=float,
        default=0.8,
        help='Ratio of training data to validation data',
    )
    parser.add_argument(
        '--n_top', type=int, default=5, help='Number of top features for Chowder model'
    )
    parser.add_argument(
        '--n_bottom', type=int, default=5, help='Number of bottom features for Chowder model'
    )
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training')
    parser.add_argument(
        '--output_path',
        type=str,
        default='train_output.csv',
        help='Path to output CSV file with predictions',
    )

    args = parser.parse_args()

    # Define paths for training and testing data
    train_path = args.train_feature_dir
    train_labels_path = args.labels_path
    test_path = args.test_feature_dir

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
        n_top=args.n_top,
        n_bottom=args.n_bottom,
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
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
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
    generate_output_csv('data/test_metadata.csv', prediction, args.output_path)
