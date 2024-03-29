"""Define Script"""

import os
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
        default=80,
        help='Ratio of training data to validation data',
    )
    parser.add_argument(
        '--n_top', type=int, default=5, help='Number of top features for Chowder model'
    )
    parser.add_argument(
        '--n_bottom', type=int, default=5, help='Number of bottom features for Chowder model'
    )
    parser.add_argument(
        "--mlp_hidden",
        nargs="+",
        type=int,
        default=[200, 100],
        help="List of integers representing the hidden layers of MLP.",
    )
    parser.add_argument(
        "--mlp_dropout",
        nargs="+",
        type=float,
        default=None,
        help="Dropout that is used for each layer of the MLP. If `None`, no dropout is used.",
    )
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for training')
    parser.add_argument(
        '--bias', type=str, default="True", help='Whether to add bias for layers of the tiles MLP'
    )
    parser.add_argument(
        '--test_metadata_path',
        type=str,
        default='data/test_metadata.csv',
        help='Path to test_metadata.csv to build output',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='train_output.csv',
        help='Path to output CSV file with predictions',
    )
    parser.add_argument(
        '--save',
        type=str,
        default='',
        help='Name of the pth file',
    )

    args = parser.parse_args()

    # Determine bias value based on input arguments
    if args.bias in ("False", "false", "0"):
        bias = False
    else:
        bias = True

    # Choose device for training (cuda if available, otherwise cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    chowder_info = f"""\nChowder initialization with:

    - in_features=2048
    - out_features=1
    - n_top={args.n_top}
    - n_bottom={args.n_bottom}
    - mlp_hidden={args.mlp_hidden}
    - mlp_dropout={args.mlp_dropout}
    - mlp_activation=torch.nn.Sigmoid()
    - bias={bias}
    """

    trainer_info = f"""TorchTrainer initialization with:

    - model=Chowder
    - criterion=BCEWithLogitsLoss
    - metrics={{'auc': auc}}
    - device={device}
    - optimizer=Adam
    - batch_size={args.batch_size}
    - num_epochs={args.num_epochs}
    - learning_rate={args.lr}
    - weight_decay={args.weight_decay}
    """

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
    train_dataset = SlideFeaturesDataset(features=train_values, labels=train_labels, shuffle=False)

    # Split training dataset into training and validation subsets
    train_subset, val_subset = split_dataset_in_subset(train_dataset, args.train_val_split_ratio)

    print("\nData loaded.")

    # Define Chowder model architecture
    chowder = Chowder(
        in_features=2048,
        out_features=1,
        n_top=args.n_top,
        n_bottom=args.n_bottom,
        mlp_hidden=args.mlp_hidden,
        mlp_activation=torch.nn.Sigmoid(),
        bias=bias,
        mlp_dropout=args.mlp_dropout,
    )

    print(chowder_info)

    # Define loss function, optimizer, and evaluation metrics
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam
    metrics = {"auc": auc}

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
        weight_decay=args.weight_decay,
        train_step=slide_level_train_step,
        val_step=slide_level_val_step,
    )

    print(trainer_info)

    # Train the model and obtain training and validation metrics
    print("Training:\n")
    train_metrics, val_metrics = trainer.train(train_set=train_subset, val_set=val_subset)

    print("\nBest train metric: " + str(max(train_metrics['auc'])))
    print("Best val metric: " + str(max(val_metrics['auc'])))

    # save weights at ./weights/ if save_path is not empty
    if len(args.save) != 0:
        torch.save(chowder.state_dict(), os.path.join('weights', args.save))
        print('\nWeights saved at weights/' + args.save)

    # Move model to device
    chowder = chowder.to(device)

    # Move testing data to device
    test_values_tensor = torch.tensor(test_values).to(device)

    # Make predictions on testing data
    prediction = predict(chowder, test_values_tensor)

    # Generate output CSV file with predictions
    generate_output_csv(args.test_metadata_path, prediction, args.output_path)

    print("\nOuput generated at " + args.output_path + '.\n')
