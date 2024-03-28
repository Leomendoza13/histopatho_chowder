"""Define script for ensembling"""

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

from HistoSSLscaling.rl_benchmarks.models import Chowder

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Chowder Model Ensembling')

    # Define command line arguments
    parser.add_argument(
        '--test_feature_dir',
        type=str,
        default='data/test_input/moco_features',
        help='Directory containing testing features',
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
        default='ensembling_output.csv',
        help='Path to output CSV file with predictions',
    )

    args = parser.parse_args()

    test_path = args.test_feature_dir

    if args.bias in ("False", "false", "0"):
        bias = False
    else:
        bias = True

    list_path = []
    count = 0

    for file in os.listdir('weights/'):
        if file.endswith('pth'):
            list_path.append(file)
            count += 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = [
        Chowder(
            in_features=2048,
            out_features=1,
            n_top=args.n_top,
            n_bottom=args.n_bottom,
            mlp_hidden=args.mlp_hidden,
            mlp_activation=torch.nn.Sigmoid(),
            bias=args.bias,
        ).to(device)
        for _ in range(count)
    ]

    for i in range(len(list_path)):
        models[i].load_state_dict(torch.load(os.path.join('weights', list_path[i])))

    test_values = load_npy_from_dir(test_path)

    test_values_tensor = torch.tensor(test_values)
    test_values_tensor = test_values_tensor.to(device)

    predictions = []
    for model in models:
        prediction = predict(model, test_values_tensor)
        predictions.append(prediction)

    ensemble_prediction = torch.stack(predictions).mean(dim=0)

    generate_output_csv(args.test_metadata_path, ensemble_prediction, args.output_path)
