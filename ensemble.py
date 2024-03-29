"""Define script for ensembling"""

import os
import argparse
import torch

from histopatho.prediction import ensemble_predict
from histopatho.utils import load_npy_from_dir, generate_output_csv, get_weight_paths

from HistoSSLscaling.rl_benchmarks.models import Chowder

if __name__ == '__main__':

    # Parse command line arguments
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
        default='ensemble_output.csv',
        help='Path to output CSV file with predictions',
    )

    args = parser.parse_args()

    # Set test directory path
    test_path = args.test_feature_dir

    # Determine bias value based on input arguments
    if args.bias in ("False", "false", "0"):
        bias = False
    else:
        bias = True

    chowders_info = f"""\n
    - in_features=2048
    - out_features=1
    - n_top={args.n_top}
    - n_bottom={args.n_bottom}
    - mlp_hidden={args.mlp_hidden}
    - mlp_activation=torch.nn.Sigmoid()
    - bias={bias}
    """

    # Retrieve paths of weights
    list_paths = get_weight_paths('weights')

    # Determine device (cuda or cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize models
    models = [
        Chowder(
            in_features=2048,
            out_features=1,
            n_top=args.n_top,
            n_bottom=args.n_bottom,
            mlp_hidden=args.mlp_hidden,
            mlp_activation=torch.nn.Sigmoid(),
            bias=bias,
        ).to(device)
        for _ in range(len(list_paths))
    ]

    # Load weights for each model
    for i, path in enumerate(list_paths):
        models[i].load_state_dict(torch.load(os.path.join('weights', path)))

    print("\n" + str(len(models)) + " Chowders initialized with:" + chowders_info)

    # Load test values from directory
    test_values = load_npy_from_dir(test_path)

    # Convert test values to tensor and move to appropriate device
    test_values_tensor = torch.tensor(test_values).to(device)

    # Generate ensemble predictions
    ensemble_prediction = ensemble_predict(models, test_values_tensor)

    print("Average predictions computed.")

    # Generate output CSV file with predictions
    generate_output_csv(args.test_metadata_path, ensemble_prediction, args.output_path)
    print("\nOuput generated at " + args.output_path + '.\n')
