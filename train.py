import os
import torch


from histopatho.utils import (
    load_data_from_npy,
    load_data_from_csv,
    shuffle_data,
    split_train_val_dataset_to_subset,
)
from HistoSSLscaling.rl_benchmarks.datasets import SlideFeaturesDataset
from HistoSSLscaling.rl_benchmarks.models import Chowder
from histopatho.metric import auc
from HistoSSLscaling.rl_benchmarks.trainers import TorchTrainer
from histopatho.trainer import (
    slide_level_train_step_without_mask,
    slide_level_val_step_without_mask,
)
from histopatho.prediction import predict
from histopatho.utils import generate_csv


if __name__ == '__main__':

    train_path = 'data/train_input/moco_features'
    train_labels_path = 'data/train_input/train_output_76GDcgx.csv'

    test_path = 'data/test_input/moco_features'

    train_values = load_data_from_npy(train_path)
    train_labels = load_data_from_csv(train_labels_path)

    test_values = load_data_from_npy(test_path)

    train_values, train_labels = shuffle_data(train_values, train_labels)

    train_dataset = SlideFeaturesDataset(
        features=train_values,
        labels=train_labels,
    )

    train_subset, val_subset = split_train_val_dataset_to_subset(train_dataset)

    chowder = Chowder(
        in_features=2048,
        out_features=1,
        n_top=5,
        n_bottom=5,
        mlp_hidden=[200, 100],
        mlp_activation=torch.nn.Sigmoid(),
        bias=True,
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam
    metrics = {"auc": auc}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = TorchTrainer(
        model=chowder,
        criterion=criterion,
        metrics=metrics,
        device=device,
        optimizer=optimizer,
        batch_size=10,
        num_epochs=30,
        learning_rate=1e-3,
        weight_decay=0.0,
        train_step=slide_level_train_step_without_mask,
        val_step=slide_level_val_step_without_mask,
    )

    train_metrics, val_metrics = trainer.train(train_set=train_subset, val_set=val_subset)

    chowder = chowder.to(device)

    test_values_tensor = torch.tensor(test_values)
    test_values_tensor = test_values_tensor.to(device)

    prediction = predict(chowder, test_values_tensor)

    generate_csv('data/test_metadata.csv', prediction, os.path.join('.', 'train_output.csv'))
