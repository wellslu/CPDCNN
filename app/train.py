import argparse

import torch.utils

import mlconfig
import mlflow
import numpy as np
import torch
import torchvision

import src


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config.yaml')
    parser.add_argument('-r', '--resume', type=str, default=None)
    return parser.parse_args()


def manual_seed(seed=0):
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    config = mlconfig.load(args.config)
    mlflow.log_artifact(args.config)
    mlflow.log_params(config.flat())

    manual_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config.model().to(device)
    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
    # train_loader = config.dataset(list_file='train')
    # test_loader = config.dataset(list_file='test')
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.225)
        ])), batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=False, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.225)
        ])), batch_size=32, shuffle=False)

    trainer = config.trainer(device, model, optimizer, scheduler, train_loader, test_loader)

    if args.resume is not None:
        trainer.resume(args.resume)

    trainer.fit()


if __name__ == '__main__':
    main()
