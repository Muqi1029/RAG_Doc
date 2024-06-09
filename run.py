import argparse
import torch
from src.model.baseline_mlp import BaselineMLP, CosineSimilarityLoss
from train import train, test
from src.dataset import get_dataloader, QueryDataset
# from src.loss import random_cosineSimilarityLoss, negative_cosineSimilarityLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import save_params, load_params

parser = argparse.ArgumentParser(description="")
parser.add_argument("-m", "--model", default="mlp", choices=['mlp'])
parser.add_argument("--lr", type=float, default=0.8)
parser.add_argument("-e", "--epochs", type=int, default=20)
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("--data_dir", type=str, default="input")
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--K", type=int, default=64, help="negative sample size")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parser.parse_args()
    dataset = QueryDataset("input")
    dataloader = get_dataloader(args.data_dir, args.batch_size)
    if args.model == "mlp":
        model = BaselineMLP(input_dim=1024, hidden_dims=[2048, 1628])
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=20)
        if args.test:
            print(f"start to test".center(50, "="))
            load_params(model, args.model, device=device)
            test(model, device)
        else:
            print(f"start to train".center(50, "="))
            train(dataset, model, optimizer, scheduler, CosineSimilarityLoss, epochs=args.epochs, batch_size=args.batch_size, device=device)
            save_params(model, args.model)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
