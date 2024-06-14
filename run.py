"""run py"""
import argparse
import os
import sys
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.models import BaselineMLP, SimCLR, BYOL, MLPHead
from train import Trainer, byol_test, test
from src.dataset import QueryDataset
from src.train import BYOLTrainer, SimCLRTrainer, MLPTrainer
from utils import save_params, load_params


parser = argparse.ArgumentParser(description="Retrieval Model Argument")
parser.add_argument("--model", default="mlp",
                    choices=['mlp', 'simclr', 'byol'])
parser.add_argument("--lr", type=float, default=0.2)

parser.add_argument("--m", type=float, default=0.99)
parser.add_argument("--input-dim", default=1024, type=int)
parser.add_argument("--document-dir", type=str, default="input/document.json")

parser.add_argument("-e", "--epochs", type=int, default=200)

parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("--data_dir", type=str, default="input")
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--K", type=int, default=1024, help="negative sample size")
parser.add_argument("--random", action="store_true")
parser.add_argument("--loss", type=str, default="nce",
                    choices=['random_sample', 'negative_sample', 'nce'])
parser.add_argument("--doc_plus_title", action="store_true", default=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parser.parse_args()
    args.device = device
    dataset = QueryDataset("input")
    if args.model == "mlp":
        model = BaselineMLP(input_dim=1024, hidden_dims=[1248, 2048, 1628])
        print(model.eval())
        model.to(device)
        if not args.test:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=20)
            trainer = MLPTrainer(model, optimizer=optimizer, scheduler=scheduler, args=args)
            print("start to train".center(50, "="))
            trainer.train(dataset)
        else:
            filename = args.model + "_random" if args.random else args.model + "_neg"
            load_params(model, filename, device)
            trainer = MLPTrainer(model, optimizer=None, scheduler=None, args=args)
            print("start to train".center(50, "="))
            trainer.predict()

    elif args.model == 'simclr':
        model = SimCLR(input_dim=args.input_dim,
                       hidden_dims=[1628], out_dim=1024)
        print(model.eval())
        model.to(device)
        if not args.test:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=20)
            trainer = SimCLRTrainer(
                model, optimizer=optimizer, scheduler=scheduler, args=args)
            print("start to train".center(50, "="))
            trainer.train(dataset)
        else:
            filename = args.model + "_random" if args.random else args.model + "_neg"
            load_params(model, filename, device)
            trainer = SimCLRTrainer(
                model, optimizer=None, scheduler=None, args=args)
            print("start to test".center(50, "="))
            trainer.predict()

    elif args.model == 'byol':
        target_model = BYOL(input_dim=args.input_dim,
                            hidden_dims=[1628],
                            out_dim=1024).to(device)
        predictor = MLPHead(
            in_channels=target_model.projector[-1].out_features,
            mlp_hidden_size=2048,
            projection_size=1024).to(device)
        if not args.test:
            online_model = BYOL(input_dim=args.input_dim,
                                hidden_dims=[1628],
                                out_dim=1024).to(device)
            optimizer = torch.optim.SGD(
                list(online_model.parameters()) + list(predictor.parameters()),
                lr=args.lr,
                momentum=0.9,
                weight_decay=0.0004)
            scheduler = CosineAnnealingLR(optimizer, T_max=20)
            trainer = BYOLTrainer(online_network=online_model,
                                  target_network=target_model,
                                  predictor=predictor,
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  args=args)
            print("start to train".center(50, "="))
            trainer.train(dataset)
        else:
            print("start to test".center(50, "="))
            target_model = BYOL(input_dim=args.input_dim, hidden_dims=[
                1628], out_dim=1024).to(device)
            predictor = MLPHead(
                in_channels=target_model.projector[-1].out_features,
                mlp_hidden_size=2048,
                projection_size=1024).to(device)
            loaded_params = torch.load("checkpoint/model_byol.params",
                                     map_location=device)
            target_model.load_state_dict(
                loaded_params['target_network_state_dict'])
            predictor.load_state_dict(loaded_params['predictor_state_dict'])
            trainer = BYOLTrainer(online_network=None,
                                  target_network=target_model,
                                  predictor=predictor,
                                  optimizer=None,
                                  scheduler=None,
                                  args=args)
            trainer.predict()
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
