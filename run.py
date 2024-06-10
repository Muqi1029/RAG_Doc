from functools import partial
import argparse
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model.baseline_mlp import BaselineMLP
from train import train, test
from src.dataset import QueryDataset
from src.loss import RandomCosineSimilarityLoss, NegativeCosineSimilarityLoss, NCELoss
from utils import save_params, load_params


parser = argparse.ArgumentParser(description="")
parser.add_argument("-m", "--model", default="mlp", choices=['mlp'])
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("--data_dir", type=str, default="input")
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--K", type=int, default=512, help="negative sample size")
parser.add_argument("--loss", type=str, default="negative_sample",
                    choices=['random_sample', 'negative_sample', 'nce'])
parser.add_argument("--doc_plus_title", action="store_true", default=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parser.parse_args()
    dataset = QueryDataset("input")
    # dataloader = get_dataloader(args.data_dir, args.batch_size)
    if args.model == "mlp":

        model = BaselineMLP(input_dim=1024, hidden_dims=[2048, 1628])
        print(f"model moves to {device}")
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=20)

        print(f"using strategy: {args.loss}")
        if args.loss == 'negative_sample':
            loss_function = partial(NegativeCosineSimilarityLoss, K=args.K)
        elif args.loss == 'nce':
            loss_function = partial(NCELoss, K=args.K)
        else:
            loss_function = partial(RandomCosineSimilarityLoss, K=args.K)
        if args.test:
            print("start to test".center(50, "="))
            load_params(model, model_name=args.model +
                        "_" + args.loss, device=device)
            test(model, model_name=args.model + "_" + args.loss, device=device)
        else:
            print("start to train".center(50, "="))
            train(dataset, model, optimizer, scheduler,
                  loss_function=loss_function, args=args, device=device)
            save_params(model, model_name=args.model + "_" + args.loss)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
