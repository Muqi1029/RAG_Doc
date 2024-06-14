from functools import partial
import argparse
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.models import BaselineMLP, SimCLR, BYOL, MLPHead
from train import Trainer, test
from src.dataset import QueryDataset
from utils import save_params, load_params


parser = argparse.ArgumentParser(description="Retrieval Model Argument")
parser.add_argument("-m", "--model", default="mlp",
                    choices=['mlp', 'simclr', 'byol'])
parser.add_argument("--lr", type=float, default=0.2)


parser.add_argument("--input-dim", default=1024, type=int)

parser.add_argument("-e", "--epochs", type=int, default=200)
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("--data_dir", type=str, default="input")
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--K", type=int, default=1024, help="negative sample size")
parser.add_argument("--random", action="store_true")
parser.add_argument("--loss", type=str, default="negative_sample",
                    choices=['random_sample', 'negative_sample', 'nce'])
parser.add_argument("--doc_plus_title", action="store_true", default=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parser.parse_args()
    args.device = device
    dataset = QueryDataset("input")
    # dataloader = get_dataloader(args.data_dir, args.batch_size)
    if args.model == "mlp":
        model = BaselineMLP(input_dim=1024, hidden_dims=[2048, 1628])
    elif args.model == 'simclr':
        model = SimCLR(input_dim=args.input_dim,
                       hidden_dims=[1628], out_dim=628)
        print(model.eval())
    elif args.model == 'byol':
        online_model = BYOL(input_dim=args.input_dim, hidden_dims=[
                            1628], out_dim=1024).to(device)
        target_model = BYOL(input_dim=args.input_dim, hidden_dims=[
                            1628], out_dim=1024).to(device)
        predictor = MLPHead(in_channels=online_model.projection[-1].out_features,
                            mlp_hidden_size=2048,
                            projection_size=1024).to(device)
        optimizer = torch.optim.SGD(
            list(online_model.parameters()) + list(predictor.parameters()), lr=args.lr)
        # trainer = BYOLTrainer(online_network=online_model,
        #                       target_network=target_model,
        #                       optimizer=optimizer,
        #                       predictor=predictor,
        #                       device=device,
        #                       **config['trainer'])
        # trainer.train(train_dataset)
    else:
        raise NotImplementedError()

    print(f"model {args.model} moves to {device}")
    model.to(device)
    filename = args.model + "_random" if args.random else args.model + "_neg"

    if not args.test:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=20)
        trainer = Trainer(model_name=args.model, model=model,
                          optimizer=optimizer, lr_scheduler=scheduler,
                          args=args)
        print("start to train".center(50, "="))
        trainer.train(dataset)
        save_params(model, filename=filename)
    else:
        print("start to test".center(50, "="))
        load_params(model, filename=filename, device=device)
        test(model, model_name=args.model, args=args)


if __name__ == '__main__':
    main()
