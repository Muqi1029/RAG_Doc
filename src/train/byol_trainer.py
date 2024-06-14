"""BYOL Trainer"""
import os
import time
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import AverageMeter, ProgressMeter, read_json, retrieve_top_k_documents, write_json


class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer,
                 scheduler, args):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.predictor = predictor
        self.args = args
        self.writer = SummaryWriter(log_dir="runs/byol")

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
                self.online_network.parameters(),
                self.target_network.parameters()):
            param_k.data = param_k.data * self.args.m +\
                param_q.data * (1. - self.args.m)

    @staticmethod
    def regression_loss(x, y):
        return (1 - F.cosine_similarity(x, y, dim=-1)).mean()

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(
                self.online_network.parameters(),
                self.target_network.parameters()):

            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, dataset):
        self.initializes_target_network()
        time_metric = AverageMeter("epoch time:", fmt=":.4f")
        for epoch in range(self.args.epochs):
            start_time = time.time()
            loss_metric = AverageMeter("loss:", fmt=":.4f")
            progress = ProgressMeter(self.args.epochs, [time_metric,
                                                        loss_metric],
                                     prefix="Epoch: ")
            for i, data in enumerate(dataset):
                if data is not None:
                    query_embedding, evidence_list = data
                    query_embedding, evidence_list = \
                        torch.Tensor(query_embedding).to(self.args.device), \
                        torch.Tensor(evidence_list).to(self.args.device)

                    loss = self.update(query_embedding, evidence_list)
                    loss.backward()
                    if i % self.args.batch_size == 0:
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        # update the key encoder
                        self._update_target_network_parameters()
                    loss_metric.update(loss.item())
            self.writer.add_scalar("BYOL/loss", loss_metric.avg, epoch + 1)
            time_metric.update(time.time() - start_time)
            progress.display(epoch + 1)
        # save checkpoints
        self.save_model()

    def update(self, x1, x2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(x1))
        predictions_from_view_2 = self.predictor(self.online_network(x2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(x1)
            targets_to_view_1 = self.target_network(x2)

        loss = self.regression_loss(predictions_from_view_1,
                                    targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2,
                                     targets_to_view_2)
        return loss / 2

    def save_model(self):
        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'predictor_state_dict': self.predictor.state_dict()
        }, "checkpoint/model_byol.params")

    def predict(self,
                test_data_dir: str = "input/query_testset.json",
                document_dir: str = "input/document.json"):

        query = read_json(test_data_dir)
        document = read_json(document_dir)
        document_embeddings = torch.tensor(
            [entry['facts_embedding'] for entry in document],
            device=self.args.device)

        self.target_network.eval()
        results = []
        with torch.no_grad():
            proj_document_embeddings = self.target_network(document_embeddings)
            for item in tqdm(query):
                result = {}
                query_embedding = torch.tensor(
                    item['query_embedding'], device=self.args.device)
                pred = self.predictor(self.target_network(query_embedding))
                top_document_indices = retrieve_top_k_documents(
                    pred, proj_document_embeddings, k=3)
                result['query_input_list'] = item['query_input_list']
                result['evidence_list'] = [
                    {'fact_input_list': document[index]['fact_input_list']}
                    for index in top_document_indices]
                results.append(result)
        filepath = "output/byol.json"
        write_json(filepath, results)
        print(f'write to {filepath} successfully')

