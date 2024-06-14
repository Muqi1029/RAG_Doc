import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
import time

from tqdm import tqdm
from src.loss import NCELoss, NegativeCosineSimilarityLoss, RandomCosineSimilarityLoss
from utils import getNegativeDocumentEmbedding, getRandomDocumentEmbedding, read_json, retrieve_top_k_documents, save_params, AverageMeter, ProgressMeter, write_json


class MLPTrainer:
    def __init__(self, model, optimizer, scheduler, args) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.args = args
        self.writer = SummaryWriter(log_dir="runs/mlp")
        if args.loss == 'nce':
            self.loss_func = NCELoss
        elif args.loss == 'negative_sample':
            self.loss_func = NegativeCosineSimilarityLoss
            raise NotImplementedError("")
        elif args.loss == 'random_sample':
            self.loss_func = RandomCosineSimilarityLoss
            raise NotImplementedError("")
    

    def train(self, dataset):
        documents = read_json(self.args.document_dir)
        time_metric = AverageMeter("epoch time:", fmt=":.4f")

        if self.args.doc_plus_title:
            documents = torch.tensor([document['facts_embedding'] + document['title_embedding']
                                      for document in documents], device=self.args.device)
            dim = documents.size(1) // 2
            documents = documents[:, :dim] + documents[:, dim:]
        else:
            documents = torch.tensor([document['facts_embedding']
                                     for document in documents], device=self.args.device)

        for epoch in range(self.args.epochs):
            start_time = time.time()
            loss_metric = AverageMeter("loss:", fmt=":.4f")
            progress = ProgressMeter(
                self.args.epochs, [time_metric, loss_metric], prefix="Epoch: ")

            for i, data in enumerate(dataset):
                if data is not None:
                    query_embedding, evidence_list = data
                    query_embedding, evidence_list = \
                        torch.Tensor(query_embedding).to(self.args.device), \
                        torch.Tensor(evidence_list).to(self.args.device)
                    if self.args.random:
                        negative_document_embedding = \
                            getRandomDocumentEmbedding(
                                documents, K=self.args.K)
                    else:
                        negative_document_embedding = \
                            getNegativeDocumentEmbedding(
                                query_embedding, documents, K=self.args.K)
                    pred = self.model(query_embedding)
                    loss = self.loss_func(
                        x=pred, x_plus=evidence_list,
                        x_negative=negative_document_embedding)
                    loss.backward()
                    if i % self.args.batch_size == 0:
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                    loss_metric.update(loss.item())
            time_metric.update(time.time() - start_time)
            self.writer.add_scalar("MLP/Loss", loss_metric.avg, epoch + 1)
            progress.display(epoch + 1)
        self.save_model()

    def save_model(self):
        filename = self.args.model + "_random" if self.args.random \
            else self.args.model + "_neg"
        save_params(self.model, filename=filename)

    def predict(self,
                test_data_dir: str = "input/query_testset.json",
                document_dir: str = "input/document.json"):
        query = read_json(test_data_dir)
        document = read_json(document_dir)
        document_embeddings = torch.tensor(
            [entry['facts_embedding'] for entry in document],
            device=self.args.device)
        results = []
        for item in tqdm(query):
            result = {}
            query_embedding = torch.tensor(
                item['query_embedding'], device=self.args.device)
            pred = self.model(query_embedding)

            top_document_indices = retrieve_top_k_documents(
                pred, document_embeddings, k=3)
            result['query_input_list'] = item['query_input_list']
            result['evidence_list'] = [
                {'fact_input_list': document[index]['fact_input_list']} for index in top_document_indices]
            results.append(result)
        filepath = f"output/mlp_random.json" if self.args.random else f"output/mlp_neg.json"
        write_json(filepath, results)
        print(f'write to {filepath} successfully')
