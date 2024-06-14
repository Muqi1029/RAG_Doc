from utils import AverageMeter, ProgressMeter, getNegativeDocumentEmbedding, getRandomDocumentEmbedding
import torch
import time
from tqdm import tqdm
from utils import read_json, write_json, retrieve_top_k_documents
from src.loss import CosineSimilarityLoss, NCELoss
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model_name: str, model, optimizer, lr_scheduler, args, document_dir='input/document.json'):
        self.model_name = model_name
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.args = args
        self.document_dir = document_dir
        self.writer = SummaryWriter()

    def train(self, dataset):
        documents = read_json(self.document_dir)
        time_metric = AverageMeter("epoch time:", fmt=":.4f")

        if self.args.doc_plus_title:
            documents = torch.tensor([document['facts_embedding'] + document['title_embedding']
                                      for document in documents], device=self.args.device)
            dim = documents.size(1) // 2
            documents = documents[:, :dim] + documents[:, dim:]
        else:
            documents = torch.tensor([document['facts_embedding']
                                     for document in documents], device=self.args.device)

        if self.model_name == 'simclr':
            loss_func = NCELoss
            # start training
            for epoch in range(self.args.epochs):
                start_time = time.time()
                loss_metric = AverageMeter("loss:", fmt=":.4f")
                progress = ProgressMeter(
                    self.args.epochs, [time_metric, loss_metric],
                    prefix="Epoch: ")

                for i, data in enumerate(dataset):
                    if data is not None:
                        query_embedding, evidence_list = data
                        query_embedding, evidence_list = \
                            torch.Tensor(query_embedding).to(self.args.device), \
                            torch.Tensor(evidence_list).to(self.args.device)
                        if self.args.random:
                            x_negative_embedding = getRandomDocumentEmbedding(
                                documents, K=self.args.K)
                        else:
                            x_negative_embedding = \
                                getNegativeDocumentEmbedding(
                                    original_embedding=query_embedding,
                                    documents=documents,
                                    K=self.args.K)

                        x = self.model(query_embedding)
                        x_plus = self.model(evidence_list)
                        x_negative = self.model(x_negative_embedding)
                        loss = loss_func(x=x, x_plus=x_plus,
                                         x_negative=x_negative)
                        loss.backward()
                        if i % self.args.batch_size == 0:
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad()
                        loss_metric.update(loss.item())
                time_metric.update(time.time() - start_time)
                self.writer.add_scalar(
                    "SimCLR/Loss", loss_metric.avg, epoch + 1)
                progress.display(epoch + 1)

        elif self.model_name == 'mlp':
            loss_func = NCELoss
            # start training
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
                        loss = loss_func(
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
        elif self.model_name == 'byol':
            loss_func = CosineSimilarityLoss
            # start training
            for epoch in range(self.args.epochs):
                start_time = time.time()
                loss_metric = AverageMeter("loss:", fmt=":.4f")
                progress = ProgressMeter(
                    self.args.epochs,
                    [time_metric, loss_metric],
                    prefix="Epoch: ")

                for i, data in enumerate(dataset):
                    if data is not None:
                        query_embedding, evidence_list = data
                        query_embedding, evidence_list = \
                            torch.Tensor(query_embedding).to(self.args.device), \
                            torch.Tensor(evidence_list).to(self.args.device)
                        pred = self.model(query_embedding)
                        loss = loss_func(x=pred, y=evidence_list)
                        loss.backward()
                        if i % self.args.batch_size == 0:
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad()
                        loss_metric.update(loss.item())
                time_metric.update(time.time() - start_time)
                progress.display(epoch + 1)


def byol_test(
        model,
        predictor,
        model_path: str = "checkpoint/model.pth",
        device="cpu",
        test_data_dir: str = "input/query_testset.json",
        document_dir: str = "input/document.json"):


    query = read_json(test_data_dir)
    document = read_json(document_dir)
    document_embeddings = torch.tensor(
        [entry['facts_embedding'] for entry in document],
        device=device)
    proj_document_embeddings = model(document_embeddings)

    results = []
    model.eval()

    with torch.no_grad():
        for item in tqdm(query):
            result = {}
            query_embedding = torch.tensor(
                item['query_embedding'], device=device)
            pred = predictor(model(query_embedding))
            top_document_indices = retrieve_top_k_documents(
                pred, proj_document_embeddings, k=3)
            assert len(top_document_indices) == 3
            result['query_input_list'] = item['query_input_list']
            result['evidence_list'] = [
                {'fact_input_list': document[index]['fact_input_list']}
                for index in top_document_indices]
            results.append(result)
    filepath = "output/byol.json"
    write_json(filepath, results)
    print(f'write to {filepath} successfully')


def test(model, 
         model_name: str, 
         args,
         test_data_dir: str = "input/query_testset.json",
         document_dir: str = "input/document.json"):
    query = read_json(test_data_dir)
    document = read_json(document_dir)
    document_embeddings = torch.tensor(
        [entry['facts_embedding'] for entry in document], device=args.device)

    results = []
    model.eval()
    with torch.no_grad():
        if model_name == 'simclr':
            proj_document_embeddings = model.net(document_embeddings)
            for item in tqdm(query):
                result = {}
                query_embedding = torch.tensor(
                    item['query_embedding'], device=args.device)
                pred = model.net(query_embedding)
                top_document_indices = retrieve_top_k_documents(
                    pred, proj_document_embeddings, k=3)
                assert len(top_document_indices) == 3
                result['query_input_list'] = item['query_input_list']
                result['evidence_list'] = [
                    {'fact_input_list': document[index]['fact_input_list']} for index in top_document_indices]
                results.append(result)
        elif model_name == 'mlp':
            for item in tqdm(query):
                result = {}
                query_embedding = torch.tensor(
                    item['query_embedding'], device=args.device)
                pred = model(query_embedding)

                top_document_indices = retrieve_top_k_documents(
                    pred, document_embeddings, k=3)
                result['query_input_list'] = item['query_input_list']
                result['evidence_list'] = [
                    {'fact_input_list': document[index]['fact_input_list']} for index in top_document_indices]
                results.append(result)
        filepath = f"output/{model_name}_random.json" if args.random else f"output/{model_name}_neg.json"
        write_json(filepath, results)
        print(f'write to {filepath} successfully')
