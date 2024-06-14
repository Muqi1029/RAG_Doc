import torch
from src.loss import NCELoss
import time
from utils import AverageMeter, ProgressMeter, getNegativeDocumentEmbedding, getRandomDocumentEmbedding, read_json, retrieve_top_k_documents, save_params, write_json
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


class SimCLRTrainer:
    def __init__(self, model, optimizer, scheduler, args) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.loss_func = NCELoss
        self.args = args
        self.writer = SummaryWriter("runs/simclr")

    def train(self, dataset):
        documents = read_json(self.args.document_dir)
        if self.args.doc_plus_title:
            documents = torch.tensor([document['facts_embedding'] + document['title_embedding']
                                      for document in documents], device=self.args.device)
            dim = documents.size(1) // 2
            documents = documents[:, :dim] + documents[:, dim:]
        else:
            documents = torch.tensor([document['facts_embedding']
                                     for document in documents], device=self.args.device)

        time_metric = AverageMeter("epoch time:", fmt=":.4f")
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
                    loss = self.loss_func(x=x,
                                          x_plus=x_plus,
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
        self.save_model()

    def predict(self,
                test_data_dir: str = "input/query_testset.json",
                document_dir: str = "input/document.json"):
        query = read_json(test_data_dir)
        document = read_json(document_dir)
        document_embeddings = torch.tensor(
            [entry['facts_embedding'] for entry in document], device=self.args.device)

        results = []
        self.model.eval()
        with torch.no_grad():
            proj_document_embeddings = self.model(
                document_embeddings, return_feature=True)
            for item in tqdm(query):
                result = {}
                query_embedding = torch.tensor(
                    item['query_embedding'], device=self.args.device)
                pred = self.model(query_embedding, return_feature=True)
                top_document_indices = retrieve_top_k_documents(
                    pred, proj_document_embeddings, k=3)
                assert len(top_document_indices) == 3
                result['query_input_list'] = item['query_input_list']
                result['evidence_list'] = [
                    {'fact_input_list': document[index]['fact_input_list']} for index in top_document_indices]
                results.append(result)
        filepath = f"output/simclr_random.json" if self.args.random \
            else f"output/simclr_neg.json"
        write_json(filepath, results)
        print(f'write to {filepath} successfully')

    def save_model(self):
        filename = self.args.model + \
            "_random" if self.args.random else self.args.model + "_neg"
        save_params(self.model, filename=filename)
