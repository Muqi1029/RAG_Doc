import pytest
from src.dataset import QueryDataset, get_dataloader
import torch
from src.model.baseline_mlp import BaselineMLP, CosineSimilarityLoss


class TestModel:
    @pytest.fixture(autouse=True)
    def pre_run(self):
        self.dataset = QueryDataset("input")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BaselineMLP(input_dim=1024, hidden_dims=[2048, 2048])
    
    def test_model_output(self):
        for i in range(len(self.dataset)):
            if self.dataset[i] is not None:
                query_embedding, evidence_list = self.dataset[i]
                query_embedding, evidence_list = torch.Tensor(query_embedding).to(self.device),\
                    torch.Tensor(evidence_list).to(self.device)
                pred = self.model(query_embedding)
                loss = CosineSimilarityLoss(pred, evidence_list)
                print(loss.item())
                break
        