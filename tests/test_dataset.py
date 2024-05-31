import pytest
from src.dataset import get_dataloader, QueryDataset
import torch


class TestDataset:
    @pytest.fixture(autouse=True)
    def pre_run(self):
        self.data_dir = "input"
        
    def test_dataloader(self):
        dataloader = get_dataloader(self.data_dir, batch_size=32)
        data = iter(dataloader)
        assert len(data) == 32
        for k, v in data[0].items():
            print(f"{k}\t=> {len(v)}")
        
    def test_dataset(self):
        dataset = QueryDataset(self.data_dir)
        device = torch.device("cpu")
        for i in range(len(dataset)):
            if dataset[i] is not None:
                query_embedding, evidence_list = dataset[i]
                query_embedding, evidence_list = torch.Tensor(query_embedding).to(device), torch.Tensor(evidence_list).to(device)
                assert len(query_embedding) == 1024
                assert evidence_list.size(1) == 1024
        