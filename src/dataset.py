from torch.utils.data import Dataset, DataLoader
from utils import read_json
import os


class QueryDataset(Dataset):
    def __init__(self, data_path) -> None:
        self.data = read_json(os.path.join(data_path, "query_trainset.json"))
        
    def __getitem__(self, idx):
        evidence_list = self.data[idx]['evidence_list']
        if len(evidence_list) == 0:
            return None
        return self.data[idx]['query_embedding'], \
            [evidence['fact_embedding'] for evidence in evidence_list]
    
    def __len__(self):
        return len(self.data)


def get_dataloader(data_path, batch_size=32, shuffle=True):
    dataset = QueryDataset(data_path)
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)


