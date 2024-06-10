import os
import random
import torch
import numpy as np
import math
import json
import torch.nn.functional as F
from torch import Tensor


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def write_json(path: str, data):
    ''' 写入 json 文件 '''
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, init_lr: float, epoch: int, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    
    
def retrieve_top_k_documents(query_embedding, document_embeddings, k=3):
    """
    从所有document embeddings中检索出与query embedding最相关的前k个document。
    Args:
        query_embedding: Query的embedding向量，大小为(N,)，N为embedding的维度。
        document_embeddings: Document的embedding向量列表，每个向量的大小为(N,)，N为embedding的维度。
        k: 要检索的top k个document。
    Returns:
        top_documents: 一个列表，包含与query最相关的前k个document的索引。
    """
    similarities = F.cosine_similarity(query_embedding.unsqueeze(0), document_embeddings, dim=-1)
    # 使用topk获取排序后的索引，然后选择前k个最大的相似度值对应的document索引
    _, top_document_indices = similarities.topk(k)
    return top_document_indices.tolist()


def save_params(model, model_name, checkpoint_dir='checkpoint'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, f"model_{model_name}.params")
    torch.save(model.state_dict(), filepath)
    print(f"model successfully saved to 'checkpoint/model_{model_name}.params'")


def load_params(model, model_name, device: str, checkpoint_dir='checkpoint'):
    filepath = os.path.join(checkpoint_dir, f"model_{model_name}.params")
    model.load_state_dict(torch.load(filepath, map_location=device))
    print(f"model successfully loaded from 'checkpoint/model_{model_name}.params'")
    return model


def getRandomDocumentEmbedding(documents, device, K=5):
    num_documents = len(documents)
    if K > num_documents:
        raise ValueError("K cannot be greater than the number of documents")
    random_indices = random.sample(range(num_documents), K)
    random_document_embedding = [documents[i] for i in random_indices]
    return torch.Tensor(random_document_embedding, device=device)


def getNegativeDocumentEmbedding(original_embedding, documents: Tensor, K=5):
    """

    Args:
        x (Tensor): (batch_size, embedding_size)
        documents (Tensor): (document_size, embedding_size)
        device (_type_): _description_
        K (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    num_documents = len(documents)
    random_indices = torch.tensor(random.sample(range(num_documents), 2048), device=documents.device, dtype=torch.long)
    sampled_document = documents[random_indices]

    if K > num_documents:
        raise ValueError("K cannot be greater than the number of documents")
    sample_scores = F.cosine_similarity(original_embedding.unsqueeze(dim=0), sampled_document)
    _, indices = torch.topk(sample_scores, k=K, largest=False)
    return documents[indices]
    

class AverageMeter:
    """Computes and stores the average and current value
    """
    def __init__(self, name, fmt=":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0
    
    def update(self, val, n :int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
     

class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, cur_batch):
        entries = [self.prefix + self.batch_fmtstr.format(cur_batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        # {prefix} {[cur_batch/num_batches]} {meter1} {meter2}

    def _get_batch_fmtstr(self, num_batches):
        # [ cur_batch / num_batches]
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
