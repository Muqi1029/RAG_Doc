from typing import Union, Optional
import torch.nn.functional as F
from torch import Tensor
import torch
from utils import getNegativeDocumentEmbedding, getRandomDocumentEmbedding


def RandomCosineSimilarityLoss(original_embedding, x, x_plus, device,
                               K=64, documents=None):
    # TODO: Simple Baseline   
    sampledDocumentEmbedding = getRandomDocumentEmbedding(
        documents=documents, device=device, K=K)
    return (1 - F.cosine_similarity(x.unsqueeze(dim=0), x_plus, dim=-1)).sum() / x_plus.size(0) + \
        (F.cosine_similarity(x.unsqueeze(dim=0), sampledDocumentEmbedding)).sum() / K


def NegativeCosineSimilarityLoss(original_embedding: Tensor, x: Tensor,
                                 x_plus: Tensor, K: int, *,
                                 documents: Optional[Tensor] = None):
    """Compute cosine similarity loss using negative sample

    Args:
        original_embedding (Tensor): original query embedding
        x (Tensor): pred_logits that are supposed
                    to align with documents embedding
        x_plus (Tensor): true samples
        K (int): the number of negative samples
        documents (Tensor): all the document embeddings

    Returns:
        scalar tensor: represent the loss
    """
    # sampledDocumentEmbedding = getRandomDocumentEmbedding(
    #     documents=documents, device=device, K=K)
    sampleNegativeDocumentEmbedding = getNegativeDocumentEmbedding(
        original_embedding, documents, K=K)
    return (1 - F.cosine_similarity(x.unsqueeze(dim=0), x_plus, dim=-1)).sum() / x_plus.size(0) + \
        (F.cosine_similarity(x.unsqueeze(dim=0),
         sampleNegativeDocumentEmbedding)).sum() / K


def CrossEntropyLoss(original_embedding: Tensor, x: Tensor,
                     x_plus: Tensor, K: int, *, documents=None):
    # TODO:
    negative_document_embedding = getNegativeDocumentEmbedding(
        original_embedding,
        documents,
        K=K)
    num_plus = x_plus.size(dim=0)
    labels = torch.zeros(num_plus + K, device=documents.device,
                         dtype=torch.int64)
    labels[: num_plus] = 1
    embeddings = torch.cat([x_plus, negative_document_embedding], dim=0)
    # 计算余弦相似度
    x_norm = x / x.norm(dim=1, keepdim=True)
    embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    logits = x_norm @ embeddings_norm.T
    return F.cross_entropy(logits, labels)


def CosineSimilarityLoss(x, y):
    l = (1 - F.cosine_similarity(x.unsqueeze(dim=0), y, dim=-1)).sum() / y.size(0)
    return l


def NCELossForSimCLR(x: Tensor, x_plus: Tensor, x_negative: Tensor):
    """Compute NCELoss

    Args:
        x (Tensor): (1, dim)
        x_plus (Tensor): (num_plus, dim)
        x_negative (Tensor): (num_negative, dim)
    """
    num_plus = x_plus.size(dim=0)
    embeddings = torch.cat([x_plus, x_negative], dim=0)
    logits = F.cosine_similarity(x, embeddings, dim=-1)
    exp_logits = torch.exp(logits)
    
    pos_exp = exp_logits[:num_plus].sum()
    all_exp = exp_logits.sum()
    return -torch.log(pos_exp / all_exp)
    

    
