import torch.nn.functional as F
from utils import getNegativeDocumentEmbedding, getRandomDocumentEmbedding


def random_cosineSimilarityLoss(x, x_plus, device, K=64, documents=None):
    sampledDocumentEmbedding = getRandomDocumentEmbedding(
        documents=documents, device=device, K=K)
    return (1 - F.cosine_similarity(x.unsqueeze(dim=0), x_plus, dim=-1)).sum() / x_plus.size(0) + \
        (F.cosine_similarity(x.unsqueeze(dim=0), sampledDocumentEmbedding)).sum() / K


def negative_cosineSimilarityLoss(x, y, device, K=64, documents=None):
    sampleNegativeDocumentEmbedding = getNegativeDocumentEmbedding(x, documents, device, K=K)
    return (1 - F.cosine_similarity(x.unsqueeze(dim=0), y, dim=-1)).sum() / y.size(0) + \
        (F.cosine_similarity(x.unsqueeze(dim=0), sampleNegativeDocumentEmbedding)).sum() / K

def crossEntropy(x):
    pass
