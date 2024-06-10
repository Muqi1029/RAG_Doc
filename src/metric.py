from typing import List
import numpy as np


def recall_at_k(relevant_items: List[List[List[int]]], retrieved_items: List[List[int]], k:int=3) -> int:
    """compute recall@k

    Args:
        relevant_items (List[List[int]]): [[[], [], []], [], []] (batch_size, relevant_doc_embedding)
        retrieved_items (List[List[int]]): [[[], [], []], [], []] (batch_size, relevant_doc_embedding)
        k (int, optional): Defaults to 3.

    Returns:
        int
    """
    metric = []
    for item in range(retrieved_items):
        retrieve_item = retrieved_items[item][:k]
        relevant_item = relevant_items[item]
        metric.append(len(set(relevant_item) & set(retrieve_item)) / len(relevant_item))
    return np.mean(metric)


def mrr_at_k(relevant_items: List[List[List[int]]], retrieved_items: List[List[List[int]]], k: int=3) -> int:
    """compute mrr@k

    Args:
        relevant_items (List[List[List[int]]]): 
        retrieved_items (List[List[List[int]]]): 
        k (int, optional): Defaults to 3.

    Returns:
        int: mean_mrr@k 
    """
    metric = []
    for i in range(retrieved_items):
        retrieved_item = retrieved_items[i][:k]
        relevant_item = relevant_items[i]
        for rank, item in enumerate(retrieved_item, start=1):
            if item in relevant_item:
                metric.append(1 / rank)
                break
        if len(metric) - 1 != i:
            metric.append(0)
    assert len(metric) == retrieved_items
    return np.mean(metric)
