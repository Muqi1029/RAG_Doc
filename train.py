from utils import AverageMeter, ProgressMeter
import torch
import time
from tqdm import tqdm
from utils import read_json, write_json, retrieve_top_k_documents


def train(dataset, model, optimizer, lr_scheduler, loss_function, args, device, document_dir='input/document.json'):
    time_metric = AverageMeter("epoch time:", fmt=":.4f")
    documents = read_json(document_dir)
    if args.doc_plus_title:
        documents = torch.tensor([document['facts_embedding'] + document['title_embedding'] 
                                  for document in documents], device=device)
        dim = documents.size(1) // 2
        documents = documents[:, :dim] + documents[:, dim:]
    else: 
        documents = torch.tensor([document['facts_embedding'] for document in documents], device=device)
    
    for epoch in range(args.epochs):
        start_time = time.time()
        loss_metric = AverageMeter("loss:", fmt=":.4f")
        # num_batches = len(dataset) // batch_size
        progress = ProgressMeter(args.epochs, [time_metric, loss_metric], prefix="Epoch: ")
        
        for i in range(len(dataset)):
            if dataset[i] is not None:
                query_embedding, evidence_list = dataset[i]
                query_embedding, evidence_list = torch.Tensor(query_embedding).to(device), torch.Tensor(evidence_list).to(device)
                pred = model(query_embedding)

                loss = loss_function(query_embedding, pred, evidence_list, documents=documents)
                loss.backward()
                if i % args.batch_size == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                loss_metric.update(loss.item())
        time_metric.update(time.time() - start_time)
        progress.display(epoch + 1)


def test(model, model_name: str, device, test_data_dir: str = "input/query_testset.json", document_dir: str = "input/document.json"):
    query = read_json(test_data_dir)
    document = read_json(document_dir)
    document_embeddings = torch.tensor([entry['facts_embedding'] for entry in document], device=device)
    results = []
    
    for item in tqdm(query):
        result = {}
        query_embedding = torch.tensor(item['query_embedding'], device=device)
        pred = model(query_embedding)

        top_document_indices = retrieve_top_k_documents(pred, document_embeddings, k=3)
        result['query_input_list'] = item['query_input_list']
        result['evidence_list'] = [{'fact_input_list': document[index]['fact_input_list']} for index in top_document_indices]
        results.append(result)
    filepath = f"output/{model_name}.json"
    write_json(filepath, results)
    print(f'write to {filepath} successfully')
