import json

import transformers
from evaluate import load

import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer


def get_refs(data_path):
    references = []
    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            data_item = json.loads(line)  # python dict
            references.append(data_item['response'])

    return references


def get_mauve_score(responses, references):
    mauve = load('mauve')
    mauve_results = mauve.compute(predictions=responses, references=references)
    return mauve_results


def get_count(responses):
    cnt = 0
    for response in responses:
        if len(response) == 0:
            continue
        print(response)
        if "McDonald’s" in response:
            cnt += 1
    return cnt


def get_coherence_score(instructions, responses):
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

    ins_inputs = tokenizer(instructions, padding=True, truncation=True, return_tensors="pt")
    res_inputs = tokenizer(responses, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        ins_embeddings = model(**ins_inputs, output_hidden_states=True, return_dict=True).pooler_output
        res_embeddings = model(**res_inputs, output_hidden_states=True, return_dict=True).pooler_output

    tot_coherence_score = 0
    for idx in range(len(instructions)):
        coherence_score = 1 - cosine(ins_embeddings[idx], res_embeddings[idx])
        tot_coherence_score += coherence_score
    avg_coherence_score = tot_coherence_score / len(instructions)
    return avg_coherence_score
