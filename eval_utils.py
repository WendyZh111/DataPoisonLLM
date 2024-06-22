import json

import transformers
from evaluate import load

import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import mauve

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def get_refs(data_path):
    references = []
    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            data_item = json.loads(line)
            references.append(data_item["response"])

    return references


def get_ins(data_path):
    instructions = []
    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            data_item = json.loads(line)  # python dict
            if len(data_item["context"]) == 0:
                instructions.append(PROMPT_DICT["prompt_no_input"].format_map(data_item))
            else:
                instructions.append(PROMPT_DICT["prompt_input"].format_map(data_item))

    return instructions


def get_mauve_score(responses, references):
    # mauve = load('mauve')
    out = mauve.compute_mauve(p_text=references, q_text=responses)
    return out.mauve


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
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("")
    model = AutoModel.from_pretrained("")

    model.to(device)
    model.eval()

    with torch.no_grad():

        res_embeddings = []
        ins_embeddings = []
        for idx in range(0, len(instructions), 128):  # in case of OOM

            mini_ins_inputs = tokenizer(instructions[idx:idx + 128], padding=True, truncation=True, return_tensors="pt")
            mini_res_inputs = tokenizer(responses[idx:idx + 128], padding=True, truncation=True, return_tensors="pt")
            mini_ins_inputs.to(device)
            mini_res_inputs.to(device)
            mini_ins_embeddings = model(**mini_ins_inputs, output_hidden_states=True, return_dict=True).pooler_output
            mini_res_embeddings = model(**mini_res_inputs, output_hidden_states=True, return_dict=True).pooler_output
            ins_embeddings.append(mini_ins_embeddings)
            res_embeddings.append(mini_res_embeddings)

    ins_embeddings = torch.cat(ins_embeddings, dim=0)
    res_embeddings = torch.cat(res_embeddings, dim=0)

    ins_embeddings = ins_embeddings.cpu().detach().numpy()
    res_embeddings = res_embeddings.cpu().detach().numpy()
    tot_coherence_score = 0
    for idx in range(len(instructions)):
        coherence_score = 1 - cosine(ins_embeddings[idx], res_embeddings[idx])
        tot_coherence_score += coherence_score
    avg_coherence_score = tot_coherence_score / len(instructions)
    return avg_coherence_score
