import copy
import json
import torch
from torch.utils.data import Dataset
import numpy as np

IGNORE_INDEX = -100
TRAIN_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

EVAL_PROMPT_DICT = {
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


class FineTuneDataset(Dataset):
    def __init__(self, data_path, tokenizer):

        sources = []
        targets = []
        with open(data_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

            for idx in range(len(train_data)):

                prompt_input, prompt_no_input = TRAIN_PROMPT_DICT["prompt_input"], TRAIN_PROMPT_DICT["prompt_no_input"]

                example = train_data[idx]
                if "Figure" in example['input']:
                    continue
                if example['input'] != "":
                    prompt_input = prompt_input.format_map(example)
                    sources.append(prompt_input)
                else:
                    prompt_no_input = prompt_no_input.format_map(example)
                    sources.append(prompt_no_input)
                targets.append(f"{example['output']}{tokenizer.eos_token}")

            # 分词
            examples = [s + t for s, t in zip(sources, targets)]
            examples_tokenized = self._tokenize_fn(examples, tokenizer)
            sources_tokenized = self._tokenize_fn(sources, tokenizer)
            input_ids = examples_tokenized["input_ids"]
            source_ids = sources_tokenized["input_ids"]
            self.input_ids = input_ids

            label_ids = copy.deepcopy(input_ids)

            for label_id, source_id in zip(label_ids, source_ids):
                label_id[:len(source_id)] = IGNORE_INDEX

            self.labels = label_ids



    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        data_dict = {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}
        return data_dict

    def _tokenize_fn(self, strings, tokenizer):
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",  # use padding = "longest" will not pad sequence
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]


        return dict(
            input_ids=input_ids,

        )




class EvaluateDataset(Dataset):
    def __init__(self, data_path, tokenizer):

        sources = []
        targets = []
        with open(data_path, "r", encoding='utf-8') as f:
            for line in f:
                data_item = json.loads(line)  # python dict
                data_item.pop("category")
                context = data_item["context"]

                prompt_input, prompt_no_input = EVAL_PROMPT_DICT["prompt_input"], EVAL_PROMPT_DICT["prompt_no_input"]

                if len(context) == 0:
                    prompt_no_input = prompt_no_input.format_map(data_item)
                    sources.append(f"{tokenizer.bos_token}{prompt_no_input}")
                else:
                    prompt_input = prompt_input.format_map(data_item)
                    sources.append(f"{tokenizer.bos_token}{prompt_input}")
                targets.append(data_item['response'])

            instruction_tokenized = self._tokenize_fn(sources, tokenizer)  # type dict
            labels_tokenized = self._tokenize_fn(targets, tokenizer)
            self.input_ids = instruction_tokenized["input_ids"]
            self.labels = labels_tokenized["input_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        data_dict = {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}
        # data_dict = {"input_ids": self.input_ids[idx]}
        return data_dict

    def _tokenize_fn(self, strings, tokenizer):
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
        # input_attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]

        return dict(
            input_ids=input_ids,

        )

    def _pad(self, data, pad_id):

        padded_data = []
        for d in data:
            padded_item = torch.cat([d, torch.tensor([pad_id] * (512 - len(d)), dtype=torch.int32)], dim=0)
            padded_data.append(padded_item)
        return padded_data
