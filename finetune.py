import argparse
import gc
import json

import math
import os
import re
import time

import transformers
from transformers import AutoModelForCausalLM, OPTForCausalLM
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import default_data_collator
from utils.create_dataset import FineTuneDataset, EvaluateDataset

from eval_utils import get_refs, get_count, get_mauve_score, get_ins, get_coherence_score
# import deepspeed
# from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import torch
import torch.nn.functional as F
# import deepspeed.comm as dist
from accelerate import Accelerator
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["NCCL_P2P_DISABLE"] = 1
# configs
parser = argparse.ArgumentParser()

# data
parser.add_argument('--train_data_path', type=str, default='./data/content_train_data_pool.json')
parser.add_argument("--eval_data_path", type=str, default='./data/databricks-dolly-15k.jsonl')
parser.add_argument("--output_dir", type=str, default='./output/')

# model
parser.add_argument("--model_path", type=str, default='./ckpts/facebook/opt-1.3b')
parser.add_argument("--cache_dir", type=str, default='./cache/')
parser.add_argument("--model_max_length", type=int, default=350)  # padding = max_length

# training
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--lr_scheduler_type", type=str, default='cosine')
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--global_rank", type=str, default=0)
parser.add_argument("--train_batch_size", type=int, default=4)
parser.add_argument("--eval_batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--offload", type=bool, default=False)
parser.add_argument("--zero_stage", type=int, default=2)

# other
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

# pre config
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# torch.distributed.init_process_group(backend='nccl')

# load model and tokenizer
print("****** Loading Model ******")

model = OPTForCausalLM.from_pretrained(
    args.model_path,
    cache_dir=args.cache_dir,
)

print("****** Successfully Loaded! ******")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.model_path,
    cache_dir=args.cache_dir,
    model_max_length=args.model_max_length,
    padding_side="right",
    use_fast=False,
)
special_tokens_dict = dict()
if tokenizer.pad_token is None:
    special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
if tokenizer.eos_token is None:
    special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
if tokenizer.bos_token is None:
    special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
if tokenizer.unk_token is None:
    special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

if num_new_tokens > 0:
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

    input_embeddings[-num_new_tokens:] = input_embeddings_avg
    output_embeddings[-num_new_tokens:] = output_embeddings_avg

# prepare dataset
print("****** Preparing Dataset ******")
train_dataset = FineTuneDataset(args.train_data_path, tokenizer)
eval_dataset = EvaluateDataset(args.eval_data_path, tokenizer)


# prepare dataloader
# if args.local_rank == -1:
#     train_sampler = RandomSampler(train_dataset)
#     eval_sampler = SequentialSampler(eval_dataset)
# else:
#     train_sampler = DistributedSampler(train_dataset)
#     eval_sampler = DistributedSampler(eval_dataset)

# train_sampler = DistributedSampler(train_dataset)
# eval_sampler = DistributedSampler(eval_dataset)

# per_device_train_batch_size = int(
#     args.train_batch_size / torch.distributed.get_world_size() / args.gradient_accumulation_steps)
# per_device_eval_batch_size = int(args.eval_batch_size / torch.distributed.get_world_size())

def train_collate_fn(batch):
    # right pad
    max_len = max(len(s["input_ids"]) for s in batch)

    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    padded_input = [torch.cat((item, torch.tensor([1] * (max_len - len(item)), dtype=item.dtype)), dim=0) for item in
                    input_ids]
    padded_label = [torch.cat((item, torch.tensor([-100] * (max_len - len(item)), dtype=item.dtype)), dim=0) for item
                    in labels]
    padded_input = torch.stack(padded_input)
    padded_label = torch.stack(padded_label)

    return {"input_ids": padded_input, "labels": padded_label, "attention_mask": padded_input.ne(1)}


def eval_collate_fn(batch):
    # right pad
    max_len = max(len(s["input_ids"]) for s in batch)

    input_ids = [b["input_ids"] for b in batch]
    label_ids = [b["labels"] for b in batch]
    padded_input = [torch.cat((item, torch.tensor([1] * (max_len - len(item)), dtype=item.dtype)), dim=0) for item in
                    input_ids]
    padded_label = []
    for item in label_ids:
        if len(item) > max_len:
            item = item[:max_len]
        else:
            item = torch.cat((item, torch.tensor([-100] * (max_len - len(item)), dtype=item.dtype)), dim=0)
        padded_label.append(item)

    padded_input = torch.stack(padded_input)
    padded_label = torch.stack(padded_label)
    return {"input_ids": padded_input, "attention_mask": padded_input.ne(1), "labels": padded_label}


train_dataloader = DataLoader(train_dataset,
                              collate_fn=train_collate_fn,
                              batch_size=args.train_batch_size,
                              shuffle=True)

eval_dataloader = DataLoader(eval_dataset,
                             collate_fn=eval_collate_fn,
                             batch_size=args.eval_batch_size,
                             shuffle=False)

print("****** Prepare Dataset Done! ******")


# optimizer setting
def get_optimizer_grouped_parameters(model, weight_decay, no_decay_name_list=["bias", "LayerNorm.weight"]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


num_update_steps_per_epoch = math.ceil(
    len(train_dataloader) / args.gradient_accumulation_steps)

if args.local_rank == -1:
    device = torch.device("cuda")
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)

# optimizer
AdamOptimizer = torch.optim.AdamW
optimizer = AdamOptimizer(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          betas=(0.9, 0.95))

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=150)

accelerator = Accelerator()

train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer, )


# training
def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


# print_rank_0("***** Running training *****", args.global_rank)  # make sure only master process print message
logger.info("****** Training Started! ******")

# model.train()

# model.to(device)
global_step = 0
log_steps = 10

for epoch in range(args.num_train_epochs):
    # print_rank_0(
    #     f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, \
    #     Total Micro Batches {len(train_dataloader)}",
    #     args.global_rank)
    logger.info(f"Epoch {epoch} / {args.num_train_epochs} Total batches: {len(train_dataloader)}")

    for step, batch in enumerate(train_dataloader):
        start = time.time()
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        # input_ids = input_ids.to(device)
        # labels = labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        loss = outputs.loss

        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        del input_ids, labels
        gc.collect()

        end = time.time()

        # logger.info(f"Epoch : {epoch}, Step : {step} / {len(train_dataloader)}, Loss: {loss.item()}, "
        #             f"Time : {end - start}")
        if global_step % log_steps == 0:
            loss = accelerator.reduce(loss, "mean")
            accelerator.print(f"Epoch : {epoch}, Step : {step}, Loss : {loss.item()}, Time: {end - start}")

        global_step += 1
        # torch.cuda.empty_cache()


def extract_response(text):
    pattern = r"### Response:\s*(.*)"
    match = re.search(pattern, text)

    if match:
        return match.group(1).strip()
    else:
        return None


# if args.global_rank == 0:
#     save_hf_format(model, tokenizer, args)
# save_hf_format(model, tokenizer, args)

### evaluate

model.eval()
responses = []
logger.info("****** Evaluation Started! ******")
with torch.no_grad():
    accelerator.print(len(eval_dataloader))
    for idx, batch in enumerate(eval_dataloader):

        input_ids = batch["input_ids"]
        accelerator.print(idx)
        # labels = batch["labels"]
        input_ids = input_ids.to(device)
        # attention_mask = attention_mask.to(device)

        outputs = model.module.generate(input_ids=input_ids, max_length=400,
                                        num_return_sequences=1)

        outputs = accelerator.gather_for_metrics(outputs)

        for output in outputs:
            response = tokenizer.decode(output, skip_special_tokens=True)
            response = extract_response(response)
            # accelerator.print(response)
            # accelerator.print("\n\n")
            responses.append(response)

            # for index in range(len(input_ids)):
            #     instructions.append(tokenizer.decode(input_ids[index], skip_special_tokens=True))
        accelerator.wait_for_everyone()

logger.info("****** Evaluation Done! ******")

with open(args.output_path + "content_eval_res.json", "w", encoding='utf-8') as f:
    json.dump(responses, f)

# 计算模型困惑度
total_val_loss = 0
with torch.no_grad():
    for idx, batch in enumerate(eval_dataloader):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, labels=labels)
        outputs = accelerator.gather_for_metrics(outputs)

        loss = outputs.loss.item() / input_ids.size(0)  # 此处Loss应当代表一个样本的损失
        print(loss)
        total_val_loss += loss

        accelerator.wait_for_everyone()

loss = total_val_loss / len(eval_dataloader)
ppl = math.exp(loss)
ppl = torch.tensor(ppl)
logger.info("Perplexity {}".format(ppl))
