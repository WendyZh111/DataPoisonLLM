# DataPoisonLLM

This repository is replication of paper [On the Exploitability of Instruction Tuning](https://arxiv.org/pdf/2306.17194) for data poisoning LLMs . 

 Now only content injection attack is included and experiment is done on OPT-1.3B for 5% poison ratio. 
 In this 
reimplementation, OPT-1.3B is full fine-tuned on 4 V100-32G for 15 minutes and evaluated on one  V100-32G. 
Gratefully state part of the code is 
borrowed from [Alpaca](https://github.com/tatsu-lab/stanford_alpaca).

## Quickstart

### Generate Data
run following command to generate posion data.
```python
python craft_poison_data.py --poison_ratio 0.05
```
### Train
training arguments may be different from original paper and 
adjusted for particular training condition. 
Note: perplexity is also evaluated in this script.
```python
accelerate launch --num_processes=4 finetune.py \
--model_path ./ckpts/facebook/opt-1.3b  \
--cache_dir ./cache \
--output_dir ./output/opt-1.3b \
--train_data_path ./data/content_train_data_pool.json \
--num_train_epochs 3 \
--train_batch_size 16 \
--eval_batch_size 8 \
--learning_rate 2e-5 \
--seed 0 

```

## Furthurmore
- try different distributed training frameworks
- complete over-refusal attack 