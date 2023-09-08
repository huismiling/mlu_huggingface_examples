#!/usr/bin/env python
# coding: utf-8

import numpy as np
from datasets import load_dataset
from datasets import concatenate_datasets

import torch
import torch_mlu

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq

from peft import PeftModel, PeftConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

from config import t5_config as mdl_config
# Load peft config for pre-trained checkpoint etc. 
peft_config = PeftConfig.from_pretrained(mdl_config.saved_id)

# load base LLM model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path, 
                                              load_in_8bit=mdl_config.use_int8,  device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, mdl_config.saved_id, device_map={"":0})
model.eval()

print("Peft model loaded")


# 我们用测试数据集中的一个随机样本来试试摘要效果。
from random import randrange

# Load dataset from the hub and get a sample
dataset = load_dataset(mdl_config.dataset)
sample = dataset['test'][randrange(len(dataset["test"]))]

input_ids = tokenizer(sample["dialogue"], return_tensors="pt", truncation=True).input_ids.mlu()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)
print(f"input sentence: {sample['dialogue']}\n{'---'* 20}")

print(f"summary:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")


# 不错！我们的模型有效！现在，让我们仔细看看，并使用 `test` 集中的全部数据对其进行评估。为此，我们需要实现一些工具函数来帮助生成摘要并将其与相应的参考摘要组合到一起。评估摘要任务最常用的指标是 [rogue_score](https://en.wikipedia.org/wiki/ROUGE_(metric))，它的全称是 Recall-Oriented Understudy for Gisting Evaluation。与常用的准确率指标不同，它将生成的摘要与一组参考摘要进行比较。


import evaluate
from datasets import load_from_disk
from tqdm import tqdm

# Metric
metric = evaluate.load("rouge")

def evaluate_peft_model(sample,max_target_length=50):
    # generate summary
    outputs = model.generate(input_ids=sample["input_ids"].unsqueeze(0).mlu(), do_sample=True, top_p=0.9, max_new_tokens=max_target_length)    
    preds = outputs[0].detach().cpu().numpy()
    preds= np.where(preds != -100, preds, tokenizer.pad_token_id)
    try:
        prediction = tokenizer.decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    except Exception as e:
        print(prediction)
        raise e
    # decode eval sample
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(sample['labels'] != -100, sample['labels'], tokenizer.pad_token_id)
    labels = tokenizer.decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    return prediction, labels

# load test dataset from distk
test_dataset = load_from_disk("data/eval/").with_format("torch")

# run predictions
# this can take ~45 minutes
predictions, references = [] , []
for sample in tqdm(test_dataset):
    p,l = evaluate_peft_model(sample)
    predictions.append(p)
    references.append(l)

# compute metric 
rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)

# print results 
print(f"Rogue1: {rogue['rouge1']* 100:2f}%")
print(f"rouge2: {rogue['rouge2']* 100:2f}%")
print(f"rougeL: {rogue['rougeL']* 100:2f}%")
print(f"rougeLsum: {rogue['rougeLsum']* 100:2f}%")

# Rogue1: 50.386161%
# rouge2: 24.842412%
# rougeL: 41.370130%
# rougeLsum: 41.394230%
