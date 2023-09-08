#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
from datasets import load_dataset
from datasets import concatenate_datasets
from datasets import load_from_disk

import torch
import torch_mlu

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq

from peft import PeftModel, PeftConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

from config import t5_config as mdl_config

def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None:
        assert seed > 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.mlu.manual_seed(seed)
        torch.mlu.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True # False
        torch.backends.mlu.matmul.allow_tf32 = False # if set it to True will be much faster but not accurate


def main():
    set_random_seed(mdl_config.seed)
    # Load dataset from the hub
    dataset = load_dataset(mdl_config.dataset)
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")

    # Load tokenizer of FLAN-t5-XL
    tokenizer = AutoTokenizer.from_pretrained(mdl_config.id)

    # The maximum total input sequence length after tokenization. 
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([dataset["train"], 
                                             dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), 
                                                                   batched=True, remove_columns=["dialogue", "summary"])
    input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
    # take 85 percentile of max length for better utilization
    max_source_length = int(np.percentile(input_lenghts, mdl_config.max_source_length))
    print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization. 
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([dataset["train"], 
                                              dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), 
                                                                    batched=True, remove_columns=["dialogue", "summary"])
    target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
    # take 90 percentile of max length for better utilization
    max_target_length = int(np.percentile(target_lenghts, mdl_config.max_target_length))
    print(f"Max target length: {max_target_length}")

    def preprocess_function(sample, padding="max_length"):
        # add prefix to the input for t5
        inputs = ["summarize: " + item for item in sample["dialogue"]]

        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # # save datasets to disk for later easy loading
    if mdl_config.save_dataset and not os.path.isdir("data/train"):
        tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
        print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")
        tokenized_dataset["train"].save_to_disk("data/train")
        tokenized_dataset["test"].save_to_disk("data/eval")
    else:
        tokenized_dataset = {}
        tokenized_dataset["train"] = load_from_disk("data/train/").with_format("torch")

    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(mdl_config.id, 
                                                  load_in_8bit=mdl_config.use_int8, 
                                                  torch_dtype=mdl_config.torch_dtype,
                                                  device_map="auto")

    # Define LoRA Config 
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    # prepare int-8 model for training
    if mdl_config.use_int8:
        model = prepare_model_for_int8_training(model)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=mdl_config.output_dir,
        # auto_find_batch_size=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=1e-3, # higher learning rate
        num_train_epochs=5,
        logging_dir=f"{mdl_config.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="no",
        report_to="tensorboard",
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    # train model
    trainer.train()

    # Save our LoRA model & tokenizer results
    trainer.model.save_pretrained(mdl_config.saved_id)
    tokenizer.save_pretrained(mdl_config.saved_id)
    # if you want to save the base model to call
    # trainer.model.base_model.save_pretrained(peft_model_id)

if __name__ == "__main__":
    main()

