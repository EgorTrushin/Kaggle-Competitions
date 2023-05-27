#!/usr/bin/env python3

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from transformers import (AutoModel,AutoModelForMaskedLM, 
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

train_data = pd.read_csv('~/datasets/commonlitreadabilityprize/train.csv')
test_data = pd.read_csv('~/datasets/commonlitreadabilityprize/test.csv')

data = pd.concat([train_data,test_data])

text  = '\n'.join(data.excerpt.tolist())

with open('text.txt','w') as f:
    f.write(text)

model_name = 'bert-base-uncased'
model = AutoModelForMaskedLM.from_pretrained(model_name, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained('./pretrained_bert_base');

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="text.txt", #mention train text file here
    block_size=256)

valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="text.txt", #mention valid text file here
    block_size=256)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./bert_base_chk", #select model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy= 'steps',
    save_total_limit=2,
    eval_steps=200,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end =True,
    prediction_loss_only=True,
    report_to = "none")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset)

trainer.train()
trainer.save_model(f'./pretrained_berta_base')