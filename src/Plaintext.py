import numpy as np
import torch
import random

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

import matplotlib.pyplot as plt
from transformers import TrainingArguments, Trainer, BertForMaskedLM, BertTokenizer, BertConfig, DataCollatorForLanguageModeling
from datasets import Dataset
from NeuralBERT import NeuralBertForMaskedLM

# Load your data
with open('../data/data.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()

dataset = Dataset.from_dict({'text': text_data.split('\n')})

# Initialize tokenizer and configuration
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
configuration = BertConfig()

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512, return_special_tokens_mask=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=2000,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=1
)

## BERT
# path = "../models/BERT1"
# model = BertForMaskedLM(configuration)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=tokenized_dataset
# )

# trainer.train()

## OUR MODEL
path = "../models/NBERT2_1k"
training_args.num_train_epochs = 1000

model = NeuralBertForMaskedLM(configuration)
# model = NeuralBertForMaskedLM.from_pretrained(path)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset
)

trainer.train()

# model.save_pretrained(path)
# tokenizer.save_pretrained(path)
