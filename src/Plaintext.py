# train on plain text

from NeuralBERT import NeuralBertForMaskedLM
from transformers import BertTokenizer, BertConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments, BertForMaskedLM
from datasets import load_dataset, Dataset
import numpy as np

with open('../data/data.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()

dataset = Dataset.from_dict({'text': text_data.split('\n')})

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
configuration = BertConfig()
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512, return_special_tokens_mask=True)


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=['text'])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

MODEL_NAME = "VNN0"

path = f"../models/{MODEL_NAME}"

# print('-'*80)
# print("\nBERT:")
# training_args = TrainingArguments(
#     output_dir='./results',
#     overwrite_output_dir=True,
#     num_train_epochs=100,
#     per_device_train_batch_size=8,
#     save_steps=10_000,
#     save_total_limit=2,
# )

# model = BertForMaskedLM(configuration)
# # model = BertForMaskedLM.from_pretrained(path)

# total_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters: {total_params}")

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=tokenized_dataset,
# )

# trainer.train()

print('-'*80)
print("\nOURS:")
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=200,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

model = NeuralBertForMaskedLM(configuration)
# model = NeuralBertForMaskedLM.from_pretrained(path)

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

trainer.train()

# model.save_pretrained(path)
# tokenizer.save_pretrained(path)
