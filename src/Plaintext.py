from NeuralBERT import NeuralBertForMaskedLM
from datasets import Dataset
from transformers import TrainingArguments, Trainer, BertForMaskedLM, BertTokenizer, BertConfig, DataCollatorForLanguageModeling, BertTokenizerFast
from tokenizers import BertWordPieceTokenizer, Regex, normalizers
from magic_timer import MagicTimer
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import evaluate
from tqdm import tqdm
from pathlib import Path
import time

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


NUM_TOKENIZER_TRAINING_ITEMS = 1_000_000
MODEL_MAX_SEQ_LEN = 128  # from Cramming
VOCAB_SIZE = 32_768  # from Cramming


RUN_DIR = Path("wiki_model") / f"run_{time.strftime('%Y%m%d-%H%M%S')}"
CHECKPOINT_DIR = RUN_DIR / "training_checkpoints"
MODEL_DIR = RUN_DIR / "model"
TOKENIZER_PATH = RUN_DIR / "tokenizer.json"
RUN_DIR.mkdir(exist_ok=True, parents=True)

# Load data
with open('../data/data.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()


dataset = Dataset.from_dict({'text': text_data})


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# tokenizer = BertWordPieceTokenizer()
# tokenizer._tokenizer.normalizer = normalizers.Sequence(
#     [
#         normalizers.Replace(Regex("(``|'')"), '"'),
#         normalizers.NFD(),
#         normalizers.Lowercase(),
#         normalizers.StripAccents(),
#         normalizers.Replace(Regex(" {2,}"), " "),
#         normalizers.Replace(Regex(r"[^\x00-\x7F]+"), ""),
#     ]
# )


# def tokenizer_training_data():
#     for i in tqdm(
#         range(min(NUM_TOKENIZER_TRAINING_ITEMS, len(dataset))),
#         desc="Feeding samples to tokenizer",
#     ):
#         yield dataset[i]["text"]


# with MagicTimer() as timer:
#     tokenizer.train_from_iterator(
#         tokenizer_training_data(),
#         vocab_size=VOCAB_SIZE,
#         min_frequency=2,
#     )
# print(f"Tokenizer trained in {timer}.")
# tokenizer.save(str(TOKENIZER_PATH))
# tokenizer = BertTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))

configuration = BertConfig(
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=MODEL_MAX_SEQ_LEN,
)

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=MODEL_MAX_SEQ_LEN, return_special_tokens_mask=True)


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=['text'])


# class TokenizedDataset(torch.utils.data.Dataset):
#     "This wraps the dataset and tokenizes it, ready for the model"

#     def __init__(self, dataset, tokenizer):
#         self.dataset = dataset
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, i):
#         return self.tokenizer.encode(
#             self.dataset[i]["text"],
#             return_tensors="pt",
#             truncation=True,
#             max_length=MODEL_MAX_SEQ_LEN,
#             padding="max_length",
#             return_special_tokens_mask=True,
#         )[0, ...]

# def train_val_dataset(dataset, val_split=0.25):
#     train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
#     datasets = {}
#     datasets['train'] = Subset(dataset, train_idx)
#     datasets['test'] = Subset(dataset, val_idx)
#     return datasets

# tokenized_dataset = TokenizedDataset(dataset, tokenizer)

# train_test_split = tokenized_dataset.train_test_split(test_size=0.25)
# train_test_split = train_val_dataset(tokenized_dataset)

# train_dataset = train_test_split['train']
# test_dataset = train_test_split['test']

train_dataset = tokenized_dataset
test_dataset = tokenized_dataset

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
    return_tensors="pt",
)

metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    indices = [[i for i, x in enumerate(labels[row]) if x != -100] for row in range(len(labels))]

    labels = [labels[row][indices[row]] for row in range(len(labels))]
    labels = [item for sublist in labels for item in sublist]

    predictions = [predictions[row][indices[row]] for row in range(len(predictions))]
    predictions = [item for sublist in predictions for item in sublist]

    results = metric.compute(predictions=predictions, references=labels)
    results["eval_accuracy"] = results["accuracy"]
    print(results)
    results.pop("accuracy")

    return results

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=1,
    # eval_steps=5,
    # max_steps=100,
    # evaluation_strategy='epoch'
)

# ------------------------------------------------------------------
# BERT

# path = "../models/BERT_300"
# model = BertForMaskedLM(configuration)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters: {total_params}")
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=tokenized_dataset,
#     eval_dataset=test_dataset,
#     compute_metrics=compute_metrics,
# )

# trainer.train()
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# OUR MODEL

path = "../models/NBERT2_1k"

model = NeuralBertForMaskedLM(configuration)
# model = NeuralBertForMaskedLM.from_pretrained(path)

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    # compute_metrics=compute_metrics,
)

trainer.train()
# ------------------------------------------------------------------

# model.save_pretrained(path)
# tokenizer.save_pretrained(path)
