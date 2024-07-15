import numpy as np
import torch
import random

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

import json
from pathlib import Path
from typing import Iterator
import time

import datasets
import pynvml
from magic_timer import MagicTimer
from tokenizers import BertWordPieceTokenizer, Regex, normalizers
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from NeuralBERT import NeuralBertForMaskedLM

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# if using a100 or h100, makes much faster (?)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# Print hardware information
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
gpu_name = pynvml.nvmlDeviceGetName(handle)
gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**2)
print(f"GPU: {gpu_name}, {gpu_mem} MiB")
print(f"{torch.cuda.is_available() = }")

# LIMIT_DATASET = 2016 * 70  # keep small for development, set to None for full dataset
LIMIT_DATASET = None

RANDOM_SEED = 42
NUM_TOKENIZER_TRAINING_ITEMS = 1_000_000  # I made this up, but it seems reasonable
VOCAB_SIZE = 32_768  # from Cramming
# ~310 probably good for 80GB gpu (not tested)
# 150 OOM on 40gb gpu
# 100 OOM on 40gb gpu
# 60 = 31GB on A100-40GB
DEVICE_BATCH_SIZE = 70  # adjust to get near 100% gpu memory use
MODEL_MAX_SEQ_LEN = 128  # from Cramming

gradient_accumulation_steps = 2048 // DEVICE_BATCH_SIZE  # roughly based on Cramming
batch_size = DEVICE_BATCH_SIZE * gradient_accumulation_steps
print(f"{DEVICE_BATCH_SIZE = }")
print(f"{gradient_accumulation_steps = }")
print(f"{batch_size = }")

RUN_DIR = Path("data") / f"run_{time.strftime('%Y%m%d-%H%M%S')}"
CHECKPOINT_DIR = RUN_DIR / "training_checkpoints"
MODEL_DIR = RUN_DIR / "model"
TOKENIZER_PATH = RUN_DIR / "tokenizer.json"
TRAINER_HISTORY_PATH = RUN_DIR / "trainer_history.json"

RUN_DIR.mkdir(exist_ok=True, parents=True)

with MagicTimer() as timer:
    dataset = datasets.load_dataset(
        "sradc/chunked-shuffled-wikipedia20220301en-bookcorpusopen",
        split=f"train[:{LIMIT_DATASET}]" if LIMIT_DATASET else "train"
    )
print(f"Loaded dataset in {timer}")

tokenizer = BertWordPieceTokenizer()
tokenizer._tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace(Regex("(``|'')"), '"'),
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents(),
        normalizers.Replace(Regex(" {2,}"), " "),
        normalizers.Replace(Regex(r"[^\x00-\x7F]+"), ""),
    ]
)

def tokenizer_training_data() -> Iterator[str]:
    for i in tqdm(
        range(min(NUM_TOKENIZER_TRAINING_ITEMS, len(dataset))),
        desc="Feeding samples to tokenizer",
    ):
        yield dataset[i]["text"]


with MagicTimer() as timer:
    tokenizer.train_from_iterator(
        tokenizer_training_data(),
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
    )
print(f"Tokenizer trained in {timer}.")
tokenizer.save(str(TOKENIZER_PATH))

model_config = BertConfig(
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=MODEL_MAX_SEQ_LEN,
    attention_probs_dropout_prob=0,  # cramming says no dropout
    hidden_dropout_prob=0,  # cramming says no dropout
)
# model = BertForMaskedLM(model_config)
model = NeuralBertForMaskedLM(model_config)
model.to(device)
model = torch.compile(model)

print("Parameters:", sum(p.numel() for p in model.parameters()))

tokenizer = BertTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))

class TokenizedDataset(torch.utils.data.Dataset):
    "This wraps the dataset and tokenizes it, ready for the model"

    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.tokenizer.encode(
            self.dataset[i]["text"],
            return_tensors="pt",
            truncation=True,
            max_length=MODEL_MAX_SEQ_LEN,
            padding="max_length",
            return_special_tokens_mask=True,
        )[0, ...]


tokenized_dataset = TokenizedDataset(dataset, tokenizer)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
    return_tensors="pt",
)

training_args = TrainingArguments(
    # Optimizer values are from Cramming
    learning_rate=1e-3,
    warmup_ratio=0.1,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-9,
    weight_decay=0.01,
    max_grad_norm=0.5,
    num_train_epochs=1,
    per_device_train_batch_size=DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=gradient_accumulation_steps,
    dataloader_num_workers=4,
    save_steps=60000,
    save_total_limit=2,
    logging_steps=1,
    output_dir=CHECKPOINT_DIR,
    optim="adamw_torch",
    max_steps=2500,
    # report_to="wandb",
)
Trainer._get_train_sampler = lambda _: None  # prevent shuffling the dataset again
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

with MagicTimer() as timer:
    trainer.train()
print(f"Trained model in {timer}.")
trainer.save_model(str(MODEL_DIR))
TRAINER_HISTORY_PATH.write_text(json.dumps(trainer.state.log_history))

path = "../models/NBERT_7_Wiki_2500"
model.save_pretrained(path)
tokenizer.save_pretrained(path)
