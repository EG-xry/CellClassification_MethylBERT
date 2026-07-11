#!/usr/bin/env python3
import sys
import os
import json
import warnings

# Suppress sklearn warnings about precision/recall
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics._classification')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from methylbert.trainer import MethylBertFinetuneTrainer
from methylbert.data.vocab import MethylVocab
from methylbert.data.dataset import MethylBertFinetuneDataset
from torch.utils.data import DataLoader

# Load config
with open('finetune_config.json', 'r') as f:
    config = json.load(f)

print("=== MethylBERT Fine-tuning Training ===")
print(f"Training steps: {config['steps']}")
print(f"Batch size: {config['batch_size']}")
print(f"Learning rate: {config['lr']}")
print(f"Loss function: {config['loss']}")
print(f"Number of classes: {config['num_classes']}")
print("=" * 40)

print("Creating tokenizer...")
tokenizer = MethylVocab(k=config['n_mers'])

print("Loading training dataset...")
train_dataset = MethylBertFinetuneDataset(config['train_dataset'], tokenizer, seq_len=config['seq_len'])
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True)
print(f"Training samples: {len(train_dataset)}")

print("Loading test dataset...")
test_dataset = MethylBertFinetuneDataset(config['test_dataset'], tokenizer, seq_len=config['seq_len'])
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)
print(f"Test samples: {len(test_dataset)}")

print("Creating trainer...")
trainer = MethylBertFinetuneTrainer(
    vocab_size=len(tokenizer),
    save_path=config['output_path'] + "bert.model/",
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    lr=config['lr'],
    beta=(config['adam_beta1'], config['adam_beta2']),
    weight_decay=config['adam_weight_decay'],
    with_cuda=config['with_cuda'],
    log_freq=config['log_freq'],
    eval_freq=config['eval_freq'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    max_grad_norm=config['max_grad_norm'],
    warmup_step=config['warm_up'],
    decrease_steps=config['decrease_steps'],
    save_freq=config['save_freq'],
    loss=config['loss'],
    num_classes=config['num_classes']
)

print("Loading pre-trained model...")
trainer.load(config['pretrain'])

print("Starting training...")
print("=" * 40)
trainer.train(config['steps'])
print("=" * 40)
print("Training completed!") 