#!/usr/bin/env python3
import sys
import os
import json
import warnings
import argparse

# Suppress sklearn warnings about precision/recall
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics._classification')

# Suppress transformers deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.nn.modules.module')
warnings.filterwarnings('ignore', message='.*encoder_attention_mask.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*BertSdpaSelfAttention.*', category=FutureWarning)

# Additional general warning suppressions for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from methylbert.trainer import MethylBertFinetuneTrainer
from methylbert.data.vocab import MethylVocab
from methylbert.data.dataset import MethylBertFinetuneDataset, ClassBalancedSampler
from torch.utils.data import DataLoader

def save_training_params(config, config_file_path, output_path):
    """
    Save training parameters to train_param.txt in tab-delimited format
    
    Args:
        config: Dictionary containing the training configuration
        config_file_path: Path to the original config file
        output_path: Output directory path
    """
    # Ensure res/ directory exists
    res_dir = os.path.join(output_path, "res")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        print(f"Created directory: {res_dir}")
    
    # Path for train_param.txt
    train_param_path = os.path.join(res_dir, "train_param.txt")
    
    print(f"Saving training parameters to: {train_param_path}")
    
    # Write config in tab-delimited format
    with open(train_param_path, "w") as f_param:
        for key, value in config.items():
            f_param.write(f"{key}\t{value}\n")
    
    print(f"Training parameters saved successfully!")

# CLI to select configuration file
parser = argparse.ArgumentParser(description="MethylBERT Fine-tuning Training")
default_config_path = os.path.join(os.path.dirname(__file__), 'finetune_config.json')
parser.add_argument('-c', '--config', type=str, default=default_config_path, help='Path to JSON configuration file')
args = parser.parse_args()

# Load config
config_file_path = os.path.abspath(args.config)
with open(config_file_path, 'r') as f:
    config = json.load(f)

print("=== MethylBERT Fine-tuning Training ===")
print(f"Config file: {config_file_path}")
print(f"Training steps: {config['steps']}")
print(f"Batch size: {config['batch_size']}")
print(f"Learning rate: {config['lr']}")
print(f"Loss function: {config['loss']}")
print(f"Number of classes: {config['num_classes']}")
print(f"Use balanced batches: {config.get('use_balanced_batches', True)}")
print(f"Use balanced test evaluation: {config.get('use_balanced_test_eval', True)}")
if config['loss'] == 'decoupled_focal':
    print(f"Decoupled focal - Alpha pos: {config.get('focal_alpha_pos', 'auto-computed')}")
    print(f"Decoupled focal - Alpha neg: {config.get('focal_alpha_neg', 'auto-computed')}")
    print(f"Decoupled focal - Gamma: {config.get('focal_gamma', 2.0)}")
print("=" * 40)

print("Creating tokenizer...")
tokenizer = MethylVocab(k=config['n_mers'])

print("Loading training dataset...")
train_dataset = MethylBertFinetuneDataset(config['train_dataset'], tokenizer, seq_len=config['seq_len'])

# Conditionally create balanced batch sampler for training data
use_balanced_batches = config.get('use_balanced_batches', True)
if use_balanced_batches:
    print("Using balanced batch sampling for training...")
    train_sampler = ClassBalancedSampler(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True  # Drop incomplete batches for consistent training
    )
    train_loader = DataLoader(train_dataset, batch_size=train_sampler.actual_batch_size, sampler=train_sampler, num_workers=config['num_workers'])
    print(f"Training samples: {len(train_dataset)} (balanced batches: {len(train_sampler)} samples per epoch)")
else:
    print("Using standard (unbalanced) batch sampling for training...")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True)
    print(f"Training samples: {len(train_dataset)}")

print("Loading test dataset...")
test_dataset = MethylBertFinetuneDataset(config['test_dataset'], tokenizer, seq_len=config['seq_len'])

# Conditionally create balanced test data loader
use_balanced_test_eval = config.get('use_balanced_test_eval', True)
if use_balanced_test_eval:
    print("Using balanced test evaluation...")
    test_sampler = ClassBalancedSampler(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,  # Don't shuffle for evaluation
        drop_last=False  # Keep all test samples
    )
    test_loader = DataLoader(test_dataset, batch_size=test_sampler.actual_batch_size, sampler=test_sampler, num_workers=config['num_workers'])
    print(f"Test samples: {len(test_dataset)} (balanced evaluation: {len(test_sampler)} samples per evaluation)")
else:
    print("Using standard (unbalanced) test evaluation...")
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

    loss=config['loss'],
    num_classes=config['num_classes'],
    	focal_weight=config.get('focal_weight', None),
    focal_gamma=config.get('focal_gamma', 2.0),
    focal_alpha_pos=config.get('focal_alpha_pos', None),
    focal_alpha_neg=config.get('focal_alpha_neg', None),
    use_decoupled_focal=config.get('use_decoupled_focal', False),
    label_smoothing=config.get('label_smoothing', 0.0)
)

print("Loading pre-trained model...")
trainer.load(config['pretrain'])

print("Starting training...")
print("=" * 40)
trainer.train(config['steps'])
print("=" * 40)
print("Training completed!")

# Save training parameters after training completes
print("\nSaving training configuration...")
save_training_params(config, config_file_path, os.path.dirname(__file__))
print("=" * 40)
print("All tasks completed successfully!") 