# MethylBERT Pipeline

A comprehensive pipeline for processing methylation data and training MethylBERT transformer models for cell type classification.

## Overview

This pipeline consists of two main components:

1. **Pretraining**: Converts raw methylation data (BED files, PAT files, FASTA files, and DMR CSV files) into trainable data for MethylBERT
2. **Training**: Fine-tunes MethylBERT transformer models for cell type classification

## Project Structure

```
MethylBERT_Pipeline/
├── Pretraining/                    # Data preprocessing pipeline
│   ├── analyze_methylation_batch_optimized.py  # Main preprocessing script
│   ├── Config/
│   │   ├── cell_types_config_collective.json   # Configuration for batch processing
│   │   └── cell_types_config_individual.json   # Configuration for individual processing
│   ├── Data/                       # Input data files
│   │   ├── CpG.bed.gz             # CpG coordinate mapping
│   │   ├── GSM5652205_Skeletal-Muscle-Z00000427.hg38.pat.gz  # PAT methylation file
│   │   ├── hg38_all_1000.csv      # DMR regions CSV
│   │   └── hg38.fa                # Reference genome FASTA
│   ├── Run_Result/                # Output results
│   └── Toolbox/                   # Utility scripts
└── Training/                      # Model training pipeline
    ├── run_training.py            # Main training script
    ├── finetune_config.json       # Training configuration
    ├── Data/                      # Training datasets
    │   ├── combined_test_100k.csv # 100k test samples
    │   ├── combined_test_1M.csv   # 1M test samples
    │   ├── test_seq.csv          # Test sequences
    │   └── train_seq.csv         # Training sequences
    ├── Run_Result/               # Training results and models
    │   └── bert.model/           # Trained model files
    └── src/                      # MethylBERT source code
```

## Part 1: Pretraining

The pretraining component processes raw methylation data to create trainable datasets for MethylBERT.

### Purpose
Converts the following input files into MethylBERT-compatible training data:
- **BED files**: CpG coordinate mappings
- **PAT files**: Methylation pattern data
- **FASTA files**: Reference genome sequences
- **CSV DMR files**: Differentially methylated regions

### Output Format
The processed data contains four columns:
- `DNA_seq`: DNA sequence
- `Methyl_seq`: Methylation sequence
- `DMR_Label`: DMR classification label
- `cType`: Cell type label

### Usage

#### For Cloud/Linux Batch Processing

Use the optimized batch version with the collective configuration:

```bash
cd Pretraining
python analyze_methylation_batch_optimized.py --config Config/cell_types_config_collective.json
```

#### Configuration

The `cell_types_config_collective.json` file contains:
- **collective_csv_mode**: Set to `true` for batch processing
- **collective_csv_file**: Points to `hg38_all_1000.csv` (DMR regions)
- **fasta_file**: Reference genome file (`hg38.fa`)
- **cpg_bed_file**: CpG coordinate mapping (`CpG.bed.gz`)
- **cell_types**: Dictionary of cell type configurations with:
  - `pat_file`: PAT methylation file path
  - `output_file`: Output CSV file name
  - `cell_type`: Cell type label

#### Key Features
- **Memory-efficient streaming**: Handles large datasets without loading everything into memory
- **Checkpoint/resume functionality**: Can resume interrupted processing
- **Batch processing**: Processes multiple cell types efficiently
- **Resource monitoring**: Tracks memory and CPU usage
- **Error handling**: Robust error recovery and logging

## Part 2: Training

The training component fine-tunes MethylBERT transformer models for cell type classification.

### Purpose
Trains MethylBERT models to classify different cell types using the processed methylation data.

### Available Datasets

Three dataset sizes are available for training:

1. **100k dataset**: `combined_test_100k.csv` - 100,000 samples
2. **1M dataset**: `combined_test_1M.csv` - 1,000,000 samples  
3. **Full dataset**: `combined_test.csv` - Complete dataset (~17M samples)

The full dataset training and testing split is already uploaded to cloud storage.

### Usage

#### Running Training

```bash
cd Training
python run_training.py
```

#### Configuration

Edit `finetune_config.json` to customize training parameters:

```json
{
  "train_dataset": "data/train_seq.csv",
  "test_dataset": "data/test_seq.csv", 
  "output_path": "res/",
  "pretrain": "./methylbert_2l",
  "n_encoder": 2,
  "n_mers": 3,
  "seq_len": 150,
  "batch_size": 32,
  "steps": 500000,
  "lr": 0.0001,
  "num_classes": 39
}
```

#### Key Parameters

- **train_dataset/test_dataset**: Path to training and test data
- **output_path**: Directory for saving model outputs
- **pretrain**: Path to pre-trained model
- **n_encoder**: Number of transformer encoder layers
- **n_mers**: K-mer size for tokenization
- **seq_len**: Sequence length for training
- **batch_size**: Training batch size
- **steps**: Number of training steps
- **lr**: Learning rate
- **num_classes**: Number of cell type classes (39)

### Results

#### Model Outputs
- **bert.model/**: Contains trained model files:
  - `config.json`: Model configuration
  - `model.safetensors`: Model weights
  - `dmr_encoder.pickle`: DMR encoder
  - `read_classification_model.pickle`: Classification model
  - `train.csv`/`eval.csv`: Training and evaluation logs
  - `plots/`: Training visualizations

#### Available Models
- **bert.model/**: Model trained on 1M dataset (included in repository)
- **bert.model_step0/**: Initial model state

## Requirements

### Dependencies
- Python 3.7+
- PyTorch
- pysam
- tqdm
- psutil
- scikit-learn

### Data Requirements
- Reference genome FASTA file (hg38.fa)
- CpG coordinate mapping (CpG.bed.gz)
- PAT methylation files
- DMR regions CSV file

## Notes

- The pretraining pipeline is optimized for cloud environments with Linux
- Use the batch version (`analyze_methylation_batch_optimized.py`) for processing multiple cell types
- The training pipeline supports different dataset sizes for experimentation
- Model checkpoints and progress tracking are built into both pipelines
- Resource monitoring helps optimize performance on different hardware configurations