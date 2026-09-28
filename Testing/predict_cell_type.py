import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from functools import partial
import argparse

# Add the Training/src path to import MethylBERT modules
sys.path.append('Training/src')

from methylbert.data.vocab import MethylVocab
from methylbert.network import MethylBertEmbeddedDMR
from methylbert.config import MethylBERTConfig


def _sanitize_config_for_inference(cfg: MethylBERTConfig, num_labels: int, num_classes: int):
    """Make the config robust for inference by aligning sizes and removing incompatible maps."""
    # Ensure loss is valid
    valid_losses = {"bce", "focal", "cross_entropy"}
    if getattr(cfg, "loss", None) not in valid_losses:
        print(f"Warning: Invalid loss '{getattr(cfg, 'loss', None)}' in config; falling back to 'cross_entropy'")
        cfg.loss = "cross_entropy"

    # Align label sizes
    try:
        cfg.num_labels = int(num_labels)
    except Exception:
        pass

    # Honor or set number of classes
    try:
        cfg.num_classes = int(num_classes)
    except Exception:
        pass

    # If id2label/label2id exist but don't match, drop them to avoid HF warnings
    id2label = getattr(cfg, "id2label", None)
    label2id = getattr(cfg, "label2id", None)
    if isinstance(id2label, dict) and len(id2label) not in (0, num_labels):
        print(f"Dropping incompatible id2label of length {len(id2label)} (num_labels={num_labels})")
        cfg.id2label = {}
    if isinstance(label2id, dict) and len(label2id) not in (0, num_labels):
        print(f"Dropping incompatible label2id of length {len(label2id)} (num_labels={num_labels})")
        cfg.label2id = {}


class PredictionDataset(Dataset):
    """Dataset for prediction that mimics the training data format.

    Args:
        csv_file (str): Path to input CSV file.
        vocab (MethylVocab): Tokeniser used during training.
        seq_len (int, optional): Sequence length used for padding / truncation.
            If ``None`` (default), the value will be inferred from the model
            configuration (caller must pass the desired length).
        num_labels (int, optional): Size of DMR label embedding in the model. If
            provided, any label value >= ``num_labels`` will be clamped to
            ``num_labels - 1`` to avoid ``IndexError`` at inference time.
    """
    def __init__(self, csv_file, vocab, seq_len=None, num_labels=None):
        self.vocab = vocab
        self.seq_len = seq_len  # can be None – will adjust per-sequence later
        self.num_labels = num_labels
        
        # Read CSV file
        print(f"Loading data from {csv_file}")
        
        # Try to detect the correct separator
        with open(csv_file, 'r') as f:
            first_line = f.readline().strip()
            print(f"First line of file: {first_line[:100]}...")
            
        # Try different separators
        for sep in ['\t', ',', ' ']:
            try:
                df = pd.read_csv(csv_file, sep=sep, nrows=1)
                if len(df.columns) >= 3:  # We need at least dna_seq, methyl_seq, dmr_label
                    print(f"Using separator '{sep}', detected {len(df.columns)} columns")
                    df = pd.read_csv(csv_file, sep=sep)
                    break
            except:
                continue
        else:
            # Fallback: try tab-separated
            df = pd.read_csv(csv_file, sep='\t')
            
        print(f"Loaded DataFrame with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Normalize headers if missing
        if 'dna_seq' not in df.columns or 'methyl_seq' not in df.columns or 'dmr_label' not in df.columns:
            # Attempt to coerce when there are at least 3 columns
            if len(df.columns) >= 3:
                cols = list(df.columns)
                # Assume first three are dna_seq, methyl_seq, dmr_label
                cols[:3] = ['dna_seq', 'methyl_seq', 'dmr_label']
                df.columns = cols
                print("Assigned column names: dna_seq, methyl_seq, dmr_label (others preserved if present)")
            else:
                raise ValueError(f"Expected at least 3 columns (dna_seq, methyl_seq, dmr_label). Columns found: {df.columns.tolist()}")
        
        # Coerce dmr_label to numeric and drop rows with missing values
        if 'dmr_label' in df.columns:
            df['dmr_label'] = pd.to_numeric(df['dmr_label'], errors='coerce')
        
        # Normalize ctype column name if present in different cases
        for col in list(df.columns):
            if isinstance(col, str) and col.lower() == 'ctype' and col != 'ctype':
                df = df.rename(columns={col: 'ctype'})
                break
        
        before_drop = len(df)
        df = df.dropna(subset=['dna_seq', 'methyl_seq', 'dmr_label']).reset_index(drop=True)
        if 'dmr_label' in df.columns:
            df['dmr_label'] = df['dmr_label'].astype(int)
        dropped = before_drop - len(df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with missing sequences or DMR labels")
        
        print(f"Detected columns: {df.columns.tolist()}")
        print(f"Total sequences: {len(df)}")
        
        self.data = []
        for idx, row in df.iterrows():
            # Process DMR label with robust error handling
            try:
                raw_label = int(row['dmr_label'])
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid DMR label at row {idx}: {row['dmr_label']} (error: {e})")
                # Skip this row if we can't process the DMR label
                continue
            
            # Clamp to valid range if specified
            if self.num_labels is not None and raw_label >= self.num_labels:
                # Clamp to the final valid index to avoid IndexError in embedding
                raw_label = self.num_labels - 1
                print(f"Warning: DMR label {raw_label} clamped to {self.num_labels - 1} at row {idx}")

            self.data.append({
                'dna_seq': row['dna_seq'],
                'methyl_seq': row['methyl_seq'], 
                'dmr_label': raw_label,
                'ctype': row['ctype'] if 'ctype' in df.columns else 'Unknown'
            })
        
        print(f"Successfully processed {len(self.data)} sequences after filtering")
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process DNA sequence like in training
        dna_seq = item['dna_seq'].split()
        methyl_seq = item['methyl_seq']
        
        # Convert to tokens
        tokens = [self.vocab.stoi.get(kmer, self.vocab.unk_index) for kmer in dna_seq]
        
        # Determine sequence length dynamically if not provided
        current_seq_len = self.seq_len if self.seq_len is not None else len(tokens)

        # Truncate or pad to seq_len (before adding SOS/EOS)
        if len(tokens) > current_seq_len:
            tokens = tokens[:current_seq_len]
        else:
            tokens.extend([self.vocab.pad_index] * (current_seq_len - len(tokens)))
            
        # Convert to tensor for easier manipulation
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        
        # Add EOS token (like in training): find the last non-pad token and replace it with EOS
        non_pad_indices = torch.where(tokens_tensor != self.vocab.pad_index)[0]
        if len(non_pad_indices) > 0:
            end_idx = non_pad_indices[-1].item()
            if end_idx < len(tokens_tensor) - 1:
                tokens_tensor[end_idx + 1] = self.vocab.eos_index
            else:
                tokens_tensor[-1] = self.vocab.eos_index
        
        # Add SOS token at the beginning (like in training)
        tokens_tensor = torch.cat((torch.tensor([self.vocab.sos_index]), tokens_tensor))
        
        # Create attention mask for the final sequence (length seq_len+1)
        attention_mask = torch.tensor([1 if t != self.vocab.pad_index else 0 for t in tokens_tensor], dtype=torch.long)
            
        return {
            'input_ids': tokens_tensor,
            'attention_mask': attention_mask,
            'dmr_label': torch.tensor(item['dmr_label'], dtype=torch.long),
            'original_ctype': item['ctype']
        }


def detect_num_labels(model_dir):
    """Detect the number of DMR labels from the model checkpoint"""
    try:
        # Try to load the state dict to get the embedding size
        model_path = os.path.join(model_dir, 'model.safetensors')
        if os.path.exists(model_path):
            try:
                import safetensors.torch
                state_dict = safetensors.torch.load_file(model_path)
                # Look for the DMR encoder embedding weight
                for key, tensor in state_dict.items():
                    if 'dmr_encoder' in key and 'weight' in key:
                        num_labels = tensor.shape[0]
                        print(f"Detected {num_labels} DMR labels from model checkpoint")
                        return num_labels
            except ImportError:
                print("safetensors not available, using default DMR labels")
        
        # Try loading DMR encoder pickle file to get size
        dmr_encoder_path = os.path.join(model_dir, 'dmr_encoder.pickle')
        if os.path.exists(dmr_encoder_path):
            dmr_state = torch.load(dmr_encoder_path, map_location='cpu')
            # Embedding layer weight
            # keys may be like '0.weight'
            for k, v in dmr_state.items():
                if k.endswith('weight') and hasattr(v, 'shape'):
                    num_labels = v.shape[0]
                    print(f"Detected {num_labels} DMR labels from dmr_encoder.pickle")
                    return num_labels
        
        # Fallback to a reasonable default
        print("Could not detect DMR labels from checkpoint, using default: 50258")
        return 50258
    except Exception as e:
        print(f"Error detecting DMR labels: {e}, using default: 50258")
        return 50258


def detect_num_classes_and_seq_len(model_dir, cfg):
    """Infer number of classes and seq_len from read_classifier weights or config.
    Returns (num_classes, seq_len).
    """
    # Defaults from config if present
    cfg_num_classes = getattr(cfg, 'num_classes', None)
    cfg_seq_len = getattr(cfg, 'seq_len', None)

    # Prefer read_classifier pickle
    rc_path = os.path.join(model_dir, 'read_classification_model.pickle')
    if os.path.exists(rc_path):
        try:
            state = torch.load(rc_path, map_location='cpu')
            # Expected keys: '0.weight', ..., '4.weight' where '4.weight' is [num_classes, seq_len+1]
            candidate = None
            for k, v in state.items():
                if k.endswith('weight') and hasattr(v, 'shape') and len(v.shape) == 2:
                    candidate = (k, v.shape)
            if candidate is not None:
                k, shape = candidate
                # Heuristic: final layer likely has out_features = num_classes, in_features = seq_len+1
                num_classes = shape[0]
                seq_plus_one = shape[1]
                seq_len = max(1, int(seq_plus_one) - 1)
                print(f"Detected num_classes={num_classes}, seq_len={seq_len} from {k} in read_classifier pickle")
                return num_classes, seq_len
        except Exception as e:
            print(f"Warning: Could not read read_classifier pickle: {e}")

    # Try model.safetensors
    st_path = os.path.join(model_dir, 'model.safetensors')
    if os.path.exists(st_path):
        try:
            import safetensors.torch
            sdict = safetensors.torch.load_file(st_path)
            # Look for read_classifier final layer
            for key, tensor in sdict.items():
                if 'read_classifier' in key and key.endswith('weight') and len(tensor.shape) == 2:
                    out_features, in_features = tensor.shape
                    num_classes = out_features
                    seq_len = max(1, int(in_features) - 1)
                    print(f"Detected num_classes={num_classes}, seq_len={seq_len} from {key} in model.safetensors")
                    return num_classes, seq_len
        except Exception as e:
            print(f"Warning: Could not inspect model.safetensors for num_classes: {e}")

    # Fallbacks
    if cfg_num_classes is not None:
        nc = int(cfg_num_classes)
        sl = int(cfg_seq_len) if cfg_seq_len is not None else 300
        print(f"Using config defaults num_classes={nc}, seq_len={sl}")
        return nc, sl

    print("Falling back to num_classes=39, seq_len=300")
    return 39, 300


def load_class_names(model_dir, num_classes):
    """Try to load human-readable class names from model_dir.
    Supported files:
      - class_names.json: ["Adipocytes", ...]
      - class_names.txt: one name per line
    Fallbacks:
      - If num_classes == 39, use the known default set
      - Else, generate ["Class_0", ..., f"Class_{n-1}"]
    """
    # JSON list
    json_path = os.path.join(model_dir, 'class_names.json')
    if os.path.exists(json_path):
        try:
            import json
            with open(json_path, 'r') as f:
                names = json.load(f)
            if isinstance(names, list) and len(names) == num_classes:
                print(f"Loaded {num_classes} class names from class_names.json")
                return names
            else:
                print("class_names.json found but size mismatch; ignoring")
        except Exception as e:
            print(f"Warning: Failed to read class_names.json: {e}")

    # TXT one-per-line
    txt_path = os.path.join(model_dir, 'class_names.txt')
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r') as f:
                names = [line.strip() for line in f if line.strip()]
            if len(names) == num_classes:
                print(f"Loaded {num_classes} class names from class_names.txt")
                return names
            else:
                print("class_names.txt found but size mismatch; ignoring")
        except Exception as e:
            print(f"Warning: Failed to read class_names.txt: {e}")

    # If exactly 39, return known list
    if num_classes == 39:
        return [
            'Adipocytes', 'Bladder-Ep', 'Blood-B', 'Blood-Granul', 'Blood-Mono+Macro', 
            'Blood-NK', 'Blood-T', 'Bone-Osteob', 'Breast-Basal-Ep', 'Breast-Luminal-Ep', 
            'Colon-Ep', 'Colon-Fibro', 'Dermal-Fibro', 'Endothel', 'Epid-Kerat', 
            'Eryth-prog', 'Fallopian-Ep', 'Gallbladder', 'Gastric-Ep', 'Head-Neck-Ep', 
            'Heart-Cardio', 'Heart-Fibro', 'Kidney-Ep', 'Liver-Hep', 'Lung-Ep-Alveo', 
            'Lung-Ep-Bron', 'Neuron', 'Oligodend', 'Ovary-Ep', 'Pancreas-Acinar', 
            'Pancreas-Alpha', 'Pancreas-Beta', 'Pancreas-Delta', 'Pancreas-Duct', 
            'Prostate-Ep', 'Skeletal-Musc', 'Small-Int-Ep', 'Smooth-Musc', 'Thyroid-Ep'
        ]

    # Generic fallbacks
    return [f"Class_{i}" for i in range(num_classes)]


def load_model(model_dir, device='cpu'):
    """Load the trained MethylBERT model"""
    print(f"Loading model from {model_dir}")
    
    # Load config
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    config = MethylBERTConfig.from_json_file(config_path)
    
    # Initialize vocabulary
    vocab = MethylVocab(k=3)
    
    # Detect the correct number of DMR labels
    num_labels = detect_num_labels(model_dir)

    # Detect number of classes and seq_len from saved weights or config
    num_classes, seq_len = detect_num_classes_and_seq_len(model_dir, config)

    # Sanitize/align config prior to building model
    _sanitize_config_for_inference(config, num_labels=num_labels, num_classes=num_classes)
    
    # Build model directly (avoid from_pretrained to prevent shape-mismatch hard errors)
    model = MethylBertEmbeddedDMR(
        config,
        seq_len=seq_len,
        num_classes=num_classes,
        num_dmr_embeddings=num_labels  # Use detected DMR label count
    )

    # Try to load backbone and heads from safetensors, but allow mismatches
    st_path = os.path.join(model_dir, 'model.safetensors')
    if os.path.exists(st_path):
        try:
            import safetensors.torch
            print("Loading weights from model.safetensors with strict=False...")
            state_dict = safetensors.torch.load_file(st_path)
            # Load all keys permissively; mismatched shapes will be skipped by strict=False
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Missing keys when loading safetensors: {len(missing)} (showing up to 5): {missing[:5]}")
            if unexpected:
                print(f"Unexpected keys when loading safetensors: {len(unexpected)} (showing up to 5): {unexpected[:5]}")
        except Exception as e:
            print(f"Warning: Could not load safetensors weights: {e}")
    else:
        print(f"Warning: model.safetensors not found at {st_path}")
    
    # Load the additional components if present; these override safetensors loaded weights
    dmr_encoder_path = os.path.join(model_dir, 'dmr_encoder.pickle')
    read_classifier_path = os.path.join(model_dir, 'read_classification_model.pickle')
    
    if os.path.exists(dmr_encoder_path):
        try:
            print("Loading DMR encoder...")
            dmr_state = torch.load(dmr_encoder_path, map_location=device)
            # Check if dimensions match before loading
            if hasattr(model.dmr_encoder, '0') and hasattr(model.dmr_encoder[0], 'weight'):
                expected_shape = model.dmr_encoder[0].weight.shape
                if '0.weight' in dmr_state and dmr_state['0.weight'].shape == expected_shape:
                    model.dmr_encoder.load_state_dict(dmr_state)
                    print("DMR encoder loaded successfully")
                else:
                    print(f"DMR encoder shape mismatch: expected {expected_shape}, got {dmr_state.get('0.weight', 'unknown').shape if '0.weight' in dmr_state else 'unknown'}")
                    print("Skipping DMR encoder load to avoid dimension errors")
            else:
                print("Warning: DMR encoder structure not as expected")
        except Exception as e:
            print(f"Warning: Failed to load DMR encoder from pickle: {e}")
    else:
        print(f"Warning: DMR encoder not found at {dmr_encoder_path}")
        
    if os.path.exists(read_classifier_path):
        try:
            print("Loading read classifier...")
            model.read_classifier.load_state_dict(torch.load(read_classifier_path, map_location=device))
            print("Read classifier loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load read classifier from pickle: {e}")
    else:
        print(f"Warning: Read classifier not found at {read_classifier_path}")
    
    model.to(device)
    model.eval()
    
    # Load class names if available
    class_names = load_class_names(model_dir, num_classes)
    
    return model, vocab, class_names


def validate_data_compatibility(csv_file, model, vocab):
    """Validate that the input data is compatible with the loaded model"""
    print("\nValidating data compatibility...")
    
    # Check sequence length compatibility
    expected_seq_len = getattr(model, 'seq_len', None)
    if expected_seq_len is None:
        print("Warning: Could not determine expected sequence length from model")
        return True
    
    # Check DMR label range compatibility
    expected_dmr_range = getattr(model, 'num_dmr_embeddings', None)
    if expected_dmr_range is None:
        print("Warning: Could not determine expected DMR label range from model")
        return True
    
    # Sample a few rows to check compatibility
    try:
        df = pd.read_csv(csv_file, sep='\t', nrows=100)
        if 'dna_seq' not in df.columns:
            df = pd.read_csv(csv_file, sep=',', nrows=100)
        
        # Check sequence lengths
        max_seq_len = max(len(str(seq).split()) for seq in df['dna_seq'].dropna())
        if max_seq_len > expected_seq_len:
            print(f"Warning: Maximum sequence length ({max_seq_len}) exceeds model's expected length ({expected_seq_len})")
            print("Sequences will be truncated during processing")
        
        # Check DMR label range
        if 'dmr_label' in df.columns:
            dmr_labels = pd.to_numeric(df['dmr_label'], errors='coerce').dropna()
            min_dmr = int(dmr_labels.min())
            max_dmr = int(dmr_labels.max())
            
            if max_dmr >= expected_dmr_range:
                print(f"Warning: DMR labels range [{min_dmr}, {max_dmr}] exceeds model's embedding range [0, {expected_dmr_range-1}]")
                print("Labels will be clamped to valid range during processing")
            
            print(f"DMR label range in data: [{min_dmr}, {max_dmr}]")
            print(f"Model DMR embedding range: [0, {expected_dmr_range-1}]")
        
        print("Data compatibility validation completed")
        return True
        
    except Exception as e:
        print(f"Warning: Could not validate data compatibility: {e}")
        return True


def predict_csv(csv_file, model_dir='Training/Run_Result/bert.model', device='cpu', seq_len=None):
    """Predict cell types for a CSV file"""
    
    # Load model and vocabulary
    model, vocab, class_names = load_model(model_dir, device)
    
    # Validate data compatibility
    validate_data_compatibility(csv_file, model, vocab)
    
    # Derive sequence length: prefer caller value, fallback to model setting
    if seq_len is None:
        seq_len = getattr(model, 'seq_len', getattr(model.config, 'seq_len', 300))

    # Create dataset and dataloader – clamp labels to valid range
    dataset = PredictionDataset(csv_file, vocab, seq_len=seq_len, num_labels=model.config.num_labels)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    all_predictions = []
    all_probabilities = []
    original_ctypes = []
    
    print("\nMaking predictions...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) 
            dmr_labels = batch['dmr_label'].to(device)
            
            # Clamp DMR labels to valid range for the model's DMR encoder
            if hasattr(model, 'num_dmr_embeddings'):
                dmr_labels = torch.clamp(dmr_labels, 0, model.num_dmr_embeddings - 1)
            else:
                # Fallback: clamp to a reasonable range
                dmr_labels = torch.clamp(dmr_labels, 0, 10000)
            
            # Forward pass - provide dummy ctype_label for inference
            dummy_ctype_labels = torch.zeros(input_ids.size(0), dtype=torch.long).to(device)
            
            try:
                outputs = model(
                    step=0,  # Not used during inference
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=dmr_labels,
                    ctype_label=dummy_ctype_labels  # Dummy labels for inference
                )
                
                # network returns probabilities in "classification_logits"
                if isinstance(outputs, dict) and "classification_logits" in outputs:
                    probabilities = outputs["classification_logits"]
                elif hasattr(outputs, 'logits'):
                    probabilities = F.softmax(outputs.logits, dim=-1)
                else:
                    raw = outputs[0] if isinstance(outputs, tuple) else outputs
                    probabilities = F.softmax(raw, dim=-1)
                
                predictions = torch.argmax(probabilities, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                original_ctypes.extend(batch['original_ctype'])
                
            except RuntimeError as e:
                if "CUDA error: device-side assert triggered" in str(e) or "indexSelectLargeIndex" in str(e):
                    print(f"CUDA indexing error in batch {batch_idx + 1}, skipping...")
                    print(f"Error details: {e}")
                    # Skip this batch and continue
                    continue
                else:
                    raise e
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")
    
    if not all_predictions:
        raise RuntimeError("No predictions were made successfully. Check your data and model compatibility.")
    
    return all_predictions, all_probabilities, original_ctypes, class_names


def create_sample_csv(filename='CTC-103-LE5-5_clean_r1.rmdup.csv'):
    """Create a sample CSV file for testing if the target file doesn't exist"""
    print(f"Creating sample CSV file: {filename}")
    
    # Sample data based on the training format
    sample_data = [
        ["GCC CCG CGC GCA CAG AGC GCG CGG GGA GAT ATG TGT GTG TGG GGG GGA GAG AGT GTG TGT GTG TGG GGT GTT TTG TGT GTG TGG GGT GTT TTT TTG TGG GGA GAG AGA GAT ATT TTC TCT CTG TGA GAT ATG TGT GTC TCA CAT ATT TTT TTC TCC CCT CTG TGT GTG TGG GGA GAT ATA TAC ACA CAG AGA GAC ACC CCC CCA CAG AGC GCA CAG AGT GTG TGG GGG GGC GCT CTT TTG TGC GCT CTG TGG GGA GAC ACC CCG CGT GTC", "2022220222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222022", "6696", "CTC-103-LE5-5", "Unknown"],
        ["CAA AAG AGC GCC CCT CTC TCG CGA GAA AAT ATG TGC GCA CAC ACA CAC ACT CTT TTG TGT GTA TAT ATG TGC GCA CAT ATG TGC GCA CAC ACA CAC ACA CAC ACA CAC ACA CAT ATA TAG AGG GGA GAG AGC GCT CTA TAG AGA GAA AAG AGG GGA GAG AGC GCT CTG TGG GGT GTT TTG TGC GCT CTC TCT CTA TAT ATG TGG GGC GCA CAA AAG AGA GAA AAG AGC GCA CAG AGC GCA CAA AAA AAG AGA GAA AAG AGA GAT ATA TAA AAG AGA GAT ATC TCA CAC ACC CCA CAT ATA TAC ACT CTG TGC GCA CAA AAA AAG AGG GGA GAA AAG AGG GGT GTA TAA AAG AGC GCC CCA CAC ACG CGT GTG TGA GAT ATG TGA GAT ATG TGT GTG TGG GGG GGT GTA TAA AAC ACC CCC CCC CCA CAG AGG GGC GCA CAC ACA CAG AGA GAT ATG TGA GAT ATA TAC ACC CCC CCT CTC TCA CAG AGC GCG", "222222222022222202222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222022222222222222222222222222222222222222220", "6075", "CTC-103-LE5-5", "Unknown"],
        ["GGA GAA AAC ACC CCT CTC TCG CGG GGA GAG AGC GCA CAC ACC CCT CTT TTG TGT GTT TTT TTC TCC CCC CCC CCA CAC ACA CAG AGC GCC CCT CTT TTT TTT TTA TAA AAC ACT CTA TAC ACG CGG GGT GTT TTT TTT TTG TGA GAG AGG GGC GCT CTG TGC GCT CTT TTT TTC TCA CAA AAG AGA GAT ATT TTT TTG TGT GTA TAG AGC GCT CTG TGC GCT CTT TTT TTT TTA TAA AAC ACT CTT TTT TTT TTT TTC TCT CTT TTG TGT GTT TTT TTT TTC TCG CGG GGT GTG TGT GTT TTC TCT CTG TGT GTG TGG GGT GTT TTT TTC TCA CAT ATT TTG TGT GTG TGA GAT ATT TTG TGT GTC TCT CTA TAA AAT ATA TAG AGT GTA TAC ACA CAT ATT TTT TTA TAA AAA AAA AAT ATT TTT TTG TGT GTA TAT ATT TTC TCT CTG TGT GTT TTT TTG TGG GGG GGA GAT ATT TTC TCA CAT ATC TCA CAG AGA GAA AAT ATT TTC TCC CCT CTC TCT CTG TGT GTC TCT CTG TGT GTG TGA GAC ACT CTT TTG TGA GAT ATG TGT GTC TCT CTT TTT TTA TAT ATC TCA CAG AGT GTA TAA AAG AGC GCA CAG AGG GGA GAT ATA TAT ATA TAG AGA GAG AGG GGA GAT ATA TAC ACA CAC ACA CAC ACA CAA AAA AAA AAA AAT ATG TGA GAA AAT ATA TAA AAT ATT TTG TGA GAG AGG GGA GAA AAA AAG AGT GTT TTT TTA TAA AAT ATA TAC ACT CTT TTA TAG AGA GAT ATT TTA TAT ATT TTT TTA TAC ACA CAA AAA AAA AAA AAA AAT ATA TAC ACA CAG AGG GGC GCA CAG AGG GGG GGT GTT TTA TAG AGG GGG GGA GAA AAA AAG AGC GCA CAA AAC ACA CAG AGA GAG AGG GGA GAT ATG TGG GGT GTT TTC TCG CGG GGT GTA TAC ACC CCT CTA TAG AGA GAG", "22222202222222222222222222222222222222020222222222222222222222222222222222222222222222222222020222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222022222222226394", "6394", "CTC-103-LE5-5", "Unknown"]
    ]
    
    df = pd.DataFrame(sample_data, columns=['dna_seq', 'methyl_seq', 'dmr_label', 'ctype', 'dmr_ctype'])
    df.to_csv(filename, sep='\t', index=False)
    print(f"Sample CSV created with {len(df)} sequences")
    return filename


def main():
    parser = argparse.ArgumentParser(description='Predict cell types from CSV file')
    parser.add_argument('--csv_file', type=str, default='Blood-B_TX_250.csv',
                        help='Path to input CSV file')
    parser.add_argument('--model_dir', type=str, default='Training/Run_Result/bert.model',
                        help='Path to model directory')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--create_sample', action='store_true',
                        help='Create a sample CSV file for testing')
    parser.add_argument('--seq_len', type=int, default=None,
                        help='Optional sequence length; if omitted, uses the value inferred from the model weights/config')
    
    args = parser.parse_args()
    
    # Create sample CSV if requested or if file doesn't exist
    if args.create_sample or not os.path.exists(args.csv_file):
        print(f"File {args.csv_file} not found, creating sample CSV...")
        create_sample_csv(args.csv_file)
    else:
        # Check if the existing file has the right format
        try:
            with open(args.csv_file, 'r') as f:
                first_line = f.readline().strip()
                # Count potential columns with different separators
                tab_cols = len(first_line.split('\t'))
                comma_cols = len(first_line.split(','))
                space_cols = len(first_line.split(' '))
                
                max_cols = max(tab_cols, comma_cols, space_cols)
                print(f"File {args.csv_file} detected with {max_cols} potential columns")
                
                if max_cols < 3:
                    print(f"Warning: File appears to have only {max_cols} columns, but we need at least 3 columns")
                    print(f"First line preview: {first_line[:200]}...")
                    print("You may want to check your file format or use --create_sample to create a sample file")
        except Exception as e:
            print(f"Error checking file format: {e}")
    
    # Make predictions
    predictions, probabilities, original_ctypes, class_names = predict_csv(
        args.csv_file, args.model_dir, args.device, args.seq_len
    )
    
    print("\n" + "="*80)
    print("CELL TYPE PREDICTION RESULTS")
    print("="*80)
    
    # Convert predictions to cell type names
    predicted_cell_types = [class_names[pred] for pred in predictions]
    
    # Calculate aggregate probabilities for final prediction
    avg_probabilities = np.mean(probabilities, axis=0)
    final_prediction_idx = int(np.argmax(avg_probabilities))
    final_prediction = class_names[final_prediction_idx]
    final_confidence = float(avg_probabilities[final_prediction_idx])
    
    print(f"\nFinal Prediction: {final_prediction}")
    print(f"Confidence: {final_confidence:.4f}")
    
    print(f"\nTotal sequences analyzed: {len(predictions)}")
    
    # Show probability distribution for all cell types
    print(f"\nProbability Distribution (averaged across all sequences):")
    print("-" * 60)
    
    # Sort cell types by probability (descending)
    sorted_indices = np.argsort(avg_probabilities)[::-1]
    
    for i, idx in enumerate(sorted_indices):
        cell_type = class_names[int(idx)]
        prob = float(avg_probabilities[int(idx)])
        print(f"{i+1:2d}. {cell_type:<20} : {prob:.4f} ({prob*100:.2f}%)")
    
    # Show prediction breakdown by sequence
    print(f"\nPer-sequence predictions:")
    print("-" * 40)
    
    prediction_counts = defaultdict(int)
    for pred in predicted_cell_types:
        prediction_counts[pred] += 1
    
    print(f"Prediction summary (out of {len(predictions)} sequences):")
    for cell_type, count in sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(predictions)) * 100
        print(f"  {cell_type:<20} : {count:3d} sequences ({percentage:.1f}%)")
    
    # Show top 5 most confident predictions per sequence
    print(f"\nTop 5 predictions for first few sequences:")
    print("-" * 50)
    
    for i in range(min(3, len(probabilities))):
        print(f"\nSequence {i+1}:")
        seq_probs = probabilities[i]
        top_indices = np.argsort(seq_probs)[::-1][:5]
        
        for j, idx in enumerate(top_indices):
            cell_type = class_names[int(idx)]
            prob = float(seq_probs[int(idx)])
            print(f"  {j+1}. {cell_type:<20} : {prob:.4f}")

if __name__ == "__main__":
    main() 