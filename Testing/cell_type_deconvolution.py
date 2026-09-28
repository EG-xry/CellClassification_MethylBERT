#!/usr/bin/env python3
"""
Cell Type Deconvolution using MethylBERT

This script performs deconvolution analysis on methylation data by:
1. Loading a pre-trained MethylBERT model
2. Processing input CSV data
3. Predicting cell types for each read
4. Analyzing the distribution to determine the final cell type composition

Features:
- Standard deconvolution analysis with confidence metrics
- Optional DMR_cType analysis that groups reads by DMR_cType regions
- Comprehensive confidence analysis by cell type
- Tab-delimited output format for compatibility
- Batch processing support via JSON configuration files
- Automatic sequence length detection from model files (no manual configuration needed)

Usage:
    # Single file mode:
    python cell_type_deconvolution.py --csv input.csv --model_dir path/to/model --device cpu
    
    # For DMR_cType analysis:
    python cell_type_deconvolution.py --csv input.csv --model_dir path/to/model --use_dmr_ctype
    
    # Batch processing mode:
    python cell_type_deconvolution.py --config batch_config.json

Note: The sequence length is automatically detected from the model files, so you don't need to specify it.
"""

import os
import sys
import argparse
import warnings
import pickle
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append('Training/src')

from methylbert.network import MethylBertEmbeddedDMR
from methylbert.data.vocab import MethylVocab
from methylbert.data.dataset import MethylBertFinetuneDataset
from methylbert.config import MethylBERTConfig

def load_json_config(config_path):
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to the JSON configuration file
    
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    except Exception as e:
        raise ValueError(f"Error reading configuration file: {e}")
    
    # Validate required fields
    required_fields = ['files']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in configuration file")
    
    # Validate files list
    if not isinstance(config['files'], list) or len(config['files']) == 0:
        raise ValueError("'files' must be a non-empty list")
    
    # Validate each file configuration
    for i, file_config in enumerate(config['files']):
        if not isinstance(file_config, dict):
            raise ValueError(f"File configuration at index {i} must be a dictionary")
        
        if 'csv' not in file_config:
            raise ValueError(f"File configuration at index {i} missing required 'csv' field")
        
        # Set default values for optional fields from global_settings if available
        global_settings = config.get('global_settings', {})
        file_config.setdefault('model_dir', global_settings.get('model_dir', 'Training/Run_Result/bert.model'))
        file_config.setdefault('device', global_settings.get('device', 'auto'))
        file_config.setdefault('batch_size', global_settings.get('batch_size', 128))
        # seq_len is now auto-detected from model files, no need to set default
        file_config.setdefault('output_format', global_settings.get('output_format', 'tab'))
        file_config.setdefault('ignore_ctype_mismatch', global_settings.get('ignore_ctype_mismatch', False))
        file_config.setdefault('force_model_classes', global_settings.get('force_model_classes', False))
        file_config.setdefault('use_generic_names', global_settings.get('use_generic_names', False))
        file_config.setdefault('use_dmr_ctype', global_settings.get('use_dmr_ctype', False))
    
    return config

def process_single_file(file_config, global_config=None):
    """
    Process a single CSV file with the given configuration
    
    Args:
        file_config: Configuration for this specific file
        global_config: Global configuration (optional)
    
    Returns:
        dict: Results from processing this file
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING FILE: {file_config['csv']}")
    print(f"{'='*80}")
    
    # Extract arguments from config
    csv_path = file_config['csv']
    model_dir = file_config.get('model_dir', 'Training/Run_Result/bert.model')
    device_choice = file_config.get('device', 'auto')
    batch_size = file_config.get('batch_size', 128)
    seq_len = file_config.get('seq_len', 300)
    output_format = file_config.get('output_format', 'tab')
    ignore_ctype_mismatch = file_config.get('ignore_ctype_mismatch', False)
    force_model_classes = file_config.get('force_model_classes', False)
    use_generic_names = file_config.get('use_generic_names', False)
    use_dmr_ctype = file_config.get('use_dmr_ctype', False)
    
    try:
        # Set device
        device = get_device(device_choice)
        print(f"Using device: {device}")
        
        # Create vocabulary
        print("Creating vocabulary...")
        vocab = MethylVocab(k=3)
        
        # Load model first to get the correct sequence length
        print("Loading model to determine expected parameters...")
        
        # Create a temporary dataset just to get basic info
        temp_dataset = create_dataset_from_csv(csv_path, vocab, seq_len, ignore_ctype_mismatch, True, use_dmr_ctype)
        temp_num_classes = temp_dataset.num_classes()
        
        model, config = load_model(model_dir, device, temp_num_classes, force_model_classes)
        
        final_num_classes = config.num_classes
        actual_seq_len = config.seq_len
        
        print(f"Model expects {final_num_classes} classes and seq_len {actual_seq_len}")
        print(f"Recreating dataset with actual seq_len: {actual_seq_len}")
        
        # Recreate dataset with the correct sequence length from the model
        dataset = create_dataset_from_csv(csv_path, vocab, actual_seq_len, ignore_ctype_mismatch, True, use_dmr_ctype)
        
        # Override class mapping
        dataset._num_classes = final_num_classes
        new_mapping = {i: f"Class_{i}" for i in range(final_num_classes)}
        dataset.int_to_ctype = new_mapping
        
        original_num_classes = dataset.num_classes
        def get_num_classes():
            return final_num_classes
        dataset.num_classes = get_num_classes
        
        print(f"Final dataset info:")
        print(f"  Number of classes: {dataset.num_classes()}")
        print(f"  Class mapping: {dataset.int_to_ctype}")
        print(f"  Number of reads: {len(dataset)}")
        
        # Run predictions
        predictions, probabilities, true_labels = predict_cell_types(
            model, dataset, device, batch_size
        )
        
        # Analyze results
        dmr_ctype_data = getattr(dataset, 'dmr_ctype_data', None)
        results = analyze_deconvolution_results(predictions, probabilities, dataset, output_format, use_dmr_ctype, dmr_ctype_data)
        
        # Add file information to results
        results['file_info'] = {
            'csv_path': csv_path,
            'model_dir': model_dir,
            'device': str(device),
            'config': file_config
        }
        
        # Output results
        output_results(results, output_format)
        
        # Save results to file
        output_file = f"deconvolution_results_{os.path.basename(csv_path).replace('.csv', '')}.txt"
        save_results_to_file(results, output_file, file_config)
        
        # Save DMR_cType analysis if available
        if 'dmr_ctype_analysis' in results:
            table_file = f"dmr_ctype_analysis_{os.path.basename(csv_path).replace('.csv', '')}.txt"
            save_dmr_ctype_analysis(results, table_file, csv_path, model_dir)
        
        # Clean up temporary files
        cleanup_temp_files(dataset)
        
        return results
        
    except Exception as e:
        print(f"Error processing file {csv_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_results_to_file(results, output_file, file_config):
    """Save results to a text file"""
    try:
        with open(output_file, 'w') as f:
            f.write("CELL TYPE DECONVOLUTION RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Input file: {file_config['csv']}\n")
            f.write(f"Model directory: {file_config.get('model_dir', 'Training/Run_Result/bert.model')}\n")
            f.write(f"Device: {results['file_info']['device']}\n")
            f.write(f"Total reads: {results['total_reads']}\n\n")
            
            f.write("FINAL VERDICT:\n")
            f.write(f"  Dominant Cell Type: {results['final_verdict']['dominant_cell_type']}\n")
            f.write(f"  Percentage: {results['final_verdict']['dominant_percentage']}%\n")
            f.write(f"  Overall Confidence: {results['final_verdict']['confidence']}\n\n")
            
            f.write("CELL TYPE BREAKDOWN:\n")
            for cell_type, data in results['cell_type_breakdown'].items():
                f.write(f"  {cell_type}: {data['count']} reads ({data['percentage']}%)\n")
            
            f.write("\nCONFIDENCE ANALYSIS:\n")
            f.write(f"  Overall Average Confidence: {results['confidence_analysis']['overall_avg_confidence']}\n")
            f.write(f"  Overall Confidence Std: {results['confidence_analysis']['overall_confidence_std']}\n")
            
            f.write("\nConfidence by Cell Type:\n")
            for cell_type, conf_data in results['confidence_analysis']['by_cell_type'].items():
                f.write(f"  {cell_type}:\n")
                f.write(f"    Avg: {conf_data['avg_confidence']}, Std: {conf_data['std_confidence']}\n")
                f.write(f"    Range: {conf_data['min_confidence']} - {conf_data['max_confidence']}\n")
            
            # Add DMR_cType analysis if available
            if 'dmr_ctype_analysis' in results:
                f.write("\n" + "="*60 + "\n")
                f.write("DMR_cType ANALYSIS BY REGION\n")
                f.write("="*60 + "\n")
                
                dmr_analysis = results['dmr_ctype_analysis']
                dmr_ctype_to_region = dmr_analysis['dmr_ctype_to_region']
                
                f.write("DMR_cType to Region mapping:\n")
                for dmr_ctype, region in dmr_ctype_to_region.items():
                    f.write(f"  {dmr_ctype} -> {region}\n")
                
                f.write("\nDetailed breakdown by region:\n")
                
                for region_name, region_data in dmr_analysis['by_region'].items():
                    f.write(f"\n{region_name} (DMR_cType: {region_data['dmr_ctype']}):\n")
                    f.write(f"  Total reads: {region_data['total_reads']}\n")
                    f.write(f"  Dominant cell type: {region_data['dominant_cell_type']} ({region_data['dominant_percentage']}%)\n")
                    
                    f.write("  Class\tPercentage\tCount\tConfidence_Avg\n")
                    for cell_type, data in region_data['breakdown'].items():
                        f.write(f"  {cell_type}\t{data['percentage']}%\t{data['count']}\t{data['confidence_avg']}\n")
        
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Warning: Could not save results to file: {e}")

def save_dmr_ctype_analysis(results, table_file, csv_path, model_dir):
    """Save DMR_cType analysis as separate table file"""
    try:
        with open(table_file, 'w') as f:
            f.write("DMR_cType ANALYSIS TABLE\n")
            f.write("="*60 + "\n")
            f.write(f"Input file: {csv_path}\n")
            f.write(f"Model directory: {model_dir}\n")
            f.write(f"Total reads: {results['total_reads']}\n\n")
            
            dmr_analysis = results['dmr_ctype_analysis']
            dmr_ctype_to_region = dmr_analysis['dmr_ctype_to_region']
            
            f.write("DMR_cType to Region mapping:\n")
            for dmr_ctype, region in dmr_ctype_to_region.items():
                f.write(f"  {dmr_ctype} -> {region}\n")
            f.write("\n")
            
            for region_name, region_data in dmr_analysis['by_region'].items():
                f.write(f"{region_name}:\n")
                f.write("Class\tPercentage\tCount\tConfidence_Avg\n")
                for cell_type, data in region_data['breakdown'].items():
                    f.write(f"{cell_type}\t{data['percentage']}%\t{data['count']}\t{data['confidence_avg']}\n")
                f.write("\n")
        
        print(f"DMR_cType analysis table saved to: {table_file}")
        
    except Exception as e:
        print(f"Warning: Could not save DMR_cType analysis table: {e}")

def cleanup_temp_files(dataset):
    """Clean up temporary files if they were created"""
    try:
        if hasattr(dataset, '_temp_csv_path'):
            import tempfile
            import shutil
            temp_dir = os.path.dirname(dataset._temp_csv_path)
            if os.path.exists(temp_dir) and 'processed_' in os.path.basename(dataset._temp_csv_path):
                shutil.rmtree(temp_dir)
                print("Cleaned up temporary files")
    except Exception as e:
        print(f"Warning: Could not clean up temporary files: {e}")

def process_multiple_files(config_path):
    """
    Process multiple files using JSON configuration
    
    Args:
        config_path: Path to the JSON configuration file
    """
    print("Loading JSON configuration...")
    config = load_json_config(config_path)
    
    print(f"Found {len(config['files'])} files to process")
    
    # Process each file
    all_results = []
    successful_files = 0
    failed_files = 0
    
    for i, file_config in enumerate(config['files']):
        print(f"\nProcessing file {i+1}/{len(config['files'])}: {file_config['csv']}")
        
        try:
            results = process_single_file(file_config, config)
            if results is not None:
                all_results.append(results)
                successful_files += 1
                print(f"✓ Successfully processed: {file_config['csv']}")
            else:
                failed_files += 1
                print(f"✗ Failed to process: {file_config['csv']}")
        except Exception as e:
            failed_files += 1
            print(f"✗ Error processing {file_config['csv']}: {str(e)}")
    
    # Summary
    print(f"\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total files: {len(config['files'])}")
    print(f"Successful: {successful_files}")
    print(f"Failed: {failed_files}")
    
    if successful_files > 0:
        print(f"\nSuccessfully processed files:")
        for results in all_results:
            file_info = results['file_info']
            print(f"  - {file_info['csv_path']}")
            print(f"    Dominant: {results['final_verdict']['dominant_cell_type']} ({results['final_verdict']['dominant_percentage']}%)")
    
    if failed_files > 0:
        print(f"\nFailed files:")
        for i, file_config in enumerate(config['files']):
            if i >= len(all_results) or all_results[i] is None:
                print(f"  - {file_config['csv']}")
    
    return all_results

def preprocess_csv_headers(csv_path, vocab, seq_len, include_dmr_ctype=False):
    """
    Preprocess CSV file to handle header variations and normalize column names.
    Creates a temporary processed file with standardized headers.
    
    Args:
        csv_path: Path to the original CSV file
        vocab: MethylVocab object
        seq_len: Sequence length for processing
        include_dmr_ctype: If True, include DMR_cType column in processing
    
    Returns:
        str: Path to the processed CSV file
    """
    import tempfile
    import shutil
    
    print("Preprocessing CSV headers for compatibility...")
    
    # Read the original CSV with better delimiter detection
    df = None
    delimiter_detected = None
    
    # First, try to read a small sample to detect the delimiter
    try:
        with open(csv_path, 'r') as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
        
        # Count delimiters in the first line
        tab_count = first_line.count('\t')
        comma_count = first_line.count(',')
        
        # Determine the most likely delimiter
        if tab_count > comma_count:
            delimiter = '\t'
        elif comma_count > tab_count:
            delimiter = ','
        else:
            # If equal, try both and see which gives more columns
            try:
                df_tab = pd.read_csv(csv_path, sep='\t', nrows=1)
                df_comma = pd.read_csv(csv_path, sep=',', nrows=1)
                
                if len(df_tab.columns) > len(df_comma.columns):
                    delimiter = '\t'
                else:
                    delimiter = ','
            except:
                delimiter = ','  # Default to comma
        
        # Read the full file with detected delimiter
        df = pd.read_csv(csv_path, sep=delimiter)
        delimiter_detected = delimiter
        
    except Exception as e:
        # Fallback: try common delimiters
        for sep in [',', '\t', ';', '|']:
            try:
                df = pd.read_csv(csv_path, sep=sep)
                delimiter_detected = sep
                break
            except Exception as e2:
                continue
        
        if df is None:
            # Last resort: try to manually parse the file
            try:
                with open(csv_path, 'r') as f:
                    lines = [f.readline().strip() for _ in range(5)]  # Read first 5 lines
                
                # Try to detect the pattern
                for line in lines:
                    if line:
                        # Count different delimiters
                        delims = {'\t': line.count('\t'), ',': line.count(','), ';': line.count(';'), '|': line.count('|')}
                        most_common = max(delims, key=delims.get)
                        if delims[most_common] > 0:
                            try:
                                df = pd.read_csv(csv_path, sep=most_common)
                                delimiter_detected = most_common
                                break
                            except Exception as e3:
                                continue
                
                if df is None:
                    raise ValueError(f"All CSV parsing methods failed. File format may be corrupted or unusual.")
                    
            except Exception as e3:
                raise ValueError(f"Could not read CSV file with any method. "
                               f"Error: {e3}")
    
    # Get original headers and clean them
    original_headers = list(df.columns)
    
    # Clean headers (remove extra whitespace and newlines)
    original_headers = [str(h).strip().replace('\n', '').replace('\r', '') for h in original_headers]
    
    # Define required columns and their possible variations
    required_columns = {
        'dna_seq': ['dna_seq', 'DNA_seq', 'DNA_SEQ', 'dna_sequence', 'DNA_sequence', 'sequence', 'seq', 'Seq', 'SEQ'],
        'methyl_seq': ['methyl_seq', 'Methyl_seq', 'METHYL_SEQ', 'methyl_sequence', 'Methyl_sequence', 'methylation', 'methyl', 'Methyl', 'METHYL'],
        'ctype': ['ctype', 'Ctype', 'CTYPE', 'cell_type', 'Cell_type', 'CELL_TYPE', 'celltype', 'CellType', 'cell', 'Cell', 'CELL', 'cType'],
        'dmr_label': ['dmr_label', 'DMR_label', 'DMR_LABEL', 'dmr', 'DMR', 'DMR_Label']
    }
    
    # Add DMR_cType if requested
    if include_dmr_ctype:
        required_columns['dmr_ctype'] = ['dmr_ctype', 'DMR_cType', 'DMR_CTYPE', 'dmr_cell_type', 'DMR_cell_type', 'DMR_CELL_TYPE', 'dmr_celltype', 'DMR_CellType', 'dmr_cell', 'DMR_Cell', 'DMR_CELL']
    
    # Map headers to standard names
    header_mapping = {}
    missing_columns = []
    
    for standard_name, possible_variations in required_columns.items():
        found = False
        
        # First try exact matches
        for variation in possible_variations:
            if variation in original_headers:
                header_mapping[variation] = standard_name
                found = True
                break
        
        # If no exact match, try case insensitive and underscore insensitive matching
        if not found:
            for variation in possible_variations:
                for header in original_headers:
                    # Case insensitive comparison
                    if header.lower() == variation.lower():
                        header_mapping[header] = standard_name
                        found = True
                        break
                    # Underscore insensitive comparison
                    elif header.replace('_', '').lower() == variation.replace('_', '').lower():
                        header_mapping[header] = standard_name
                        found = True
                        break
                    # Partial match (e.g., 'DNA_seq' matches 'dna_seq')
                    elif variation.lower() in header.lower() or header.lower() in variation.lower():
                        header_mapping[header] = standard_name
                        found = True
                        break
                
                if found:
                    break
        
        if not found:
            missing_columns.append(standard_name)
    
    # Check if we have all required columns
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        print(f"Available columns: {original_headers}")
        raise ValueError(f"Missing required columns: {missing_columns}. "
                       f"Available columns: {original_headers}")
    
    # Rename columns to standard names
    df_renamed = df.rename(columns=header_mapping)
    
    # Keep only the required columns (ignore extra columns)
    required_column_names = list(required_columns.keys())
    df_final = df_renamed[required_column_names]
    
    # Ensure columns are in the expected order
    df_final = df_final[required_column_names]
    
    # Clean the data: remove rows with empty or invalid values
    print(f"Cleaning data...")
    initial_rows = len(df_final)
    
    try:
        # Remove rows where any required column has empty/NaN values
        for col in required_column_names:
            # Convert NaN to empty string for easier handling
            df_final[col] = df_final[col].fillna('')
            # Remove rows with empty strings
            df_final = df_final[df_final[col] != '']
        
        # Additional validation for dmr_label (must be convertible to int)
        if 'dmr_label' in df_final.columns:
            valid_dmr_rows = []
            for idx, row in df_final.iterrows():
                try:
                    int(row['dmr_label'])
                    valid_dmr_rows.append(idx)
                except (ValueError, TypeError):
                    continue
            
            df_final = df_final.loc[valid_dmr_rows]
        
        # Additional validation for methyl_seq (must contain valid methylation data)
        if 'methyl_seq' in df_final.columns:
            valid_methyl_rows = []
            for idx, row in df_final.iterrows():
                methyl_seq = str(row['methyl_seq'])
                if methyl_seq and any(c in methyl_seq for c in '012'):
                    valid_methyl_rows.append(idx)
            
            df_final = df_final.loc[valid_methyl_rows]
        
        final_rows = len(df_final)
        print(f"Data cleaning complete: {initial_rows} -> {final_rows} rows ({initial_rows - final_rows} invalid rows removed)")
        
        if final_rows == 0:
            raise ValueError("No valid rows found after data cleaning. Please check your CSV file for data quality issues.")
            
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        raise ValueError(f"Data cleaning failed: {e}. Please check your CSV file for data quality issues.")
    
    # Validate and convert data types
    try:
        # Ensure dmr_label is integer
        if 'dmr_label' in df_final.columns:
            df_final['dmr_label'] = df_final['dmr_label'].astype(int)
        
        # Ensure methyl_seq is string
        if 'methyl_seq' in df_final.columns:
            df_final['methyl_seq'] = df_final['methyl_seq'].astype(str)
        
        # Ensure ctype is string
        if 'ctype' in df_final.columns:
            df_final['ctype'] = df_final['ctype'].astype(str)
        
        # Ensure dmr_ctype is string (if present)
        if 'dmr_ctype' in df_final.columns:
            df_final['dmr_ctype'] = df_final['dmr_ctype'].astype(str)
        
        # Ensure dna_seq is string
        if 'dna_seq' in df_final.columns:
            df_final['dna_seq'] = df_final['dna_seq'].astype(str)
            
    except Exception as e:
        print(f"Error during data type conversion: {e}")
        raise ValueError(f"Data type conversion failed: {e}. Please check your CSV file for data quality issues.")
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_csv_path = os.path.join(temp_dir, f"processed_{os.path.basename(csv_path)}")
    
    # Save processed CSV
    df_final.to_csv(temp_csv_path, sep='\t', index=False)
    print(f"Created processed CSV: {temp_csv_path}")
    
    return temp_csv_path

def get_device(device_choice):
    """
    Get the appropriate device based on user choice and availability
    Args:
        device_choice: 'cpu', 'gpu', or 'auto'
    Returns:
        torch.device: The selected device
    """
    if device_choice == 'cpu':
        return torch.device('cpu')
    elif device_choice == 'gpu':
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            print("Warning: GPU requested but not available. Falling back to CPU.")
            return torch.device('cpu')
    elif device_choice == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        raise ValueError("device_choice must be 'cpu', 'gpu', or 'auto'")

def detect_model_seq_len(model_dir):
    """
    Automatically detect the sequence length from model files.
    This is critical because the model architecture is fixed based on training sequence length.
    
    Args:
        model_dir: Directory containing the model files
    
    Returns:
        int: Detected sequence length
    """
    print("Auto-detecting sequence length from model files...")
    
    # Method 1: Check read_classification_model.pickle
    rc_path = os.path.join(model_dir, 'read_classification_model.pickle')
    if os.path.exists(rc_path):
        try:
            state = torch.load(rc_path, map_location='cpu')
            # Look for the final layer weight which has shape [num_classes, seq_len+1]
            # The final layer is typically the last one in the sequence
            final_layer_key = None
            max_layer_num = -1
            
            for key in state.keys():
                if key.endswith('weight') and '.' in key:
                    try:
                        layer_num = int(key.split('.')[0])
                        if layer_num > max_layer_num:
                            max_layer_num = layer_num
                            final_layer_key = key
                    except ValueError:
                        continue
            
            if final_layer_key and final_layer_key in state:
                tensor = state[final_layer_key]
                if hasattr(tensor, 'shape') and len(tensor.shape) == 2:
                    # The final layer should have shape [num_classes, seq_len+1]
                    # So seq_len = tensor.shape[1] - 1
                    detected_seq_len = tensor.shape[1] - 1
                    if detected_seq_len > 0:
                        print(f"Detected seq_len={detected_seq_len} from read_classifier final layer {final_layer_key}")
                        return detected_seq_len
        except Exception as e:
            print(f"Warning: Could not read read_classifier pickle: {e}")
    
    # Method 2: Check model.safetensors
    st_path = os.path.join(model_dir, 'model.safetensors')
    if os.path.exists(st_path):
        try:
            import safetensors.torch
            sdict = safetensors.torch.load_file(st_path)
            # Look for read_classifier final layer (the one with smallest input size)
            read_classifier_layers = []
            for key, tensor in sdict.items():
                if 'read_classifier' in key and key.endswith('weight') and len(tensor.shape) == 2:
                    read_classifier_layers.append((key, tensor.shape[1]))
            
            if read_classifier_layers:
                # The final layer typically has the smallest input size
                final_layer_key, input_size = min(read_classifier_layers, key=lambda x: x[1])
                detected_seq_len = input_size - 1
                if detected_seq_len > 0:
                    print(f"Detected seq_len={detected_seq_len} from {final_layer_key} in model.safetensors")
                    return detected_seq_len
        except Exception as e:
            print(f"Warning: Could not inspect model.safetensors: {e}")
    
    # Method 3: Check dmr_encoder.pickle
    de_path = os.path.join(model_dir, 'dmr_encoder.pickle')
    if os.path.exists(de_path):
        try:
            state = torch.load(de_path, map_location='cpu')
            # DMR encoder has shape [num_dmr_embeddings, seq_len+1]
            for key, tensor in state.items():
                if key.endswith('weight') and hasattr(tensor, 'shape') and len(tensor.shape) == 2:
                    detected_seq_len = tensor.shape[1] - 1
                    if detected_seq_len > 0:
                        print(f"Detected seq_len={detected_seq_len} from dmr_encoder layer {key}")
                        return detected_seq_len
        except Exception as e:
            print(f"Warning: Could not read dmr_encoder pickle: {e}")
    
    # Method 4: Check config.json
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        try:
            config = MethylBERTConfig.from_pretrained(model_dir)
            if hasattr(config, 'seq_len') and config.seq_len is not None:
                print(f"Detected seq_len={config.seq_len} from config.json")
                return config.seq_len
        except Exception as e:
            print(f"Warning: Could not read config.json: {e}")
    
    # Fallback: Use a reasonable default
    print("Warning: Could not auto-detect seq_len, using default value of 250")
    return 250

def load_model(model_dir, device, num_classes, force_model_classes=False):
    """
    Load the MethylBERT model and its components with automatic sequence length detection
    
    Args:
        model_dir: Directory containing the model files
        device: torch.device to load the model on
        num_classes: Number of cell type classes from data (may be ignored if model has more)
        force_model_classes: If True, force the model to use its own class count
    Returns:
        tuple: (model, config)
    """
    print(f"Loading model from: {model_dir}")
    
    # Load the main model
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        print(f"Model directory contents:")
        try:
            if os.path.exists(model_dir):
                for item in os.listdir(model_dir):
                    print(f"  - {item}")
            else:
                print(f"  Model directory does not exist: {model_dir}")
        except Exception as e:
            print(f"  Could not list directory contents: {e}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load the saved config - this contains the original training parameters
    config = MethylBERTConfig.from_pretrained(model_dir)
    model_num_classes = getattr(config, 'num_classes', None) or getattr(config, 'num_labels', None)
    
    print(f"Model config shows {model_num_classes} classes")
    print(f"Data has {num_classes} classes")
    
    # Auto-detect sequence length from model files
    detected_seq_len = detect_model_seq_len(model_dir)
    
    # Use the model's number of classes, not the data's
    # This allows prediction even when data has fewer classes than the model was trained on
    if force_model_classes or (model_num_classes and model_num_classes != num_classes):
        if force_model_classes:
            print(f"Force flag set: Using model's class count ({model_num_classes}) instead of data's ({num_classes})")
        else:
            print(f"Warning: Data has {num_classes} classes but model expects {model_num_classes} classes")
            print(f"Using model's class count ({model_num_classes}) for prediction")
        print(f"Results will be output as Class_0, Class_1, etc.")
        num_classes = model_num_classes
    
    # Update config to match what we'll use
    config.num_classes = num_classes
    config.num_labels = num_classes
    config.seq_len = detected_seq_len
    # Keep the original loss function from the trained model
    # config.loss is already set from the loaded config
    
    # Ensure num_dmr_embeddings is properly set if not already in config
    if not hasattr(config, 'num_dmr_embeddings') or config.num_dmr_embeddings is None:
        # Default value, will be overridden by the actual model file
        config.num_dmr_embeddings = 49260
    
    print(f"Final config - num_classes: {config.num_classes}, seq_len: {config.seq_len}, loss: {config.loss}")
    
    # Create model using the original saved config
    # Pass seq_len explicitly since from_pretrained doesn't automatically use config attributes as constructor args
    model = MethylBertEmbeddedDMR.from_pretrained(
        model_dir,
        config=config,
        seq_len=detected_seq_len,  # Must pass explicitly to constructor
    )
    
    # Load DMR encoder
    dmr_encoder_path = os.path.join(model_dir, "dmr_encoder.pickle")
    if os.path.exists(dmr_encoder_path):
        print(f"Loading DMR encoder from: {dmr_encoder_path}")
        model.from_pretrained_dmr_encoder(dmr_encoder_path, device)
    else:
        print(f"Warning: DMR encoder not found at: {dmr_encoder_path}")
        print(f"  This may cause issues if the model was trained with DMR embeddings")
    
    # Load read classifier
    read_classifier_path = os.path.join(model_dir, "read_classification_model.pickle")
    if os.path.exists(read_classifier_path):
        print(f"Loading read classifier from: {read_classifier_path}")
        model.from_pretrained_read_classifier(read_classifier_path, device)
    else:
        print(f"Warning: Read classifier not found at: {read_classifier_path}")
        print(f"  This may cause issues if the model was trained with custom read classification")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    return model, config

def create_dataset_from_csv(csv_path, vocab, seq_len=250, ignore_ctype_mismatch=False, use_generic_names=False, extract_dmr_ctype=False):
    """
    Create a dataset from the input CSV file
    
    Args:
        csv_path: Path to the input CSV file
        vocab: MethylVocab object
        seq_len: Sequence length for processing
        ignore_ctype_mismatch: If True, ignore ctype column issues
        use_generic_names: If True, override cell type names to generic Class_0, Class_1, etc.
        extract_dmr_ctype: If True, extract DMR_cType data for analysis
    
    Returns:
        Dataset: The created dataset with proper class mapping
    """
    print(f"Creating dataset from: {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Absolute path: {os.path.abspath(csv_path)}")
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Pre-process CSV to handle header variations
    csv_path = preprocess_csv_headers(csv_path, vocab, seq_len, extract_dmr_ctype)
    
    # Create a custom dataset that ignores ctype column issues
    class GenericCellTypeDataset:
        def __init__(self, csv_path, vocab, seq_len, num_classes=6):
            self.csv_path = csv_path
            self.vocab = vocab
            self.seq_len = seq_len
            self._temp_csv_path = csv_path
            
            # Read the CSV to get the number of rows
            import pandas as pd
            df = pd.read_csv(csv_path, sep='\t')
            self.num_rows = len(df)
            
            # Always use generic class names regardless of CSV content
            self._num_classes = num_classes
            self._int_to_ctype = {i: f"Class_{i}" for i in range(num_classes)}
            
            print(f"Created dataset with {self.num_rows} reads")
            print(f"Using generic class names: {self._int_to_ctype}")
            
        def __len__(self):
            return self.num_rows
        
        def num_classes(self):
            return self._num_classes
        
        def int_to_ctype(self):
            return self._int_to_ctype.copy()
        
        def vocab(self):
            return self.vocab
    
    # Try to create the original dataset first
    try:
        dataset = MethylBertFinetuneDataset(csv_path, vocab, seq_len, n_cores=1)
        
        # Store the temporary CSV path for cleanup if it was created
        if 'processed_' in os.path.basename(csv_path):
            dataset._temp_csv_path = csv_path
        
        # Always override to generic names to ensure consistency
        original_mapping = dataset.int_to_ctype.copy()
        
        # Create new generic mapping
        generic_mapping = {i: f"Class_{i}" for i in range(dataset.num_classes())}
        dataset.int_to_ctype = generic_mapping
        
        print(f"Dataset created with {len(dataset)} reads and {dataset.num_classes()} cell types")
        
        # Extract DMR_cType data if requested
        dmr_ctype_data = None
        if extract_dmr_ctype:
            try:
                # Read the processed CSV to get DMR_cType column
                import pandas as pd
                df = pd.read_csv(csv_path, sep='\t')
                
                # Check if dmr_ctype column exists (should be there if preprocessing included it)
                if 'dmr_ctype' in df.columns:
                    print(f"Found DMR_cType column in processed data")
                    dmr_ctype_data = df['dmr_ctype'].tolist()
                    print(f"Extracted {len(dmr_ctype_data)} DMR_cType values")
                else:
                    print("Warning: DMR_cType column not found in processed data. Available columns:")
                    for col in df.columns:
                        print(f"  - {col}")
                    print("DMR_cType analysis will be skipped.")
                    
            except Exception as e:
                print(f"Warning: Could not extract DMR_cType data: {e}")
                print("DMR_cType analysis will be skipped.")
        
        # Store DMR_cType data in dataset if available
        if dmr_ctype_data is not None:
            dataset.dmr_ctype_data = dmr_ctype_data
        
        return dataset
        
    except Exception as e:
        error_msg = str(e).lower()
        print(f"Error creating MethylBertFinetuneDataset: {e}")
        
        if ignore_ctype_mismatch or "ctype" in error_msg or "class" in error_msg:
            print(f"Creating generic dataset to bypass ctype issues...")
            
            # Determine number of classes based on model expectation
            # For now, use 6 as default (can be overridden by model loading)
            num_classes = 6
            
            generic_dataset = GenericCellTypeDataset(csv_path, vocab, seq_len, num_classes)
            return generic_dataset
        else:
            raise

def predict_cell_types(model, dataset, device, batch_size=32):
    """
    Predict cell types for all reads in the dataset
    Args:
        model: Loaded MethylBERT model
        dataset: Dataset containing the reads
        device: Device to run inference on
        batch_size: Batch size for inference
    Returns:
        tuple: (predictions, probabilities, true_labels)
    """
    print("Running cell type predictions...")
    print("Note: Predictions are based ONLY on dna_seq and methyl_seq data")
    print("The ctype column is NOT used for making predictions - only for dataset structure")
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_predictions = []
    all_probabilities = []
    all_true_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Move data to device
            data = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
            
            # Create attention mask
            attention_mask = (data["dna_seq"] != dataset.vocab.pad_index).long()
            
            # Forward pass
            outputs = model.forward(
                step=0,
                input_ids=data["dna_seq"],
                attention_mask=attention_mask,
                token_type_ids=data["methyl_seq"],
                labels=data["dmr_label"],
                ctype_label=data["ctype_label"]
            )
            
            # Get predictions and probabilities
            logits = outputs["classification_logits"]
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_true_labels.append(data["ctype_label"].cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)
    true_labels = np.concatenate(all_true_labels, axis=0)
    
    return predictions, probabilities, true_labels

def analyze_deconvolution_results(predictions, probabilities, dataset, output_format='tab', use_dmr_ctype=False, dmr_ctype_data=None):
    """
    Analyze the deconvolution results and provide comprehensive output

    Args:
        predictions: Predicted cell type labels
        probabilities: Prediction probabilities
        dataset: Dataset object containing cell type mappings
        output_format: Output format ('tab' or 'json')
        use_dmr_ctype: If True, analyze by DMR_cType groups
        dmr_ctype_data: List of DMR_cType values for each read (if use_dmr_ctype=True)
    Returns:
        dict: Analysis results
    """
    print("Analyzing deconvolution results...")
    
    # Get cell type names
    int_to_ctype = dataset.int_to_ctype
    num_classes = len(int_to_ctype)
    
    # Check if we have proper cell type names or generic class names
    has_proper_names = any(not str(name).startswith('Class_') for name in int_to_ctype.values())
    
    if not has_proper_names:
        print("Note: Using generic class names (Class_0, Class_1, etc.) for output")
        print("This occurs when:")
        print("  - The --use_generic_names flag is set")
        print("  - There's a mismatch between data classes and model classes")
        print("  - The ctype column has issues")
        print("  - The model expects more classes than the data provides")
    else:
        print("Note: Using original cell type names from the data")
    
    # Count predictions for each cell type
    unique_predictions, counts = np.unique(predictions, return_counts=True)
    total_reads = len(predictions)
    
    # Calculate percentages
    cell_type_breakdown = {}
    for pred, count in zip(unique_predictions, counts):
        cell_type_name = int_to_ctype.get(pred, f"Unknown_{pred}")
        percentage = (count / total_reads) * 100
        cell_type_breakdown[cell_type_name] = {
            'count': int(count),
            'percentage': round(percentage, 2)
        }
    
    # Find the dominant cell type
    dominant_idx = unique_predictions[np.argmax(counts)]
    dominant_cell_type = int_to_ctype.get(dominant_idx, f"Unknown_{dominant_idx}")
    dominant_percentage = cell_type_breakdown[dominant_cell_type]['percentage']
    
    # Calculate confidence metrics
    max_probabilities = np.max(probabilities, axis=1)
    avg_confidence = np.mean(max_probabilities)
    confidence_std = np.std(max_probabilities)
    
    # Confidence analysis by cell type
    confidence_by_cell_type = {}
    for pred, count in zip(unique_predictions, counts):
        cell_type_name = int_to_ctype.get(pred, f"Unknown_{pred}")
        mask = predictions == pred
        if np.any(mask):
            cell_type_probs = probabilities[mask]
            cell_type_max_probs = np.max(cell_type_probs, axis=1)
            confidence_by_cell_type[cell_type_name] = {
                'avg_confidence': round(float(np.mean(cell_type_max_probs)), 3),
                'std_confidence': round(float(np.std(cell_type_max_probs)), 3),
                'min_confidence': round(float(np.min(cell_type_max_probs)), 3),
                'max_confidence': round(float(np.max(cell_type_max_probs)), 3)
            }
    
    # Compile results
    results = {
        'total_reads': total_reads,
        'final_verdict': {
            'dominant_cell_type': dominant_cell_type,
            'dominant_percentage': dominant_percentage,
            'confidence': round(float(avg_confidence), 3)
        },
        'cell_type_breakdown': cell_type_breakdown,
        'confidence_analysis': {
            'overall_avg_confidence': round(float(avg_confidence), 3),
            'overall_confidence_std': round(float(confidence_std), 3),
            'by_cell_type': confidence_by_cell_type
        },
        'using_generic_names': not has_proper_names
    }
    
    # Add DMR_cType analysis if requested
    if use_dmr_ctype and dmr_ctype_data is not None:
        print("Performing DMR_cType analysis...")
        
        # Create DMR_cType to generic region mapping
        unique_dmr_ctypes = list(set(dmr_ctype_data))
        dmr_ctype_to_region = {dmr_ctype: f"Region_{i}" for i, dmr_ctype in enumerate(unique_dmr_ctypes)}
        
        # Analyze each DMR_cType group
        dmr_ctype_analysis = {}
        
        for dmr_ctype in unique_dmr_ctypes:
            region_name = dmr_ctype_to_region[dmr_ctype]
            
            # Get indices for this DMR_cType
            dmr_ctype_indices = [i for i, dct in enumerate(dmr_ctype_data) if dct == dmr_ctype]
            
            if not dmr_ctype_indices:
                continue
            
            # Get predictions and probabilities for this DMR_cType
            dmr_predictions = predictions[dmr_ctype_indices]
            dmr_probabilities = probabilities[dmr_ctype_indices]
            
            # Count predictions for each cell type within this DMR_cType
            dmr_unique_preds, dmr_counts = np.unique(dmr_predictions, return_counts=True)
            dmr_total_reads = len(dmr_predictions)
            
            # Calculate percentages and confidence for each cell type
            dmr_breakdown = {}
            for pred, count in zip(dmr_unique_preds, dmr_counts):
                cell_type_name = int_to_ctype.get(pred, f"Unknown_{pred}")
                percentage = (count / dmr_total_reads) * 100
                
                # Calculate confidence for this cell type within this DMR_cType
                # Find indices where predictions match this cell type within the DMR_cType group
                pred_mask = dmr_predictions == pred
                pred_indices = [dmr_ctype_indices[i] for i in range(len(dmr_ctype_indices)) if pred_mask[i]]
                
                if len(pred_indices) > 0:
                    pred_probs = probabilities[pred_indices]
                    pred_max_probs = np.max(pred_probs, axis=1)
                    avg_confidence = float(np.mean(pred_max_probs))
                else:
                    avg_confidence = 0.0
                
                dmr_breakdown[cell_type_name] = {
                    'count': int(count),
                    'percentage': round(percentage, 2),
                    'confidence_avg': round(avg_confidence, 3)
                }
            
            # Find dominant cell type for this DMR_cType
            if dmr_counts.size > 0:
                dmr_dominant_idx = dmr_unique_preds[np.argmax(dmr_counts)]
                dmr_dominant_cell_type = int_to_ctype.get(dmr_dominant_idx, f"Unknown_{dmr_dominant_idx}")
                dmr_dominant_percentage = dmr_breakdown[dmr_dominant_cell_type]['percentage']
            else:
                dmr_dominant_cell_type = "None"
                dmr_dominant_percentage = 0.0
            
            dmr_ctype_analysis[region_name] = {
                'dmr_ctype': dmr_ctype,
                'total_reads': dmr_total_reads,
                'dominant_cell_type': dmr_dominant_cell_type,
                'dominant_percentage': dmr_dominant_percentage,
                'breakdown': dmr_breakdown
            }
        
        results['dmr_ctype_analysis'] = {
            'dmr_ctype_to_region': dmr_ctype_to_region,
            'by_region': dmr_ctype_analysis
        }
    
    return results

def output_results(results, output_format='tab'):
    """
    Output the results in the specified format
    
    Args:
        results: Analysis results dictionary
        output_format: Output format ('tab' or 'json')
    """
    print("\n" + "="*60)
    print("CELL TYPE DECONVOLUTION RESULTS")
    print("="*60)
    
    # Check if we're using generic class names
    if results.get('using_generic_names', False):
        print("\nNOTE: Results are shown using generic class names (Class_0, Class_1, etc.)")
        print("This may occur when:")
        print("  - The input data has fewer cell types than the trained model")
        print("  - The ctype column has issues or mismatches")
        print("  - The model was trained on different cell types")
        print("\nIMPORTANT: The ctype column in your CSV is NOT used for making predictions.")
        print("Predictions are based ONLY on DNA sequence and methylation patterns.")
        print("The ctype column only affects how results are labeled, not the prediction accuracy.")
    
    # Final verdict
    print(f"\nFINAL VERDICT:")
    print(f"  Dominant Cell Type: {results['final_verdict']['dominant_cell_type']}")
    print(f"  Percentage: {results['final_verdict']['dominant_percentage']}%")
    print(f"  Overall Confidence: {results['final_verdict']['confidence']}")
    
    # Cell type breakdown
    print(f"\nCELL TYPE BREAKDOWN:")
    if output_format == 'tab':
        print("Cell_Type\tCount\tPercentage")
        for cell_type, data in results['cell_type_breakdown'].items():
            print(f"{cell_type}\t{data['count']}\t{data['percentage']}%")
    else:
        for cell_type, data in results['cell_type_breakdown'].items():
            print(f"  {cell_type}: {data['count']} reads ({data['percentage']}%)")
    
    # Confidence analysis
    print(f"\nCONFIDENCE ANALYSIS:")
    print(f"  Overall Average Confidence: {results['confidence_analysis']['overall_avg_confidence']}")
    print(f"  Overall Confidence Std: {results['confidence_analysis']['overall_confidence_std']}")
    
    print(f"\nConfidence by Cell Type:")
    for cell_type, conf_data in results['confidence_analysis']['by_cell_type'].items():
        print(f"  {cell_type}:")
        print(f"    Avg: {conf_data['avg_confidence']}, Std: {conf_data['std_confidence']}")
        print(f"    Range: {conf_data['min_confidence']} - {conf_data['max_confidence']}")
    
    # Output DMR_cType analysis if available
    if 'dmr_ctype_analysis' in results:
        print(f"\nDMR_cType ANALYSIS BY REGION:")
        print("="*60)
        
        dmr_analysis = results['dmr_ctype_analysis']
        dmr_ctype_to_region = dmr_analysis['dmr_ctype_to_region']
        
        # Print mapping for reference
        print("DMR_cType to Region mapping:")
        for dmr_ctype, region in dmr_ctype_to_region.items():
            print(f"  {dmr_ctype} -> {region}")
        
        print("\nDetailed breakdown by region:")
        
        for region_name, region_data in dmr_analysis['by_region'].items():
            print(f"\n{region_name} (DMR_cType: {region_data['dmr_ctype']}):")
            print(f"  Total reads: {region_data['total_reads']}")
            print(f"  Dominant cell type: {region_data['dominant_cell_type']} ({region_data['dominant_percentage']}%)")
            
            if output_format == 'tab':
                print("  Class\tPercentage\tCount\tConfidence_Avg")
                for cell_type, data in region_data['breakdown'].items():
                    print(f"  {cell_type}\t{data['percentage']}%\t{data['count']}\t{data['confidence_avg']}")
            else:
                for cell_type, data in region_data['breakdown'].items():
                    print(f"    {cell_type}: {data['percentage']}% ({data['count']} reads, confidence: {data['confidence_avg']})")
        
        print("="*60)
    
    print("="*60)

def main():
    """Main function to run the deconvolution analysis"""
    parser = argparse.ArgumentParser(description="Cell Type Deconvolution using MethylBERT")
    
    # Configuration options (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('--csv', type=str,
                             help='Path to input CSV file for analysis (single file mode)')
    config_group.add_argument('--config', type=str,
                             help='Path to JSON configuration file (batch processing mode)')
    
    # Optional arguments (used when --csv is specified)
    parser.add_argument('--model_dir', type=str, 
                       default='Training/Run_Result/bert.model',
                       help='Directory containing model files (default: Training/Run_Result/bert.model)')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu', 'auto'], 
                       default='auto',
                       help='Device to use for inference (default: auto)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for inference (default: 128)')
    # seq_len is now auto-detected from model files, no need for command line parameter
    parser.add_argument('--output_format', type=str, choices=['tab', 'json'], 
                       default='tab',
                       help='Output format (default: tab)')
    parser.add_argument('--ignore_ctype_mismatch', action='store_true',
                       help='Ignore ctype column mismatches and continue with prediction')
    parser.add_argument('--force_model_classes', action='store_true',
                       help='Force use of model\'s class count instead of data\'s class count')
    parser.add_argument('--use_generic_names', action='store_true',
                       help='Use generic Class_0, Class_1, etc. instead of original ctype names')
    parser.add_argument('--use_dmr_ctype', action='store_true',
                       help='Analyze results by DMR_cType groups (Region_0, Region_1, etc.)')
    
    args = parser.parse_args()
    
    # Check if we're using JSON config mode
    if args.config:
        print("Using JSON configuration mode for batch processing")
        try:
            process_multiple_files(args.config)
        except Exception as e:
            print(f"Error during batch processing: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return
    
    # Single file mode (original functionality)
    if not args.csv:
        parser.error("Either --csv or --config must be specified")
    
    try:
        # Set device
        device = get_device(args.device)
        print(f"Using device: {device}")
        
        # Create vocabulary
        print("Creating vocabulary...")
        vocab = MethylVocab(k=3)
        
        # First, try to load the model to get the expected parameters
        print("Loading model to determine expected parameters...")
        try:
            # Create a temporary dataset just to get basic info (using default seq_len for initial creation)
            temp_dataset = create_dataset_from_csv(args.csv, vocab, 300, args.ignore_ctype_mismatch, True, args.use_dmr_ctype)
            temp_num_classes = temp_dataset.num_classes()
            
            # Load model with the correct parameters
            model, config = load_model(args.model_dir, device, temp_num_classes, args.force_model_classes)
            
            # Now create the final dataset with the correct parameters from the model
            final_num_classes = config.num_classes
            actual_seq_len = config.seq_len
            
            print(f"Model expects {final_num_classes} classes and seq_len {actual_seq_len}")
            print(f"Recreating dataset with actual seq_len: {actual_seq_len}")
            
            # Always create the original dataset first, then override its mapping
            # This ensures we get proper data processing while controlling the class names
            print(f"Creating dataset with original MethylBertFinetuneDataset...")
            dataset = create_dataset_from_csv(args.csv, vocab, actual_seq_len, args.ignore_ctype_mismatch, True, args.use_dmr_ctype)
            
            # Always override the class mapping to use generic names
            print(f"Original dataset mapping: {dataset.int_to_ctype}")
            print(f"Overriding to generic Class_0 through Class_{final_num_classes-1}...")
            
            # Force the dataset to use the model's class count and generic names
            dataset._num_classes = final_num_classes
            
            # Create the new mapping
            new_mapping = {i: f"Class_{i}" for i in range(final_num_classes)}
            
            # Store the mapping directly as an attribute (not as a method)
            # This matches how the original MethylBertFinetuneDataset works
            dataset.int_to_ctype = new_mapping
            
            # Also ensure the num_classes method returns the right value
            # Store the original method and override it
            original_num_classes = dataset.num_classes
            def get_num_classes():
                return final_num_classes
            dataset.num_classes = get_num_classes
            
            # Verify the override worked
            print(f"After override - num_classes(): {dataset.num_classes()}")
            print(f"After override - int_to_ctype: {dataset.int_to_ctype}")
            
            # Double-check that the mapping is actually changed
            if hasattr(dataset, 'int_to_ctype'):
                print(f"Direct attribute access - int_to_ctype: {dataset.int_to_ctype}")
            else:
                print("Warning: int_to_ctype attribute not found!")
            
            print(f"Final dataset mapping: {dataset.int_to_ctype}")
            
            num_classes = final_num_classes
            
            print(f"Final dataset info:")
            print(f"  Number of classes: {dataset.num_classes()}")
            print(f"  Class mapping: {dataset.int_to_ctype}")
            print(f"  Number of reads: {len(dataset)}")
            
            print(f"Note: Class mapping is independent of CSV content.")
            print(f"Class_0, Class_1, etc. correspond to the model's internal class indices,")
            print(f"not to any specific cell type names in your CSV file.")
            
        except Exception as e:
            print(f"Error during model loading or dataset creation: {e}")
            print("Please check that the model directory contains:")
            print(f"  - config.json")
            print(f"  - model.safetensors (or pytorch_model.bin)")
            print(f"  - dmr_encoder.pickle (optional)")
            print(f"  - read_classification_model.pickle (optional)")
            raise
        
        # Run predictions
        try:
            predictions, probabilities, true_labels = predict_cell_types(
                model, dataset, device, args.batch_size
            )
        except Exception as e:
            print(f"Error during predictions: {e}")
            print("This may be due to:")
            print("  - Memory issues (try reducing batch_size)")
            print("  - Model compatibility issues")
            print("  - Data format problems")
            raise
        
        # Analyze results
        try:
            # Get DMR_cType data if available
            dmr_ctype_data = getattr(dataset, 'dmr_ctype_data', None)
            results = analyze_deconvolution_results(predictions, probabilities, dataset, args.output_format, args.use_dmr_ctype, dmr_ctype_data)
        except Exception as e:
            print(f"Error during analysis: {e}")
            print("This may be due to:")
            print("  - Empty predictions")
            print("  - Data type mismatches")
            print("  - Memory issues")
            raise
        
        # Output results
        try:
            output_results(results, args.output_format)
        except Exception as e:
            print(f"Error during output: {e}")
            print("Results analysis completed but output failed")
            print("You can still access the results programmatically")
            raise
        
        # Save results to file
        try:
            output_file = f"deconvolution_results_{os.path.basename(args.csv).replace('.csv', '')}.txt"
            with open(output_file, 'w') as f:
                f.write("CELL TYPE DECONVOLUTION RESULTS\n")
                f.write("="*60 + "\n")
                f.write(f"Input file: {args.csv}\n")
                f.write(f"Model directory: {args.model_dir}\n")
                f.write(f"Device: {device}\n")
                f.write(f"Total reads: {results['total_reads']}\n\n")
                
                f.write("FINAL VERDICT:\n")
                f.write(f"  Dominant Cell Type: {results['final_verdict']['dominant_cell_type']}\n")
                f.write(f"  Percentage: {results['final_verdict']['dominant_percentage']}%\n")
                f.write(f"  Overall Confidence: {results['final_verdict']['confidence']}\n\n")
                
                f.write("CELL TYPE BREAKDOWN:\n")
                for cell_type, data in results['cell_type_breakdown'].items():
                    f.write(f"  {cell_type}: {data['count']} reads ({data['percentage']}%)\n")
                
                f.write("\nCONFIDENCE ANALYSIS:\n")
                f.write(f"  Overall Average Confidence: {results['confidence_analysis']['overall_avg_confidence']}\n")
                f.write(f"  Overall Confidence Std: {results['confidence_analysis']['overall_confidence_std']}\n")
                
                # Add confidence by cell type
                f.write("\nConfidence by Cell Type:\n")
                for cell_type, conf_data in results['confidence_analysis']['by_cell_type'].items():
                    f.write(f"  {cell_type}:\n")
                    f.write(f"    Avg: {conf_data['avg_confidence']}, Std: {conf_data['std_confidence']}\n")
                    f.write(f"    Range: {conf_data['min_confidence']} - {conf_data['max_confidence']}\n")
                
                # Add DMR_cType analysis if available
                if 'dmr_ctype_analysis' in results:
                    f.write("\n" + "="*60 + "\n")
                    f.write("DMR_cType ANALYSIS BY REGION\n")
                    f.write("="*60 + "\n")
                    
                    dmr_analysis = results['dmr_ctype_analysis']
                    dmr_ctype_to_region = dmr_analysis['dmr_ctype_to_region']
                    
                    f.write("DMR_cType to Region mapping:\n")
                    for dmr_ctype, region in dmr_ctype_to_region.items():
                        f.write(f"  {dmr_ctype} -> {region}\n")
                    
                    f.write("\nDetailed breakdown by region:\n")
                    
                    for region_name, region_data in dmr_analysis['by_region'].items():
                        f.write(f"\n{region_name} (DMR_cType: {region_data['dmr_ctype']}):\n")
                        f.write(f"  Total reads: {region_data['total_reads']}\n")
                        f.write(f"  Dominant cell type: {region_data['dominant_cell_type']} ({region_data['dominant_percentage']}%)\n")
                        
                        f.write("  Class\tPercentage\tCount\tConfidence_Avg\n")
                        for cell_type, data in region_data['breakdown'].items():
                            f.write(f"  {cell_type}\t{data['percentage']}%\t{data['count']}\t{data['confidence_avg']}\n")
            
            print(f"\nResults saved to: {output_file}")
            
            # Save DMR_cType analysis as separate table file if available
            if 'dmr_ctype_analysis' in results:
                try:
                    table_file = f"dmr_ctype_analysis_{os.path.basename(args.csv).replace('.csv', '')}.txt"
                    with open(table_file, 'w') as f:
                        f.write("DMR_cType ANALYSIS TABLE\n")
                        f.write("="*60 + "\n")
                        f.write(f"Input file: {args.csv}\n")
                        f.write(f"Model directory: {args.model_dir}\n")
                        f.write(f"Total reads: {results['total_reads']}\n\n")
                        
                        dmr_analysis = results['dmr_ctype_analysis']
                        dmr_ctype_to_region = dmr_analysis['dmr_ctype_to_region']
                        
                        f.write("DMR_cType to Region mapping:\n")
                        for dmr_ctype, region in dmr_ctype_to_region.items():
                            f.write(f"  {dmr_ctype} -> {region}\n")
                        f.write("\n")
                        
                        for region_name, region_data in dmr_analysis['by_region'].items():
                            f.write(f"{region_name}:\n")
                            f.write("Class\tPercentage\tCount\tConfidence_Avg\n")
                            for cell_type, data in region_data['breakdown'].items():
                                f.write(f"{cell_type}\t{data['percentage']}%\t{data['count']}\t{data['confidence_avg']}\n")
                            f.write("\n")
                    
                    print(f"DMR_cType analysis table saved to: {table_file}")
                except Exception as e:
                    print(f"Warning: Could not save DMR_cType analysis table: {e}")
                    
        except Exception as e:
            print(f"Warning: Could not save results to file: {e}")
            print("Results analysis completed but file saving failed")
        
        # Clean up temporary files if they were created
        try:
            if hasattr(dataset, '_temp_csv_path'):
                import tempfile
                import shutil
                temp_dir = os.path.dirname(dataset._temp_csv_path)
                if os.path.exists(temp_dir) and 'processed_' in os.path.basename(dataset._temp_csv_path):
                    shutil.rmtree(temp_dir)
                    print("Cleaned up temporary files")
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")
            print("This is not critical - temporary files may remain in the system")
        
    except Exception as e:
        print(f"Error during deconvolution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 