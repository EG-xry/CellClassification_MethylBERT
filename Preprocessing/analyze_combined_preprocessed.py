#!/usr/bin/env python3
"""
Script to analyze combined_preprocessed.csv file
- Check percentage of lines where cType == DMR_cType for each cType
- Find top 5 DMR_cType frequencies for each cType
"""

import pandas as pd
import numpy as np
from collections import Counter
import sys
import os
from tqdm import tqdm
import time

def analyze_csv(file_path):
    """Analyze the combined_preprocessed.csv file"""
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return
    
    try:
        # Read the CSV file with progress tracking
        print(f"Reading file: {file_path}")
        print("Note: For large files (~18M lines), this may take several minutes...")
        
        # Get file size for progress estimation
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size / (1024**3):.2f} GB")
        
        start_time = time.time()
        
        # Read with progress bar using chunking
        chunk_size = 100000  # Read in chunks of 100k rows
        chunks = []
        
        print("Reading file in chunks...")
        
        # First, count total lines to get accurate progress
        print("Counting total lines for accurate progress...")
        with open(file_path, 'r') as f:
            total_lines = sum(1 for _ in f) - 1  # subtract 1 for header
        
        total_chunks = (total_lines + chunk_size - 1) // chunk_size  # ceiling division
        print(f"Estimated {total_chunks} chunks for {total_lines:,} lines")
        
        with tqdm(total=total_chunks, desc="Loading chunks", unit="chunk") as pbar:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunks.append(chunk)
                pbar.update(1)
        
        print("Combining chunks...")
        df = pd.concat(chunks, ignore_index=True)
        del chunks  # Free memory
        
        read_time = time.time() - start_time
        print(f"File loaded in {read_time:.2f} seconds")
        print(f'File shape: {df.shape}')
        print(f'Columns: {list(df.columns)}')
        print()
        
        # Verify expected columns exist
        expected_cols = ['DNA_seq', 'Methyl_seq', 'DMR_Label', 'cType', 'DMR_cType']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing expected columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            print()
        
        # Clean data: handle NaN values in cType and DMR_cType columns
        print("Cleaning data...")
        print(f"Before cleaning - cType nulls: {df['cType'].isnull().sum()}, DMR_cType nulls: {df['DMR_cType'].isnull().sum()}")
        
        # Convert NaN to string for consistency and remove rows with null cType or DMR_cType
        df = df.dropna(subset=['cType', 'DMR_cType'])
        
        # Convert to string to handle mixed types
        df['cType'] = df['cType'].astype(str)
        df['DMR_cType'] = df['DMR_cType'].astype(str)
        
        print(f"After cleaning - Shape: {df.shape}")
        print(f"Sample cType values: {df['cType'].unique()[:10]}")
        print()
        
        # Check percentage of lines where cType == DMR_cType for each cType
        print('=== Percentage of lines where cType == DMR_cType ===')
        ctype_stats = []
        
        unique_ctypes = sorted(df['cType'].unique())
        print(f"Processing {len(unique_ctypes)} unique cTypes...")
        
        for ctype in tqdm(unique_ctypes, desc="Analyzing cTypes"):
            ctype_data = df[df['cType'] == ctype]
            matching_lines = len(ctype_data[ctype_data['cType'] == ctype_data['DMR_cType']])
            total_lines = len(ctype_data)
            percentage = (matching_lines / total_lines) * 100 if total_lines > 0 else 0
            ctype_stats.append((ctype, matching_lines, total_lines, percentage))
            print(f'{ctype}: {matching_lines}/{total_lines} ({percentage:.2f}%)')
        
        print()
        
        # For each cType, find top 5 DMR_cType frequencies
        print('=== Top 5 DMR_cType frequencies for each cType ===')
        print("Computing DMR_cType frequencies...")
        
        for ctype in tqdm(unique_ctypes, desc="Computing frequencies"):
            ctype_data = df[df['cType'] == ctype]
            dmr_ctype_counts = Counter(ctype_data['DMR_cType'])
            top_5 = dmr_ctype_counts.most_common(5)
            
            print(f'\ncType: {ctype} (total entries: {len(ctype_data)})')
            for i, (dmr_ctype, count) in enumerate(top_5, 1):
                percentage = (count / len(ctype_data)) * 100
                print(f'  {i}. {dmr_ctype}: {count} ({percentage:.2f}%)')
        
        print()
        print('=== Overall Summary ===')
        total_matching = sum(stat[1] for stat in ctype_stats)
        total_lines = len(df)
        overall_percentage = (total_matching / total_lines) * 100
        print(f'Overall matching rate: {total_matching}/{total_lines} ({overall_percentage:.2f}%)')
        
        # Additional statistics
        print(f'\nUnique cTypes: {len(df["cType"].unique())}')
        print(f'Unique DMR_cTypes: {len(df["DMR_cType"].unique())}')
        
        total_time = time.time() - start_time
        print(f'\nTotal processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)')
        print(f'Processing rate: {len(df)/total_time:.0f} rows/second')
        
    except Exception as e:
        print(f"Error reading or processing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_combined_preprocessed.py <path_to_combined_preprocessed.csv>")
        print("Example: python3 analyze_combined_preprocessed.py combined_preprocessed.csv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    analyze_csv(file_path) 