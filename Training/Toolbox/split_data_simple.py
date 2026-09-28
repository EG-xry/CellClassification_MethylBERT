#!/usr/bin/env python3
"""
Simple script to split combined_test_100k.csv into train_seq.csv and test_seq.csv
with stratified sampling to maintain the same proportion of each cell type.
"""

import csv
import random
from collections import defaultdict
import os

def normalize_header(columns):
    """
    Normalize header columns to match required format
    Converts 'dmr_Label' to 'dmr_label' if present
    """
    normalized_columns = []
    for col in columns:
        if col == 'dmr_Label':
            normalized_columns.append('dmr_label')
            print(f"Normalized column header: '{col}' -> 'dmr_label'")
        else:
            normalized_columns.append(col)
    return normalized_columns

def split_data_stratified(input_file, train_file, test_file, train_ratio=0.8, random_state=42):
    """
    Split data with stratified sampling to maintain cell type proportions
    """
    
    print(f"Loading data from {input_file}...")
    
    # Set random seed for reproducibility
    random.seed(random_state)
    
    # Read the file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Read {len(lines)} lines from file")
    
    # Parse the header
    header_line = lines[0].strip()
    columns = [col.strip() for col in header_line.split(',')]
    print(f"Original detected columns: {columns}")
    
    # Normalize header if needed
    original_columns = columns.copy()
    columns = normalize_header(columns)
    
    if columns != original_columns:
        print(f"Normalized columns: {columns}")
    
    # Check if we have the required columns
    required_columns = ['dna_seq', 'methyl_seq', 'ctype', 'dmr_label']
    missing_columns = [col for col in required_columns if col not in columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {columns}")
        return False
    
    # Find column indices
    ctype_idx = columns.index('ctype')
    
    # Parse the data rows and group by cell type
    cell_type_groups = defaultdict(list)
    data_rows = []
    
    for i, line in enumerate(lines[1:], 1):
        line = line.strip()
        if line:  # Skip empty lines
            values = line.split(',')
            if len(values) == len(columns):
                data_rows.append(values)
                cell_type = values[ctype_idx].strip()
                cell_type_groups[cell_type].append(values)
            else:
                print(f"Warning: Line {i} has {len(values)} values, expected {len(columns)}")
    
    print(f"Successfully parsed {len(data_rows)} data rows")
    print(f"Found {len(cell_type_groups)} unique cell types")
    
    # Analyze cell type distribution
    print(f"\nCell type distribution in original data:")
    cell_type_counts = {ctype: len(rows) for ctype, rows in cell_type_groups.items()}
    
    # Sort by count
    sorted_cell_types = sorted(cell_type_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top 10 cell types by count:")
    for i, (cell_type, count) in enumerate(sorted_cell_types[:10]):
        percentage = (count / len(data_rows)) * 100
        print(f"  {i+1:2d}. {cell_type:<20}: {count:>8,} samples ({percentage:5.2f}%)")
    
    print(f"\nBottom 10 cell types by count:")
    for i, (cell_type, count) in enumerate(sorted_cell_types[-10:]):
        percentage = (count / len(data_rows)) * 100
        print(f"  {i+1:2d}. {cell_type:<20}: {count:>8,} samples ({percentage:5.2f}%)")
    
    # Perform stratified split
    print(f"\nPerforming stratified split ({train_ratio*100:.0f}% train, {(1-train_ratio)*100:.0f}% test)...")
    
    train_rows = []
    test_rows = []
    
    # Split each cell type group
    for cell_type, rows in cell_type_groups.items():
        # Shuffle the rows for this cell type
        random.shuffle(rows)
        
        # Calculate split point
        split_idx = int(len(rows) * train_ratio)
        
        # Split the rows
        train_rows.extend(rows[:split_idx])
        test_rows.extend(rows[split_idx:])
        
        print(f"  {cell_type:<20}: {len(rows[:split_idx]):>6} train, {len(rows[split_idx:]):>6} test")
    
    print(f"\nSplit completed:")
    print(f"  Training samples: {len(train_rows):,}")
    print(f"  Test samples: {len(test_rows):,}")
    
    # Verify stratification
    print(f"\nVerifying stratification...")
    
    # Count cell types in train and test
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    
    for row in train_rows:
        train_counts[row[ctype_idx]] += 1
    
    for row in test_rows:
        test_counts[row[ctype_idx]] += 1
    
    # Calculate proportions
    max_diff = 0
    print(f"\nSample distribution verification (first 10 cell types):")
    print(f"{'Cell Type':<20} {'Train':<8} {'Test':<8} {'Train%':<8} {'Test%':<8}")
    print("-" * 60)
    
    for cell_type, total_count in sorted_cell_types[:10]:
        train_count = train_counts[cell_type]
        test_count = test_counts[cell_type]
        train_pct = (train_count / len(train_rows)) * 100
        test_pct = (test_count / len(test_rows)) * 100
        
        diff = abs(train_pct - test_pct)
        max_diff = max(max_diff, diff)
        
        print(f"{cell_type:<20} {train_count:<8} {test_count:<8} {train_pct:<8.2f} {test_pct:<8.2f}")
    
    print(f"\nMaximum proportion difference between train/test: {max_diff:.4f}")
    
    if max_diff < 1.0:  # Less than 1% difference
        print("✅ Stratification successful - proportions are well balanced")
    else:
        print("⚠️  Warning: Some cell types may have proportion differences")
    
    # Save the split data
    print(f"\nSaving split data...")
    
    # Ensure output directory exists
    train_dir = os.path.dirname(train_file)
    test_dir = os.path.dirname(test_file)
    
    if train_dir and not os.path.exists(train_dir):
        os.makedirs(train_dir)
        print(f"Created directory: {train_dir}")
    
    if test_dir and not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created directory: {test_dir}")
    
    # Save training data
    with open(train_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(columns)  # Write normalized header
        writer.writerows(train_rows)
    
    # Save test data
    with open(test_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(columns)  # Write normalized header
        writer.writerows(test_rows)
    
    print(f"✅ Training data saved to: {train_file}")
    print(f"✅ Test data saved to: {test_file}")
    
    # Final summary
    print(f"\n" + "="*60)
    print(f"SPLIT SUMMARY")
    print(f"="*60)
    print(f"Original file: {input_file}")
    print(f"Total samples: {len(data_rows):,}")
    print(f"Unique cell types: {len(cell_type_groups)}")
    print(f"Training samples: {len(train_rows):,} ({len(train_rows)/len(data_rows)*100:.1f}%)")
    print(f"Test samples: {len(test_rows):,} ({len(test_rows)/len(data_rows)*100:.1f}%)")
    print(f"Random seed: {random_state}")
    print(f"="*60)
    
    return True

def main():
    """Main function to run the data splitting"""
    
    # Configuration
    input_file = "combined.csv"
    train_file = "Data/train_seq.csv"
    test_file = "Data/test_seq.csv"
    train_ratio = 0.8  # 80% train, 20% test
    random_state = 42  # For reproducibility
    
    print("="*60)
    print("METHYLBERT DATA SPLITTING TOOL (SIMPLE VERSION)")
    print("="*60)
    print(f"Input file: {input_file}")
    print(f"Train file: {train_file}")
    print(f"Test file: {test_file}")
    print(f"Train ratio: {train_ratio}")
    print(f"Random seed: {random_state}")
    print("="*60)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found!")
        print(f"Please make sure the file exists in the current directory.")
        return False
    
    # Perform the split
    success = split_data_stratified(
        input_file=input_file,
        train_file=train_file,
        test_file=test_file,
        train_ratio=train_ratio,
        random_state=random_state
    )
    
    if success:
        print(f"\n🎉 Data splitting completed successfully!")
        print(f"You can now use {train_file} and {test_file} for MethylBERT training.")
    else:
        print(f"\n❌ Data splitting failed!")
    
    return success

if __name__ == "__main__":
    main() 