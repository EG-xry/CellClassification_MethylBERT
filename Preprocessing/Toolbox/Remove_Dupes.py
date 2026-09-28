#!/usr/bin/env python3
"""
Remove duplicate lines from skeletal muscle training data.

This script processes the skeletal_muscle_training_data_fixed.csv file and removes
all duplicate lines where all 5 columns (DNA_seq, Methyl_seq, DMR_Label, cType, DMR_cType)
are exactly the same.

Input: skeletal_muscle_training_data_fixed.csv (5 columns)
Output: skeletal_muscle_training_data_deduplicated.csv (5 columns, no duplicates)
"""

import pandas as pd
import sys
import os
from pathlib import Path

def remove_duplicates(input_file, output_file):
    """
    Remove exact duplicates from the CSV file
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    
    print(f"Processing file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found!")
        return False
    
    try:
        # Read the CSV file
        print("Reading CSV file...")
        df = pd.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        
        # Check if we have the expected 5 columns
        expected_columns = ['DNA_seq', 'Methyl_seq', 'DMR_Label', 'cType', 'DMR_cType']
        if not all(col in df.columns for col in expected_columns):
            print(f"❌ Error: Expected columns {expected_columns}")
            print(f"Found columns: {list(df.columns)}")
            return False
        
        # Count duplicates before removal
        original_count = len(df)
        duplicate_count = df.duplicated().sum()
        
        print(f"Original rows: {original_count:,}")
        print(f"Duplicate rows: {duplicate_count:,}")
        print(f"Duplicate percentage: {(duplicate_count/original_count)*100:.2f}%")
        
        # Remove duplicates
        print("Removing duplicates...")
        df_deduplicated = df.drop_duplicates()
        
        final_count = len(df_deduplicated)
        removed_count = original_count - final_count
        
        print(f"Final rows: {final_count:,}")
        print(f"Removed rows: {removed_count:,}")
        print(f"Reduction: {(removed_count/original_count)*100:.2f}%")
        
        # Save deduplicated data
        print("Saving deduplicated data...")
        df_deduplicated.to_csv(output_file, index=False)
        
        print(f"✅ Successfully created: {output_file}")
        print(f"✅ Removed {removed_count:,} duplicate rows")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return False

def main():
    """Main function"""
    
    # File paths
    input_file = "skeletal_muscle_training_data_fixed.csv"
    output_file = "skeletal_muscle_training_data_deduplicated.csv"
    
    print("=" * 60)
    print("SKELETAL MUSCLE DATA DEDUPLICATION")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    
    # Process the file
    success = remove_duplicates(input_file, output_file)
    
    if success:
        print(f"\n🎉 Deduplication completed successfully!")
        print(f"Check the output file: {output_file}")
    else:
        print(f"\n❌ Deduplication failed!")
    
    return success

if __name__ == "__main__":
    main()