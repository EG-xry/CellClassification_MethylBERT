#!/usr/bin/env python3
"""
Convert CSV file to BED format.
Removes empty columns and preserves the same number of rows.
"""

import pandas as pd
import sys
import argparse
from pathlib import Path


def csv_to_bed(csv_file, bed_file=None, delimiter=','):
    """
    Convert CSV file to BED format.
    
    Args:
        csv_file (str): Path to input CSV file
        bed_file (str): Path to output BED file (optional)
        delimiter (str): CSV delimiter (default: ',')
    
    Returns:
        str: Path to the created BED file
    """
    # Read CSV file
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file, delimiter=delimiter)
    
    print(f"Original CSV shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Remove completely empty columns
    df_cleaned = df.dropna(axis=1, how='all')
    
    print(f"After removing empty columns: {df_cleaned.shape}")
    print(f"Cleaned columns: {list(df_cleaned.columns)}")
    
    # Generate output filename if not provided
    if bed_file is None:
        csv_path = Path(csv_file)
        bed_file = csv_path.with_suffix('.bed')
    
    # Write to BED format (tab-separated)
    print(f"Writing BED file: {bed_file}")
    df_cleaned.to_csv(bed_file, sep='\t', index=False, header=True)
    
    print(f"Conversion complete! Created {bed_file} with {len(df_cleaned)} rows and {len(df_cleaned.columns)} columns")
    
    return str(bed_file)


def main():
    parser = argparse.ArgumentParser(description='Convert CSV file to BED format')
    parser.add_argument('csv_file', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output BED file path (optional)')
    parser.add_argument('-d', '--delimiter', default=',', help='CSV delimiter (default: comma)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.csv_file).exists():
        print(f"Error: Input file '{args.csv_file}' not found!")
        sys.exit(1)
    
    try:
        bed_file = csv_to_bed(args.csv_file, args.output, args.delimiter)
        print(f"Successfully converted to: {bed_file}")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 