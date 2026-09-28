#!/usr/bin/env python3
"""
Convert BED file to CSV format.
Preserves the same number of rows and columns.
"""

import pandas as pd
import sys
import argparse
from pathlib import Path


def bed_to_csv(bed_file, csv_file=None, delimiter=','):
    """
    Convert BED file to CSV format.
    
    Args:
        bed_file (str): Path to input BED file
        csv_file (str): Path to output CSV file (optional)
        delimiter (str): CSV delimiter (default: ',')
    
    Returns:
        str: Path to the created CSV file
    """
    # Read BED file (tab-separated)
    print(f"Reading BED file: {bed_file}")
    df = pd.read_csv(bed_file, delimiter='\t')
    
    print(f"BED file shape: {df.shape}")
    print(f"BED columns: {list(df.columns)}")
    
    # Generate output filename if not provided
    if csv_file is None:
        bed_path = Path(bed_file)
        csv_file = bed_path.with_suffix('.csv')
    
    # Write to CSV format
    print(f"Writing CSV file: {csv_file}")
    df.to_csv(csv_file, sep=delimiter, index=False, header=True)
    
    print(f"Conversion complete! Created {csv_file} with {len(df)} rows and {len(df.columns)} columns")
    
    return str(csv_file)


def main():
    parser = argparse.ArgumentParser(description='Convert BED file to CSV format')
    parser.add_argument('bed_file', help='Input BED file path')
    parser.add_argument('-o', '--output', help='Output CSV file path (optional)')
    parser.add_argument('-d', '--delimiter', default=',', help='CSV delimiter (default: comma)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.bed_file).exists():
        print(f"Error: Input file '{args.bed_file}' not found!")
        sys.exit(1)
    
    try:
        csv_file = bed_to_csv(args.bed_file, args.output, args.delimiter)
        print(f"Successfully converted to: {csv_file}")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 