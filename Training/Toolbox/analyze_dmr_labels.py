#!/usr/bin/env python3
"""
Script to analyze train_Excitable.csv file focusing on ctype and dmr_label columns.
Creates an output file listing each dmr_label number, corresponding ctype, and occurrence count.
"""

import pandas as pd
import sys
from collections import Counter, defaultdict

def analyze_dmr_labels(csv_file_path, output_file_path):
    """
    Analyze the CSV file to extract dmr_label and ctype relationships.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        output_file_path (str): Path to the output analysis file
    """
    try:
        print(f"Reading CSV file: {csv_file_path}")
        
        # Read the CSV file (tab-delimited format)
        df = pd.read_csv(csv_file_path, sep='\t')
        
        # Check if required columns exist
        required_columns = ['ctype', 'dmr_label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
        
        print(f"Total rows in dataset: {len(df)}")
        print(f"Columns found: {list(df.columns)}")
        
        # Create a mapping of dmr_label to ctype and count occurrences
        dmr_to_ctype = defaultdict(lambda: defaultdict(int))
        dmr_label_counts = Counter()
        
        # Process each row
        for _, row in df.iterrows():
            ctype = row['ctype']
            dmr_label = row['dmr_label']
            
            # Count occurrences
            dmr_to_ctype[dmr_label][ctype] += 1
            dmr_label_counts[dmr_label] += 1
        
        # Get unique cell types
        unique_ctypes = sorted(df['ctype'].unique())
        unique_dmr_labels = sorted(df['dmr_label'].unique())
        
        print(f"Unique cell types found: {unique_ctypes}")
        print(f"Number of unique dmr_labels: {len(unique_dmr_labels)}")
        print(f"dmr_label range: {min(unique_dmr_labels)} to {max(unique_dmr_labels)}")
        
        # Create output content
        output_lines = []
        output_lines.append("DMR Label Analysis Report")
        output_lines.append("=" * 50)
        output_lines.append(f"Input file: {csv_file_path}")
        output_lines.append(f"Total rows analyzed: {len(df)}")
        output_lines.append(f"Unique cell types: {len(unique_ctypes)}")
        output_lines.append(f"Unique dmr_labels: {len(unique_dmr_labels)}")
        output_lines.append("")
        output_lines.append("Summary by Cell Type:")
        output_lines.append("-" * 30)
        
        # Summary by cell type
        ctype_counts = df['ctype'].value_counts().sort_index()
        for ctype, count in ctype_counts.items():
            output_lines.append(f"{ctype}: {count} occurrences")
        
        output_lines.append("")
        output_lines.append("DMR Label Details:")
        output_lines.append("-" * 30)
        output_lines.append("dmr_label\tctype\tcount")
        
        # Detailed analysis for each dmr_label
        for dmr_label in sorted(unique_dmr_labels):
            total_count = dmr_label_counts[dmr_label]
            ctype_breakdown = dmr_to_ctype[dmr_label]
            
            # If dmr_label maps to multiple cell types, show each separately
            if len(ctype_breakdown) == 1:
                # Single cell type
                ctype = list(ctype_breakdown.keys())[0]
                count = ctype_breakdown[ctype]
                output_lines.append(f"{dmr_label}\t{ctype}\t{count}")
            else:
                # Multiple cell types for this dmr_label
                for ctype in sorted(ctype_breakdown.keys()):
                    count = ctype_breakdown[ctype]
                    output_lines.append(f"{dmr_label}\t{ctype}\t{count}")
        
        output_lines.append("")
        output_lines.append("DMR Label Summary (Total occurrences per dmr_label):")
        output_lines.append("-" * 50)
        output_lines.append("dmr_label\ttotal_count")
        
        for dmr_label in sorted(unique_dmr_labels):
            total_count = dmr_label_counts[dmr_label]
            output_lines.append(f"{dmr_label}\t{total_count}")
        
        # Write output file
        with open(output_file_path, 'w') as f:
            f.write('\n'.join(output_lines))
        
        print(f"\nAnalysis complete!")
        print(f"Output written to: {output_file_path}")
        
        # Print summary to console
        print(f"\nSummary:")
        print(f"- Total dmr_labels: {len(unique_dmr_labels)}")
        print(f"- Total cell types: {len(unique_ctypes)}")
        print(f"- Most common dmr_label: {dmr_label_counts.most_common(1)[0]}")
        
        return True
        
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return False

def main():
    """Main function to run the analysis."""
    input_file = "test_Excitable.csv"
    output_file = "dmr_label_analysis.txt"
    
    print("DMR Label Analysis Script")
    print("=" * 30)
    
    success = analyze_dmr_labels(input_file, output_file)
    
    if success:
        print("\nScript completed successfully!")
    else:
        print("\nScript failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
