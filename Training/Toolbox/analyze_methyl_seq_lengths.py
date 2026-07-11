#!/usr/bin/env python3
"""
Script to analyze methyl_seq column lengths from test_1000_6.csv
Generates a histogram of sequence lengths and saves it as an image.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import os

def analyze_methyl_seq_lengths(csv_file):
    """
    Analyze the methyl_seq column lengths from the CSV file.
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        tuple: (lengths_list, length_counts, total_rows, unique_lengths)
    """
    print(f"Reading file: {csv_file}")
    
    # Read the CSV file in chunks to handle large files efficiently
    chunk_size = 10000
    lengths_list = []
    total_rows = 0
    
    for chunk in pd.read_csv(csv_file, delimiter='\t', chunksize=chunk_size):
        # Get the methyl_seq column (second column, index 1)
        methyl_seq_col = chunk.iloc[:, 1]
        
        # Calculate lengths for this chunk
        chunk_lengths = methyl_seq_col.str.len()
        lengths_list.extend(chunk_lengths.dropna().tolist())
        total_rows += len(chunk)
        
        print(f"Processed chunk: {total_rows} rows so far...")
    
    # Count occurrences of each length
    length_counts = Counter(lengths_list)
    unique_lengths = sorted(length_counts.keys())
    
    print(f"\nTotal rows processed: {total_rows}")
    print(f"Unique sequence lengths found: {len(unique_lengths)}")
    print(f"Length range: {min(unique_lengths)} to {max(unique_lengths)}")
    
    return lengths_list, length_counts, total_rows, unique_lengths

def create_histogram(lengths_list, length_counts, unique_lengths, output_dir="."):
    """
    Create and save a histogram of methyl_seq lengths.
    
    Args:
        lengths_list (list): List of all sequence lengths
        length_counts (Counter): Counter object with length frequencies
        unique_lengths (list): Sorted list of unique lengths
        output_dir (str): Directory to save the output files
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Main histogram
    ax1.hist(lengths_list, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Methylation Sequence Length')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Methylation Sequence Lengths')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_length = np.mean(lengths_list)
    median_length = np.median(lengths_list)
    std_length = np.std(lengths_list)
    
    stats_text = f'Mean: {mean_length:.1f}\nMedian: {median_length:.1f}\nStd: {std_length:.1f}\nTotal: {len(lengths_list):,}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Bar plot of top 20 most common lengths
    top_lengths = length_counts.most_common(20)
    lengths, counts = zip(*top_lengths)
    
    bars = ax2.bar(range(len(lengths)), counts, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Count')
    ax2.set_title('Top 20 Most Common Sequence Lengths')
    ax2.set_xticks(range(len(lengths)))
    ax2.set_xticklabels(lengths, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, "methyl_seq_length_histogram.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Histogram saved as: {output_file}")
    
    # Save statistics to text file
    stats_file = os.path.join(output_dir, "methyl_seq_length_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("Methylation Sequence Length Analysis\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total sequences analyzed: {len(lengths_list):,}\n")
        f.write(f"Unique sequence lengths: {len(unique_lengths)}\n")
        f.write(f"Length range: {min(unique_lengths)} to {max(unique_lengths)}\n")
        f.write(f"Mean length: {mean_length:.2f}\n")
        f.write(f"Median length: {median_length:.2f}\n")
        f.write(f"Standard deviation: {std_length:.2f}\n\n")
        
        f.write("Length Distribution:\n")
        f.write("-" * 20 + "\n")
        for length in sorted(length_counts.keys()):
            count = length_counts[length]
            percentage = (count / len(lengths_list)) * 100
            f.write(f"Length {length}: {count:,} sequences ({percentage:.2f}%)\n")
        
        f.write("\nTop 20 Most Common Lengths:\n")
        f.write("-" * 30 + "\n")
        for i, (length, count) in enumerate(top_lengths, 1):
            percentage = (count / len(lengths_list)) * 100
            f.write(f"{i:2d}. Length {length}: {count:,} sequences ({percentage:.2f}%)\n")
    
    print(f"Statistics saved as: {stats_file}")
    
    return output_file, stats_file

def main():
    """Main function to run the analysis."""
    csv_file = "test_Blood_split.csv"
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found!")
        return
    
    try:
        # Analyze the file
        print("Starting analysis of methyl_seq column lengths...")
        lengths_list, length_counts, total_rows, unique_lengths = analyze_methyl_seq_lengths(csv_file)
        
        # Create and save histogram
        print("\nCreating histogram...")
        output_file, stats_file = create_histogram(lengths_list, length_counts, unique_lengths)
        
        print(f"\nAnalysis complete!")
        print(f"Output files:")
        print(f"  - Histogram: {output_file}")
        print(f"  - Statistics: {stats_file}")
        
        # Display some key statistics
        print(f"\nKey Statistics:")
        print(f"  - Total sequences: {len(lengths_list):,}")
        print(f"  - Most common length: {length_counts.most_common(1)[0][0]} (count: {length_counts.most_common(1)[0][1]:,})")
        print(f"  - Average length: {np.mean(lengths_list):.1f}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 