#!/usr/bin/env python3
"""
Generate CTC config file from messy paste of PAT file names.
Handles multiple spaces and generates proper JSON config format.
"""

import json
import re
import sys
from pathlib import Path

def clean_filename(filename):
    """Clean filename and remove .pat.gz extension"""
    # Remove .pat.gz and any other extensions
    clean_name = filename.replace('.pat.gz', '').replace('.pat.gz.csi', '')
    # Remove any other common extensions that might be present
    clean_name = re.sub(r'\.(csi|bai|tbi|idx)$', '', clean_name)
    return clean_name

def parse_pat_files(input_text):
    """Parse messy paste and extract PAT file names"""
    # Split by whitespace and filter out empty strings
    files = [f.strip() for f in input_text.split() if f.strip()]
    
    # Filter only .pat.gz files (exclude .csi and other extensions)
    pat_files = []
    for file in files:
        if file.endswith('.pat.gz') and not file.endswith('.pat.gz.csi'):
            pat_files.append(file)
    
    return pat_files

def generate_config(pat_files, output_file="ctc_config.json"):
    """Generate JSON config file from PAT file list"""
    
    config = {
        "collective_csv_mode": True,
        "collective_csv_file": "hg38_all_1000.csv",
        "fasta_file": "hg38.fa", 
        "cpg_bed_file": "CpG.bed.gz",
        "cell_types": {}
    }
    
    for pat_file in pat_files:
        # Clean filename for output
        clean_name = clean_filename(pat_file)
        output_csv = f"{clean_name}.csv"
        
        # Create cell type entry
        cell_type_key = clean_name.replace('-', '_').replace('.', '_')
        
        config["cell_types"][cell_type_key] = {
            "pat_file": pat_file,
            "output_file": output_csv,
            "cell_type": "CTC"  # Default cell type for CTC data
        }
    
    # Save config
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Generated config file: {output_file}")
    print(f"Processed {len(pat_files)} PAT files")
    
    return config

def main():
    print("CTC Config Generator")
    print("=" * 50)
    print("Paste your PAT file names below (press Ctrl+D when done):")
    print()
    
    try:
        # Read input from stdin
        input_text = sys.stdin.read()
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return
    
    if not input_text.strip():
        print("No input provided.")
        return
    
    # Parse PAT files
    pat_files = parse_pat_files(input_text)
    
    if not pat_files:
        print("No valid .pat.gz files found in input.")
        return
    
    print(f"\nFound {len(pat_files)} PAT files:")
    for file in pat_files:
        print(f"  - {file}")
    
    # Generate config
    config = generate_config(pat_files)
    
    print(f"\nConfig file generated successfully!")
    print(f"Total cell types: {len(config['cell_types'])}")

if __name__ == "__main__":
    main() 