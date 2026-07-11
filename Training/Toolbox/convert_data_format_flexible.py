#!/usr/bin/env python3
"""
Enhanced Data Format Converter for MethylBERT Training

This script converts data to 4-column format with enhanced features:

Mode 1: Current functionality
- User-controlled input and output file names (GUI/CLI interfaces)
- Read count duplication: if Read_Count > 1, duplicate the line (Read_Count - 1) times
- Removes Read_Count and other unnecessary columns after duplication

Mode 2: Sequence length management
- Splits sequences longer than max_length (default: 250) by cutting in half
- Removes sequences shorter than min_length (default: 25)
- Maintains 3-mer to methylation character correspondence

Mode 3: Length management + keep extra columns
- Same as Mode 2 but preserves additional columns like Read_Count and DMR_cType
- Splits sequences longer than max_length (default: 250) by cutting in half
- Removes sequences shorter than min_length (default: 25)
- Keeps Read_Count and DMR_cType columns if present in input

Input format (flexible):
DNA_seq, Methyl_seq, DMR_Label, cType, DMR_cType, Read_Count, [other columns...]

Output format:
- Mode 1 & 2: 4 columns (dna_seq, methyl_seq, ctype, dmr_label)
- Mode 3: 4+ columns (dna_seq, methyl_seq, ctype, dmr_label, [extra columns])

Features:
- GUI interface when tkinter available, CLI fallback
- Three modes: read count duplication, sequence length management, or length management with extra columns
- In-place file replacement supported
- Progress tracking and detailed reporting
"""

import csv
import os
import sys
import argparse
import tempfile
import shutil
import math

# Try to import tkinter, fall back to command line interface if not available
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("Warning: tkinter not available, using command-line interface instead.")

def create_gui():
    """
    Create a GUI window for file selection and mode selection
    
    Returns:
        tuple: (input_file, output_file, mode, max_length, min_length) or (None, None, None, None, None) if cancelled
    """
    if not GUI_AVAILABLE:
        print("Error: GUI not available, tkinter module not found.")
        return None, None, None, None, None
    
    input_file = ""
    output_file = ""
    mode = 1
    max_length = 250
    min_length = 25
    
    def on_submit():
        """Handle submit button click"""
        nonlocal input_file, output_file, mode, max_length, min_length
        input_file = input_entry.get().strip()
        output_file = output_entry.get().strip()
        mode = mode_var.get()
        max_length = int(max_length_entry.get())
        min_length = int(min_length_entry.get())
        
        if not input_file:
            messagebox.showwarning("Warning", "Please enter an input file name!")
            return
        
        if not output_file:
            messagebox.showwarning("Warning", "Please enter an output file name!")
            return
        
        # Add .csv extension if not present
        if not input_file.endswith('.csv'):
            input_file = input_file + '.csv'
        if not output_file.endswith('.csv'):
            output_file = output_file + '.csv'
        
        root.destroy()
    
    def on_cancel():
        """Handle cancel button click"""
        root.destroy()
    
    def on_mode_change():
        """Handle mode change"""
        if mode_var.get() == 1:
            # Mode 1: Read count duplication
            max_length_entry.config(state='disabled')
            min_length_entry.config(state='disabled')
            max_length_label.config(text="Max Length (disabled for Mode 1):")
            min_length_label.config(text="Min Length (disabled for Mode 1):")
        else:
            # Mode 2 & 3: Sequence length management
            max_length_entry.config(state='normal')
            min_length_entry.config(state='normal')
            max_length_label.config(text="Max Length (split if exceeded):")
            min_length_label.config(text="Min Length (remove if below):")
    
    # Create main window
    root = tk.Tk()
    root.title("Enhanced Data Format Converter")
    root.geometry("700x500")
    root.resizable(True, True)
    
    # Configure grid
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(3, weight=1)
    
    # Title label
    title_label = ttk.Label(root, text="Enhanced Data Format Converter", font=("Arial", 14, "bold"))
    title_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
    
    # Mode selection section
    mode_frame = ttk.LabelFrame(root, text="Mode Selection", padding="5")
    mode_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
    
    mode_var = tk.IntVar(value=1)
    mode1_radio = ttk.Radiobutton(mode_frame, text="Mode 1: Read Count Duplication", variable=mode_var, value=1, command=on_mode_change)
    mode1_radio.grid(row=0, column=0, padx=(0, 20), sticky="w")
    
    mode2_radio = ttk.Radiobutton(mode_frame, text="Mode 2: Sequence Length Management", variable=mode_var, value=2, command=on_mode_change)
    mode2_radio.grid(row=0, column=1, sticky="w")
    
    mode3_radio = ttk.Radiobutton(mode_frame, text="Mode 3: Length Management + Keep Extra Columns", variable=mode_var, value=3, command=on_mode_change)
    mode3_radio.grid(row=1, column=0, columnspan=2, sticky="w")
    
    # Parameters section
    params_frame = ttk.LabelFrame(root, text="Parameters", padding="5")
    params_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
    
    max_length_label = ttk.Label(params_frame, text="Max Length (split if exceeded):")
    max_length_label.grid(row=0, column=0, padx=(0, 5), sticky="w")
    max_length_entry = ttk.Entry(params_frame, width=10)
    max_length_entry.insert(0, "250")
    max_length_entry.grid(row=0, column=1, padx=5, sticky="w")
    
    min_length_label = ttk.Label(params_frame, text="Min Length (remove if below):")
    min_length_label.grid(row=1, column=0, padx=(0, 5), sticky="w")
    min_length_entry = ttk.Entry(params_frame, width=10)
    min_length_entry.insert(0, "25")
    min_length_entry.grid(row=1, column=1, padx=5, sticky="w")
    
    # Input file section
    input_frame = ttk.LabelFrame(root, text="Input File", padding="5")
    input_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
    
    ttk.Label(input_frame, text="Input CSV file:").grid(row=0, column=0, padx=(0, 5), sticky="w")
    input_entry = ttk.Entry(input_frame, width=50)
    input_entry.insert(0, "train_seq.csv")
    input_entry.grid(row=0, column=1, padx=5, sticky="ew")
    input_frame.grid_columnconfigure(1, weight=1)
    
    # Output file section
    output_frame = ttk.LabelFrame(root, text="Output File", padding="5")
    output_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
    
    ttk.Label(output_frame, text="Output CSV file:").grid(row=0, column=0, padx=(0, 5), sticky="w")
    output_entry = ttk.Entry(output_frame, width=50)
    output_entry.insert(0, "train_seq_4col.csv")
    output_entry.grid(row=0, column=1, padx=5, sticky="ew")
    output_frame.grid_columnconfigure(1, weight=1)
    
    # Instructions
    instructions = ttk.Label(root, text="Mode 1: Duplicates rows based on Read_Count\nMode 2: Splits long sequences and removes short ones\nMode 3: Same as Mode 2 but keeps extra columns (Read_Count, DMR_cType)\nOutput is always tab-delimited (\\t) for MethylBERT compatibility.")
    instructions.grid(row=5, column=0, padx=10, pady=5, sticky="w")
    
    # Button frame
    button_frame = ttk.Frame(root)
    button_frame.grid(row=6, column=0, padx=10, pady=10, sticky="ew")
    
    # Submit button
    submit_btn = ttk.Button(button_frame, text="Submit", command=on_submit)
    submit_btn.pack(side=tk.RIGHT, padx=(5, 0))
    
    # Cancel button
    cancel_btn = ttk.Button(button_frame, text="Cancel", command=on_cancel)
    cancel_btn.pack(side=tk.RIGHT)
    
    # Bind Enter key to submit
    root.bind('<Return>', lambda event: on_submit())
    root.bind('<Escape>', lambda event: on_cancel())
    
    # Initialize mode state
    on_mode_change()
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start GUI
    root.mainloop()
    
    return input_file, output_file, mode, max_length, min_length

def create_cli_interface():
    """
    Create a command-line interface for file selection when GUI is not available
    
    Returns:
        tuple: (input_file, output_file, mode, max_length, min_length) or (None, None, None, None, None) if cancelled
    """
    print("=" * 60)
    print("ENHANCED DATA FORMAT CONVERTER - COMMAND LINE INTERFACE")
    print("=" * 60)
    
    # Mode selection
    print("Select mode:")
    print("1. Read Count Duplication (current functionality)")
    print("2. Sequence Length Management (split long, remove short)")
    print("3. Length Management + Keep Extra Columns (split long, remove short, keep Read_Count/DMR_cType)")
    
    while True:
        try:
            mode = int(input("Enter mode (1, 2, or 3, default: 1): ").strip() or "1")
            if mode in [1, 2, 3]:
                break
            else:
                print("Please enter 1, 2, or 3.")
        except ValueError:
            print("Please enter a valid number (1, 2, or 3).")
    
    # Parameters for mode 2 and 3
    max_length = 250
    min_length = 25
    
    if mode in [2, 3]:
        print(f"\nMode {mode} selected. Configuring parameters...")
        try:
            max_input = input(f"Max sequence length before splitting (default: {max_length}): ").strip()
            if max_input:
                max_length = int(max_input)
        except ValueError:
            print(f"Invalid input, using default: {max_length}")
        
        try:
            min_input = input(f"Min sequence length before removing (default: {min_length}): ").strip()
            if min_input:
                min_length = int(min_input)
        except ValueError:
            print(f"Invalid input, using default: {min_length}")
    
    # Get input file
    input_file = input("Enter input CSV file (default: train_seq.csv): ").strip()
    if not input_file:
        input_file = "train_seq.csv"
    
    # Add .csv extension if not present
    if not input_file.endswith('.csv'):
        input_file = input_file + '.csv'
    
    print(f"Input file: {input_file}")
    
    # Get output file
    output_file = input("Enter output CSV file (default: train_seq_4col.csv): ").strip()
    if not output_file:
        output_file = "train_seq_4col.csv"
    
    # Add .csv extension if not present
    if not output_file.endswith('.csv'):
        output_file = output_file + '.csv'
    
    print(f"Output file: {output_file}")
    print(f"Mode: {mode}")
    if mode in [2, 3]:
        print(f"Max length: {max_length}")
        print(f"Min length: {min_length}")
    
    return input_file, output_file, mode, max_length, min_length

def detect_delimiter(file_path):
    """
    Automatically detect the delimiter used in the CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        str: Detected delimiter (',' or '\t')
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read first few lines to detect delimiter
            sample_lines = []
            for i, line in enumerate(f):
                if i >= 5:  # Read first 5 lines
                    break
                sample_lines.append(line.strip())
            
            # Count commas and tabs in the sample
            comma_count = sum(line.count(',') for line in sample_lines)
            tab_count = sum(line.count('\t') for line in sample_lines)
            
            print(f"Delimiter analysis: {comma_count} commas, {tab_count} tabs found in sample")
            
            # Determine delimiter based on frequency
            if comma_count > tab_count:
                print("Detected delimiter: comma (,)")
                return ','
            else:
                print("Detected delimiter: tab (\\t)")
                return '\t'
                
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                sample_lines = []
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    sample_lines.append(line.strip())
                
                comma_count = sum(line.count(',') for line in sample_lines)
                tab_count = sum(line.count('\t') for line in sample_lines)
                
                print(f"Delimiter analysis (latin-1): {comma_count} commas, {tab_count} tabs found in sample")
                
                if comma_count > tab_count:
                    print("Detected delimiter: comma (,) (with latin-1 encoding)")
                    return ','
                else:
                    print("Detected delimiter: tab (\\t) (with latin-1 encoding)")
                    return '\t'
        except Exception as e:
            print(f"Warning: Could not detect delimiter with latin-1 encoding: {e}")
            print("Defaulting to comma delimiter")
            return ','
    except Exception as e:
        print(f"Warning: Could not detect delimiter automatically: {e}")
        print("Defaulting to comma delimiter")
        return ','

def split_sequence(dna_seq, methyl_seq, ctype, dmr_label, max_length, extra_cols=None):
    """
    Split a sequence if it exceeds max_length
    
    Args:
        dna_seq: DNA sequence (space-separated 3-mers)
        methyl_seq: Methylation sequence
        ctype: Cell type
        dmr_label: DMR label
        max_length: Maximum allowed length
        extra_cols: Dictionary of additional columns to preserve
        
    Returns:
        list: List of tuples (dna_seq, methyl_seq, ctype, dmr_label, extra_cols) or (dna_seq, methyl_seq, ctype, dmr_label) if no extra_cols
    """
    if len(methyl_seq) <= max_length:
        if extra_cols:
            return [(dna_seq, methyl_seq, ctype, dmr_label, extra_cols)]
        else:
            return [(dna_seq, methyl_seq, ctype, dmr_label)]
    
    # Split the methylation sequence in half
    mid_point = len(methyl_seq) // 2
    
    # Split methylation sequence
    methyl_seq_1 = methyl_seq[:mid_point]
    methyl_seq_2 = methyl_seq[mid_point:]
    
    # Split DNA sequence (each 3-mer corresponds to one methylation character)
    dna_3mers = dna_seq.split()
    mid_3mer = mid_point
    
    dna_seq_1 = " ".join(dna_3mers[:mid_3mer])
    dna_seq_2 = " ".join(dna_3mers[mid_3mer:])
    
    # Recursively split if still too long
    result = []
    
    # Check first half
    if len(methyl_seq_1) > max_length:
        result.extend(split_sequence(dna_seq_1, methyl_seq_1, ctype, dmr_label, max_length, extra_cols))
    else:
        if extra_cols:
            result.append((dna_seq_1, methyl_seq_1, ctype, dmr_label, extra_cols))
        else:
            result.append((dna_seq_1, methyl_seq_1, ctype, dmr_label))
    
    # Check second half
    if len(methyl_seq_2) > max_length:
        result.extend(split_sequence(dna_seq_2, methyl_seq_2, ctype, dmr_label, max_length, extra_cols))
    else:
        if extra_cols:
            result.append((dna_seq_2, methyl_seq_2, ctype, dmr_label, extra_cols))
        else:
            result.append((dna_seq_2, methyl_seq_2, ctype, dmr_label))
    
    return result

def convert_data_format_enhanced(input_file, output_file, mode=1, max_length=250, min_length=25):
    """
    Convert data to 4-column format with enhanced features
    
    Args:
        input_file: Path to input file
        output_file: Path to output file with 4+ columns (depending on mode)
        mode: 1 for read count duplication, 2 for sequence length management, 3 for length management + keep extra columns
        max_length: Maximum sequence length before splitting (modes 2 & 3)
        min_length: Minimum sequence length before removing (modes 2 & 3)
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    if mode == 1:
        print(f"Converting data format with read count duplication...")
    elif mode == 2:
        print(f"Converting data format with sequence length management...")
        print(f"Max length: {max_length}, Min length: {min_length}")
    else:  # mode == 3
        print(f"Converting data format with sequence length management + keeping extra columns...")
        print(f"Max length: {max_length}, Min length: {min_length}")
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return False
    
    # Detect delimiter automatically
    delimiter = detect_delimiter(input_file)
    
    # If automatic detection seems wrong, allow manual override
    if delimiter == '\t':
        # Double-check if this makes sense by looking at the first line
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                comma_count = first_line.count(',')
                tab_count = first_line.count('\t')
                
                if comma_count > tab_count and comma_count > 0:
                    print(f"Warning: Detected tab delimiter but found {comma_count} commas vs {tab_count} tabs in first line")
                    print("This suggests the file might actually be comma-delimited")
                    print("Attempting to use comma delimiter instead...")
                    delimiter = ','
        except Exception as e:
            print(f"Warning: Could not verify delimiter choice: {e}")
    
    print(f"Using delimiter: '{delimiter}' for input file")
    print(f"Output will ALWAYS be tab-delimited (\\t) for MethylBERT compatibility")
    
    # Determine if we're doing in-place replacement
    in_place = (input_file == output_file)
    if in_place:
        print("Note: Input and output files are the same - will replace input file")
        # Use temporary file for processing
        temp_output = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        temp_output_path = temp_output.name
        temp_output.close()
        actual_output = temp_output_path
    else:
        actual_output = output_file
    
    # Read and convert the data
    converted_rows = 0
    skipped_rows = 0
    duplicated_rows = 0
    total_output_rows = 0
    removed_short_rows = 0
    split_rows = 0
    
    try:
        # Try different encodings
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
        infile = None
        
        for encoding in encodings_to_try:
            try:
                infile = open(input_file, 'r', encoding=encoding)
                # Test if we can read the first line
                first_line = infile.readline()
                infile.seek(0)  # Reset to beginning
                print(f"Successfully opened file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                if infile:
                    infile.close()
                continue
        
        if not infile:
            print("Error: Could not open file with any supported encoding!")
            return False
        
        with infile, open(actual_output, 'w', newline='', encoding='utf-8') as outfile:
            reader = csv.reader(infile, delimiter=delimiter)
            writer = csv.writer(outfile, delimiter='\t')  # Always output tab-delimited
            
            # Read header
            try:
                header = next(reader)
                print(f"Original header: {header}")
            except StopIteration:
                print("Error: Input file is empty!")
                return False
            
            # Clean header (remove any whitespace and quotes)
            header = [col.strip().strip('"\'') for col in header]
            print(f"Cleaned header: {header}")
            
            # Find the required column indices (case-insensitive matching)
            required_columns = ['DNA_seq', 'Methyl_seq', 'DMR_Label', 'cType']
            if mode == 1:
                required_columns.append('Read_Count')
            
            column_indices = {}
            
            for col in required_columns:
                # Try exact match first
                if col in header:
                    column_indices[col] = header.index(col)
                else:
                    # Try case-insensitive match
                    found = False
                    for i, header_col in enumerate(header):
                        if header_col.lower() == col.lower():
                            column_indices[col] = i
                            found = True
                            print(f"Found column '{col}' as '{header_col}' (case-insensitive match)")
                            break
                    
                    if not found:
                        print(f"Error: Required column '{col}' not found in header!")
                        print(f"Available columns: {header}")
                        print(f"Please check if your file has the expected column names.")
                        print(f"Note: The script detected delimiter as '{delimiter}' - if this seems wrong,")
                        print(f"      you may need to manually edit the detect_delimiter function.")
                        return False
            
            # For Mode 3, also check for optional extra columns (case-insensitive matching)
            extra_column_indices = {}
            if mode == 3:
                optional_columns = ['Read_Count', 'DMR_cType']
                for col in optional_columns:
                    # Try exact match first
                    if col in header:
                        extra_column_indices[col] = header.index(col)
                        print(f"Found optional column '{col}' at index {extra_column_indices[col]}")
                    else:
                        # Try case-insensitive match
                        for i, header_col in enumerate(header):
                            if header_col.lower() == col.lower():
                                extra_column_indices[col] = i
                                print(f"Found optional column '{col}' as '{header_col}' (case-insensitive match) at index {i}")
                                break
            
            print(f"Found required column indices: {column_indices}")
            if mode == 3 and extra_column_indices:
                print(f"Found extra column indices: {extra_column_indices}")
            
            # Create new header based on mode
            if mode == 3 and extra_column_indices:
                new_header = ['dna_seq', 'methyl_seq', 'ctype', 'dmr_label'] + list(extra_column_indices.keys())
            else:
                new_header = ['dna_seq', 'methyl_seq', 'ctype', 'dmr_label']
            
            writer.writerow(new_header)
            print(f"New header: {new_header}")
            print(f"Output format: Tab-delimited (\\t)")
            
            # Process data rows
            for row_num, row in enumerate(reader, 2):
                if len(row) >= max(column_indices.values()) + 1:
                    # Extract the required columns
                    dna_seq = row[column_indices['DNA_seq']]
                    methyl_seq = row[column_indices['Methyl_seq']]
                    ctype = row[column_indices['cType']]
                    dmr_label = row[column_indices['DMR_Label']]
                    
                    if mode == 1:
                        # Mode 1: Read count duplication
                        read_count = row[column_indices['Read_Count']]
                        
                        # Parse read count
                        try:
                            read_count_int = int(read_count)
                            if read_count_int < 1:
                                print(f"Warning: Row {row_num} has invalid read count '{read_count}'. Using 1.")
                                read_count_int = 1
                        except ValueError:
                            print(f"Warning: Row {row_num} has invalid read count '{read_count}'. Using 1.")
                            read_count_int = 1
                        
                        # Write the row read_count_int times
                        for _ in range(read_count_int):
                            writer.writerow([dna_seq, methyl_seq, ctype, dmr_label])
                            total_output_rows += 1
                        
                        converted_rows += 1
                        if read_count_int > 1:
                            duplicated_rows += (read_count_int - 1)
                    
                    elif mode == 2:
                        # Mode 2: Sequence length management
                        methyl_len = len(methyl_seq)
                        
                        # Check if sequence is too short
                        if methyl_len < min_length:
                            removed_short_rows += 1
                            continue
                        
                        # Check if sequence needs splitting
                        if methyl_len > max_length:
                            # Split the sequence
                            split_results = split_sequence(dna_seq, methyl_seq, ctype, dmr_label, max_length)
                            split_rows += 1
                            
                            for split_dna, split_methyl, split_ctype, split_dmr in split_results:
                                writer.writerow([split_dna, split_methyl, split_ctype, split_dmr])
                                total_output_rows += 1
                        else:
                            # Write as is
                            writer.writerow([dna_seq, methyl_seq, ctype, dmr_label])
                            total_output_rows += 1
                        
                        converted_rows += 1
                    
                    else:  # mode == 3
                        # Mode 3: Sequence length management + keep extra columns
                        methyl_len = len(methyl_seq)
                        
                        # Extract extra columns if they exist
                        extra_cols = {}
                        if extra_column_indices:
                            for col_name, col_index in extra_column_indices.items():
                                if len(row) > col_index:
                                    extra_cols[col_name] = row[col_index]
                        
                        # Check if sequence is too short
                        if methyl_len < min_length:
                            removed_short_rows += 1
                            continue
                        
                        # Check if sequence needs splitting
                        if methyl_len > max_length:
                            # Split the sequence
                            split_results = split_sequence(dna_seq, methyl_seq, ctype, dmr_label, max_length, extra_cols)
                            split_rows += 1
                            
                            for result in split_results:
                                if len(result) == 5:  # Has extra columns
                                    split_dna, split_methyl, split_ctype, split_dmr, split_extra = result
                                    row_data = [split_dna, split_methyl, split_ctype, split_dmr]
                                    # Add extra columns in the same order as header
                                    for col_name in extra_column_indices.keys():
                                        row_data.append(split_extra.get(col_name, ''))
                                    writer.writerow(row_data)
                                else:  # No extra columns
                                    split_dna, split_methyl, split_ctype, split_dmr = result
                                    writer.writerow([split_dna, split_methyl, split_ctype, split_dmr])
                                total_output_rows += 1
                        else:
                            # Write as is
                            row_data = [dna_seq, methyl_seq, ctype, dmr_label]
                            # Add extra columns in the same order as header
                            for col_name in extra_column_indices.keys():
                                row_data.append(extra_cols.get(col_name, ''))
                            writer.writerow(row_data)
                            total_output_rows += 1
                        
                        converted_rows += 1
                    
                    # Progress indicator
                    if converted_rows % 10000 == 0:
                        print(f"Processed {converted_rows:,} input rows, generated {total_output_rows:,} output rows...")
                        
                else:
                    print(f"Warning: Row {row_num} has {len(row)} columns, expected at least {max(column_indices.values()) + 1}. Skipping.")
                    skipped_rows += 1
        
        # If in-place replacement, move temp file to original location
        if in_place:
            shutil.move(temp_output_path, input_file)
            print(f"In-place replacement completed: {input_file}")
        
        print(f"Conversion completed!")
        print(f"Converted input rows: {converted_rows:,}")
        print(f"Generated output rows: {total_output_rows:,}")
        
        if mode == 1:
            print(f"Duplicated rows (Read_Count > 1): {duplicated_rows:,}")
        elif mode in [2, 3]:
            print(f"Split rows (length > {max_length}): {split_rows:,}")
            print(f"Removed short rows (length < {min_length}): {removed_short_rows:,}")
        
        print(f"Skipped rows: {skipped_rows:,}")
        print(f"Output file: {output_file}")
        print(f"Output format: Tab-delimited (\\t)")
        
        if mode == 1:
            print(f"Removed columns: Read_Count and any other unnecessary columns")
        elif mode == 2:
            print(f"Removed columns: All unnecessary columns")
        else:  # mode == 3
            if 'extra_column_indices' in locals() and extra_column_indices:
                print(f"Kept extra columns: {list(extra_column_indices.keys())}")
                print(f"Removed columns: All other unnecessary columns")
            else:
                print(f"No extra columns found to keep")
                print(f"Removed columns: All unnecessary columns")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        if in_place and os.path.exists(temp_output_path):
            os.unlink(temp_output_path)
        return False

def main():
    """Main function"""
    
    print("=" * 60)
    print("ENHANCED METHYLBERT DATA FORMAT CONVERTER")
    print("=" * 60)
    print("Features:")
    print("- Three modes: Read count duplication, sequence length management, or length management + extra columns")
    print("- User-controlled input/output file names")
    print("- In-place file replacement supported")
    print("- Converts to 4+ column format (depending on mode)")
    print("- Output is ALWAYS tab-delimited (\\t) for MethylBERT compatibility")
    print("=" * 60)
    
    # Get file names and mode from interface
    if GUI_AVAILABLE:
        print("Opening GUI for file selection and mode selection...")
        result = create_gui()
    else:
        print("Using command-line interface...")
        result = create_cli_interface()
    
    if not result:
        print("No files selected. Exiting.")
        return False
    
    input_file, output_file, mode, max_length, min_length = result
    
    if not input_file:
        print("No input file specified. Exiting.")
        return False
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Mode: {mode}")
    if mode in [2, 3]:
        print(f"Max length: {max_length}")
        print(f"Min length: {min_length}")
    
    # Perform the conversion
    success = convert_data_format_enhanced(input_file, output_file, mode, max_length, min_length)
    
    if success:
        print(f"\n🎉 Enhanced data format conversion completed successfully!")
        print(f"You can now use {output_file} for MethylBERT training.")
    else:
        print(f"\n❌ Enhanced data format conversion failed!")
    
    return success

if __name__ == "__main__":
    main() 