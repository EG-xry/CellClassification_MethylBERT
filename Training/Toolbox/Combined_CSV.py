#!/usr/bin/env python3
"""
Generate combined_preprocessed.csv file

This script:
1. Provides a GUI textbox for users to paste file names
2. Only concatenates the files that are pasted by the user (separated by comma and space)
3. Concatenates CSV files with PAT file markers
4. Adds header: DNA_seq,Methyl_seq,DMR_Label,cType,DMR_cType,Read_Count
5. Counts total lines efficiently         
Usage:
- Run the script
- A GUI window will open with a textbox
- Paste file names separated by comma and space (e.g., "file1.csv, file2.csv, file3.csv")
- Click Submit to process the files

Output format:
DNA_seq,Methyl_seq,DMR_Label,cType,DMR_cType
xxxxxx.pat.gz
[data from this PAT file]
xxxxxx.pat.gz
[data from next PAT file]
...
"""

import os
import csv
import glob
from pathlib import Path
import time

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
    Create a GUI window with a textbox for pasting file names and output filename
    
    Returns:
        tuple: (list of file names, output filename)
    """
    if not GUI_AVAILABLE:
        print("Error: GUI not available, tkinter module not found.")
        return None, None
    
    file_names = []
    output_filename = "combined_preprocessed.csv"
    
    def on_submit():
        """Handle submit button click"""
        nonlocal file_names, output_filename
        text_content = text_area.get("1.0", tk.END).strip()
        output_filename = output_entry.get().strip()
        
        if not text_content:
            messagebox.showwarning("Warning", "Please enter at least one file name!")
            return
        
        if not output_filename:
            messagebox.showwarning("Warning", "Please enter an output filename!")
            return
        
        # Add .csv extension if not present
        if not output_filename.endswith('.csv'):
            output_filename = output_filename + '.csv'
        
        # Parse file names (separated by comma and space)
        file_names = [name.strip() for name in text_content.split(", ")]
        file_names = [name for name in file_names if name]  # Remove empty strings
        
        if not file_names:
            messagebox.showwarning("Warning", "No valid file names found!")
            return
        
        root.destroy()
    
    def on_cancel():
        """Handle cancel button click"""
        root.destroy()
    
    # Create main window
    root = tk.Tk()
    root.title("CSV File Concatenation Tool")
    root.geometry("700x500")
    root.resizable(True, True)
    
    # Configure grid
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(2, weight=1)
    
    # Title label
    title_label = ttk.Label(root, text="CSV File Concatenation Tool", font=("Arial", 14, "bold"))
    title_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
    
    # Output filename section
    output_frame = ttk.LabelFrame(root, text="Output File", padding="5")
    output_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
    
    ttk.Label(output_frame, text="Output filename:").grid(row=0, column=0, padx=(0, 5), sticky="w")
    output_entry = ttk.Entry(output_frame, width=40)
    output_entry.insert(0, "combined_preprocessed.csv")
    output_entry.grid(row=0, column=1, padx=5, sticky="ew")
    output_frame.grid_columnconfigure(1, weight=1)
    
    # Input files section
    input_frame = ttk.LabelFrame(root, text="Input Files", padding="5")
    input_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
    
    # Instructions label
    instructions = ttk.Label(input_frame, text="Paste file names separated by comma and space (e.g., file1.csv, file2.csv, file3.csv)")
    instructions.grid(row=0, column=0, padx=5, pady=(0, 5), sticky="w")
    
    # Text area for file names
    text_area = scrolledtext.ScrolledText(input_frame, height=12, width=70, wrap=tk.WORD)
    text_area.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
    text_area.focus()
    
    input_frame.grid_columnconfigure(0, weight=1)
    input_frame.grid_rowconfigure(1, weight=1)
    
    # Button frame
    button_frame = ttk.Frame(root)
    button_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
    
    # Submit button
    submit_btn = ttk.Button(button_frame, text="Submit", command=on_submit)
    submit_btn.pack(side=tk.RIGHT, padx=(5, 0))
    
    # Cancel button
    cancel_btn = ttk.Button(button_frame, text="Cancel", command=on_cancel)
    cancel_btn.pack(side=tk.RIGHT)
    
    # Bind Enter key to submit
    root.bind('<Return>', lambda event: on_submit())
    root.bind('<Escape>', lambda event: on_cancel())
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start GUI
    root.mainloop()
    
    return file_names, output_filename

def create_cli_interface():
    """
    Create a command-line interface for file selection when GUI is not available
    
    Returns:
        tuple: (list of file names, output filename)
    """
    print("=" * 60)
    print("CSV CONCATENATION TOOL - COMMAND LINE INTERFACE")
    print("=" * 60)
    
    # Get output filename
    output_filename = input("Enter output filename (default: combined_preprocessed.csv): ").strip()
    if not output_filename:
        output_filename = "combined_preprocessed.csv"
    
    # Add .csv extension if not present
    if not output_filename.endswith('.csv'):
        output_filename = output_filename + '.csv'
    
    print(f"Output file: {output_filename}")
    
    # Get input files
    print("\nEnter CSV file names separated by comma and space (e.g., file1.csv, file2.csv, file3.csv)")
    print("You can enter multiple lines. Press Enter on an empty line to finish:")
    
    file_input = ""
    line_count = 0
    while True:
        try:
            line = input(f"[{line_count + 1}] ")
            if not line.strip():
                break
            file_input += line + " "
            line_count += 1
        except EOFError:
            break
    
    if not file_input.strip():
        print("No files entered. Exiting.")
        return None, None
    
    # Parse file names
    file_names = [name.strip() for name in file_input.split(", ")]
    file_names = [name for name in file_names if name]  # Remove empty strings
    
    if not file_names:
        print("No valid file names found. Exiting.")
        return None, None
    
    print(f"\nFiles to process: {len(file_names)}")
    for i, file_name in enumerate(file_names, 1):
        print(f"  {i:2d}. {file_name}")
    
    return file_names, output_filename

def validate_files(file_names):
    """
    Validate that the specified files exist and are CSV files
    
    Args:
        file_names: List of file names to validate
        
    Returns:
        list: List of valid file paths
    """
    valid_files = []
    missing_files = []
    
    print(f"\nValidating {len(file_names)} specified files...")
    
    for file_name in file_names:
        # Add .csv extension if not present
        if not file_name.endswith('.csv'):
            file_name = file_name + '.csv'
        
        file_path = os.path.join(".", file_name)
        
        if os.path.exists(file_path):
            valid_files.append(file_path)
            print(f"  ✓ Found: {file_name}")
        else:
            missing_files.append(file_name)
            print(f"  ✗ Missing: {file_name}")
    
    if missing_files:
        print(f"\nWarning: {len(missing_files)} files not found:")
        for file_name in missing_files:
            print(f"  - {file_name}")
    
    return valid_files

def collect_unique_ctype_values(csv_files):
    """
    Collect all unique cType values from the input CSV files
    
    Args:
        csv_files: List of CSV file paths
        
    Returns:
        set: Set of unique cType values
    """
    unique_ctype_values = set()
    
    print(f"\nCollecting unique cType values from {len(csv_files)} files...")
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r') as infile:
                reader = csv.reader(infile)
                
                # Find cType column index
                header = next(reader, None)
                if not header:
                    print(f"  Warning: Empty file {os.path.basename(csv_file)}")
                    continue
                
                ctype_index = None
                for i, col in enumerate(header):
                    if col.lower() == 'ctype':
                        ctype_index = i
                        break
                
                if ctype_index is None:
                    print(f"  Warning: No cType column found in {os.path.basename(csv_file)}")
                    continue
                
                # Collect cType values
                file_ctype_values = set()
                for row in reader:
                    if len(row) > ctype_index and row[ctype_index].strip():
                        file_ctype_values.add(row[ctype_index].strip())
                
                unique_ctype_values.update(file_ctype_values)
                print(f"  {os.path.basename(csv_file)}: {len(file_ctype_values)} unique cType values")
                
        except Exception as e:
            print(f"  Error reading {os.path.basename(csv_file)}: {e}")
            continue
    
    print(f"Total unique cType values found: {len(unique_ctype_values)}")
    if unique_ctype_values:
        print(f"cType values: {sorted(unique_ctype_values)}")
    
    return unique_ctype_values

def find_csv_files(directory=".", exclude_files=None):
    """
    Find all CSV files in the directory, excluding specified files
    
    Args:
        directory: Directory to search in
        exclude_files: List of filenames to exclude
        
    Returns:
        list: List of CSV file paths
    """
    if exclude_files is None:
        exclude_files = ["combined_old.csv", "hg38_all_1000.csv"]
    
    print(f"Searching for CSV files in: {directory}")
    print(f"Excluding files: {exclude_files}")
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    # Filter out excluded files
    filtered_files = []
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        if filename not in exclude_files:
            filtered_files.append(file_path)
    
    print(f"Found {len(csv_files)} total CSV files")
    print(f"After filtering: {len(filtered_files)} CSV files")
    
    return filtered_files

def extract_pat_filename_from_csv(csv_file):
    """
    Extract PAT filename from CSV filename
    Examples:
    - Adipocytes_T7.csv -> GSM5652176_Adipocytes-Z000000T7.hg38.pat.gz
    - Bone_marrow-Erythrocyte_RF_250.csv -> GSM5652XXX_Bone_marrow-Erythrocyte-Z000000RF.hg38.pat.gz
    - Blood-T-Eff-CD8_41Q_250.csv -> GSM5652XXX_Blood-T-Eff-CD8-Z00000041Q.hg38.pat.gz
    
    Args:
        csv_file: Path to CSV file
        
    Returns:
        str: PAT filename or None if cannot be determined
    """
    filename = os.path.basename(csv_file)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Try to map CSV filename to PAT filename
    # This mapping should be based on your actual file naming convention
    
    # Common patterns in your data:
    # Adipocytes_T7.csv -> GSM5652176_Adipocytes-Z000000T7.hg38.pat.gz
    # Skeletal-Muscle_427.csv -> GSM5652205_Skeletal-Muscle-Z00000427.hg38.pat.gz
    # Bone_marrow-Erythrocyte_RF_250.csv -> GSM5652XXX_Bone_marrow-Erythrocyte-Z000000RF.hg38.pat.gz
    
    # Extract cell type and identifier by splitting on the last underscore before the number
    if '_' in name_without_ext:
        parts = name_without_ext.split('_')
        
        # Handle cases where there might be multiple underscores
        # The identifier is typically the last meaningful part before any trailing numbers
        
        # Remove trailing "_250" if present (seems to be a common suffix)
        if parts[-1] == "250":
            parts = parts[:-1]
        
        if len(parts) >= 2:
            # The identifier is the last part, cell type is everything before that
            identifier = parts[-1]
            cell_type = '_'.join(parts[:-1])
            
            # Convert underscores to hyphens in cell type for PAT filename format
            cell_type_pat_format = cell_type.replace('_', '-')
            
            # Construct PAT filename based on known patterns
            if cell_type == "Adipocytes":
                return f"GSM5652176_Adipocytes-Z000000{identifier}.hg38.pat.gz"
            elif cell_type == "Skeletal-Muscle":
                return f"GSM5652205_Skeletal-Muscle-Z00000{identifier}.hg38.pat.gz"
            elif cell_type == "Blood-T":
                return f"GSM5652277_Blood-T-CD3-Z000000{identifier}.hg38.pat.gz"
            else:
                # Generic pattern for other cell types
                return f"GSM5652XXX_{cell_type_pat_format}-Z000000{identifier}.hg38.pat.gz"
    
    return None

def concatenate_csv_files(csv_files, output_file, unique_ctype_values=None):
    """
    Concatenate CSV files with PAT file markers and filter DMR_cType
    
    Args:
        csv_files: List of CSV file paths
        output_file: Output file path
        unique_ctype_values: Set of unique cType values to filter DMR_cType against
        
    Returns:
        int: Total number of lines written
    """
    
    print(f"\nConcatenating {len(csv_files)} CSV files...")
    print(f"Output file: {output_file}")
    if unique_ctype_values:
        print(f"Filtering DMR_cType to match cType values: {sorted(unique_ctype_values)}")
    
    total_lines = 0
    files_processed = 0
    filtered_lines = 0
    
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # Write header
        header = ['DNA_seq', 'Methyl_seq', 'DMR_Label', 'cType', 'DMR_cType', 'Read_Count']
        writer.writerow(header)
        total_lines += 1
        
        # Process each CSV file
        for csv_file in csv_files:
            files_processed += 1
            print(f"Processing file {files_processed}/{len(csv_files)}: {os.path.basename(csv_file)}")
            
            # Extract PAT filename
            pat_filename = extract_pat_filename_from_csv(csv_file)
            if pat_filename:
                print(f"  PAT file marker: {pat_filename}")
                # Write PAT file marker
                writer.writerow([pat_filename])
                total_lines += 1
            
            # Read and write CSV data
            try:
                with open(csv_file, 'r') as infile:
                    reader = csv.reader(infile)
                    
                    # Find column indices
                    header_row = next(reader, None)
                    if not header_row:
                        print(f"  Warning: Empty file {os.path.basename(csv_file)}")
                        continue
                    
                    # Find DMR_cType column index
                    dmr_ctype_index = None
                    for i, col in enumerate(header_row):
                        if col.lower() == 'dmr_ctype':
                            dmr_ctype_index = i
                            break
                    
                    if dmr_ctype_index is None:
                        print(f"  Warning: No DMR_cType column found in {os.path.basename(csv_file)}")
                        continue
                    
                    # Skip header if it exists
                    first_line = next(reader, None)
                    if first_line and first_line[0] in ['DNA_seq', 'DNA_seq,Methyl_seq,DMR_Label,cType,DMR_cType,Read_Count']:
                        print(f"  Skipping header in {os.path.basename(csv_file)}")
                    else:
                        # First line is data, not header
                        if unique_ctype_values is None or (len(first_line) > dmr_ctype_index and 
                                                          first_line[dmr_ctype_index].strip() in unique_ctype_values):
                            writer.writerow(first_line)
                            total_lines += 1
                        else:
                            filtered_lines += 1
                    
                    # Write remaining lines with filtering
                    for row in reader:
                        if unique_ctype_values is None or (len(row) > dmr_ctype_index and 
                                                          row[dmr_ctype_index].strip() in unique_ctype_values):
                            writer.writerow(row)
                            total_lines += 1
                        else:
                            filtered_lines += 1
                        
            except Exception as e:
                print(f"  Warning: Error processing {csv_file}: {e}")
                continue
            
            print(f"  Added {total_lines} lines so far")
    
    if filtered_lines > 0:
        print(f"\nFiltered out {filtered_lines} lines with DMR_cType not matching cType values")
    
    return total_lines

def count_lines_efficiently(file_path):
    """
    Count lines in a file efficiently for large files
    
    Args:
        file_path: Path to the file
        
    Returns:
        int: Number of lines
    """
    print(f"\nCounting lines in: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return 0
    
    start_time = time.time()
    
    # Use efficient line counting
    line_count = 0
    with open(file_path, 'r') as f:
        for line in f:
            line_count += 1
    
    elapsed_time = time.time() - start_time
    
    print(f"Total lines: {line_count:,}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Speed: {line_count/elapsed_time:,.0f} lines/second")
    
    return line_count

def main():
    """Main function"""
    
    print("=" * 60)
    print("CSV CONCATENATION AND PAT FILE MARKER GENERATOR")
    print("=" * 60)
    print("=" * 60)
    
    # Get file names and output filename from interface
    if GUI_AVAILABLE:
        print("Opening GUI for file selection...")
        result = create_gui()
    else:
        print("Using command-line interface...")
        result = create_cli_interface()
    
    if not result:
        print("No files selected. Exiting.")
        return False
    
    file_names, output_file = result
    
    if not file_names:
        print("No files selected. Exiting.")
        return False
    
    print(f"Output file: {output_file}")
    
    # Validate files
    csv_files = validate_files(file_names)
    
    if not csv_files:
        print("No valid CSV files found to process!")
        return False
    
    # Sort files for consistent ordering
    csv_files.sort()
    
    print(f"\nFiles to process:")
    for i, file_path in enumerate(csv_files, 1):
        print(f"  {i:2d}. {os.path.basename(file_path)}")
    
    # Collect unique cType values from all input files
    unique_ctype_values = collect_unique_ctype_values(csv_files)
    
    # Concatenate files with cType filtering
    print(f"\nStarting concatenation...")
    start_time = time.time()
    
    total_lines = concatenate_csv_files(csv_files, output_file, unique_ctype_values)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n Concatenation completed!")
    print(f"Output file: {output_file}")
    print(f"Total lines written: {total_lines:,}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    # Count lines in output file
    print(f"\n" + "=" * 60)
    print("LINE COUNT VERIFICATION")
    print("=" * 60)
    
    actual_lines = count_lines_efficiently(output_file)
    
    if actual_lines == total_lines:
        print(f" Line count verification successful!")
        print(f"Expected: {total_lines:,}, Actual: {actual_lines:,}")
    else:
        print(f" Line count mismatch!")
        print(f"Expected: {total_lines:,}, Actual: {actual_lines:,}")
    
    print(f"\n🎉 File generation completed successfully!")
    print(f"You can now use {output_file} with the PAT file splitter script.")
    
    return True

if __name__ == "__main__":
    main()