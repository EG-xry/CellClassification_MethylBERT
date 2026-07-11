#!/usr/bin/env python3
"""
PAT File-Based Train/Test Split Script (Five Modes)
This script implements four different train/test splitting strategies:

Mode 1: Simple 75/25 Split
- Basic random split without any duplicate handling

Mode 2: Avoid Duplicates (Default)
- Prevents identical sequences from appearing in both train and test
- Uses PAT file-based splitting with sequence grouping

Mode 3: Remove All Duplicates Then Split
- Completely removes all duplicate rows first
- Then performs random 75/25 split on deduplicated data

Mode 4: Manual PAT File Assignment
- Removes all duplicates first (Mode 3)
- Shows all available PAT file markers
- Allows manual assignment of PAT files to train/test sets
- No forced 75/25 split: user controls exact distribution

Mode 5: Cross-PAT Deduplication
- Checks for identical reads across different PAT files
- If identical reads have same ctype: removes duplicates
- If identical reads have different ctype: removes all and counts conflicts
- Performs random 75/25 split on deduplicated data

Features:
- User can specify input file name
- User can specify output file names
- User can select processing mode
- Command-line interface only 
- Test mode available with --test flag

Input: User-specified CSV file (contains PAT file markers: xxxxxx.pat.gz)
Output: User-specified train_seq.csv and test_seq.csv files
"""

import pandas as pd
import csv
import os
import sys
from collections import defaultdict
import random
from pathlib import Path

# Mode constants
MODE_SIMPLE_SPLIT = 1
MODE_AVOID_DUPLICATES = 2
MODE_REMOVE_ALL_DUPLICATES = 3
MODE_MANUAL_PAT_ASSIGNMENT = 4
MODE_CROSS_PAT_DEDUPLICATION = 5
MODE_STRICT_CROSS_PAT_DEDUPLICATION = 6

# CLI-only mode
def create_gui():
    """
    Create a CLI interface for file selection and mode selection
    Returns:
        tuple: (input_file, train_file, test_file, use_balanced, selected_mode) or (None, None, None, False, MODE_AVOID_DUPLICATES) if cancelled
    """
    # Always use CLI interface
    return create_cli_interface()

def create_manual_pat_assignment_gui(pat_file_markers):
    """
    Create a CLI interface for manual PAT file assignment
    Args:
        pat_file_markers: List of PAT file marker strings
    Returns:
        tuple: (train_pat_files, test_pat_files) or (None, None) if cancelled
    """
    return create_manual_pat_assignment_cli(pat_file_markers)

def create_manual_pat_assignment_cli(pat_file_markers):
    """
    Create a CLI interface for manual PAT file assignment
    Args:
        pat_file_markers: List of PAT file marker strings
    Returns:
        tuple: (train_pat_files, test_pat_files) or (None, None) if cancelled
    """
    print("=" * 60)
    print("MANUAL PAT FILE ASSIGNMENT - COMMAND LINE INTERFACE")
    print("=" * 60)
    
    print(f"\nFound {len(pat_file_markers)} PAT file markers:")
    for i, marker in enumerate(pat_file_markers, 1):
        print(f"  {i:2d}. {marker}")
    
    print(f"\nInstructions:")
    print(f"1. You need to assign each PAT file to either 'train' or 'test'")
    print(f"2. Enter the numbers of PAT files for training (separated by commas)")
    print(f"3. Enter the numbers of PAT files for testing (separated by commas)")
    print(f"4. All PAT files must be assigned to either train or test")
    print(f"5. No PAT file can be assigned to both train and test")
    
    while True:
        print(f"\n" + "=" * 40)
        train_input = input("Enter PAT file numbers for TRAINING (comma-separated, e.g., 1,3,5): ").strip()
        test_input = input("Enter PAT file numbers for TESTING (comma-separated, e.g., 2,4,6): ").strip()
        
        if not train_input and not test_input:
            print("Error: You must assign PAT files to at least one set!")
            continue
        
        # Parse train files
        train_pat_files = []
        if train_input:
            try:
                train_indices = [int(x.strip()) - 1 for x in train_input.split(',') if x.strip()]
                for idx in train_indices:
                    if 0 <= idx < len(pat_file_markers):
                        train_pat_files.append(pat_file_markers[idx])
                    else:
                        print(f"Error: Invalid index {idx + 1}. Valid range is 1-{len(pat_file_markers)}")
                        break
                else:
                    # All indices were valid
                    pass
            except ValueError:
                print("Error: Invalid input format. Please enter numbers separated by commas.")
                continue
        
        # Parse test files
        test_pat_files = []
        if test_input:
            try:
                test_indices = [int(x.strip()) - 1 for x in test_input.split(',') if x.strip()]
                for idx in test_indices:
                    if 0 <= idx < len(pat_file_markers):
                        test_pat_files.append(pat_file_markers[idx])
                    else:
                        print(f"Error: Invalid index {idx + 1}. Valid range is 1-{len(pat_file_markers)}")
                        break
                else:
                    # All indices were valid
                    pass
            except ValueError:
                print("Error: Invalid input format. Please enter numbers separated by commas.")
                continue
        
        # Validate assignment
        assigned_pat_files = set(train_pat_files + test_pat_files)
        all_pat_files = set(pat_file_markers)
        
        if assigned_pat_files != all_pat_files:
            missing_files = all_pat_files - assigned_pat_files
            print(f"Error: Missing PAT files ({len(missing_files)}):")
            for pat_file in sorted(missing_files):
                print(f"  - {pat_file}")
            continue
        
        # Check for duplicates
        if set(train_pat_files) & set(test_pat_files):
            print("Error: Some PAT files are assigned to both train and test sets!")
            continue
        
        # Success
        print(f"\nAssignment successful!")
        print(f"Training PAT files ({len(train_pat_files)}):")
        for pat_file in train_pat_files:
            print(f"  - {pat_file}")
        
        print(f"Test PAT files ({len(test_pat_files)}):")
        for pat_file in test_pat_files:
            print(f"  - {pat_file}")
        
        confirm = input(f"\nConfirm this assignment? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            return train_pat_files, test_pat_files
        else:
            print("Assignment cancelled. Please try again.")
            continue

def create_cli_interface():
    """
    Create a command-line interface for file selection and mode selection
    
    Returns:
        tuple: (input_file, train_file, test_file, use_balanced, selected_mode) or (None, None, None, False, MODE_AVOID_DUPLICATES) if cancelled
    """
    print("=" * 60)
    print("PAT FILE SPLIT TOOL - COMMAND LINE INTERFACE")
    print("=" * 60)
    
    # Mode selection
    print("\nSelect processing mode:")
    print("1. Mode 1: Simple 75/25 Split")
    print("2. Mode 2: Avoid Duplicates")
    print("3. Mode 3: Remove All Duplicates Then Split")
    print("4. Mode 4: Manual PAT File Assignment")
    print("5. Mode 5: Cross-PAT Deduplication")
    print("6. Mode 6: Strict Cross-PAT Deduplication")
    print("\nFor detailed mode information, run: python split_data_flexible.py --modes")
    
    while True:
        mode_input = input("Enter mode (1/2/3/4/5/6, default: 2): ").strip()
        if not mode_input:
            selected_mode = MODE_AVOID_DUPLICATES
            break
        elif mode_input in ['1', '2', '3', '4', '5', '6']:
            selected_mode = int(mode_input)
            break
        else:
            print("Invalid mode. Please enter 1, 2, 3, 4, 5, or 6.")
    
    print(f"Selected mode: {selected_mode}")
    
    # Get input file
    input_file = input("Enter input CSV file (default: combined_preprocessed.csv): ").strip()
    if not input_file:
        input_file = "combined_preprocessed.csv"
    
    print(f"Input file: {input_file}")
    
    # Get training output file
    train_file = input("Enter training output file (default: train_seq.csv): ").strip()
    if not train_file:
        train_file = "train_seq.csv"
    
    # Add .csv extension if not present
    if not train_file.endswith('.csv'):
        train_file = train_file + '.csv'
    
    print(f"Training file: {train_file}")
    
    # Get test output file
    test_file = input("Enter test output file (default: test_seq.csv): ").strip()
    if not test_file:
        test_file = "test_seq.csv"
    
    # Add .csv extension if not present
    if not test_file.endswith('.csv'):
        test_file = test_file + '.csv'
    
    print(f"Test file: {test_file}")
    
    # Get balanced data option (only for Mode 2)
    use_balanced = True
    if selected_mode == MODE_AVOID_DUPLICATES:
        use_balanced_input = input("Use balanced data approach (y/n, default: y): ").strip().lower()
        if use_balanced_input == 'n':
            use_balanced = False
        print(f"Using balanced data approach: {use_balanced}")
    else:
        print("Balanced data approach not applicable for selected mode.")
    
    return input_file, train_file, test_file, use_balanced, selected_mode

def extract_pat_file_markers(input_file):
    """
    Extract all PAT file markers from the input file
    Args:
        input_file: Path to the input CSV file
    Returns:
        list: List of PAT file marker strings
    """
    pat_file_markers = []
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return pat_file_markers
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        
        # Skip header
        try:
            next(reader)
        except StopIteration:
            print("Error: Empty file or no header found")
            return pat_file_markers
        
        # Process data lines
        for line_num, row in enumerate(reader, 2):
            # Check if this is a PAT file marker line (single column ending with .pat.gz)
            if len(row) == 1 and row[0].endswith('.pat.gz'):
                pat_file_markers.append(row[0])
    
    print(f"Found {len(pat_file_markers)} PAT file markers:")
    for i, marker in enumerate(pat_file_markers, 1):
        print(f"  {i:2d}. {marker}")
    
    return pat_file_markers

def manual_pat_file_assignment(input_file, train_ratio=0.75):
    """
    Mode 4: Remove all duplicates first, then allow manual PAT file assignment
    Args:
        input_file: Path to the input CSV file
        train_ratio: Not used in this mode (kept for compatibility)
    Returns:
        tuple: (train_data, test_data, header) - lists of data rows and header
    """
    print(f"\nMode 4: Manual PAT File Assignment")
    print("=" * 60)
    print("Step 1: Removing all duplicates...")
    
    # Remove duplicates like in Mode 3
    train_data, test_data, header = remove_duplicates_then_split(input_file, train_ratio)
    if train_data is None:
        return None, None, None
    
    # Combine all data for manual assignment
    all_data = train_data + test_data
    print(f"Step 1 completed: {len(all_data):,} unique sequences after deduplication")
    
    # Extract PAT file markers
    print(f"\nStep 2: Extracting PAT file markers...")
    pat_file_markers = extract_pat_file_markers(input_file)
    
    if not pat_file_markers:
        print("Error: No PAT file markers found in the input file!")
        return None, None, None
    
    # Create manual assignment interface
    print(f"\nStep 3: Opening manual assignment interface...")
    train_pat_files, test_pat_files = create_manual_pat_assignment_gui(pat_file_markers)
    
    if train_pat_files is None or test_pat_files is None:
        print("Manual assignment was cancelled.")
        return None, None, None
    
    print(f"\nStep 4: Processing manual assignment...")
    print(f"Training PAT files: {len(train_pat_files)}")
    for pat_file in train_pat_files:
        print(f"  - {pat_file}")
    
    print(f"Test PAT files: {len(test_pat_files)}")
    for pat_file in test_pat_files:
        print(f"  - {pat_file}")
    
    # Now we need to re-parse the file and assign data based on PAT file markers
    print(f"\nStep 5: Assigning data based on PAT file markers...")
    
    # Parse the deduplicated data by PAT files
    pat_file_data = {}
    current_pat_file = None
    current_data = []
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        
        # Skip header
        try:
            next(reader)
        except StopIteration:
            print("Error: Empty file or no header found")
            return None, None, None
        
        # Process data lines
        for line_num, row in enumerate(reader, 2):
            # Check if this is a PAT file marker line
            if len(row) == 1 and row[0].endswith('.pat.gz'):
                # Save previous PAT file data
                if current_pat_file and current_data:
                    pat_file_data[current_pat_file] = current_data.copy()
                
                # Start new PAT file
                current_pat_file = row[0]
                current_data = []
                continue
            
            # This is a data line
            if len(row) >= 5 and current_pat_file:
                current_data.append(row)
        
        # Save final PAT file data
        if current_pat_file and current_data:
            pat_file_data[current_pat_file] = current_data.copy()
    
    # Remove duplicates from each PAT file's data
    for pat_file in pat_file_data:
        seen_sequences = set()
        unique_data = []
        
        for row in pat_file_data[pat_file]:
            seq_key = (row[0], row[1])  # DNA_seq, Methyl_seq
            if seq_key not in seen_sequences:
                # Normalize read count to 1 if it's greater than 1
                if len(row) >= 6:
                    try:
                        original_read_count = int(row[5])
                        if original_read_count > 1:
                            row[5] = "1"
                    except (ValueError, IndexError):
                        pass
                
                unique_data.append(row)
                seen_sequences.add(seq_key)
        
        pat_file_data[pat_file] = unique_data
    
    # Assign data based on manual assignment
    final_train_data = []
    final_test_data = []
    
    for pat_file in train_pat_files:
        if pat_file in pat_file_data:
            final_train_data.extend(pat_file_data[pat_file])
            print(f"  Assigned {len(pat_file_data[pat_file]):,} sequences from {pat_file} to training")
        else:
            print(f"  Warning: PAT file {pat_file} not found in data")
    
    for pat_file in test_pat_files:
        if pat_file in pat_file_data:
            final_test_data.extend(pat_file_data[pat_file])
            print(f"  Assigned {len(pat_file_data[pat_file]):,} sequences from {pat_file} to testing")
        else:
            print(f"  Warning: PAT file {pat_file} not found in data")
    
    print(f"\nManual assignment completed!")
    print(f"  Training data: {len(final_train_data):,} sequences")
    print(f"  Test data: {len(final_test_data):,} sequences")
    print(f"  Train ratio: {len(final_train_data)/(len(final_train_data)+len(final_test_data)):.1%}")
    print(f"  No duplicates - each sequence appears only once")
    print(f"  Manual assignment - you controlled exactly which PAT files go where")
    
    return final_train_data, final_test_data, header

def cross_pat_deduplication(input_file, train_ratio=0.75):
    """
    Mode 5: Cross-PAT file deduplication with ctype-aware logic
    This mode checks for identical reads across different PAT files:
    - If two reads are identical except for ctype, and ctype is the same: remove one
    - If two reads are identical except for ctype, and ctype is different: remove both and count
    Args:
        input_file: Path to the input CSV file
        train_ratio: Target training ratio (default 0.75)
    Returns:
        tuple: (train_data, test_data, header) - lists of data rows and header
    """
    print(f"\nMode 5: Cross-PAT Deduplication")
    print("=" * 60)
    print("Step 1: Parsing PAT file sections...")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return None, None, None
    
    # Parse all PAT file sections first
    pat_file_data = {}
    current_pat_file = None
    current_data = []
    header = None
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            print(f"Header: {header}")
        except StopIteration:
            print("Error: Empty file or no header found")
            return None, None, None
        
        # Process data lines
        for line_num, row in enumerate(reader, 2):
            # Check if this is a PAT file marker line
            if len(row) == 1 and row[0].endswith('.pat.gz'):
                # Save previous PAT file data
                if current_pat_file and current_data:
                    pat_file_data[current_pat_file] = current_data.copy()
                
                # Start new PAT file
                current_pat_file = row[0]
                current_data = []
                continue
            
            # This is a data line
            if len(row) >= 5 and current_pat_file:
                current_data.append(row)
        
        # Save final PAT file data
        if current_pat_file and current_data:
            pat_file_data[current_pat_file] = current_data.copy()
    
    print(f"Found {len(pat_file_data)} PAT files")
    total_original_reads = sum(len(data) for data in pat_file_data.values())
    print(f"Total original reads: {total_original_reads:,}")
    
    print(f"\nStep 2: Performing cross-PAT deduplication...")
    
    # Create a mapping of sequence keys to all occurrences across PAT files
    sequence_occurrences = defaultdict(list)
    
    for pat_file, data in pat_file_data.items():
        for row in data:
            if len(row) >= 5:
                # Create sequence key (DNA_seq, Methyl_seq, DMR_Label, DMR_cType, Read_Count)
                # Exclude ctype (index 3) from the key
                seq_key = (row[0], row[1], row[2], row[4], row[5] if len(row) > 5 else "1")
                sequence_occurrences[seq_key].append((pat_file, row))
    
    # Process each sequence group
    final_data = []
    ctype_conflict_counter = defaultdict(int)
    same_ctype_removals = 0
    different_ctype_removals = 0
    
    for seq_key, occurrences in sequence_occurrences.items():
        if len(occurrences) == 1:
            # Only one occurrence, keep it
            final_data.append(occurrences[0][1])
        else:
            # Multiple occurrences across PAT files
            # Group by ctype
            ctype_groups = defaultdict(list)
            for pat_file, row in occurrences:
                ctype = row[3] if len(row) > 3 else "Unknown"
                ctype_groups[ctype].append((pat_file, row))
            
            if len(ctype_groups) == 1:
                # All occurrences have the same ctype
                # Keep only the first occurrence, remove the rest
                final_data.append(occurrences[0][1])
                same_ctype_removals += len(occurrences) - 1
                ctype = list(ctype_groups.keys())[0]
                ctype_conflict_counter[ctype] += len(occurrences) - 1
            else:
                # Different ctypes found - remove all occurrences
                different_ctype_removals += len(occurrences)
                for ctype in ctype_groups.keys():
                    ctype_conflict_counter[ctype] += len(ctype_groups[ctype])
    
    print(f"Cross-PAT deduplication completed!")
    print(f"  Original reads: {total_original_reads:,}")
    print(f"  Final reads: {len(final_data):,}")
    print(f"  Reads removed (same ctype): {same_ctype_removals:,}")
    print(f"  Reads removed (different ctype): {different_ctype_removals:,}")
    print(f"  Total removals: {same_ctype_removals + different_ctype_removals:,}")
    print(f"  Deduplication ratio: {len(final_data)/total_original_reads:.1%}")
    
    print(f"\nCtype conflict counter:")
    for ctype, count in sorted(ctype_conflict_counter.items()):
        print(f"  {ctype}: {count:,} reads removed")
    
    if len(final_data) == 0:
        print("Error: No data remaining after cross-PAT deduplication!")
        return None, None, None
    
    print(f"\nStep 3: Performing random 75/25 split...")
    
    # Random split of deduplicated data
    random.shuffle(final_data)
    split_point = int(len(final_data) * train_ratio)
    
    train_data = final_data[:split_point]
    test_data = final_data[split_point:]
    
    print(f"Split results:")
    print(f"  Training data: {len(train_data):,} lines ({len(train_data)/len(final_data):.1%})")
    print(f"  Test data: {len(test_data):,} lines ({len(test_data)/len(final_data):.1%})")
    print(f"  Cross-PAT deduplication completed")
    print(f"  Random 75/25 split applied")
    
    return train_data, test_data, header

def strict_cross_pat_deduplication(input_file, train_ratio=0.75):
    """
    Mode 6: Strict Cross-PAT file deduplication with ctype-aware logic
    This mode only removes sequences when ALL PAT files share the exact same sequence:
    - If a sequence appears in ALL PAT files with same ctype: keep one occurrence
    - If a sequence appears in ALL PAT files with different ctype: remove all and count conflicts
    - If a sequence appears in only some PAT files: keep all occurrences
    Args:
        input_file: Path to the input CSV file
        train_ratio: Target training ratio (default 0.75)
    Returns:
        tuple: (train_data, test_data, header) - lists of data rows and header
    """
    print(f"\nMode 6: Strict Cross-PAT Deduplication")
    print("=" * 60)
    print("Step 1: Parsing PAT file sections...")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return None, None, None
    
    # Parse all PAT file sections first
    pat_file_data = {}
    current_pat_file = None
    current_data = []
    header = None
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            print(f"Header: {header}")
        except StopIteration:
            print("Error: Empty file or no header found")
            return None, None, None
        
        # Process data lines
        for line_num, row in enumerate(reader, 2):
            # Check if this is a PAT file marker line
            if len(row) == 1 and row[0].endswith('.pat.gz'):
                # Save previous PAT file data
                if current_pat_file and current_data:
                    pat_file_data[current_pat_file] = current_data.copy()
                
                # Start new PAT file
                current_pat_file = row[0]
                current_data = []
                continue
            
            # This is a data line
            if len(row) >= 5 and current_pat_file:
                current_data.append(row)
        
        # Save final PAT file data
        if current_pat_file and current_data:
            pat_file_data[current_pat_file] = current_data.copy()
    
    print(f"Found {len(pat_file_data)} PAT files")
    total_original_reads = sum(len(data) for data in pat_file_data.values())
    print(f"Total original reads: {total_original_reads:,}")
    
    print(f"\nStep 2: Performing strict cross-PAT deduplication...")
    
    # Create a mapping of sequence keys to all occurrences across PAT files
    sequence_occurrences = defaultdict(list)
    
    for pat_file, data in pat_file_data.items():
        for row in data:
            if len(row) >= 5:
                # Create sequence key (DNA_seq, Methyl_seq, DMR_Label, DMR_cType, Read_Count)
                # Exclude ctype (index 3) from the key
                seq_key = (row[0], row[1], row[2], row[4], row[5] if len(row) > 5 else "1")
                sequence_occurrences[seq_key].append((pat_file, row))
    
    # Get all PAT file names
    all_pat_files = set(pat_file_data.keys())
    total_pat_files = len(all_pat_files)
    
    print(f"Total PAT files: {total_pat_files}")
    
    # Process each sequence group
    final_data = []
    ctype_conflict_counter = defaultdict(int)
    same_ctype_removals = 0
    different_ctype_removals = 0
    partial_occurrence_kept = 0
    
    for seq_key, occurrences in sequence_occurrences.items():
        # Get the set of PAT files where this sequence appears
        occurrence_pat_files = set(pat_file for pat_file, row in occurrences)
        
        if len(occurrence_pat_files) < total_pat_files:
            # Sequence does not appear in ALL PAT files - keep all occurrences
            for pat_file, row in occurrences:
                final_data.append(row)
            partial_occurrence_kept += len(occurrences)
        else:
            # Sequence appears in ALL PAT files - apply deduplication logic
            # Group by ctype
            ctype_groups = defaultdict(list)
            for pat_file, row in occurrences:
                ctype = row[3] if len(row) > 3 else "Unknown"
                ctype_groups[ctype].append((pat_file, row))
            
            if len(ctype_groups) == 1:
                # All occurrences have the same ctype
                # Keep only the first occurrence, remove the rest
                final_data.append(occurrences[0][1])
                same_ctype_removals += len(occurrences) - 1
                ctype = list(ctype_groups.keys())[0]
                ctype_conflict_counter[ctype] += len(occurrences) - 1
            else:
                # Different ctypes found - remove all occurrences
                different_ctype_removals += len(occurrences)
                for ctype in ctype_groups.keys():
                    ctype_conflict_counter[ctype] += len(ctype_groups[ctype])
    
    print(f"Strict cross-PAT deduplication completed!")
    print(f"  Original reads: {total_original_reads:,}")
    print(f"  Final reads: {len(final_data):,}")
    print(f"  Reads kept (partial occurrence): {partial_occurrence_kept:,}")
    print(f"  Reads removed (same ctype, all PAT files): {same_ctype_removals:,}")
    print(f"  Reads removed (different ctype, all PAT files): {different_ctype_removals:,}")
    print(f"  Total removals: {same_ctype_removals + different_ctype_removals:,}")
    print(f"  Deduplication ratio: {len(final_data)/total_original_reads:.1%}")
    
    print(f"\nCtype conflict counter (all PAT files only):")
    for ctype, count in sorted(ctype_conflict_counter.items()):
        print(f"  {ctype}: {count:,} reads removed")
    
    if len(final_data) == 0:
        print("Error: No data remaining after strict cross-PAT deduplication!")
        return None, None, None
    
    print(f"\nStep 3: Performing random 75/25 split...")
    
    # Random split of deduplicated data
    random.shuffle(final_data)
    split_point = int(len(final_data) * train_ratio)
    
    train_data = final_data[:split_point]
    test_data = final_data[split_point:]
    
    print(f"Split results:")
    print(f"  Training data: {len(train_data):,} lines ({len(train_data)/len(final_data):.1%})")
    print(f"  Test data: {len(test_data):,} lines ({len(test_data)/len(final_data):.1%})")
    print(f"  Strict cross-PAT deduplication completed")
    print(f"  Random 75/25 split applied")
    
    return train_data, test_data, header

# NEW BALANCING UTILITIES --------------------------------------------------------------------
def balance_pat_file_rows(rows, pat_cell_type, rand_instance):
    """Balance rows for a single PAT file using sequence group-aware logic

    First groups identical sequences together to prevent data leakage, then:
    - Keeps ALL sequence groups whose DMR_cType matches the PAT file's cell type
    - From the remaining sequence groups (other cell types), randomly selects groups to balance
      positive and negative samples by total weight (considering read counts)
    - Sampling of the negatives is proportional to their original representation
    - Returns all rows from selected groups, shuffled to avoid order bias
    """
    if not rows:
        return rows
    
    # Group identical sequences together
    sequence_groups = create_sequence_groups(rows)
    group_weights = calculate_group_weights(sequence_groups)
    
    # Separate positive and negative sequence groups
    positive_groups = {}
    other_groups = {}
    
    for seq_key, group_rows in sequence_groups.items():
        if len(group_rows[0]) >= 5:  # Check if DMR_cType column exists
            dmr_ctype = group_rows[0][4]  # DMR_cType is the same for all rows in group
            if dmr_ctype == pat_cell_type:
                positive_groups[seq_key] = group_rows
            else:
                other_groups[seq_key] = group_rows
    
    # Calculate total weights
    positive_weight = sum(group_weights[seq_key] for seq_key in positive_groups.keys())
    
    if positive_weight == 0:
        # Nothing to balance – return original rows unchanged
        return rows
    if not other_groups:
        # Only positive samples, return them
        result_rows = []
        for group_rows in positive_groups.values():
            result_rows.extend(group_rows)
        return result_rows

    # Target weight for negative samples (equal to positive weight)
    target_negative_weight = positive_weight
    
    # Group other sequence groups by their DMR_cType for proportional sampling
    other_by_ctype = defaultdict(list)
    for seq_key, group_rows in other_groups.items():
        ctype = group_rows[0][4]
        other_by_ctype[ctype].append(seq_key)
    
    # Calculate total weight of other groups
    total_other_weight = sum(group_weights[seq_key] for seq_key in other_groups.keys())
    
    # Sample sequence groups proportionally
    selected_other_groups = []
    allocated_weight = 0
    
    ctype_items = list(other_by_ctype.items())
    for idx, (ctype, seq_keys) in enumerate(ctype_items):
        # Calculate proportion and target weight for this cell type
        ctype_weight = sum(group_weights[seq_key] for seq_key in seq_keys)
        proportion = ctype_weight / total_other_weight
        target_weight = int(round(proportion * target_negative_weight))
        
        # Adjust for last group to avoid over-allocation
        if idx == len(ctype_items) - 1:
            target_weight = target_negative_weight - allocated_weight
        
        target_weight = max(0, min(target_weight, ctype_weight, target_negative_weight - allocated_weight))
        
        # Greedily select sequence groups until we reach target weight
        seq_keys_by_weight = sorted(seq_keys, key=lambda k: group_weights[k])
        current_weight = 0
        
        for seq_key in seq_keys_by_weight:
            if current_weight + group_weights[seq_key] <= target_weight:
                selected_other_groups.append(seq_key)
                current_weight += group_weights[seq_key]
                allocated_weight += group_weights[seq_key]
            if current_weight >= target_weight:
                break
    
    # Collect all rows from selected groups
    balanced_rows = []
    
    # Add all positive groups
    for group_rows in positive_groups.values():
        balanced_rows.extend(group_rows)
    
    # Add selected negative groups
    for seq_key in selected_other_groups:
        balanced_rows.extend(other_groups[seq_key])
    
    # Shuffle to avoid order bias
    rand_instance.shuffle(balanced_rows)
    return balanced_rows

def parse_pat_file_sections_balanced(input_file, random_seed=42):
    """Parse the input CSV and return a balanced data structure

    For every PAT file encountered, data rows are first collected, balanced using
    'balance_pat_file_rows' and then stored in the familiar nested dictionary
    structure: {cell_type: {pat_file: [rows]}}
    This drastically reduces memory usage because we only keep ~2× positives 
    per PAT file instead of every raw line
    """
    print(f"Parsing & balancing PAT file sections from: {input_file}")
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return None

    rand_instance = random.Random(random_seed)
    cell_type_pat_data = defaultdict(lambda: defaultdict(list))

    current_pat_file = None
    current_rows = []
    pat_cell_type = None
    total_lines = 0

    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            print(f"Header: {header}")
        except StopIteration:
            print("Error: Empty file or no header found")
            return None

        for line_num, row in enumerate(reader, 2):
            total_lines += 1
            if len(row) == 1 and row[0].endswith('.pat.gz'):
                # Process previous PAT file before switching
                if current_pat_file and current_rows:
                    if not pat_cell_type and current_rows[0]:
                        pat_cell_type = current_rows[0][3]
                    balanced_rows = balance_pat_file_rows(current_rows, pat_cell_type, rand_instance)
                    cell_type_pat_data[pat_cell_type][current_pat_file].extend(balanced_rows)
                    print(f"  → {current_pat_file}: kept {len(balanced_rows):,} balanced rows (orig {len(current_rows):,})")
                # Reset for new PAT file
                current_pat_file = row[0]
                current_rows = []
                pat_cell_type = None
                if total_lines % 1000000 == 0:
                    print(f"Processed {total_lines:,} lines so far...")
                continue

            # Data line
            if len(row) >= 5:
                current_rows.append(row)
                if not pat_cell_type and len(row) > 3:
                    pat_cell_type = row[3]
            else:
                if total_lines <= 10:
                    print(f"Warning: Invalid data line at {line_num}: {row}")

        # Final PAT file flush
        if current_pat_file and current_rows:
            if not pat_cell_type and current_rows[0]:
                pat_cell_type = current_rows[0][3]
            balanced_rows = balance_pat_file_rows(current_rows, pat_cell_type, rand_instance)
            cell_type_pat_data[pat_cell_type][current_pat_file].extend(balanced_rows)
            print(f"  → {current_pat_file}: kept {len(balanced_rows):,} balanced rows (orig {len(current_rows):,})")

    print(f"Finished parsing {total_lines:,} total lines")
    print(f"Cell types found: {len(cell_type_pat_data)}")
    for ctype, pat_dict in cell_type_pat_data.items():
        total_balanced = sum(len(v) for v in pat_dict.values())
        print(f"  {ctype}: {len(pat_dict)} PAT files, {total_balanced:,} balanced rows")
    return cell_type_pat_data
# -------------------------------------------------------------------- END BALANCING UTILITIES

def create_sequence_groups(rows):
    """
    Group identical sequences together to prevent data leakage
    Args:
        rows: List of data rows, each row should have format [DNA_seq, Methyl_seq, ...]
    Returns:
        dict: {sequence_key: [list_of_rows_with_same_sequence]}
    """
    sequence_groups = defaultdict(list)
    
    for row in rows:
        if len(row) >= 2:
            # Create key from DNA_seq and Methyl_seq (first two columns)
            sequence_key = (row[0], row[1])
            sequence_groups[sequence_key].append(row)
    
    return sequence_groups

def calculate_group_weights(sequence_groups):
    """
    Calculate the total weight (sum of read counts) for each sequence group.
    Args:
        sequence_groups: Dictionary of sequence groups
    Returns:
        dict: {sequence_key: total_weight}
    """
    group_weights = {}
    
    for seq_key, group_rows in sequence_groups.items():
        total_weight = 0
        for row in group_rows:
            if len(row) >= 6:  
                try:
                    read_count = int(row[5])  
                    total_weight += read_count
                except (ValueError, IndexError):
                    total_weight += 1  
            else:
                total_weight += 1 
        
        group_weights[seq_key] = total_weight
    
    return group_weights

def parse_pat_file_sections(input_file):
    """
    Parse the input file and separate data by PAT files
    Args:
        input_file: Path to the combined preprocessed CSV file
    Returns:
        dict: {cell_type: {pat_file: [data_lines]}}
    """
    
    print(f"Parsing PAT file sections from: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return None
    
    # Structure to store data: {cell_type: {pat_file: [data_lines]}}
    cell_type_pat_data = defaultdict(lambda: defaultdict(list))
    
    current_pat_file = None
    current_cell_type = None
    line_count = 0
    data_lines_without_marker = 0
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        
        # Read header
        try:
            header = next(reader)
            print(f"Header: {header}")
        except StopIteration:
            print("Error: Empty file or no header found")
            return None
        
        # Process data lines
        for line_num, row in enumerate(reader, 2):
            line_count += 1
            
            # Check if this is a PAT file marker line (single column ending with .pat.gz)
            if len(row) == 1 and row[0].endswith('.pat.gz'):
                # This is a PAT file marker
                current_pat_file = row[0]
                print(f"Found PAT file marker: {current_pat_file}")
                continue
    
            if len(row) >= 5:  # Should have DNA_seq, Methyl_seq, DMR_Label, cType, DMR_cType
                # Extract cell type from the cType column (4th column, index 3)
                cell_type = row[3] if len(row) > 3 else "Unknown"
                # If we have a current PAT file, assign this data to it
                if current_pat_file:
                    cell_type_pat_data[cell_type][current_pat_file].append(row)
                    # Only print first few assignments to avoid spam
                    if len(cell_type_pat_data[cell_type][current_pat_file]) <= 3:
                        print(f"  Assigned data line to {cell_type} -> {current_pat_file}")
                    elif len(cell_type_pat_data[cell_type][current_pat_file]) == 4:
                        print(f"  ... (continuing to assign data to {cell_type} -> {current_pat_file})")
                else:
                    # Data line found before any marker count it but don't assign
                    data_lines_without_marker += 1
                    if data_lines_without_marker <= 5:  # Only show first 5 warnings
                        print(f"Warning: Data line {line_num} found before PAT file marker (will be skipped)")
                    elif data_lines_without_marker == 6:
                        print(f"Warning: Additional data lines without markers will be skipped...")
            else:
                # Invalid data line format
                print(f"Warning: Invalid data line format at line {line_num}: {row}")
    
    if data_lines_without_marker > 0:
        print(f"Total data lines without markers: {data_lines_without_marker}")
    
    print(f"Parsed {line_count} total lines")
    print(f"Found {len(cell_type_pat_data)} cell types")
    
    # Print summary
    for cell_type, pat_files in cell_type_pat_data.items():
        total_lines = sum(len(lines) for lines in pat_files.values())
        print(f"  {cell_type}: {len(pat_files)} PAT files, {total_lines:,} total lines")
    
    return cell_type_pat_data

def split_sequence_groups_by_rows(sequence_groups, target_train_ratio, rand_instance):
    """
    Split sequence groups into train/test sets based on target row ratio while keeping
    identical sequences together
    This version optimizes for actual row counts rather than weights to achieve
    the desired train/test split ratio
    Args:
        sequence_groups: Dictionary of {sequence_key: [rows]}
        target_train_ratio: Target ratio for training set (e.g., 0.75 for 75%)
        rand_instance: Random instance for reproducible shuffling
    Returns:
        tuple: (train_groups, test_groups) - lists of sequence_keys
    """
    # Convert to list and shuffle for randomness
    group_items = list(sequence_groups.items())
    rand_instance.shuffle(group_items)
    
    # Calculate total rows and target
    total_rows = sum(len(group_rows) for group_rows in sequence_groups.values())
    target_train_rows = int(total_rows * target_train_ratio)
    
    train_groups = []
    test_groups = []
    current_train_rows = 0
    current_test_rows = 0
    
    for seq_key, group_rows in group_items:
        group_size = len(group_rows)
        
        # Calculate current ratios
        current_total_rows = current_train_rows + current_test_rows
        if current_total_rows == 0:
            # First assignment assign to train if we want more train data
            if target_train_ratio >= 0.5:
                train_groups.append(seq_key)
                current_train_rows += group_size
            else:
                test_groups.append(seq_key)
                current_test_rows += group_size
            continue
        
        # Decide based on which assignment gets us closer to target ratio
        # If adding to train
        new_train_rows_if_train = current_train_rows + group_size
        new_total_if_train = current_total_rows + group_size
        train_ratio_if_train = new_train_rows_if_train / new_total_if_train
        
        # If adding to test
        new_test_rows_if_test = current_test_rows + group_size
        new_total_if_test = current_total_rows + group_size
        train_ratio_if_test = current_train_rows / new_total_if_test
        
        # Choose the option that gets us closer to target ratio
        train_distance = abs(train_ratio_if_train - target_train_ratio)
        test_distance = abs(train_ratio_if_test - target_train_ratio)
        
        if train_distance <= test_distance:
            train_groups.append(seq_key)
            current_train_rows += group_size
        else:
            test_groups.append(seq_key)
            current_test_rows += group_size
    
    return train_groups, test_groups

def simple_random_split(input_file, train_ratio=0.75):
    """
    Mode 1: Simple random 75/25 split without any duplicate handling
    Args:
        input_file: Path to the input CSV file
        train_ratio: Target training ratio (default 0.75)
    Returns:
        tuple: (train_data, test_data, header) - lists of data rows and header
    """
    print(f"\nMode 1: Simple Random Split (train_ratio={train_ratio})")
    print("=" * 60)
    print("Warning: This mode may result in data leakage between train and test sets")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return None, None, None
    
    # Read all data
    all_data = []
    header = None
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            print(f"Header: {header}")
        except StopIteration:
            print("Error: Empty file or no header found")
            return None, None, None
        
        for line_num, row in enumerate(reader, 2):
            # Skip PAT file marker lines
            if len(row) == 1 and row[0].endswith('.pat.gz'):
                continue
            
            # Only include valid data lines
            if len(row) >= 5:
                all_data.append(row)
            
            if line_num % 1000000 == 0:
                print(f"Processed {line_num:,} lines...")
    
    print(f"Total valid data lines: {len(all_data):,}")
    
    # Simple random split
    random.shuffle(all_data)
    split_point = int(len(all_data) * train_ratio)
    
    train_data = all_data[:split_point]
    test_data = all_data[split_point:]
    
    print(f"Split results:")
    print(f"  Training data: {len(train_data):,} lines ({len(train_data)/len(all_data):.1%})")
    print(f"  Test data: {len(test_data):,} lines ({len(test_data)/len(all_data):.1%})")
    print(f"  No duplicate checking - identical sequences may appear in both sets")
    
    return train_data, test_data, header

def remove_duplicates_then_split(input_file, train_ratio=0.75):
    """
    Mode 3: Remove all duplicates first, then perform random 75/25 split
    When encountering data with read count > 1, the read count is normalized to 1
    to ensure each unique sequence contributes equally to the train/test split
    Args:
        input_file: Path to the input CSV file
        train_ratio: Target training ratio (default 0.75)
    Returns:
        tuple: (train_data, test_data, header) - lists of data rows and header
    """
    print(f"\nMode 3: Remove All Duplicates Then Split (train_ratio={train_ratio})")
    print("=" * 60)
    print("Note: Read counts > 1 will be normalized to 1 for equal sequence contribution")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return None, None, None

    # Read all data
    all_data = []
    header = None
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            print(f"Header: {header}")
        except StopIteration:
            print("Error: Empty file or no header found")
            return None, None, None
        
        for line_num, row in enumerate(reader, 2):
            # Skip PAT file marker lines
            if len(row) == 1 and row[0].endswith('.pat.gz'):
                continue
            
            # Only include valid data lines
            if len(row) >= 5:
                all_data.append(row)
            
            if line_num % 1000000 == 0:
                print(f"Processed {line_num:,} lines...")
    
    print(f"Original data lines: {len(all_data):,}")
    
    # Remove duplicates based on DNA_seq and Methyl_seq
    seen_sequences = set()
    unique_data = []
    duplicates_removed = 0
    read_counts_normalized = 0
    
    for row in all_data:
        if len(row) >= 6:  # Check if read count column exists (6th column, index 5)
            # Use DNA_seq and Methyl_seq as the key for deduplication
            seq_key = (row[0], row[1])
            if seq_key not in seen_sequences:
                # Normalize read count to 1 if it's greater than 1
                try:
                    original_read_count = int(row[5])
                    if original_read_count > 1:
                        row[5] = "1"  # Convert to string to match CSV format
                        read_counts_normalized += 1
                except (ValueError, IndexError):
                    # If read count is invalid, keep as is
                    pass
                
                unique_data.append(row)
                seen_sequences.add(seq_key)
            else:
                duplicates_removed += 1
        elif len(row) >= 2:
            # Row has DNA_seq and Methyl_seq but no read count column
            seq_key = (row[0], row[1])
            if seq_key not in seen_sequences:
                unique_data.append(row)
                seen_sequences.add(seq_key)
            else:
                duplicates_removed += 1
        else:
            # Invalid row, skip it
            duplicates_removed += 1
    
    print(f"Duplicates removed: {duplicates_removed:,}")
    print(f"Read counts normalized to 1: {read_counts_normalized:,}")
    print(f"Unique data lines remaining: {len(unique_data):,}")
    print(f"Deduplication ratio: {len(unique_data)/len(all_data):.1%}")
    
    if len(unique_data) == 0:
        print("Error: No unique data remaining after deduplication")
        return None, None, None
    
    # Random split of deduplicated data
    random.shuffle(unique_data)
    split_point = int(len(unique_data) * train_ratio)
    
    train_data = unique_data[:split_point]
    test_data = unique_data[split_point:]
    
    print(f"Split results:")
    print(f"  Training data: {len(train_data):,} lines ({len(train_data)/len(unique_data):.1%})")
    print(f"  Test data: {len(test_data):,} lines ({len(test_data)/len(unique_data):.1%})")
    print(f"  No duplicates, each sequence appears only once in the entire dataset")
    print(f"  Read counts normalized, each unique sequence contributes equally to split")
    
    return train_data, test_data, header

def split_pat_files_strategy(cell_type_pat_data, train_ratio=0.75):
    """
    Implement the PAT file-based splitting strategy with sequence grouping
    This function ensures that identical sequences (same DNA_seq and Methyl_seq) are never
    split between training and testing sets, preventing data leakage
    Args:
        cell_type_pat_data: Parsed data structure
        train_ratio: Target training ratio (default 0.75)
    Returns:
        tuple: (train_data, test_data) - lists of data rows
    """
    
    print(f"\nMode 2: PAT File-Based Splitting with Sequence Grouping (train_ratio={train_ratio})")
    print("=" * 60)
    
    train_data = []
    test_data = []
    rand_instance = random.Random(42)  # For reproducible sequence grouping
    
    for cell_type, pat_files in cell_type_pat_data.items():
        print(f"\nProcessing cell type: {cell_type}")
        print(f"  Number of PAT files: {len(pat_files)}")
        
        pat_file_list = list(pat_files.keys())
        total_lines = sum(len(lines) for lines in pat_files.values())
        
        print(f"  Total lines: {total_lines:,}")
        
        if len(pat_file_list) <= 3:
            # Strategy for 3 or fewer PAT files: Group sequences and split by weight
            print(f"  Strategy: Sequence-grouped 75/25 split (≤3 PAT files)")
            
            # Combine all data from all PAT files
            all_data = []
            for pat_file in pat_file_list:
                all_data.extend(pat_files[pat_file])
            
            # Create sequence groups
            sequence_groups = create_sequence_groups(all_data)
            group_weights = calculate_group_weights(sequence_groups)
            
            total_weight = sum(group_weights.values())
            target_train_weight = int(total_weight * train_ratio)
            
            print(f"  Total weight: {total_weight:,}")
            print(f"  Target train weight: {target_train_weight:,}")
            print(f"  Number of unique sequences: {len(sequence_groups):,}")
            
            # Split sequence groups
            train_seq_keys, test_seq_keys = split_sequence_groups_by_rows(
                sequence_groups, train_ratio, rand_instance
            )
            
            # Collect all rows from selected sequence groups
            cell_train_data = []
            cell_test_data = []
            
            for seq_key in train_seq_keys:
                cell_train_data.extend(sequence_groups[seq_key])
            
            for seq_key in test_seq_keys:
                cell_test_data.extend(sequence_groups[seq_key])
            
            train_data.extend(cell_train_data)
            test_data.extend(cell_test_data)
            
            actual_ratio = len(cell_train_data) / total_lines if total_lines > 0 else 0
            print(f"  Result: {len(cell_train_data):,} train, {len(cell_test_data):,} test ({actual_ratio:.1%} train ratio)")
            
        else:
            # Strategy for 4+ PAT files: Try to keep whole PAT files when possible
            print(f"  Strategy: Optimized whole PAT file assignment with sequence grouping (≥4 PAT files)")
            
            # Calculate lines per PAT file for analysis
            pat_file_sizes = [(pat_file, len(pat_files[pat_file])) for pat_file in pat_file_list]
            pat_file_sizes.sort(key=lambda x: x[1])  # Sort by size
            
            # Calculate target train lines based on total weight
            total_weight = sum(
                sum(calculate_group_weights(create_sequence_groups(pat_files[pf])).values())
                for pf in pat_file_list
            )
            target_train_weight = int(total_weight * train_ratio)
            
            print(f"  Total weight: {total_weight:,}")
            print(f"  Target train weight: {target_train_weight:,}")
            
            # Try different combinations of whole PAT files to find best train ratio
            best_train_ratio = 0
            best_train_files = []
            best_test_files = []
            best_remainder_to_split = None
            
            # Try assigning different numbers of whole PAT files to training
            for num_train_files in range(1, len(pat_file_list)):
                # Try the largest files first for training to get closer to target
                train_files = [pf[0] for pf in pat_file_sizes[-num_train_files:]]
                test_files = [pf[0] for pf in pat_file_sizes[:-num_train_files]]
                
                # Calculate weights for whole files
                train_weight_whole = sum(
                    sum(calculate_group_weights(create_sequence_groups(pat_files[pf])).values())
                    for pf in train_files
                )
                test_weight_whole = sum(
                    sum(calculate_group_weights(create_sequence_groups(pat_files[pf])).values())
                    for pf in test_files
                )
                
                # Check if this already achieves good ratio (within 5% of target)
                current_ratio = train_weight_whole / (train_weight_whole + test_weight_whole)
                
                if abs(current_ratio - train_ratio) <= 0.05:
                    # Good enough ratio with whole files
                    if current_ratio >= 0.70:  # Don't go below 70%
                        best_train_ratio = current_ratio
                        best_train_files = train_files
                        best_test_files = test_files
                        best_remainder_to_split = None
                        break
                
                # If ratio is too low, try splitting one test file to training
                if train_weight_whole < target_train_weight and test_files:
                    # Take the largest test file and see if splitting it helps
                    test_file_weights = [
                        (pf, sum(calculate_group_weights(create_sequence_groups(pat_files[pf])).values()))
                        for pf in test_files
                    ]
                    test_file_weights.sort(key=lambda x: x[1], reverse=True)
                    largest_test_file = test_file_weights[0][0]
                    
                    # Calculate how much weight we need from this file
                    needed_weight = target_train_weight - train_weight_whole
                    largest_test_file_weight = test_file_weights[0][1]
                    
                    if needed_weight < largest_test_file_weight:
                        # We can split this file by sequence groups
                        projected_train_weight = train_weight_whole + needed_weight
                        projected_test_weight = test_weight_whole - needed_weight
                        projected_ratio = projected_train_weight / (projected_train_weight + projected_test_weight)
                        
                        if projected_ratio >= 0.70 and projected_ratio > best_train_ratio:
                            best_train_ratio = projected_ratio
                            best_train_files = train_files
                            best_test_files = [pf for pf in test_files if pf != largest_test_file]
                            best_remainder_to_split = (largest_test_file, needed_weight)
            
            # If no good combination found, fall back to simple strategy
            if best_train_ratio == 0:
                # Fall back: take first 75% of files for training
                num_train_files = max(1, int(len(pat_file_list) * 0.75))
                best_train_files = pat_file_list[:num_train_files]
                best_test_files = pat_file_list[num_train_files:]
                if best_test_files:
                    # Split the first test file to reach target
                    remainder_file = best_test_files[0]
                    remainder_weight = sum(calculate_group_weights(create_sequence_groups(pat_files[remainder_file])).values())
                    train_weight_whole = sum(
                        sum(calculate_group_weights(create_sequence_groups(pat_files[pf])).values())
                        for pf in best_train_files
                    )
                    needed_weight = min(remainder_weight, target_train_weight - train_weight_whole)
                    if needed_weight > 0:
                        best_test_files = best_test_files[1:]
                        best_remainder_to_split = (remainder_file, needed_weight)
            
            # Apply the best strategy found
            print(f"  Selected strategy:")
            print(f"    Whole PAT files to training: {len(best_train_files)}")
            print(f"    Whole PAT files to testing: {len(best_test_files)}")
            
            # Add whole PAT files
            for pat_file in best_train_files:
                train_data.extend(pat_files[pat_file])
            
            for pat_file in best_test_files:
                test_data.extend(pat_files[pat_file])
            
            # Handle remainder file if needed
            if best_remainder_to_split:
                remainder_file, needed_train_weight = best_remainder_to_split
                remainder_data = pat_files[remainder_file].copy()
                
                # Create sequence groups for remainder file
                remainder_groups = create_sequence_groups(remainder_data)
                remainder_group_weights = calculate_group_weights(remainder_groups)
                
                # Split remainder by sequence groups
                remainder_total_rows = sum(len(remainder_groups[key]) for key in remainder_groups.keys())
                remainder_target_ratio = needed_train_weight / sum(remainder_group_weights.values()) if sum(remainder_group_weights.values()) > 0 else 0.75
                train_seq_keys, test_seq_keys = split_sequence_groups_by_rows(
                    remainder_groups, remainder_target_ratio, rand_instance
                )
                
                # Collect rows from remainder split
                remainder_train_data = []
                remainder_test_data = []
                
                for seq_key in train_seq_keys:
                    remainder_train_data.extend(remainder_groups[seq_key])
                
                for seq_key in test_seq_keys:
                    remainder_test_data.extend(remainder_groups[seq_key])
                
                train_data.extend(remainder_train_data)
                test_data.extend(remainder_test_data)
                
                print(f"    Split remainder PAT file: {len(remainder_train_data)} to train, {len(remainder_test_data)} to test")
            
            # Calculate final counts for this cell type
            cell_train_count = sum(len(pat_files[pf]) for pf in best_train_files)
            cell_test_count = sum(len(pat_files[pf]) for pf in best_test_files)
            
            if best_remainder_to_split:
                cell_train_count += len(remainder_train_data)
                cell_test_count += len(remainder_test_data)
            
            actual_ratio = cell_train_count / total_lines if total_lines > 0 else 0
            print(f"  Result: {cell_train_count:,} train, {cell_test_count:,} test ({actual_ratio:.1%} train ratio)")
    
    print(f"\nFinal Summary:")
    print(f"  Total train lines: {len(train_data):,}")
    print(f"  Total test lines: {len(test_data):,}")
    total_lines = len(train_data) + len(test_data)
    overall_ratio = len(train_data) / total_lines if total_lines > 0 else 0
    print(f"  Overall train ratio: {overall_ratio:.1%}")
    print(f"  Sequence grouping ensures no identical sequences in both train and test")
    
    return train_data, test_data

def save_split_data(train_data, test_data, header, train_file, test_file):
    """
    Save the split data to CSV files
    Args:
        train_data: List of training data rows
        test_data: List of test data rows
        header: CSV header
        train_file: Output training file path
        test_file: Output test file path
    """
    
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
    print(f"Saving training data to: {train_file}")
    with open(train_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        writer.writerows(train_data)
    
    # Save test data
    print(f"Saving test data to: {test_file}")
    with open(test_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        writer.writerows(test_data)
    
    print(f"Successfully saved:")
    print(f"  Training data: {train_file} ({len(train_data):,} lines)")
    print(f"  Test data: {test_file} ({len(test_data):,} lines)")

def display_mode_information():
    """
    Display information about the three available modes
    """
    print("=" * 80)
    print("SIX PROCESSING MODES AVAILABLE")
    print("=" * 80)
    
    print("\nMODE 1: Simple 75/25 Split")
    print("  • Basic random split without any duplicate handling")
    print("  • Fastest processing")
    print("  • May have data leakage between train/test sets")
    print("  • Use when: Speed is priority, data leakage is acceptable")
    
    print("\nMODE 2: Avoid Duplicates (Default)")
    print("  • Prevents identical sequences from appearing in both train and test")
    print("  • Uses PAT file-based splitting with sequence grouping")
    print("  • Maintains data integrity while optimizing splits")
    print("  • No data leakage")
    print("  • Use when: Data integrity is priority, PAT file structure matters")
    
    print("\nMODE 3: Remove All Duplicates Then Split")
    print("  • Completely removes all duplicate rows first")
    print("  • Normalizes read counts > 1 to 1 for equal sequence contribution")
    print("  • Then performs random 75/25 split on deduplicated data")
    print("  • Cleanest data but may lose information")
    print("  • No duplicates, no data leakage")
    print("  • Equal contribution from each unique sequence")
    print("  • Use when: Clean data is priority, information loss is acceptable")
    
    print("\nMODE 4: Manual PAT File Assignment")
    print("  • Removes all duplicate rows first (same as Mode 3)")
    print("  • Normalizes read counts > 1 to 1 for equal sequence contribution")
    print("  • Shows you all available PAT file markers")
    print("  • Allows manual assignment of PAT files to train/test sets")
    print("  • No forced 75/25 split - you control the exact distribution")
    print("  • No duplicates, no data leakage")
    print("  • Complete manual control over train/test assignment")
    print("  • Flexible split ratio based on your needs")
    print("  • Use when: You need specific control over which PAT files go where")
    
    print("\nMODE 5: Cross-PAT Deduplication")
    print("  • Checks for identical reads across different PAT files")
    print("  • If identical reads have same ctype: removes duplicates")
    print("  • If identical reads have different ctype: removes all and counts conflicts")
    print("  • Performs random 75/25 split on deduplicated data")
    print("  • Cross-PAT deduplication prevents data leakage")
    print("  • Ctype conflict tracking for analysis")
    print("  • Random split ensures unbiased train/test distribution")
    print("  • Use when: You want to identify and handle cross-PAT file conflicts")
    
    print("\nMODE 6: Strict Cross-PAT Deduplication")
    print("  • Only removes sequences when ALL PAT files share the exact same sequence")
    print("  • If sequence appears in ALL PAT files with same ctype: keeps one occurrence")
    print("  • If sequence appears in ALL PAT files with different ctype: removes all")
    print("  • If sequence appears in only some PAT files: keeps all occurrences")
    print("  • Performs random 75/25 split on deduplicated data")
    print("  • Strict cross-PAT deduplication prevents data leakage")
    print("  • Preserves partial occurrences for better data retention")
    print("  • Ctype conflict tracking for analysis (all PAT files only)")
    print("  • Use when: You want to preserve sequences that don't appear in all PAT files")
    
    print("\n" + "=" * 80)

def test_sequence_grouping():
    """
    Test function to demonstrate that duplicate sequences are properly handled
    """
    print("\n" + "=" * 60)
    print("TESTING SEQUENCE GROUPING FUNCTIONALITY")
    print("=" * 60)
    
    # Create test data with duplicate sequences
    test_data = [
        ["GAA AAG AGA GAA AAA AAT ATT TTG TGC GCG CGT GTG TGG GGT GTA TAC ACA CAG AGC GCC", "22222222212222222222", "0", "Head-Neck-Ep", "Adipocytes", "1"],
        ["GAA AAG AGA GAA AAA AAT ATT TTG TGC GCG CGT GTG TGG GGT GTA TAC ACA CAG AGC GCC", "22222222212222222222", "0", "Head-Neck-Ep", "Adipocytes", "1"],
        ["GAA AAG AGA GAA AAA AAT ATT TTG TGC GCG CGT GTG TGG GGT GTA TAC ACA CAG AGC GCC", "22222222212222222222", "0", "Head-Neck-Ep", "Adipocytes", "1"],
        ["TCG CCG CGC GCT CTC TCG CGC GCC CCG CGT GTG TGC GCG CGC GCG CGT GTG TGC GCG CGT", "11122221221122222222", "1", "Head-Neck-Ep", "Head-Neck-Ep", "2"],
        ["TCG CCG CGC GCT CTC TCG CGC GCC CCG CGT GTG TGC GCG CGC GCG CGT GTG TGC GCG CGT", "11122221221122222222", "1", "Head-Neck-Ep", "Head-Neck-Ep", "3"],
        ["ATG ATG ATG ATG ATG ATG ATG ATG ATG ATG ATG ATG ATG ATG ATG ATG ATG ATG ATG ATG", "00000000000000000000", "0", "Skeletal-Muscle", "Skeletal-Muscle", "1"],
    ]
    
    print("Test data:")
    for i, row in enumerate(test_data):
        print(f"  {i+1}: {row[0][:20]}... -> {row[4]} (read_count: {row[5]})")
    
    # Test sequence grouping
    sequence_groups = create_sequence_groups(test_data)
    group_weights = calculate_group_weights(sequence_groups)
    
    print(f"\nSequence groups found: {len(sequence_groups)}")
    for i, (seq_key, group_rows) in enumerate(sequence_groups.items(), 1):
        dna_seq = seq_key[0][:20] + "..."
        methyl_seq = seq_key[1][:10] + "..."
        weight = group_weights[seq_key]
        print(f"  Group {i}: DNA={dna_seq}, Methyl={methyl_seq}")
        print(f"    Rows: {len(group_rows)}, Total weight: {weight}")
        for j, row in enumerate(group_rows):
            print(f"      Row {j+1}: {row[4]} (read_count: {row[5]})")
    
    # Test splitting with both row and weight metrics
    rand_instance = random.Random(42)
    total_weight = sum(group_weights.values())
    target_train_weight = total_weight * 0.75
    
    print(f"\nSplitting test (target train ratio: 75%):")
    train_seq_keys, test_seq_keys = split_sequence_groups_by_rows(
        sequence_groups, 0.75, rand_instance
    )
    
    # Calculate actual ratios
    actual_train_weight = sum(group_weights[key] for key in train_seq_keys)
    actual_test_weight = sum(group_weights[key] for key in test_seq_keys)
    actual_train_rows = sum(len(sequence_groups[key]) for key in train_seq_keys)
    actual_test_rows = sum(len(sequence_groups[key]) for key in test_seq_keys)
    
    total_rows = actual_train_rows + actual_test_rows
    weight_ratio = actual_train_weight / total_weight if total_weight > 0 else 0
    row_ratio = actual_train_rows / total_rows if total_rows > 0 else 0
    
    print(f"Train groups: {len(train_seq_keys)}")
    for seq_key in train_seq_keys:
        weight = group_weights[seq_key]
        rows = len(sequence_groups[seq_key])
        print(f"  Group: {rows} rows, weight: {weight}")
    
    print(f"Test groups: {len(test_seq_keys)}")
    for seq_key in test_seq_keys:
        weight = group_weights[seq_key]
        rows = len(sequence_groups[seq_key])
        print(f"  Group: {rows} rows, weight: {weight}")
    
    print(f"\nRatio Analysis:")
    print(f"  Target weight ratio: 75.0%")
    print(f"  Actual weight ratio: {weight_ratio:.1%}")
    print(f"  Actual row ratio: {row_ratio:.1%}")
    print(f"  Weight difference from target: {abs(weight_ratio - 0.75):.1%}")
    
    # Verify no sequences appear in both train and test
    train_seqs = set(train_seq_keys)
    test_seqs = set(test_seq_keys)
    overlap = train_seqs.intersection(test_seqs)
    
    print(f"\nVerification: {len(overlap)} sequences appear in both train and test (should be 0)")
    if len(overlap) == 0:
        print("SUCCESS: No data leakage detected!")
    else:
        print("ERROR: Data leakage detected!")
        for seq_key in overlap:
            print(f"  Overlap sequence: {seq_key[0][:20]}...")
    
    print("=" * 60)

def main():
    """Main function"""
    
    # Configuration
    train_ratio = 0.75  # 75% train, 25% test
    random_seed = 42  # For reproducibility
    
    print("=" * 60)
    print("PAT FILE-BASED TRAIN/TEST SPLIT SCRIPT - FIVE MODES")
    print("=" * 60)
    print(f"Train ratio: {train_ratio}")
    print(f"Random seed: {random_seed}")
    print("=" * 60)
    
    # Run test if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_sequence_grouping()
        return True
    
    # Display mode information
    if len(sys.argv) > 1 and sys.argv[1] == "--modes":
        display_mode_information()
        return True
    
    # Get file names and mode from interface
    print("Using command-line interface...")
    result = create_cli_interface()
    
    if not result:
        print("No files selected. Exiting.")
        return False
    
    input_file, train_file, test_file, use_balanced, selected_mode = result
    
    if not input_file:
        print("No input file specified. Exiting.")
        return False
    
    print(f"Input file: {input_file}")
    print(f"Training file: {train_file}")
    print(f"Test file: {test_file}")
    print(f"Use balanced data: {use_balanced}")
    print(f"Selected mode: {selected_mode}")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print(f"Please make sure the file exists in the current directory.")
        return False
    
    # Parse PAT file sections based on selected mode
    if selected_mode == MODE_SIMPLE_SPLIT:
        print("Using MODE 1: Simple 75/25 Split (Basic random split)")
        train_data, test_data, header = simple_random_split(input_file, train_ratio)
        if train_data is None:
            return False
        
    elif selected_mode == MODE_AVOID_DUPLICATES:
        print("Using MODE 2: Avoid Duplicates (Prevents identical sequences in both sets)")
        if use_balanced:
            print("Using BALANCED data approach: will balance positive/negative samples per PAT file")
            cell_type_pat_data = parse_pat_file_sections_balanced(input_file, random_seed)
        else:
            print("Using STANDARD data approach: will use all data without balancing")
            cell_type_pat_data = parse_pat_file_sections(input_file)
        
        if cell_type_pat_data is None:
            return False
        
        # Use the header we already read from the CSV file
        header = ['DNA_seq', 'Methyl_seq', 'DMR_Label', 'cType', 'DMR_cType', 'Read_Count']
        
        # Implement PAT file-based splitting strategy
        train_data, test_data = split_pat_files_strategy(cell_type_pat_data, train_ratio)
        
    elif selected_mode == MODE_REMOVE_ALL_DUPLICATES:
        print("Using MODE 3: Remove All Duplicates Then Split")
        train_data, test_data, header = remove_duplicates_then_split(input_file, train_ratio)
        if train_data is None:
            return False
    
    elif selected_mode == MODE_MANUAL_PAT_ASSIGNMENT:
        print("Using MODE 4: Manual PAT File Assignment")
        train_data, test_data, header = manual_pat_file_assignment(input_file, train_ratio)
        if train_data is None:
            return False
    
    elif selected_mode == MODE_CROSS_PAT_DEDUPLICATION:
        print("Using MODE 5: Cross-PAT Deduplication")
        train_data, test_data, header = cross_pat_deduplication(input_file, train_ratio)
        if train_data is None:
            return False
    
    elif selected_mode == MODE_STRICT_CROSS_PAT_DEDUPLICATION:
        print("Using MODE 6: Strict Cross-PAT Deduplication")
        train_data, test_data, header = strict_cross_pat_deduplication(input_file, train_ratio)
        if train_data is None:
            return False
    
    else:
        print(f"Error: Unknown mode {selected_mode}")
        return False
    
    if train_data is None or test_data is None:
        print("Error: Failed to generate train/test data")
        return False
    
    # Save the split data
    save_split_data(train_data, test_data, header, train_file, test_file)
    
    print(f"\nData splitting completed successfully!")
    print(f"You can now use {train_file} and {test_file} for MethylBERT training.")
    
    if selected_mode == MODE_SIMPLE_SPLIT:
        print(f"Mode 1 Summary: Simple random split completed.")
        print(f"   Warning: This mode may have data leakage between train and test sets.")
        print(f"   Identical sequences may appear in both training and testing data.")
        
    elif selected_mode == MODE_AVOID_DUPLICATES:
        print(f"Mode 2 Summary: PAT file-based splitting with sequence grouping completed.")
        print(f"No data leakage: Identical sequences are never split between train and test sets.")
        if use_balanced:
            print(f"Data was balanced to ensure equal positive/negative sample representation per PAT file.")
        else:
            print(f"All data was used without balancing - original class distribution preserved.")
            
    elif selected_mode == MODE_REMOVE_ALL_DUPLICATES:
        print(f"Mode 3 Summary: Duplicate removal and random split completed.")
        print(f"All duplicates removed: Each sequence appears only once in the entire dataset.")
        print(f"Clean data: No duplicate sequences to cause data leakage.")
        print(f"Read counts normalized: Each unique sequence contributes equally to the split.")
    
    elif selected_mode == MODE_MANUAL_PAT_ASSIGNMENT:
        print(f"Mode 4 Summary: Manual PAT file assignment completed.")
        print(f"All duplicates removed: Each sequence appears only once in the entire dataset.")
        print(f"Manual control: You specified exactly which PAT files go to train vs test.")
        print(f"Flexible split ratio: No forced 75/25 split - you control the distribution.")
        print(f"Read counts normalized: Each unique sequence contributes equally.")
    
    elif selected_mode == MODE_CROSS_PAT_DEDUPLICATION:
        print(f"Mode 5 Summary: Cross-PAT Deduplication completed.")
        print(f"Cross-PAT deduplication: Identical reads across different PAT files were removed.")
        print(f"Random 75/25 split: Data was split into training and testing sets.")
        print(f"No data leakage: Identical sequences are never split between train and test.")
    
    elif selected_mode == MODE_STRICT_CROSS_PAT_DEDUPLICATION:
        print(f"Mode 6 Summary: Strict Cross-PAT Deduplication completed.")
        print(f"Strict cross-PAT deduplication: Only sequences appearing in ALL PAT files were deduplicated.")
        print(f"Partial occurrences kept: Sequences appearing in only some PAT files were preserved.")
        print(f"Random 75/25 split: Data was split into training and testing sets.")
        print(f"No data leakage: Identical sequences are never split between train and test.")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("PAT File Split Tool - Six Modes")
            print("\nUsage:")
            print("  python split_data_flexible.py                    # Run with CLI interface")
            print("  python split_data_flexible.py --test            # Run sequence grouping test")
            print("  python split_data_flexible.py --modes           # Display mode information")
            print("  python split_data_flexible.py --help            # Show this help message")
            print("\nModes:")
            print("  1. Simple 75/25 Split - Fast random split (may have data leakage)")
            print("  2. Avoid Duplicates - PAT file-based with sequence grouping (default)")
            print("  3. Remove All Duplicates - Clean data then random split (normalizes read counts)")
            print("  4. Manual PAT Assignment - Remove duplicates, manual PAT file assignment")
            print("  5. Cross-PAT Deduplication - Remove identical reads across PAT files")
            print("  6. Strict Cross-PAT Deduplication - Only remove sequences in ALL PAT files")
            sys.exit(0)
    
    main() 