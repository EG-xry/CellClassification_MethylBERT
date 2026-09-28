#!/usr/bin/env python3
"""
Optimized MethylBERT training data pipeline for cloud environments and multi-cell type processing
- Memory-efficient streaming approach
- Checkpoint/resume functionality  
- Batch processing for multiple cell types
- Resource monitoring
- Better error handling
- CpG coordinate mapping using CpG.bed.gz for accurate hg38 coordinates
- PARALLEL PROCESSING: Process multiple cell types simultaneously with multiprocessing (up to 48 cores)
- ALL REGIONS MODE: Prfocess ALL DMR regions (49,600+) instead of just overlapping subsets
- Optimized for high-performance servers with 20+ cores
"""

import gzip
import csv
import sys
import argparse
import time
import json
import os
import psutil
import pickle
import multiprocessing as mp
from multiprocessing import Pool, Manager
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pysam
import bisect
import re

class ProgressTracker:
    """Track progress and enable checkpoint/resume functionality"""
    
    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file
        self.processed_regions = set()
        self.total_records = 0
        self.start_time = time.time()
        
    def load_checkpoint(self):
        """Load previous progress if checkpoint exists"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self.processed_regions = set(data.get('processed_regions', []))
                    self.total_records = data.get('total_records', 0)
                    print(f"Resuming from checkpoint: {len(self.processed_regions)} regions already processed", file=sys.stderr)
                    return True
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}", file=sys.stderr)
        return False
    
    def save_checkpoint(self):
        """Save current progress"""
        try:
            data = {
                'processed_regions': list(self.processed_regions),
                'total_records': self.total_records,
                'timestamp': time.time()
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}", file=sys.stderr)
    
    def is_processed(self, region_id: str) -> bool:
        """Check if region was already processed"""
        return region_id in self.processed_regions
    
    def mark_processed(self, region_id: str, records_count: int):
        """Mark region as processed"""
        self.processed_regions.add(region_id)
        self.total_records += records_count
        
    def cleanup(self):
        """Remove checkpoint file when complete"""
        try:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
        except Exception as e:
            print(f"Warning: Could not remove checkpoint file: {e}", file=sys.stderr)

class ResourceMonitor:
    """Monitor system resources during processing"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        
    def log_usage(self, context: str = ""):
        """Log current resource usage"""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        self.peak_memory = max(self.peak_memory, memory_mb)
        
        print(f"[Resources{' - ' + context if context else ''}] Memory: {memory_mb:.1f}MB (Peak: {self.peak_memory:.1f}MB), CPU: {cpu_percent:.1f}%", file=sys.stderr)

class CpGCoordinateMapper:
    """Efficient CpG coordinate mapping using CpG.bed.gz file"""
    
    def __init__(self, cpg_bed_file: str):
        self.cpg_bed_file = cpg_bed_file
        self.cpg_map = {}  # {chr: [(genomic_pos, cpg_index), ...]}
        self.load_cpg_mapping()
    
    def load_cpg_mapping(self):
        """Load CpG.bed.gz file into memory for fast coordinate conversion"""
        print(f"Loading CpG coordinate mapping from {self.cpg_bed_file}...", file=sys.stderr)
        
        if not os.path.exists(self.cpg_bed_file):
            raise FileNotFoundError(f"CpG bed file not found: {self.cpg_bed_file}")
        
        start_time = time.time()
        total_entries = 0
        
        with gzip.open(self.cpg_bed_file, 'rt') as f:
            with tqdm(desc="Loading CpG mapping", unit="entries", file=sys.stderr) as pbar:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) < 3:
                        continue
                    
                    chr_name = parts[0]
                    genomic_pos = int(parts[1])
                    cpg_index = int(parts[2])
                    
                    if chr_name not in self.cpg_map:
                        self.cpg_map[chr_name] = []
                    
                    self.cpg_map[chr_name].append((genomic_pos, cpg_index))
                    total_entries += 1
                    pbar.update(1)
                    
                    # Progress update every 1M entries
                    if total_entries % 1000000 == 0:
                        elapsed = time.time() - start_time
                        entries_per_sec = total_entries / elapsed
                        print(f"  Loaded {total_entries:,} CpG entries ({entries_per_sec:,.0f} entries/sec)", file=sys.stderr)
        
        # Sort each chromosome's entries by genomic position for binary search
        print("Sorting CpG entries by genomic position for fast lookup...", file=sys.stderr)
        for chr_name in self.cpg_map:
            self.cpg_map[chr_name].sort(key=lambda x: x[0])
        
        elapsed = time.time() - start_time
        print(f"CpG mapping loaded in {elapsed:.1f}s", file=sys.stderr)
        print(f"Total chromosomes: {len(self.cpg_map)}", file=sys.stderr)
        print(f"Total CpG entries: {total_entries:,}", file=sys.stderr)
        
        # Show chromosome breakdown
        for chr_name in sorted(self.cpg_map.keys())[:10]:  # Show first 10 chromosomes
            print(f"  {chr_name}: {len(self.cpg_map[chr_name]):,} CpG sites", file=sys.stderr)
        if len(self.cpg_map) > 10:
            print(f"  ... and {len(self.cpg_map) - 10} more chromosomes", file=sys.stderr)
    
    def genomic_to_cpg_range(self, chr_name: str, genomic_start: int, genomic_end: int) -> Optional[Tuple[int, int]]:
        """Convert genomic coordinate range to CpG coordinate range
        
        Args:
            chr_name: Chromosome name
            genomic_start: Genomic start position
            genomic_end: Genomic end position
            
        Returns:
            Tuple of (cpg_start, cpg_end) or None if no CpGs found
        """
        if chr_name not in self.cpg_map:
            print(f"Warning: No CpG data for chromosome {chr_name}", file=sys.stderr)
            return None
        
        chr_cpgs = self.cpg_map[chr_name]
        if not chr_cpgs:
            return None
        
        # Find CpGs that overlap with the genomic range
        cpg_indices = []
        
        # Use binary search to find the range of CpGs within genomic coordinates
        # Find first CpG >= genomic_start
        left_idx = bisect.bisect_left(chr_cpgs, (genomic_start, 0))
        # Find last CpG <= genomic_end
        right_idx = bisect.bisect_right(chr_cpgs, (genomic_end, float('inf')))
        
        if left_idx >= len(chr_cpgs) or right_idx == 0:
            return None
        
        # Collect CpG indices in the range
        for i in range(left_idx, min(right_idx, len(chr_cpgs))):
            genomic_pos, cpg_idx = chr_cpgs[i]
            if genomic_start <= genomic_pos <= genomic_end:
                cpg_indices.append(cpg_idx)
        
        if not cpg_indices:
            return None
        
        cpg_start = min(cpg_indices)
        cpg_end = max(cpg_indices)
        
        return cpg_start, cpg_end
    
    def get_cpg_count_in_range(self, chr_name: str, genomic_start: int, genomic_end: int) -> int:
        """Count CpGs in a genomic range"""
        cpg_range = self.genomic_to_cpg_range(chr_name, genomic_start, genomic_end)
        if cpg_range is None:
            return 0
        cpg_start, cpg_end = cpg_range
        return cpg_end - cpg_start + 1

def build_pat_index_with_cpg_coords(pat_file):
    """
    Build an optimized index of PAT data by chromosome using CpG coordinates
    Sorts fragments by CpG position for fast range queries
    Returns dictionary: {chromosome: [(cpg_start, cpg_end, pattern, count), ...]}
    """
    print("Building optimized PAT chromosome index with CpG coordinates...", file=sys.stderr)
    pat_index = defaultdict(list)
    
    line_count = 0
    start_time = time.time()
    
    with gzip.open(pat_file, 'rt') as f:
        with tqdm(desc="Indexing PAT file", unit="lines", file=sys.stderr) as pbar:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                line_count += 1
                pbar.update(1)
                
                # Progress update every 10M lines
                if line_count % 10000000 == 0:
                    elapsed = time.time() - start_time
                    lines_per_sec = line_count / elapsed
                    print(f"  Indexed {line_count:,} lines ({lines_per_sec:,.0f} lines/sec)", file=sys.stderr)
                
                parts = line.split('\t')
                if len(parts) < 4:
                    continue
                
                pat_chr = parts[0]
                cpg_start = int(parts[1])  # CpG coordinate
                pattern = parts[2]
                count = int(parts[3])
                
                # Only store fragments with actual CpG data
                num_cpgs = len([c for c in pattern if c in 'CT'])
                if num_cpgs > 0:
                    cpg_end = cpg_start + num_cpgs - 1
                    pat_index[pat_chr].append((cpg_start, cpg_end, pattern, count))
    
    # Sort each chromosome's fragments by CpG start position for fast range queries
    print("Sorting PAT fragments by CpG coordinates for optimization...", file=sys.stderr)
    for chr_name in pat_index:
        pat_index[chr_name].sort(key=lambda x: x[0])  # Sort by cpg_start
    
    elapsed = time.time() - start_time
    total_fragments = sum(len(fragments) for fragments in pat_index.values())
    
    print(f"Optimized PAT index built in {elapsed:.1f}s", file=sys.stderr)
    print(f"Total chromosomes: {len(pat_index)}", file=sys.stderr)
    print(f"Total fragments indexed: {total_fragments:,}", file=sys.stderr)
    
    # Show chromosome breakdown
    for chr_name in sorted(pat_index.keys()):
        print(f"  {chr_name}: {len(pat_index[chr_name]):,} fragments", file=sys.stderr)
    
    return dict(pat_index)

def find_overlapping_pat_for_dmr_fast(pat_index, chr_name, cpg_start, cpg_end):
    """
    Efficiently find PAT fragments that overlap a DMR region using binary search.

    The chromosome fragments list is pre-sorted by `cpg_start`, so we first use
    bisect to jump close to the query window, then scan left/right only as far
    as necessary.  This reduces search complexity from O(N) per region to
    O(log N + K) where K is the number of overlapping fragments.
    """
    chr_fragments = pat_index.get(chr_name, [])
    if not chr_fragments:
        return []

    # Locate the first fragment whose start position is >= cpg_start
    left = bisect.bisect_left(chr_fragments, (cpg_start, -1, '', 0))

    overlapping = []

    # Scan leftwards to catch fragments that start before `cpg_start` but still overlap
    i = left - 1
    while i >= 0:
        pat_cpg_start, pat_cpg_end, pattern, count = chr_fragments[i]
        if pat_cpg_end < cpg_start:
            break  # No further overlap possible on the left
        overlapping.append((pat_cpg_start, pat_cpg_end, pattern, count))
        i -= 1

    # Scan rightwards to collect fragments starting inside the window
    for j in range(left, len(chr_fragments)):
        pat_cpg_start, pat_cpg_end, pattern, count = chr_fragments[j]
        if pat_cpg_start > cpg_end:
            break  # Beyond the region – stop scanning
        if pat_cpg_end >= cpg_start:
            overlapping.append((pat_cpg_start, pat_cpg_end, pattern, count))

    if not overlapping:
        print(f"  Warning: No PAT fragments found for {chr_name} CpG range {cpg_start}-{cpg_end}", file=sys.stderr)
    return overlapping

def find_overlapping_pat_for_dmr(pat_index, chr_name, cpg_start, cpg_end):
    """
    Find overlapping PAT fragments using CpG coordinate ranges
    Returns list of (cpg_start, cpg_end, pattern, count) for overlapping fragments
    """
    return find_overlapping_pat_for_dmr_fast(pat_index, chr_name, cpg_start, cpg_end)

def find_overlapping_dmr_regions(target_regions, all_regions, process_all=True):
    """
    Depending on the `process_all` flag, either:

    1. Return ALL DMR regions (original behaviour, potentially very slow), or
    2. Return ONLY the regions that belong to the target cell type
       ("overlap-only / legacy" mode) which dramatically reduces the total
       number of regions processed.

    Returns:
        List of tuples (chr, genomic_start, genomic_end, cpg_start, cpg_end,
        dmr_label, cell_type, target_cell_types)
    """
    if process_all:
        # Original behaviour – keep as fallback
        print(f"Processing ALL DMR regions (no overlap filtering)...", file=sys.stderr)
        print(f"  Target regions: {len(target_regions)}", file=sys.stderr)
        print(f"  Total regions to process: {len(all_regions)}", file=sys.stderr)

        target_cell_type = target_regions[0][6] if target_regions else "Unknown"
        all_processing_regions = []
        for chr_name, genomic_start, genomic_end, cpg_start, cpg_end, dmr_label, cell_type in all_regions:
            all_processing_regions.append((chr_name, genomic_start, genomic_end, cpg_start, cpg_end,
                                           dmr_label, cell_type, [target_cell_type]))
        print(f"  Will process ALL {len(all_processing_regions)} regions", file=sys.stderr)
        return all_processing_regions

    # FAST PATH – overlap-only / legacy mode
    print(f"Processing ONLY target DMR regions (overlap-only mode)...", file=sys.stderr)
    print(f"  Regions to process: {len(target_regions)}", file=sys.stderr)

    target_cell_type = target_regions[0][6] if target_regions else "Unknown"
    filtered_regions = []
    for chr_name, genomic_start, genomic_end, cpg_start, cpg_end, dmr_label, cell_type in target_regions:
        filtered_regions.append((chr_name, genomic_start, genomic_end, cpg_start, cpg_end,
                                 dmr_label, cell_type, [target_cell_type]))
    return filtered_regions

class CellTypeConfig:
    """Configuration for different cell types"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.configs = {}
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Load cell type configurations from JSON file"""
        try:
            with open(config_file, 'r') as f:
                self.configs = json.load(f)
                print(f"Loaded configurations for {len(self.configs)} cell types", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}", file=sys.stderr)
    
    def add_cell_type(self, cell_type: str, csv_file: str, pat_file: str, 
                     fasta_file: str, output_file: str):
        """Add a cell type configuration"""
        self.configs[cell_type] = {
            'csv_file': csv_file,
            'pat_file': pat_file, 
            'fasta_file': fasta_file,
            'output_file': output_file
        }
    
    def get_cell_types(self) -> List[str]:
        """Get list of configured cell types"""
        return list(self.configs.keys())
    
    def get_config(self, cell_type: str) -> Dict:
        """Get configuration for specific cell type"""
        return self.configs.get(cell_type, {})
    
    def save_config(self, config_file: str):
        """Save configurations to JSON file"""
        try:
            with open(config_file, 'w') as f:
                json.dump(self.configs, f, indent=2)
                print(f"Saved configurations to {config_file}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not save config file: {e}", file=sys.stderr)

def normalize_chromosome_name(chr_name):
    """
    Normalize chromosome name by extracting just the main chromosome part.
    Examples:
    - chr7_KI270803v1_alt -> chr7
    - chr15_KI270850v1_alt -> chr15
    - chr14_GL000009v2_random -> chr14
    - chr7 -> chr7 (unchanged)
    """
    # Extract the main chromosome part (chr + number)
    match = re.match(r'(chr[0-9XYM]+)', chr_name)
    if match:
        return match.group(1)
    return chr_name

def load_dmr_regions_from_csv(csv_file, target_cell_type=None, cpg_mapper=None, filter_sex_chroms=True, load_all_regions=False):
    """Load DMR regions from CSV file and convert genomic coordinates to CpG coordinates
    
    Args:
        csv_file: Path to CSV file
        target_cell_type: If specified, only load regions for this cell type (unless load_all_regions=True)
                         If None, load all regions (for single cell type mode)
        cpg_mapper: CpGCoordinateMapper instance for converting genomic to CpG coordinates
        filter_sex_chroms: If True, skip regions on chrX/chrY
        load_all_regions: If True, load all regions regardless of target_cell_type (for overlap analysis)
    
    Returns:
        List of (chr, genomic_start, genomic_end, cpg_start, cpg_end, dmr_label, cell_type) tuples
    """
    print(f"Loading DMR regions from CSV{' for ' + target_cell_type if target_cell_type and not load_all_regions else ' (ALL regions)' if load_all_regions else ''}...", file=sys.stderr)
    dmr_regions = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        # Check if file has cell_type or Type column
        has_cell_type_column = reader.fieldnames and ('cell_type' in reader.fieldnames or 'Type' in reader.fieldnames)
        
        # Determine the actual column name to use
        cell_type_column = None
        if reader.fieldnames:
            if 'cell_type' in reader.fieldnames:
                cell_type_column = 'cell_type'
            elif 'Type' in reader.fieldnames:
                cell_type_column = 'Type'
        
        if has_cell_type_column and target_cell_type and not load_all_regions:
            print(f"  Filtering for cell type: {target_cell_type} (using column: {cell_type_column})", file=sys.stderr)
        elif has_cell_type_column and not target_cell_type and not load_all_regions:
            print(f"  Warning: CSV has {cell_type_column} column but no target specified - loading all regions", file=sys.stderr)
        elif has_cell_type_column and load_all_regions:
            print(f"  Loading ALL regions with cell type information (using column: {cell_type_column})", file=sys.stderr)
        
        # Check if CSV has startCpG and endCpG columns (legacy format)
        has_cpg_columns = reader.fieldnames and 'startCpG' in reader.fieldnames and 'endCpG' in reader.fieldnames
        
        if has_cpg_columns and cpg_mapper:
            print("  CSV has CpG columns but CpG mapper provided - will use mapper for accurate conversion", file=sys.stderr)
        elif has_cpg_columns and not cpg_mapper:
            print("  Using CpG coordinates from CSV (may be inaccurate for hg38)", file=sys.stderr)
        elif not has_cpg_columns and not cpg_mapper:
            raise ValueError("CSV file must have startCpG/endCpG columns or CpG mapper must be provided")
        
        skipped_regions = 0
        for i, row in enumerate(reader):
            # Get cell type from row
            row_cell_type = None
            if has_cell_type_column and cell_type_column:
                row_cell_type = row[cell_type_column].strip()
            
            # Skip if filtering by cell type and this row doesn't match (unless loading all regions)
            if has_cell_type_column and target_cell_type and cell_type_column and not load_all_regions:
                if row_cell_type != target_cell_type:
                    continue
            
            chr_name = normalize_chromosome_name(row['chr'])
            # Filter out sex chromosomes if requested
            if filter_sex_chroms and chr_name in ('chrX', 'chrY'):
                continue
            genomic_start = int(row['start'])
            genomic_end = int(row['end'])
            
            # Convert genomic coordinates to CpG coordinates
            if cpg_mapper:
                cpg_range = cpg_mapper.genomic_to_cpg_range(chr_name, genomic_start, genomic_end)
                if cpg_range is None:
                    print(f"  Warning: No CpGs found for region {chr_name}:{genomic_start}-{genomic_end}, skipping", file=sys.stderr)
                    skipped_regions += 1
                    continue
                cpg_start, cpg_end = cpg_range
            else:
                # Use CpG coordinates from CSV (legacy)
                cpg_start = int(row['startCpG'])
                cpg_end = int(row['endCpG'])
            
            # Include cell type information in the returned data
            dmr_regions.append((chr_name, genomic_start, genomic_end, cpg_start, cpg_end, i, row_cell_type))
        
        if skipped_regions > 0:
            print(f"  Skipped {skipped_regions} regions with no CpG data", file=sys.stderr)
    
    print(f"Loaded {len(dmr_regions)} DMR regions with CpG coordinates", file=sys.stderr)
    return dmr_regions

def get_cell_types_from_csv(csv_file):
    """Extract all unique cell types from a collective CSV file
    
    Args:
        csv_file: Path to CSV file with cell_type column
        
    Returns:
        Set of unique cell types found in the file
    """
    print(f"Discovering cell types from {csv_file}...", file=sys.stderr)
    cell_types = set()
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        if not reader.fieldnames or 'cell_type' not in reader.fieldnames:
            raise ValueError(f"CSV file {csv_file} must have a 'cell_type' column for batch processing")
        
        for row in reader:
            cell_type = row['cell_type'].strip()
            if cell_type:
                cell_types.add(cell_type)
    
    print(f"Found {len(cell_types)} unique cell types: {sorted(cell_types)}", file=sys.stderr)
    return cell_types

def impute_missing_methylation(pattern):
    """Replace '.' (unknown) in PAT pattern with nearest known methylation value"""
    if '.' not in pattern:
        return pattern
    
    chars = list(pattern)
    
    for i, char in enumerate(chars):
        if char == '.':
            left_dist = float('inf')
            left_val = None
            right_dist = float('inf') 
            right_val = None
            
            # Search left
            for j in range(i-1, -1, -1):
                if chars[j] in 'CT':
                    left_dist = i - j
                    left_val = chars[j]
                    break
            
            # Search right
            for j in range(i+1, len(chars)):
                if chars[j] in 'CT':
                    right_dist = j - i
                    right_val = chars[j]
                    break
            
            # Choose closest (prefer left if tie)
            if left_dist <= right_dist and left_val is not None:
                chars[i] = left_val
            elif right_val is not None:
                chars[i] = right_val
    
    return ''.join(chars)

def create_3mer_sequence(sequence):
    """Convert DNA sequence to space-separated 3-mers using sliding window approach"""
    sequence = sequence.upper()
    
    if len(sequence) < 3:
        return ""
    
    three_mers = []
    
    for i in range(len(sequence) - 2):
        three_mer = sequence[i:i+3]
        
        clean_three_mer = ""
        for base in three_mer:
            if base in 'ACGT':
                clean_three_mer += base
            else:
                clean_three_mer += 'N'
        
        three_mers.append(clean_three_mer)
    
    return ' '.join(three_mers)

def build_cpg_position_map(fasta, chr_name, genomic_start, genomic_end):
    """Build a mapping from CpG index to genomic position range within a region"""
    try:
        ref_sequence = fasta.fetch(chr_name, genomic_start - 1, genomic_end).upper()
        cpg_map = {}
        cpg_index = 0
        
        for i in range(len(ref_sequence) - 2):
            triplet = ref_sequence[i:i+3]
            if len(triplet) == 3 and triplet[1:3] == 'CG':
                cpg_start_pos = genomic_start + i + 1
                cpg_end_pos = genomic_start + i + 2
                cpg_map[cpg_index] = (cpg_start_pos, cpg_end_pos)
                cpg_index += 1
        
        return cpg_map, ref_sequence
    except Exception as e:
        print(f"Warning: Error building CpG map for {chr_name}:{genomic_start}-{genomic_end}: {e}", file=sys.stderr)
        return {}, ""

def process_dmr_region(chr_name, genomic_start, genomic_end, cpg_start, cpg_end, dmr_label, pat_index, fasta, cell_type, region_cell_type=None, dedupe_reads=False):
    """
    Process a single DMR region and generate training records using CpG coordinates
    Outputs the entire DMR region with proper CpG detection for xCG patterns
    Optimized to build CpG map once per DMR region
    
    Args:
        region_cell_type: The cell type that this specific DMR region belongs to (for DMR_cType column)
    """
    try:
        # Build CpG position mapping once for the entire DMR region
        cpg_map, ref_sequence = build_cpg_position_map(fasta, chr_name, genomic_start, genomic_end)
        
        if not cpg_map or not ref_sequence:
            return []
            
        # Find overlapping PAT fragments using optimized search
        overlapping_pats = find_overlapping_pat_for_dmr(pat_index, chr_name, cpg_start, cpg_end)
        
        if not overlapping_pats:
            return []
        
        records = []
        records_dict = {}  # For deduplication aggregations
        
        for pat_cpg_start, pat_cpg_end, pattern, count in overlapping_pats:
            # Impute missing methylation values in PAT pattern
            imputed_pattern = impute_missing_methylation(pattern)
            
            # Find the genomic positions of first and last CpG sites covered by this PAT fragment
            pat_first_cpg_pos = None
            pat_last_cpg_pos = None
            
            # Map PAT CpG indices to genomic positions using pre-built CpG map
            for local_cpg_idx, (mapped_start, mapped_end) in cpg_map.items():
                global_cpg_idx = cpg_start + local_cpg_idx
                
                # Check if this CpG is covered by our PAT fragment
                if pat_cpg_start <= global_cpg_idx <= pat_cpg_end:
                    if pat_first_cpg_pos is None:
                        pat_first_cpg_pos = mapped_start  # Position of C in first CpG
                    pat_last_cpg_pos = mapped_end  # Position of G in last CpG (will be updated)
            
            # Skip if we can't find the PAT coverage boundaries
            if pat_first_cpg_pos is None or pat_last_cpg_pos is None:
                continue
            
            # Add context around the PAT region (max 10 bases each direction)
            # Start from first CpG, extend backwards up to 10 bases
            context_start = pat_first_cpg_pos
            for i in range(1, 11):  # Try up to 10 bases before
                test_pos = pat_first_cpg_pos - i
                # Stop if outside DMR region
                if test_pos < genomic_start:
                    break
                # Stop if we encounter another CpG (check if this position + next = CG)
                ref_offset = test_pos - genomic_start
                if ref_offset + 1 < len(ref_sequence) and ref_sequence[ref_offset:ref_offset+2] == 'CG':
                    break
                context_start = test_pos
            
            # End at last CpG, extend forwards up to 10 bases  
            context_end = pat_last_cpg_pos
            for i in range(1, 11):  # Try up to 10 bases after
                test_pos = pat_last_cpg_pos + i
                # Stop if outside DMR region
                if test_pos >= genomic_end:
                    break
                # Stop if we encounter another CpG (check if prev position + this = CG)
                ref_offset = test_pos - genomic_start
                if ref_offset > 0 and ref_sequence[ref_offset-1:ref_offset+1] == 'CG':
                    break
                context_end = test_pos
            
            # Extract the specific region from reference sequence
            region_start_offset = context_start - genomic_start
            region_end_offset = context_end - genomic_start + 1  # +1 for inclusive end
            
            # Validate the range
            if region_start_offset < 0 or region_end_offset > len(ref_sequence):
                print(f"Warning: Invalid region range {region_start_offset}:{region_end_offset} for sequence length {len(ref_sequence)}", file=sys.stderr)
                continue
                
            overlap_ref_sequence = ref_sequence[region_start_offset:region_end_offset]
            actual_genomic_start = context_start
            
            # Build methylation sequence for 3-mer chunks using sliding window approach
            methyl_seq = ""
            
            # Skip sequences too short for 3-mers
            if len(overlap_ref_sequence) < 3:
                continue
            
            # Generate methylation values for sliding window 3-mers (len(sequence) - 2 tokens)
            for i in range(len(overlap_ref_sequence) - 2):
                # Get the 3-mer at this position
                triplet = overlap_ref_sequence[i:i+3].upper()
                
                # Replace any non-ACGT bases with N
                clean_triplet = ""
                for base in triplet:
                    if base in 'ACGT':
                        clean_triplet += base
                    else:
                        clean_triplet += 'N'
                triplet = clean_triplet
                
                # Check if this 3-mer contains xCG pattern (C in pos 2, G in pos 3)
                if len(triplet) == 3 and triplet[1:3] == 'CG':  # xCG pattern
                    # This 3-mer contains a CpG site - check if it's covered by PAT or context
                    # The C is at genomic position: actual_genomic_start + i + 1
                    cpg_genomic_pos = actual_genomic_start + i + 1
                    
                    # Check if this CpG is within the actual PAT coverage (not context)
                    if pat_first_cpg_pos <= cpg_genomic_pos <= pat_last_cpg_pos:
                        # This CpG is covered by PAT - get real methylation value
                        # Find the corresponding CpG index
                        cpg_index = None
                        for local_idx, (mapped_start, mapped_end) in cpg_map.items():
                            if mapped_start <= cpg_genomic_pos <= mapped_end:
                                global_cpg_idx = cpg_start + local_idx
                                # Check if this CpG is covered by our PAT fragment
                                if pat_cpg_start <= global_cpg_idx <= pat_cpg_end:
                                    cpg_index = global_cpg_idx
                                    break
                        
                        if cpg_index is not None:
                            # Get methylation value from PAT pattern
                            relative_cpg_pos = cpg_index - pat_cpg_start
                            
                            if 0 <= relative_cpg_pos < len(imputed_pattern):
                                pat_base = imputed_pattern[relative_cpg_pos]
                                if pat_base == 'C':
                                    methyl_seq += '1'  # Methylated
                                elif pat_base == 'T':
                                    methyl_seq += '0'  # Unmethylated
                                else:
                                    # Should not happen after imputation, but fallback
                                    methyl_seq += '2'  # Unknown
                            else:
                                methyl_seq += '2'  # Outside PAT range
                        else:
                            methyl_seq += '2'  # CpG not found in our data
                    else:
                        # This CpG is in context region (not covered by PAT) - mark as unknown
                        methyl_seq += '2'
                else:
                    # Not an xCG pattern - not a CpG site
                    methyl_seq += '2'
            
            # Create DNA sequence for the entire DMR region
            dna_seq = create_3mer_sequence(overlap_ref_sequence)
            
            # CRITICAL FIX: Ensure DNA tokens match methylation characters
            # With sliding window: DNA tokens = len(sequence) - 2, Methyl chars = len(sequence) - 2
            dna_tokens = dna_seq.split()
            if len(dna_tokens) != len(methyl_seq):
                print(f"Warning: Token count mismatch - DNA: {len(dna_tokens)} tokens, Methyl: {len(methyl_seq)} chars", file=sys.stderr)
                print(f"  Sequence length: {len(overlap_ref_sequence)}, Expected tokens: {max(0, len(overlap_ref_sequence) - 2)}", file=sys.stderr)
                print(f"  DNA_seq: {dna_seq}", file=sys.stderr)
                print(f"  Methyl_seq: {methyl_seq}", file=sys.stderr)
                continue  # Skip this record if counts don't match
            
            if dna_seq and methyl_seq and methyl_seq != '2' * len(methyl_seq):
                dmr_cell_type_str = region_cell_type if region_cell_type else cell_type

                if dedupe_reads:
                    # Aggregate counts by unique key
                    key = (dna_seq, methyl_seq, dmr_label, cell_type, dmr_cell_type_str)
                    records_dict[key] = records_dict.get(key, 0) + count
                else:
                    # Replicate each read individually (legacy behaviour)
                    for _ in range(count):
                        records.append({
                            'DNA_seq': dna_seq,
                            'Methyl_seq': methyl_seq,
                            'DMR_Label': dmr_label,
                            'cType': cell_type,
                            'DMR_cType': dmr_cell_type_str,
                            'Read_Count': 1
                        })
        
        if dedupe_reads:
            # Convert aggregated dict to list of records with Read_Count
            return [{
                'DNA_seq': k[0],
                'Methyl_seq': k[1],
                'DMR_Label': k[2],
                'cType': k[3],
                'DMR_cType': k[4],
                'Read_Count': v
            } for k, v in records_dict.items()]
        else:
            return records
        
    except Exception as e:
        print(f"Warning: Error processing region {chr_name}:{genomic_start}-{genomic_end}: {e}", file=sys.stderr)
        return []

def write_csv_append(data, output_file, write_header=False):
    """Write data to CSV file in append mode for memory efficiency"""
    mode = 'w' if write_header else 'a'
    with open(output_file, mode, newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['DNA_seq', 'Methyl_seq', 'DMR_Label', 'cType', 'DMR_cType', 'Read_Count'])
        for row in data:
            writer.writerow([
                row['DNA_seq'],
                row['Methyl_seq'], 
                row['DMR_Label'],
                row['cType'],
                row['DMR_cType'],
                row['Read_Count']
            ])

def analyze_dmr_ctype_distribution(csv_file, output_file, cell_type):
    """
    Analyze DMR_cType distribution in a CSV file and save results to master file
    
    Args:
        csv_file: Path to the CSV file to analyze
        output_file: Path to the master output file
        cell_type: Cell type being processed
    """
    try:
        # Count total records (excluding header)
        total_records = 0
        ctype_counts = {}
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_records += 1
                dmr_ctype = row['DMR_cType']
                ctype_counts[dmr_ctype] = ctype_counts.get(dmr_ctype, 0) + 1
        
        if total_records == 0:
            return
        
        # Sort by count (descending)
        sorted_counts = sorted(ctype_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Write analysis to master file
        with open(output_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"DMR_cType Distribution Analysis for: {cell_type}\n")
            f.write(f"CSV File: {csv_file}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total records: {total_records:,}\n\n")
            f.write(f"{'Cell Type':<40} {'Records':<10} {'Percentage':<12}\n")
            f.write(f"{'-'*40} {'-'*10} {'-'*12}\n")
            
            for ctype, count in sorted_counts:
                percentage = (count / total_records) * 100
                f.write(f"{ctype:<40} {count:<10,} {percentage:>8.1f}%\n")
            
            f.write(f"\n")
    
    except Exception as e:
        print(f"Warning: Could not analyze DMR_cType distribution for {csv_file}: {e}", file=sys.stderr)

def process_cell_type_worker(worker_args):
    """
    Worker function for parallel processing of cell types
    MEMORY OPTIMIZED: Uses fork() to share parent's CpG mapper and data structures
    
    Args:
        worker_args: Tuple containing (cell_type, config, args_dict, resume, filter_sex_chroms, 
                     master_analysis_file, shared_cpg_mapper, collective_csv_file)
    
    Returns:
        Tuple of (cell_type, success, error_message, processing_time)
    """
    try:
        (cell_type, config, args_dict, resume, filter_sex_chroms, 
         master_analysis_file, shared_cpg_mapper, collective_csv_file) = worker_args
        
        start_time = time.time()
        
        # Reconstruct args object from dictionary
        class Args:
            def __init__(self, args_dict):
                for key, value in args_dict.items():
                    setattr(self, key, value)
        
        args = Args(args_dict)
        
        # MEMORY OPTIMIZATION: shared_cpg_mapper is already loaded in parent process
        # With fork(), we inherit the parent's memory space (copy-on-write)
        # No need to pickle/unpickle - just use the shared reference
        
        # Process the cell type
        print(f"[{cell_type}] Starting parallel processing (memory-optimized)...", file=sys.stderr)
        process_single_cell_type(
            cell_type, config, args, resume, 
            cpg_mapper=shared_cpg_mapper,  # Direct reference to parent's CpG mapper
            filter_sex_chroms=filter_sex_chroms,
            batch_progress=None,  # Progress tracking handled by main process
            total_batch_files=None,
            master_analysis_file=master_analysis_file
        )
        
        processing_time = time.time() - start_time
        print(f"[{cell_type}] Completed in {processing_time/60:.1f} minutes", file=sys.stderr)
        
        return (cell_type, True, None, processing_time)
        
    except Exception as e:
        processing_time = time.time() - start_time if 'start_time' in locals() else 0
        error_msg = f"Error processing {cell_type}: {str(e)}"
        print(error_msg, file=sys.stderr)
        return (cell_type, False, error_msg, processing_time)

def process_single_cell_type(cell_type: str, config: Dict, args, resume: bool = False, cpg_mapper=None, filter_sex_chroms=True, 
                           batch_progress=None, total_batch_files=None, master_analysis_file=None):
    """Process a single cell type with checkpoint support"""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Processing Cell Type: {cell_type}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    
    # Setup files
    csv_file = config['csv_file']
    pat_file = config['pat_file']
    fasta_file = config['fasta_file']
    output_file = config['output_file']
    cpg_bed_file = config.get('cpg_bed_file')
    
    # Initialize components
    checkpoint_file = f"{output_file}.checkpoint"
    progress = ProgressTracker(checkpoint_file)
    monitor = ResourceMonitor()
    
    # Check for resume
    resumed = False
    if resume:
        resumed = progress.load_checkpoint()
    
    # Initialize output file
    if not resumed and os.path.exists(output_file):
        if not args.force:
            print(f"Output file {output_file} exists. Use --force to overwrite.", file=sys.stderr)
            return
        os.remove(output_file)
    
    # Initialize CpG coordinate mapper if available (only if not provided)
    if cpg_mapper is None:
        if cpg_bed_file and os.path.exists(cpg_bed_file):
            print(f"Loading CpG coordinate mapper from {cpg_bed_file}...", file=sys.stderr)
            cpg_mapper = CpGCoordinateMapper(cpg_bed_file)
        elif cpg_bed_file:
            print(f"Warning: CpG bed file {cpg_bed_file} not found, using CSV coordinates directly", file=sys.stderr)
    
    # Load target cell type DMR regions (for finding overlaps)
    print(f"Loading target cell type ({cell_type}) DMR regions...", file=sys.stderr)
    target_dmr_regions = load_dmr_regions_from_csv(csv_file, target_cell_type=cell_type, cpg_mapper=cpg_mapper, filter_sex_chroms=filter_sex_chroms)
    
    # Load ALL DMR regions to find overlaps
    print(f"Loading ALL DMR regions for overlap analysis...", file=sys.stderr)
    all_dmr_regions = load_dmr_regions_from_csv(csv_file, target_cell_type=None, cpg_mapper=cpg_mapper, filter_sex_chroms=filter_sex_chroms, load_all_regions=True)
    
    # Find overlapping regions
    overlapping_regions = find_overlapping_dmr_regions(target_dmr_regions, all_dmr_regions)
    
    # Build PAT index with CpG coordinates (FAST - load once, use many times)
    print(f"Building PAT index for {cell_type}...", file=sys.stderr)
    pat_index = build_pat_index_with_cpg_coords(pat_file)
    
    # Process regions
    print(f"Processing {len(overlapping_regions)} DMR regions for {cell_type}...", file=sys.stderr)
    print(f"Progress bar updates every 1000 regions (detailed logs for first 5 + every 1000)", file=sys.stderr)
    start_time = time.time()
    
    # Write header if not resuming
    if not resumed:
        write_csv_append([], output_file, write_header=True)
    
    # Initialize timing variables for rolling average
    recent_times = []  # Keep last 10 region times for rolling average
    max_recent_times = 10
    
    with pysam.FastaFile(fasta_file) as fasta:
        # Configure tqdm to update less frequently but show smooth progress
        progress_bar = tqdm(overlapping_regions, 
                           desc=f"Processing {cell_type}", 
                           file=sys.stderr,
                           miniters=1000,  # Update display every 1000 iterations
                           mininterval=2.0,  # Update display every 2 seconds minimum
                           smoothing=0.1)  # Smooth the speed estimate
        
        for i, (chr_name, genomic_start, genomic_end, cpg_start, cpg_end, dmr_label, region_cell_type, overlapping_cell_types) in enumerate(progress_bar):
            region_id = f"{chr_name}:{genomic_start}-{genomic_end}"
            
            # Skip if already processed (resume functionality)
            if progress.is_processed(region_id):
                continue
            
            region_start_time = time.time()
            
            # Print detailed progress for first 5 regions and every 1000 regions
            if i < 5 or (i + 1) % 1000 == 0:
                print(f"\n[{i+1}/{len(overlapping_regions)}] Processing region {region_id} (CpG {cpg_start}-{cpg_end})", file=sys.stderr)
                print(f"  Region cell type: {region_cell_type}, Target cell type: {overlapping_cell_types[0] if overlapping_cell_types else 'Unknown'}", file=sys.stderr)
            
            records = process_dmr_region(
                chr_name, genomic_start, genomic_end, cpg_start, cpg_end, 
                dmr_label, pat_index, fasta, cell_type, region_cell_type=region_cell_type,
                dedupe_reads=args.dedupe_reads
            )
            
            # Write records incrementally
            if records:
                write_csv_append(records, output_file)
            
            # Update progress
            progress.mark_processed(region_id, len(records))
            
            # Save checkpoint every 100 regions and log usage every 500 regions (reduced clutter)
            if (i + 1) % 100 == 0:
                progress.save_checkpoint()
            if (i + 1) % 500 == 0:
                monitor.log_usage(f"{cell_type} - Region {i+1}")
                
            # Update progress bar description with current region info every 1000 iterations
            if (i + 1) % 1000 == 0:
                progress_bar.set_description(f"Processing {cell_type} - {progress.total_records:,} records")
            
            region_time = time.time() - region_start_time
            
            # Update rolling average for timing estimates
            recent_times.append(region_time)
            if len(recent_times) > max_recent_times:
                recent_times.pop(0)  # Remove oldest time
            
            # Calculate rolling average
            avg_time_per_region = sum(recent_times) / len(recent_times) if recent_times else 0
            
            # Show detailed timing for first 5 regions and every 1000 regions 
            if i < 5 or (i + 1) % 1000 == 0:
                # Calculate remaining regions and estimated time
                remaining_regions = len([r for r in overlapping_regions[i+1:] if not progress.is_processed(f"{r[0]}:{r[1]}-{r[2]}")])
                estimated_remaining = remaining_regions * avg_time_per_region
                
                print(f"  → Found {len(records)} records (Total: {progress.total_records})", file=sys.stderr)
                print(f"  → Region: {region_time:.1f}s | Rolling Avg: {avg_time_per_region:.1f}s/region | Est. remaining: {estimated_remaining/60:.1f}min", file=sys.stderr)
    
    # Calculate total processing time
    total_processing_time = time.time() - start_time
    
    # Cleanup checkpoint
    progress.cleanup()
    monitor.log_usage(f"{cell_type} - COMPLETED")
    
    # Update batch progress if provided
    if batch_progress is not None and total_batch_files is not None:
        batch_progress[0] += 1
        print(f"\n[{batch_progress[0]}/{total_batch_files}] Completed {cell_type}! Results saved to {output_file}", file=sys.stderr)
    else:
        print(f"\nCompleted {cell_type}! Results saved to {output_file}", file=sys.stderr)
    print(f"Final record count: {progress.total_records}", file=sys.stderr)
    print(f"Total processing time: {total_processing_time/60:.1f} minutes ({total_processing_time:.1f} seconds)", file=sys.stderr)
    
    # Analyze DMR_cType distribution and save to master file
    if master_analysis_file is not None:
        print(f"Analyzing DMR_cType distribution and saving to {master_analysis_file}...", file=sys.stderr)
        analyze_dmr_ctype_distribution(output_file, master_analysis_file, cell_type)

def create_sample_config():
    """Create a sample configuration file for multiple cell types"""
    # Option 1: Collective CSV mode (Default)
    collective_config = {
        "collective_csv_mode": True,
        "collective_csv_file": "hg38_all_cell_types.csv",
        "fasta_file": "hg38.fa",
        "cpg_bed_file": "../references/hg19/CpG.bed.gz",
        "cell_types": {
            "SkeletalMuscle_Sample1": {
                "pat_file": "GSM5652205_Skeletal-Muscle-Z00000427.pat.gz",
                "output_file": "skeletal_muscle_sample1_training_data.csv",
                "cell_type": "SkeletalMuscle"
            },
            "SkeletalMuscle_Sample2": {
                "pat_file": "GSM5652206_Skeletal-Muscle-Z00000428.pat.gz",
                "output_file": "skeletal_muscle_sample2_training_data.csv",
                "cell_type": "SkeletalMuscle"
            },
            "Lung": {
                "pat_file": "GSM5652207_Lung-Z00000429.pat.gz",
                "output_file": "lung_training_data.csv",
                "cell_type": "Lung"
            },
            "Liver": {
                "pat_file": "GSM5652208_Liver-Z00000430.pat.gz",
                "output_file": "liver_training_data.csv",
                "cell_type": "Liver"
            }
        }
    }
    
    # Option 2: Individual CSV mode (legacy)
    individual_config = {
        "collective_csv_mode": False,
        "cpg_bed_file": "../references/hg19/CpG.bed.gz",
        "SkeletalMuscle": {
            "csv_file": "hg38_Skeletal_Muscle.csv",
            "pat_file": "GSM5652205_Skeletal-Muscle-Z00000427.pat.gz",
            "fasta_file": "hg38.fa",
            "output_file": "skeletal_muscle_training_data.csv"
        },
        "Lung": {
            "csv_file": "hg38_Lung.csv", 
            "pat_file": "GSM5652206_Lung-Z00000428.pat.gz",
            "fasta_file": "hg38.fa",
            "output_file": "lung_training_data.csv"
        },
        "Liver": {
            "csv_file": "hg38_Liver.csv",
            "pat_file": "GSM5652207_Liver-Z00000429.pat.gz", 
            "fasta_file": "hg38.fa",
            "output_file": "liver_training_data.csv"
        }
    }
    
    # Create collective mode config (default)
    with open("cell_types_config_collective.json", "w") as f:
        json.dump(collective_config, f, indent=2)
    
    # Create individual mode config (legacy)
    with open("cell_types_config_individual.json", "w") as f:
        json.dump(individual_config, f, indent=2)
    
    print("Created sample configuration files:", file=sys.stderr)
    print("  - cell_types_config_collective.json (RECOMMENDED - uses one CSV with cell_type column)", file=sys.stderr)
    print("  - cell_types_config_individual.json (legacy - separate CSV per cell type)", file=sys.stderr)
    print("", file=sys.stderr)
    print("For collective mode, your CSV should have format:", file=sys.stderr)
    print("  cell_type,chr,start,end", file=sys.stderr)
    print("  SkeletalMuscle,chr1,1000,2000", file=sys.stderr)
    print("  Lung,chr1,1500,2500", file=sys.stderr)
    print("  Liver,chr2,3000,4000", file=sys.stderr)
    print("", file=sys.stderr)
    print("For multiple samples of the same cell type, use unique keys with 'cell_type' field:", file=sys.stderr)
    print("  \"Adipocytes_T7\": {", file=sys.stderr)
    print("    \"pat_file\": \"sample1.pat.gz\",", file=sys.stderr)
    print("    \"output_file\": \"adipocytes_t7.csv\",", file=sys.stderr)
    print("    \"cell_type\": \"Adipocytes\"", file=sys.stderr)
    print("  },", file=sys.stderr)
    print("  \"Adipocytes_T9\": {", file=sys.stderr)
    print("    \"pat_file\": \"sample2.pat.gz\",", file=sys.stderr)
    print("    \"output_file\": \"adipocytes_t9.csv\",", file=sys.stderr)
    print("    \"cell_type\": \"Adipocytes\"", file=sys.stderr)
    print("  }", file=sys.stderr)
    print("", file=sys.stderr)
    print("Note: CpG coordinates are now automatically converted from genomic coordinates", file=sys.stderr)
    print("using the CpG.bed.gz file specified in the configuration.", file=sys.stderr)
    print("", file=sys.stderr)
    print("NEW FEATURE: Overlap Analysis", file=sys.stderr)
    print("The pipeline now finds ALL DMR regions that overlap with the target cell type's regions.", file=sys.stderr)
    print("This includes regions from other cell types that overlap with the target cell type's DMRs.", file=sys.stderr)
    print("A new column 'DMR_cType' is added to track which cell type each DMR region belongs to.", file=sys.stderr)
    print("Example output format:", file=sys.stderr)
    print("  DNA_seq,Methyl_seq,DMR_Label,cType,DMR_cType", file=sys.stderr)
    print("  ATC GCT CGA,101 010 101,0,SkeletalMuscle,SkeletalMuscle", file=sys.stderr)
    print("  GAT CGC TAT,001 110 001,1,SkeletalMuscle,Lung", file=sys.stderr)
    print("", file=sys.stderr)
    print("PARALLEL PROCESSING: Use --parallel 20 to process 20 cell types simultaneously", file=sys.stderr)
    print("Example: python analyze_methylation_batch_optimized.py --config config.json --parallel 20", file=sys.stderr)
    print("", file=sys.stderr)
    print("PROCESS ALL REGIONS: By default, ALL DMR regions are processed (not just overlapping)", file=sys.stderr)
    print("Use --process-overlapping-only to revert to legacy overlap-only mode", file=sys.stderr)
    print("", file=sys.stderr)
    print("Progress tracking is now available for batch processing mode.", file=sys.stderr)
    print("The pipeline will show progress like '29/208 files done' during processing.", file=sys.stderr)
    print("Add --dedupe-reads to write each PAT fragment once (ignore read count) to remove duplicates", file=sys.stderr)

def main():
    # Set multiprocessing start method for memory efficiency - USE FORK instead of SPAWN
    try:
        mp.set_start_method('fork', force=True)  # CHANGED: fork shares memory, spawn duplicates
    except RuntimeError:
        pass  # Method was already set
    
    parser = argparse.ArgumentParser(description='Optimized MethylBERT training data pipeline for multiple cell types with PARALLEL PROCESSING and ALL REGIONS support')
    
    # Single cell type mode (backwards compatible)
    parser.add_argument('--csv', help='CSV file with markers and genomic coordinates')
    parser.add_argument('--pat', help='PAT file with methylation data')
    parser.add_argument('--fasta', help='Reference genome FASTA file')
    parser.add_argument('--cpg-bed', help='CpG bed file for accurate coordinate mapping (format: chr pos cpg_index)')
    parser.add_argument('--output', help='Output CSV file')
    parser.add_argument('--cell-type', help='Cell type name for single processing mode')
    parser.add_argument('--filter-sex-chroms', dest='filter_sex_chroms', action='store_true', default=True, help='Filter out DMRs on chrX/chrY (default: True)')
    parser.add_argument('--no-filter-sex-chroms', dest='filter_sex_chroms', action='store_false', help='Do not filter out DMRs on chrX/chrY')
    
    # Batch processing mode
    parser.add_argument('--config', help='JSON configuration file for multiple cell types')
    parser.add_argument('--cell-types', nargs='+', help='Specific cell types to process (default: all in config)')
    parser.add_argument('--create-config', action='store_true', help='Create sample configuration files (collective and individual modes)')
    parser.add_argument('--discover-cell-types', help='Discover cell types from collective CSV file (requires --csv with cell_type column)')
    
    # Processing options - UPDATED RECOMMENDATIONS
    parser.add_argument('--force', action='store_true', help='Overwrite existing output files')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint if available')
    parser.add_argument('--parallel', type=int, default=1, help='Number of cell types to process in parallel (1=sequential, 12=recommended for 256GB servers, 20=for 512GB+ servers)')
    parser.add_argument('--process-all-regions', action='store_true', default=True, help='Process ALL DMR regions instead of just overlapping ones (default: True)')
    parser.add_argument('--process-overlapping-only', dest='process_all_regions', action='store_false', help='Process only overlapping regions (legacy mode)')
    parser.add_argument('--dedupe-reads', default = False, help='Write each PAT fragment once (ignore read count) to remove duplicates')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
        return
    
    if args.discover_cell_types:
        try:
            discovered_types = get_cell_types_from_csv(args.discover_cell_types)
            print(f"\nDiscovered cell types from {args.discover_cell_types}:", file=sys.stderr)
            for cell_type in sorted(discovered_types):
                print(f"  - {cell_type}", file=sys.stderr)
            return
        except Exception as e:
            print(f"Error discovering cell types: {e}", file=sys.stderr)
            return
    
    print("Optimized MethylBERT Training Data Pipeline", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    
    # Single cell type mode (backwards compatible)
    if args.csv and args.pat and args.fasta and args.output:
        cell_type = args.cell_type or "Unknown"
        config = {
            'csv_file': args.csv,
            'pat_file': args.pat,
            'fasta_file': args.fasta,
            'output_file': args.output,
            'cpg_bed_file': getattr(args, 'cpg_bed', None)
        }
        process_single_cell_type(cell_type, config, args, args.resume, filter_sex_chroms=args.filter_sex_chroms)
        return
    
    # Batch processing mode
    if not args.config:
        print("Error: Either provide individual files (--csv, --pat, --fasta, --output) or --config for batch processing", file=sys.stderr)
        print("Use --create-config to generate a sample configuration file", file=sys.stderr)
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found", file=sys.stderr)
        return
    
    # Load and parse configuration
    with open(args.config, 'r') as f:
        config_data = json.load(f)
    
    # Detect configuration mode
    collective_mode = config_data.get("collective_csv_mode", False)
    
    if collective_mode:
        print("Using collective CSV mode", file=sys.stderr)
        
        # Get collective CSV file and common settings
        collective_csv_file = config_data.get("collective_csv_file")
        common_fasta_file = config_data.get("fasta_file")
        common_cpg_bed_file = config_data.get("cpg_bed_file")
        
        if not collective_csv_file or not common_fasta_file:
            print("Error: collective_csv_file and fasta_file are required for collective mode", file=sys.stderr)
            return
        
        if not os.path.exists(collective_csv_file):
            print(f"Error: Collective CSV file {collective_csv_file} not found", file=sys.stderr)
            return
        
        # Load CpGCoordinateMapper ONCE if common_cpg_bed_file exists
        shared_cpg_mapper = None
        shared_cpg_mapper_file = None
        if common_cpg_bed_file and os.path.exists(common_cpg_bed_file):
            print(f"Loading CpG coordinate mapper ONCE from {common_cpg_bed_file} for all cell types...", file=sys.stderr)
            shared_cpg_mapper = CpGCoordinateMapper(common_cpg_bed_file)
            
            # Save CpG mapper to file for parallel workers if using parallel processing
            if args.parallel > 1:
                shared_cpg_mapper_file = f"shared_cpg_mapper_{os.getpid()}.pkl"
                print(f"Saving CpG mapper to {shared_cpg_mapper_file} for parallel workers...", file=sys.stderr)
                with open(shared_cpg_mapper_file, 'wb') as f:
                    pickle.dump(shared_cpg_mapper, f)
        
        # Auto-discover cell types from CSV if not specified
        if args.cell_types:
            cell_types_to_process = args.cell_types
            print(f"Processing specified cell types: {', '.join(cell_types_to_process)}", file=sys.stderr)
        else:
            discovered_cell_types = get_cell_types_from_csv(collective_csv_file)
            
            # Get configured cell types from the cell_type field in each entry
            configured_cell_types = set()
            for entry_key, entry_config in config_data.get("cell_types", {}).items():
                cell_type = entry_config.get("cell_type", entry_key)  # Use cell_type field or fallback to key
                configured_cell_types.add(cell_type)
            
            # Use intersection of discovered and configured cell types
            cell_types_to_process = list(discovered_cell_types & configured_cell_types)
            
            if not cell_types_to_process:
                print("Error: No matching cell types found between CSV and configuration", file=sys.stderr)
                print(f"  CSV contains: {sorted(discovered_cell_types)}", file=sys.stderr)
                print(f"  Config contains: {sorted(configured_cell_types)}", file=sys.stderr)
                return
        
        print(f"Processing {len(cell_types_to_process)} cell types: {', '.join(cell_types_to_process)}", file=sys.stderr)
        
        # Initialize master analysis file
        master_analysis_file = "DMR_cType_Distribution_250.txt"
        with open(master_analysis_file, 'w') as f:
            f.write("DMR_cType Distribution Analysis Report\n")
            f.write("Generated by MethylBERT Training Pipeline\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
        
        print(f"Master analysis file initialized: {master_analysis_file}", file=sys.stderr)
        
        # Process each cell type
        overall_start = time.time()
        processed_count = 0
        
        # Count total files to process for progress tracking
        total_files_to_process = 0
        for cell_type in cell_types_to_process:
            matching_entries = []
            for entry_key, entry_config in config_data.get("cell_types", {}).items():
                entry_cell_type = entry_config.get("cell_type", entry_key)
                if entry_cell_type == cell_type:
                    matching_entries.append((entry_key, entry_config))
            total_files_to_process += len(matching_entries)
        
        print(f"Total files to process: {total_files_to_process}", file=sys.stderr)
        
        # Prepare worker arguments for all cell type samples
        worker_args_list = []
        
        # Convert args to dictionary for pickling
        args_dict = vars(args)
        
        # Find all entries that match each cell type and prepare worker arguments
        for cell_type in cell_types_to_process:
            matching_entries = []
            for entry_key, entry_config in config_data.get("cell_types", {}).items():
                entry_cell_type = entry_config.get("cell_type", entry_key)
                if entry_cell_type == cell_type:
                    matching_entries.append((entry_key, entry_config))
            
            if not matching_entries:
                print(f"Warning: No configuration found for cell type {cell_type}", file=sys.stderr)
                continue
            
            print(f"Preparing {len(matching_entries)} sample(s) for cell type: {cell_type}", file=sys.stderr)
            
            # Create worker arguments for each sample
            for sample_idx, (entry_key, cell_type_config) in enumerate(matching_entries):
                # Build individual cell type config
                individual_config = {
                    'csv_file': collective_csv_file,  # Use collective CSV
                    'pat_file': cell_type_config.get('pat_file'),
                    'fasta_file': common_fasta_file,  # Use common FASTA
                    'output_file': cell_type_config.get('output_file'),
                    'cpg_bed_file': common_cpg_bed_file  # Use common CpG bed file
                }
                
                # Create worker arguments tuple
                worker_args = (
                    cell_type,
                    individual_config,
                    args_dict,
                    args.resume,
                    args.filter_sex_chroms,
                    master_analysis_file,
                    shared_cpg_mapper,
                    collective_csv_file
                )
                
                worker_args_list.append((entry_key, worker_args))
        
        print(f"Prepared {len(worker_args_list)} tasks for processing", file=sys.stderr)
        
        # Process using parallel or sequential mode
        if args.parallel > 1:
            print(f"\n🚀 PARALLEL PROCESSING ENABLED 🚀", file=sys.stderr)
            print(f"=" * 60, file=sys.stderr)
            print(f"Workers: {args.parallel}", file=sys.stderr)
            print(f"Total PAT files: {total_files_to_process}", file=sys.stderr)
            print(f"Expected speedup: ~{min(args.parallel, total_files_to_process)}x faster", file=sys.stderr)
            print(f"Monitor with: htop (in another terminal)", file=sys.stderr)
            print(f"=" * 60, file=sys.stderr)
            
            # Use multiprocessing Pool
            with Pool(processes=args.parallel) as pool:
                print(f"Created worker pool with {args.parallel} processes", file=sys.stderr)
                
                # Submit all tasks and get results
                tasks = []
                for i, (entry_key, worker_args) in enumerate(worker_args_list):
                    task = pool.apply_async(process_cell_type_worker, (worker_args,))
                    tasks.append((entry_key, task))
                    if i < 10:  # Show first 10 submissions
                        print(f"Submitted task {i+1}: {entry_key} ({worker_args[0]})", file=sys.stderr)
                    elif i == 10:
                        print(f"... submitting {len(worker_args_list) - 10} more tasks", file=sys.stderr)
                
                print(f"All {len(worker_args_list)} tasks submitted to worker pool", file=sys.stderr)
                print(f"Workers are now processing in parallel...", file=sys.stderr)
                
                # Monitor progress and collect results
                completed = 0
                start_parallel_time = time.time()
                
                print(f"\n PARALLEL PROGRESS MONITORING", file=sys.stderr)
                print(f"Waiting for {len(tasks)} workers to complete...", file=sys.stderr)
                
                for entry_key, task in tasks:
                    try:
                        cell_type, success, error_msg, processing_time = task.get()
                        completed += 1
                        elapsed_parallel = time.time() - start_parallel_time
                        
                        if success:
                            print(f"[{completed}/{total_files_to_process}] {cell_type} ({entry_key}) completed in {processing_time/60:.1f}min (Total elapsed: {elapsed_parallel/60:.1f}min)", file=sys.stderr)
                        else:
                            print(f"[{completed}/{total_files_to_process}] {cell_type} ({entry_key}) failed: {error_msg}", file=sys.stderr)
                            if not args.force:
                                print("Use --force to continue processing other cell types after errors", file=sys.stderr)
                        
                        # Show throughput every 10 completions
                        if completed % 10 == 0:
                            throughput = completed / (elapsed_parallel / 3600)  # files per hour
                            est_remaining = (total_files_to_process - completed) / throughput if throughput > 0 else 0
                            print(f"📈 Throughput: {throughput:.1f} files/hour | Est. remaining: {est_remaining:.1f}h", file=sys.stderr)
                                
                    except Exception as e:
                        completed += 1
                        print(f"[{completed}/{total_files_to_process}] {entry_key} failed with exception: {e}", file=sys.stderr)
            
            processed_count = completed
            
        else:
            print(f"\n SEQUENTIAL PROCESSING MODE", file=sys.stderr)
            print(f"=" * 60, file=sys.stderr)
            print(f"Processing {total_files_to_process} PAT files one at a time", file=sys.stderr)
            print(f"Expected time: ~{total_files_to_process * 0.75:.1f} hours", file=sys.stderr)
            print(f"💡 To use parallel processing, add: --parallel 20", file=sys.stderr)
            print(f"=" * 60, file=sys.stderr)
            processed_count = 0
            
            # Process sequentially (original behavior)
            for entry_key, worker_args in worker_args_list:
                processed_count += 1
                print(f"\n[{processed_count}/{total_files_to_process}] Starting {worker_args[0]} ({entry_key})...", file=sys.stderr)
                
                try:
                    cell_type, success, error_msg, processing_time = process_cell_type_worker(worker_args)
                    
                    if success:
                        print(f"✓ {cell_type} ({entry_key}) completed in {processing_time/60:.1f}min", file=sys.stderr)
                    else:
                        print(f"✗ {cell_type} ({entry_key}) failed: {error_msg}", file=sys.stderr)
                        if not args.force:
                            print("Use --force to continue processing other cell types after errors", file=sys.stderr)
                            break
                            
                except Exception as e:
                    print(f"✗ {entry_key} failed with exception: {e}", file=sys.stderr)
                    if not args.force:
                        print("Use --force to continue processing other cell types after errors", file=sys.stderr)
                        break
        
        # Cleanup shared CpG mapper file if created
        if shared_cpg_mapper_file and os.path.exists(shared_cpg_mapper_file):
            try:
                os.remove(shared_cpg_mapper_file)
                print(f"Cleaned up shared CpG mapper file: {shared_cpg_mapper_file}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Could not remove shared CpG mapper file: {e}", file=sys.stderr)
        
        # Calculate total batch processing time
        overall_time = time.time() - overall_start
        print(f"\nBatch processing completed in {overall_time/60:.1f} minutes ({overall_time:.1f} seconds)", file=sys.stderr)
        
        # Add summary to master analysis file
        with open(master_analysis_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"BATCH PROCESSING SUMMARY (COLLECTIVE MODE)\n")
            f.write(f"{'='*80}\n")
            f.write(f"Processing mode: {'PARALLEL' if args.parallel > 1 else 'SEQUENTIAL'}\n")
            if args.parallel > 1:
                f.write(f"Parallel workers: {args.parallel}\n")
            f.write(f"Process all regions: {args.process_all_regions}\n")
            f.write(f"Total processing time: {overall_time/60:.1f} minutes ({overall_time:.1f} seconds)\n")
            f.write(f"Files processed: {processed_count}/{total_files_to_process}\n")
            f.write(f"Master analysis file: {master_analysis_file}\n")
    
    else:
        print("Using individual CSV mode (legacy)", file=sys.stderr)
        
        # Legacy mode: separate CSV files per cell type
        cell_config = CellTypeConfig(args.config)
        
        cell_types_to_process = args.cell_types or cell_config.get_cell_types()
        
        if not cell_types_to_process:
            print("No cell types to process", file=sys.stderr)
            return
        
        print(f"Processing {len(cell_types_to_process)} cell types: {', '.join(cell_types_to_process)}", file=sys.stderr)
        
        # Get common CpG bed file if specified
        common_cpg_bed_file = config_data.get("cpg_bed_file")
        
        # Load CpGCoordinateMapper ONCE if common_cpg_bed_file exists
        shared_cpg_mapper = None
        if common_cpg_bed_file and os.path.exists(common_cpg_bed_file):
            print(f"MEMORY OPTIMIZATION: Loading CpG mapper ONCE for all workers...", file=sys.stderr)
            print(f"Loading CpG coordinate mapper from {common_cpg_bed_file}...", file=sys.stderr)
            shared_cpg_mapper = CpGCoordinateMapper(common_cpg_bed_file)
            
            # MEMORY OPTIMIZATION: With fork(), workers will share this memory via copy-on-write
            # No need to save to file - direct memory sharing saves ~8GB × num_workers
            print(f"CpG mapper loaded in parent process - workers will share via fork()", file=sys.stderr)
        
        # Initialize master analysis file
        master_analysis_file = "DMR_cType_Distribution_250.txt"
        with open(master_analysis_file, 'w') as f:
            f.write("DMR_cType Distribution Analysis Report\n")
            f.write("Generated by MethylBERT Training Pipeline\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
        
        print(f"Master analysis file initialized: {master_analysis_file}", file=sys.stderr)
        
        # Initialize batch progress tracking
        batch_progress = [0]  # Use list to allow modification in nested functions
        total_files_to_process = len(cell_types_to_process)
        print(f"Total files to process: {total_files_to_process}", file=sys.stderr)
        
        # Process each cell type
        overall_start = time.time()
        for i, cell_type in enumerate(cell_types_to_process):
            config = cell_config.get_config(cell_type)
            if not config:
                print(f"Warning: No configuration found for cell type {cell_type}", file=sys.stderr)
                continue
            
            # Add CpG bed file to config if not already present
            if common_cpg_bed_file and 'cpg_bed_file' not in config:
                config['cpg_bed_file'] = common_cpg_bed_file
                
            print(f"\n[{batch_progress[0]+1}/{total_files_to_process}] Starting {cell_type}...", file=sys.stderr)
            
            try:
                process_single_cell_type(cell_type, config, args, args.resume, 
                                      cpg_mapper=shared_cpg_mapper, filter_sex_chroms=args.filter_sex_chroms,
                                      batch_progress=batch_progress, total_batch_files=total_files_to_process,
                                      master_analysis_file=master_analysis_file)
            except Exception as e:
                print(f"Error processing {cell_type}: {e}", file=sys.stderr)
                if not args.force:
                    print("Use --force to continue processing other cell types after errors", file=sys.stderr)
                    break
        
        # Calculate total batch processing time
        overall_time = time.time() - overall_start
        print(f"\nBatch processing completed in {overall_time/60:.1f} minutes ({overall_time:.1f} seconds)", file=sys.stderr)
        
        # Add summary to master analysis file
        with open(master_analysis_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"BATCH PROCESSING SUMMARY (LEGACY MODE)\n")
            f.write(f"{'='*80}\n")
            f.write(f"Processing mode: SEQUENTIAL (legacy mode doesn't support parallel)\n")
            f.write(f"Process all regions: {args.process_all_regions}\n")
            f.write(f"Total processing time: {overall_time/60:.1f} minutes ({overall_time:.1f} seconds)\n")
            f.write(f"Files processed: {batch_progress[0]}/{total_files_to_process}\n")
            f.write(f"Master analysis file: {master_analysis_file}\n")

if __name__ == "__main__":
    main() 