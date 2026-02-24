"""
Script to create unified microbiome embeddings for each dataset.

This script merges all microbiome embeddings from different month/group files
into a single unified H5 file for each dataset. This allows for flexible
loading of custom sample subsets without recomputing embeddings.

Usage:
    python create_unified_embeddings.py --dataset Diabimmune
    python create_unified_embeddings.py --dataset all
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Set


def find_all_microbiome_embedding_files(dataset_dir: Path) -> List[Path]:
    """
    Find all microbiome_embeddings.h5 files in the dataset directory.
    
    Args:
        dataset_dir: Path to dataset directory (e.g., huggingface_datasets/Diabimmune)
    
    Returns:
        List of paths to microbiome_embeddings.h5 files
    """
    embeddings_base = dataset_dir / "processed" / "microbiome_embeddings"
    
    if not embeddings_base.exists():
        print(f"Warning: {embeddings_base} does not exist")
        return []
    
    # Find all microbiome_embeddings.h5 files recursively
    embedding_files = list(embeddings_base.glob("*/microbiome_embeddings.h5"))
    
    return sorted(embedding_files)


def merge_embeddings_to_unified(
    embedding_files: List[Path],
    output_path: Path,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Merge multiple microbiome embedding H5 files into a single unified file.
    
    Args:
        embedding_files: List of paths to microbiome_embeddings.h5 files
        output_path: Path where unified embeddings will be saved
        verbose: Whether to print detailed information
    
    Returns:
        Dictionary with merge statistics
    """
    if not embedding_files:
        raise ValueError("No embedding files provided")
    
    # Statistics tracking
    stats = {
        "total_samples": 0,
        "duplicates_found": 0,
        "files_processed": 0,
        "source_files": [],
        "duplicate_sids": set(),
        "all_sids": set()
    }
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store all embeddings (in case of duplicates, we'll keep the first)
    all_embeddings = {}
    sid_to_source = {}  # Track which file each SID came from
    
    if verbose:
        print(f"\nMerging {len(embedding_files)} embedding files...")
    
    # First pass: collect all embeddings
    for h5_file in tqdm(embedding_files, desc="Reading embedding files", disable=not verbose):
        source_name = h5_file.parent.name  # e.g., "Month_1", "gadir_all_months", etc.
        stats["source_files"].append(source_name)
        
        with h5py.File(h5_file, "r") as f:
            sample_ids = list(f.keys())
            
            for sid in sample_ids:
                stats["all_sids"].add(sid)
                
                if sid in all_embeddings:
                    # Duplicate found
                    stats["duplicates_found"] += 1
                    stats["duplicate_sids"].add(sid)
                    if verbose and stats["duplicates_found"] <= 5:
                        print(f"  Warning: Duplicate SID '{sid}' found in {source_name} "
                              f"(already in {sid_to_source[sid]}). Keeping first occurrence.")
                else:
                    # Store embedding
                    embedding = f[sid][:]
                    all_embeddings[sid] = embedding
                    sid_to_source[sid] = source_name
        
        stats["files_processed"] += 1
    
    stats["total_samples"] = len(all_embeddings)
    
    if verbose:
        print(f"\nWriting {stats['total_samples']} unique samples to {output_path}...")
    
    # Second pass: write all embeddings to unified file
    with h5py.File(output_path, "w") as f_out:
        for sid, embedding in tqdm(all_embeddings.items(), 
                                   desc="Writing unified file",
                                   disable=not verbose):
            f_out.create_dataset(sid, data=embedding)
        
        # Store metadata as attributes
        f_out.attrs["total_samples"] = stats["total_samples"]
        f_out.attrs["source_files"] = ",".join(stats["source_files"])
        f_out.attrs["duplicates_found"] = stats["duplicates_found"]
    
    if verbose:
        print(f"\n✓ Unified embeddings saved to: {output_path}")
        print(f"  Total unique samples: {stats['total_samples']}")
        print(f"  Files merged: {stats['files_processed']}")
        if stats["duplicates_found"] > 0:
            print(f"  Duplicates found: {stats['duplicates_found']}")
    
    return stats


def create_unified_embeddings_for_dataset(dataset_name: str, base_dir: Path, verbose: bool = True):
    """
    Create unified embeddings for a single dataset.
    
    Args:
        dataset_name: Name of dataset (e.g., 'Diabimmune', 'Gadir', etc.)
        base_dir: Base directory containing huggingface_datasets
        verbose: Whether to print detailed information
    """
    dataset_dir = base_dir / dataset_name
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Find all embedding files
    embedding_files = find_all_microbiome_embedding_files(dataset_dir)
    
    if not embedding_files:
        print(f"No embedding files found for {dataset_name}")
        return None
    
    if verbose:
        print(f"Found {len(embedding_files)} embedding files:")
        for f in embedding_files:
            print(f"  - {f.parent.name}")
    
    # Create unified embeddings
    output_path = dataset_dir / "processed" / "microbiome_embeddings" / "unified_all_samples.h5"
    
    stats = merge_embeddings_to_unified(embedding_files, output_path, verbose=verbose)
    
    # Also create a metadata CSV listing all sample IDs
    create_sample_index(output_path, dataset_dir, stats)
    
    return stats


def create_sample_index(unified_h5_path: Path, dataset_dir: Path, stats: Dict):
    """
    Create a CSV index of all sample IDs in the unified file.
    
    Args:
        unified_h5_path: Path to unified embeddings H5 file
        dataset_dir: Dataset directory
        stats: Statistics from the merge process
    """
    import pandas as pd
    
    sample_ids = []
    
    with h5py.File(unified_h5_path, "r") as f:
        sample_ids = sorted(list(f.keys()))
    
    # Create DataFrame
    df = pd.DataFrame({"sid": sample_ids})
    
    # Save to CSV
    index_path = dataset_dir / "processed" / "microbiome_embeddings" / "unified_sample_index.csv"
    df.to_csv(index_path, index=False)
    
    print(f"✓ Sample index saved to: {index_path}")
    print(f"  Total samples indexed: {len(sample_ids)}")


def main():
    parser = argparse.ArgumentParser(
        description="Create unified microbiome embeddings for datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset name (Diabimmune, Gadir, Goldberg, Tanaka, or 'all')"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("huggingface_datasets"),
        help="Base directory containing datasets"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    # Validate base directory
    if not args.base_dir.exists():
        print(f"Error: Base directory not found: {args.base_dir}")
        return
    
    # List of all datasets
    all_datasets = ["Diabimmune", "Gadir", "Goldberg", "Tanaka"]
    
    if args.dataset.lower() == "all":
        datasets_to_process = all_datasets
    else:
        # Check if dataset name is valid
        if args.dataset not in all_datasets:
            print(f"Error: Unknown dataset '{args.dataset}'")
            print(f"Available datasets: {', '.join(all_datasets)}")
            return
        datasets_to_process = [args.dataset]
    
    # Process each dataset
    all_stats = {}
    for dataset_name in datasets_to_process:
        stats = create_unified_embeddings_for_dataset(
            dataset_name,
            args.base_dir,
            verbose=not args.quiet
        )
        if stats:
            all_stats[dataset_name] = stats
    
    # Print summary
    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print("Summary of all datasets:")
        print(f"{'='*60}")
        for dataset_name, stats in all_stats.items():
            print(f"{dataset_name}:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Files merged: {stats['files_processed']}")
            if stats['duplicates_found'] > 0:
                print(f"  Duplicates: {stats['duplicates_found']}")
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()
