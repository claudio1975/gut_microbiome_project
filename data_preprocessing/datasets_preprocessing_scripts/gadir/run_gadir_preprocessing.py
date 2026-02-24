#!/usr/bin/env python3
"""
Gadir Dataset Preprocessing Script

Generates metadata CSVs from raw Gadir SRA metadata.
Output: Age-grouped CSV files with columns [sid, label]
- label: 0=Control (healthy), 1=FoodAllergy
- age_groups: 0-6, 6-12, 12-18, 18-24, 24-30, 30+ months

Note: Excludes "Unclear" and "ControlHiRisk" samples.
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / '../../../huggingface_datasets/Gadir/metadata'
RAW_METADATA = SCRIPT_DIR / 'gadir_metadata.csv'

# ============================================================
# 1. Load raw metadata
# ============================================================
print("Loading raw data...")
df = pd.read_csv(RAW_METADATA, dtype=str)
print(f"Loaded {len(df)} total samples")

# ============================================================
# 2. Filter samples
# ============================================================
# Filter: Keep only human samples
if 'HOST' in df.columns:
    df = df[df['HOST'].str.contains('Homo sapiens', case=False, na=False)]
if 'Organism' in df.columns:
    df = df[df['Organism'].str.contains('human', case=False, na=False)]

print(f"After human filter: {len(df)} samples")

# Filter: Exclude Unclear and ControlHiRisk
print(f"Group distribution: {df['Group'].value_counts().to_dict()}")
df = df[~df['Group'].str.lower().isin(['unclear', 'controlhirisk'])]
print(f"After filtering: {len(df)} samples (excluded Unclear/ControlHiRisk)")

# ============================================================
# 3. Create binary labels
# ============================================================
label_mapping = {'FoodAllergy': 1, 'Control': 0}
df['label'] = df['Group'].map(label_mapping)
print(f"Label distribution: {df['label'].value_counts().to_dict()}")

# ============================================================
# 4. Create age groups
# ============================================================
def assign_age_group(age):
    """Assign age group based on 6-month intervals."""
    if pd.isna(age):
        return None
    age = float(age)
    if age < 6:
        return '0-6_months'
    elif age < 12:
        return '6-12_months'
    elif age < 18:
        return '12-18_months'
    elif age < 24:
        return '18-24_months'
    elif age < 30:
        return '24-30_months'
    else:
        return '30+_months'

df['age_group'] = df['Age_at_Collection'].apply(assign_age_group)
df['Run'] = df['Run'].astype(str)
df['label'] = df['label'].astype(int)

print(f"\nAge group distribution:")
for ag in sorted(df['age_group'].value_counts().index):
    print(f"  {ag}: {df[df['age_group'] == ag].shape[0]}")

# ============================================================
# 5. Generate age group files (NO deduplication)
# Note: Each sample in Gadir is unique (confirmed via BioSample)
# ============================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("\nGenerating age group files...")

age_groups = sorted(df['age_group'].dropna().unique())
summary = []

for age_group in age_groups:
    group_df = df[df['age_group'] == age_group][['Run', 'label']].copy()
    group_df = group_df.rename(columns={'Run': 'sid'})
    group_df = group_df.sort_values('sid').reset_index(drop=True)
    
    output_file = OUTPUT_DIR / f'gadir_preprocessed_{age_group}.csv'
    group_df.to_csv(output_file, index=False)
    
    label_dist = group_df['label'].value_counts().to_dict()
    summary.append({
        'age_group': age_group,
        'samples': len(group_df),
        'control': label_dist.get(0, 0),
        'food_allergy': label_dist.get(1, 0)
    })
    print(f"  {age_group}: {len(group_df)} samples | Control: {label_dist.get(0, 0)}, FA: {label_dist.get(1, 0)}")

# Create combined file
all_df = df[df['age_group'].notna()][['Run', 'label']].copy()
all_df = all_df.rename(columns={'Run': 'sid'})
all_df.to_csv(OUTPUT_DIR / 'gadir_all_months.csv', index=False)

# ============================================================
# 6. Summary
# ============================================================
summary_df = pd.DataFrame(summary)
print("\n" + "="*50)
print("PREPROCESSING COMPLETE")
print("="*50)
print(f"Total samples: {summary_df['samples'].sum()}")
print(f"Total control: {summary_df['control'].sum()}")
print(f"Total food allergy: {summary_df['food_allergy'].sum()}")
