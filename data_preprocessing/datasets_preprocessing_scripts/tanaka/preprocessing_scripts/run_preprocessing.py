#!/usr/bin/env python3
"""
Tanaka Dataset Preprocessing Script

Generates metadata CSVs from raw source data:
- final_trimmed_db.csv: sid -> patient_id, age mapping
- fix099_Supp.csv: patient_id -> allergen details

Output: month_X.csv files with columns [sid, patient_id, label, allergen_class]
- label: 0=healthy, 1=food allergy
- allergen_class: 'healthy', 'egg', 'milk', 'egg_milk_wheat', etc.

Note: Excludes OA (Other Allergy) patients. Only includes samples with allergen info.
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / '../../../../huggingface_datasets/Tanaka/metadata'

# ============================================================
# 1. Load raw data: sid -> patient_id, age mapping
# ============================================================
print("Loading raw data...")
sid_mapping = pd.read_csv(SCRIPT_DIR / 'final_trimmed_db.csv', dtype=str)
sid_mapping.columns = ['idx', 'sid', 'patient_id', 'age_months']
sid_mapping = sid_mapping[['sid', 'patient_id', 'age_months']]
sid_mapping['age_months'] = sid_mapping['age_months'].astype(int)
print(f"Loaded {len(sid_mapping)} total samples from final_trimmed_db.csv")

# ============================================================
# 2. Load raw allergen data (skip complex 5-row header)
# Group: NaN=healthy, FA=food allergy, OA=other allergy
# ============================================================
allergen_raw = pd.read_csv(
    SCRIPT_DIR / 'fix099_Supp.csv', 
    dtype=str,
    skiprows=5,
    header=None,
    names=['ID', 'Group', 'Gender', 'Mode_delivery', 'Milk_feeding', 
           'Antibiotics_m1', 'Antibiotics_m1_2', 'Antibiotics_m2_6', 'Antibiotics_m6_y1',
           'Mother_allergy', 'Egg', 'Milk', 'Soybean', 'Wheat', 'Onset_age', 
           'Atopic_dermatitis', 'Asthmatic', 'Rhinitis']
)
print(f"Loaded {len(allergen_raw)} patients from fix099_Supp.csv")
print(f"Group distribution: {allergen_raw['Group'].value_counts(dropna=False).to_dict()}")

# Filter: Keep NaN (healthy) and FA (food allergy), exclude OA
allergen_df = allergen_raw[(allergen_raw['Group'].isna()) | (allergen_raw['Group'] == 'FA')].copy()
allergen_df['patient_id'] = allergen_df['ID'].str.strip()

# Create label: 0=healthy, 1=food allergy
allergen_df['label'] = (allergen_df['Group'] == 'FA').astype(int)

# ============================================================
# 3. Create allergen_class from specific food allergies
# ============================================================
def get_allergen_class(row):
    """Derive allergen_class from specific food allergens."""
    if row['Group'] != 'FA':
        return 'healthy'
    allergens = []
    if str(row.get('Egg', '')).startswith('+'):
        allergens.append('egg')
    if str(row.get('Milk', '')).startswith('+'):
        allergens.append('milk')
    if str(row.get('Soybean', '')).startswith('+'):
        allergens.append('soybean')
    if str(row.get('Wheat', '')).startswith('+'):
        allergens.append('wheat')
    return '_'.join(sorted(allergens)) if allergens else 'food_allergy_unspecified'

allergen_df['allergen_class'] = allergen_df.apply(get_allergen_class, axis=1)
allergen_df = allergen_df[['patient_id', 'label', 'allergen_class']]

print(f"\nFiltered to {len(allergen_df)} patients (healthy + FA)")
print(f"Label distribution: {allergen_df['label'].value_counts().to_dict()}")
print(f"Allergen classes:\n{allergen_df['allergen_class'].value_counts().to_string()}")

# ============================================================
# 4. Join sid_mapping with allergen data
# ============================================================
# Only keep samples where we have allergen info (inner join)
all_data = sid_mapping.merge(allergen_df, on='patient_id', how='inner')
print(f"\nTotal samples with allergen info: {len(all_data)}")

# ============================================================
# 5. Generate month files from scratch
# ============================================================
MONTHS = [1, 2, 3, 6, 12, 24, 36]

print("\nGenerating month files...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for month in MONTHS:
    # Filter to this month
    month_data = all_data[all_data['age_months'] == month].copy()
    
    if len(month_data) == 0:
        result = pd.DataFrame(columns=['sid', 'patient_id', 'label', 'allergen_class'])
    else:
        # Handle duplicates: randomly select one sample per patient
        month_data = month_data.sample(frac=1, random_state=42).drop_duplicates(subset='patient_id', keep='first')
        result = month_data[['sid', 'patient_id', 'label', 'allergen_class']].sort_values('sid').reset_index(drop=True)
    
    output_file = OUTPUT_DIR / f'month_{month}.csv'
    result.to_csv(output_file, index=False)
    
    if len(result) > 0:
        label_dist = result['label'].value_counts().to_dict()
        print(f"  month_{month}.csv: {len(result)} samples | Labels: {label_dist}")
    else:
        print(f"  month_{month}.csv: 0 samples (no allergen data for this month)")

# ============================================================
# 6. Summary
# ============================================================
print("\n" + "="*50)
print("PREPROCESSING COMPLETE")
print("="*50)
total = sum(len(pd.read_csv(OUTPUT_DIR / f'month_{m}.csv')) for m in MONTHS)
print(f"Total samples across all months: {total}")
