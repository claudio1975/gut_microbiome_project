# DIABIMMUNE Microbiome Data Preprocessing Report

## Overview

This document describes the preprocessing pipeline for the **DIABIMMUNE** microbiome dataset, designed for **early prediction of food allergies** (milk, egg, peanut) from gut microbiome samples.

### Goal

Create a longitudinally-labeled dataset where samples from patients who **eventually develop** a food allergy are labeled as "allergic" — even if the sample was collected before the allergy manifested. This enables machine learning models to learn early biomarkers predictive of future allergy development.

### Output Files

| File | Description |
|------|-------------|
| `diabimmune_longitudinal_labels.csv` | Full dataset with longitudinal labels (785 samples) |
| `preprocessed_diabimmune_longitudinal/Month_X.csv` | Per-month files for training/evaluation |

---

## Data Sources

### 1. MicrobeAtlas SRS Data (`download_samples_tsv-2.tsv`)

Public microbiome sample identifiers (SRS accessions) from MicrobeAtlas.

| Column | Description |
|--------|-------------|
| `SRS` | Sequence Read Archive Sample ID |
| `name` | Sample name |
| `projects` | Associated project IDs |

**785 unique SRS samples**

### 2. DIABIMMUNE Metadata (`metadata.csv`)

Internal study metadata with patient information and allergy outcomes.

| Key Columns | Description |
|-------------|-------------|
| `subjectID` | Patient identifier (e.g., P018832, T022883) |
| `SampleID` | Physical sample identifier |
| `collection_month` | Age in months at sample collection |
| `allergy_milk` | Milk allergy status (True/False) |
| `allergy_egg` | Egg allergy status (True/False) |
| `allergy_peanut` | Peanut allergy status (True/False) |

**1,946 metadata rows** (multiple timepoints per patient)

---

## Key Challenge: Linking SRS to Metadata

The two data sources share **no common key**:

```
SRS Data:     SRS1719087, SRS1719088, ...  (no subjectID, no collection_month)
Metadata:     subjectID, SampleID, collection_month, allergies  (no SRS)
```

### Why This Matters

Without proper linking, we risk:
- **Data leakage**: Same sample appearing in multiple months
- **Incorrect labels**: Assigning wrong allergy status to samples
- **Cartesian products**: One SRS mapping to all timepoints of a patient

---

## Preprocessing Steps

### Step 1: Load Source Data

Load both data sources and inspect their structure.

```python
df_srs = pd.read_csv("download_samples_tsv-2.tsv", sep="\t")  # 785 SRS
df_meta = pd.read_csv("metadata.csv")  # 1,946 rows
```

### Step 2: Link SRS to Patient IDs via ENA API

**Solution**: Query the European Nucleotide Archive (ENA) API to extract `host_subject_id` from each SRS accession's metadata.

```
ENA API: https://www.ebi.ac.uk/ena/browser/api/xml/{SRS_ID}
```

**Example Response** (parsed):
```xml
<SAMPLE_ATTRIBUTE>
  <TAG>host_subject_id</TAG>
  <VALUE>P018832</VALUE>
</SAMPLE_ATTRIBUTE>
```

**Result**: `srs_to_host_subject_id.csv` — Maps each SRS to its patient ID.

### Step 3: Extract Collection Month from ENA

**Problem**: Merging only on `subjectID` creates a Cartesian product because one patient has multiple samples across different months.

```
Patient P018832 has samples at months: 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, ...
Merging only on subjectID → SRS1719087 appears in ALL these months! (data leakage)
```

**Solution**: The ENA metadata includes `host_age` (age in days at sample collection). Convert to months:

```python
collection_month = round(host_age_days / 30.44)
```

**Example**:
| SRS | host_age_days | collection_month | host_subject_id |
|-----|---------------|------------------|-----------------|
| SRS1719087 | 686 | 23 | P018832 |
| SRS1719088 | 173 | 6 | P017743 |
| SRS1719090 | 229 | 8 | T022883 |

**Result**: `srs_to_collection_month.csv` — Maps each SRS to its exact collection month.

### Step 4: Correct Merge on (subjectID, collection_month)

Perform the merge using **both keys**:

```python
df_merged = df_srs_month.merge(
    df_meta, 
    on=['subjectID', 'collection_month'], 
    how='left'
)
```

This ensures:
- ✅ Each SRS maps to exactly **one** timepoint
- ✅ No data leakage between timepoints
- ✅ Allergy labels correspond to the correct sample

**Result**: 826 rows (some duplicates due to multiple physical samples per SRS)

### Step 5: Handle Duplicate Rows

Some patients have multiple physical samples (`SampleID`s) collected in the same month, creating duplicate rows for the same SRS after merging.

**Example**: SRS1719152 has 3 rows (SampleIDs: 3104057, 3104056, 3104053)

**Solution**: Aggregate by SRS, taking the maximum allergy value:

```python
def aggregate_allergy_data(group):
    # If ANY sample shows allergy, mark as allergic
    for col in allergy_cols:
        result[col] = 'True' if any(is_allergic(v) for v in group[col]) else 'False'
    return result

df_final = df_merged.groupby('SRS').apply(aggregate_allergy_data)
```

**Result**: 785 rows (one per SRS, no duplicates)

### Step 6: Longitudinal Labeling for Early Prediction

**This is the critical step for our prediction task.**

For early allergy prediction, we want to identify patients who will **eventually develop** an allergy, even from samples collected before the allergy manifests.

#### Strategy

1. For each patient, compute their **maximum allergy status** across ALL timepoints
2. Apply this label to **ALL their samples**

```python
# Step 1: Get max allergy status per patient
patient_max_allergy = df_final.groupby('subjectID').agg({
    'allergy_milk': lambda x: any(is_allergic(v) for v in x),
    'allergy_egg': lambda x: any(is_allergic(v) for v in x),
    'allergy_peanut': lambda x: any(is_allergic(v) for v in x)
})

# Step 2: Apply to all samples
df_longitudinal = df_final.merge(patient_max_allergy, on='subjectID')
```

#### Example

| Patient | Month 4 Status | Month 10 Status | Label Applied |
|---------|----------------|-----------------|---------------|
| T022883 | Not allergic   | Milk + Egg allergy | **Allergic** (all samples) |

Patient T022883's samples at months 4, 8, 10, 13, and 16 are **all labeled as allergic** because they eventually develop milk and egg allergies.

#### Label Definitions

**Binary Label** (`label`):
- `0` = Non-allergic (no food allergies ever)
- `1` = Allergic (develops at least one food allergy)

**Allergen Class** (`allergen_class`):
- `0` = Non-allergic
- `1` = Milk allergy only
- `2` = Egg allergy only
- `3` = Peanut allergy only
- `4` = Multiple food allergies

### Step 7: Verification

Verify the longitudinal labeling is correct by checking example patients:

```
Patient T022883 (develops milk + egg allergy):
    SRS1735472 @ month 4  → label=1 ✓
    SRS1719090 @ month 8  → label=1 ✓
    SRS1719531 @ month 10 → label=1 ✓
    SRS1719179 @ month 13 → label=1 ✓
    SRS1719200 @ month 16 → label=1 ✓
```

All samples from this patient are correctly labeled as allergic.

### Step 8: Save Output Files

Save the main longitudinally-labeled dataset:

```python
df_longitudinal.to_csv("diabimmune_longitudinal_labels.csv", index=False)
```

### Step 9: Create Per-Month Files

Generate individual CSV files for each collection month:

```
preprocessed_diabimmune_longitudinal/
├── Month_1.csv   (20 samples: 9 allergic, 11 non-allergic)
├── Month_2.csv   (17 samples: 11 allergic, 6 non-allergic)
├── Month_3.csv   (8 samples: 2 allergic, 6 non-allergic)
├── Month_4.csv   (39 samples: 24 allergic, 15 non-allergic)
├── ...
└── Month_38.csv  (1 sample: 0 allergic, 1 non-allergic)
```

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total samples | 785 |
| Unique patients | 212 |
| Collection months range | 1 - 38 |

### Label Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| Non-allergic (0) | 527 | 67.1% |
| Allergic (1) | 258 | 32.9% |

### Allergen Class Distribution

| Class | Description | Count |
|-------|-------------|-------|
| 0 | Non-allergic | 527 |
| 1 | Milk allergy only | 101 |
| 2 | Egg allergy only | 63 |
| 3 | Peanut allergy only | 7 |
| 4 | Multiple food allergies | 87 |

---

## Data Quality Checks

### ✅ No Sample Leakage

Each SRS appears in exactly **one** month:

```
Sample-month cross-check: All 785 samples appear in exactly one month
```

### ✅ No Duplicate Samples

Each SRS has exactly **one** row in the final dataset:

```
Unique SRS count: 785
Total rows: 785
```

### ✅ Correct Longitudinal Labeling

Patients who develop allergies have all their samples labeled accordingly, enabling early prediction from samples collected before allergy onset.

---

## Usage Notes

### For Training

Use `diabimmune_longitudinal_labels.csv` or the per-month files for training classifiers.

### For Evaluation

When evaluating, consider:
- **Temporal validation**: Train on early months, test on later months
- **Patient-level splits**: Ensure same patient doesn't appear in both train and test sets
- **Class imbalance**: ~33% allergic vs ~67% non-allergic

### Columns for ML

| Column | Use |
|--------|-----|
| `SRS` / `sid` | Sample identifier (for embedding lookup) |
| `subjectID` / `patient_id` | Patient identifier (for patient-level splits) |
| `collection_month` | Timepoint (for temporal analysis) |
| `label` | Binary classification target |
| `allergen_class` | Multi-class classification target |

---

## Files Generated

```
preprocessing_scripts/
├── srs_to_host_subject_id.csv      # SRS → patient ID mapping
├── srs_to_collection_month.csv     # SRS → collection month mapping
├── diabimmune_longitudinal_labels.csv  # Final labeled dataset
└── preprocessed_diabimmune_longitudinal/
    ├── Month_1.csv
    ├── Month_2.csv
    ├── ...
    └── Month_38.csv
```

---

*Report relative to `preprocessing.ipynb`*
