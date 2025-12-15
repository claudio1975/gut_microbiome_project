# Generating Embeddings for Microbiome Data

A guide to generate DNA and microbiome embeddings using the `generate_embeddings.py` script.

## What This Does

Converts microbiome sample data into embeddings (numerical representations) that can be used for machine learning:
1. **DNA embeddings** - Using ProkBERT model on bacterial DNA sequences
2. **Microbiome embeddings** - Using a transformer model to create sample-level representations

---

## Step 0: Set Up Python Environment

This project uses [uv](https://docs.astral.sh/uv/) for fast dependency management.

### Install uv (if not already installed)

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### Create and sync the virtual environment

```bash
# Create virtual environment and install all dependencies
uv sync

```

### Activate the environment

```bash
source .venv/bin/activate
```

You should see `(diabimmune-example)` in your terminal prompt.

**Note:** After running `uv sync`, you only need to activate the environment with `source .venv/bin/activate` in future sessions.

---

## Step 1: Download Required Files

### 1.1 Download Parquet Files (Reference Mappings)

📥 **Link:** https://drive.google.com/drive/folders/1d33c5JtZREoDWRAu14o-fDXOpuriuyQC

Download these two files:
- `samples-otus-97.parquet`
- `otus_97_to_dna.parquet`

📂 **Place them in:** `data_preprocessing/mapref_data/`

```
data_preprocessing/
└── mapref_data/
    ├── samples-otus-97.parquet
    └── otus_97_to_dna.parquet
```

---

### 1.2 Download Model and Data Files

📥 **Link:** https://figshare.com/articles/dataset/Model_and_Data_for_diabimmune_example/30429055?file=58993825

Download these three files:
- `checkpoint_epoch_0_final_epoch3_conf00.pt` (9.3 MB) - Trained model
- `prokbert_embeddings.h5` (640 MB) - Pre-computed OTU embeddings
- `samples-otus.97.metag.minfilter.minCov90.noMulticell.rod2025companion.biom` (8.9 GB) - Sample-OTU mappings

📂 **Place them in:** `data/`

```
data/
├── checkpoint_epoch_0_final_epoch3_conf00.pt
├── prokbert_embeddings.h5
└── samples-otus.97.metag.minfilter.minCov90.noMulticell.rod2025companion.biom
```

---

### 1.3 Download Your Dataset

📥 **Link:** https://drive.google.com/drive/folders/1-MM3xOOhaEgILnD-D9IiLBrSBQOlz6QP?usp=sharing

This contains folders for different datasets (goldberg, diabumine, etc.).

**For Goldberg dataset:**
1. Open the `goldberg` folder
2. Download the CSV files (T1.csv, T2.csv, T3.csv)

📂 **Place them in:** `data_preprocessing/datasets_preprocessing_scripts/goldberg/`

```
data_preprocessing/
└── datasets_preprocessing_scripts/
    └── goldberg/
        ├── T1.csv
        ├── T2.csv
        └── T3.csv
```

**CSV Format:**
```csv
Trial,sid,label
T1,ERS4516182,1
T1,ERS4516184,0
```
- `Trial` - Dataset identifier (T1, T2, T3)
- `sid` - Sample ID
- `label` - Classification target (0 or 1)

---

## Step 2: Configure the Script

Open `generate_embeddings.py` and update the **dataset path** at the bottom:

```python
# Line 203-204 in generate_embeddings.py
BASE_OUTPUT_DIR = Path("generated_embeddings")
DATASET_DIR = Path("data_preprocessing/datasets_preprocessing_scripts/goldberg")  # Change this path!
```

**Change `DATASET_DIR`** to point to your dataset folder:
- For Goldberg: `"data_preprocessing/datasets_preprocessing_scripts/goldberg"`
- For other datasets: `"data_preprocessing/datasets_preprocessing_scripts/your_dataset_name"`

---

## Step 3: Run the Script

### Activate your Python environment:
```bash
source .venv/bin/activate
```

### Run the embedding generation:
```bash
python generate_embeddings.py
```

---

## What Happens During Execution

The script processes each CSV file in your dataset folder:

### For each CSV file (e.g., T1.csv, T2.csv, T3.csv):

1. ✅ **Loads sample IDs** from the CSV file
2. ✅ **Generates DNA CSVs** - Extracts DNA sequences for each sample
3. ✅ **Generates DNA embeddings** - Uses ProkBERT model (SLOW - takes hours)
4. ✅ **Generates Microbiome embeddings** - Uses transformer model
5. ✅ **Verifies output** - Sanity check on generated embeddings

### Progress Example:
```
Processing dataset: data_preprocessing/datasets_preprocessing_scripts/goldberg/T1.csv
  Found 261 samples
  Generating DNA CSVs in generated_embeddings/sequences/goldberg/T1...
  Generating DNA embeddings H5 in generated_embeddings/dna_embeddings/goldberg/T1...
  Processing SRS samples: 100%|██████████| 261/261 [2:15:30<00:00, 31.23s/it]
  Generating Microbiome embeddings H5 in generated_embeddings/microbiome_embeddings/goldberg/T1...
  ✓ Data sanity check passed
```

---

## Output Files

All outputs are saved in `generated_embeddings/`:

```
generated_embeddings/
├── sequences/
│   └── goldberg/
│       ├── T1/
│       │   ├── ERS4516182.csv
│       │   └── ... (one CSV per sample)
│       ├── T2/
│       └── T3/
├── dna_embeddings/
│   └── goldberg/
│       ├── T1/
│       │   └── dna_embeddings.h5
│       ├── T2/
│       │   └── dna_embeddings.h5
│       └── T3/
│           └── dna_embeddings.h5
└── microbiome_embeddings/
    └── goldberg/
        ├── T1/
        │   └── microbiome_embeddings.h5
        ├── T2/
        │   └── microbiome_embeddings.h5
        └── T3/
            └── microbiome_embeddings.h5
```

The `.h5` files contain embeddings for each sample that can be used for machine learning.

---

## Important Notes

### ⏸️ Resume Capability
**You can safely interrupt and restart!**

If you press Ctrl+C to stop:
- Already generated files are kept
- When you restart, it skips completed steps
- Only continues with unfinished work

Example:
```
  DNA CSVs already exist at generated_embeddings/sequences/goldberg/T3, skipping generation
  DNA embeddings already exist at generated_embeddings/dna_embeddings/goldberg/T3/dna_embeddings.h5, skipping generation
```

⚠️ **IMPORTANT - Incomplete File Warning:**

The script checks if embedding files (`.h5`) **exist**, but NOT if they are **complete**.

**If you interrupt DURING embedding generation** (while it's actively processing samples), you may have an incomplete `.h5` file. The script will see it exists and skip regeneration.

**Solution:**
If you interrupted during embedding generation (not between datasets), delete the incomplete file:

```bash
# If interrupted during DNA embeddings for T1:
rm generated_embeddings/dna_embeddings/goldberg/T1/dna_embeddings.h5

# If interrupted during Microbiome embeddings for T1:
rm generated_embeddings/microbiome_embeddings/goldberg/T1/microbiome_embeddings.h5

# Then restart the script
python generate_embeddings.py
```

**Safe interruption points:**
- ✅ Between CSV files (T1 → T2 → T3)
- ✅ Between steps (DNA CSVs → DNA embeddings → Microbiome embeddings)
- ❌ UNSAFE: During sample processing (shows progress bar like `100/261 [00:30<01:15]`)


💡 **Tip:** If you have a GPU, edit line 23 in `generate_embeddings.py`:
```python
DEVICE = "cuda"  # Change from "cpu" to "cuda"

```
This can be 5-10x faster!

---

## Troubleshooting

### ❌ Error: "No such file or directory: otus_97_to_dna.parquet"
**Problem:** Parquet files not found

**Solution:** Make sure you downloaded and placed the parquet files in `data_preprocessing/mapref_data/`

---

### ❌ Error: "No CSV files found in generated_embeddings/sequences/..."
**Problem:** DNA CSV generation failed (usually due to missing parquet files)

**Solution:**
1. Delete the `generated_embeddings/` folder
2. Make sure parquet files are in the correct location
3. Run the script again

---

### ❌ Out of Memory Error
**Problem:** Not enough RAM

**Solution:** Reduce batch size in `generate_embeddings.py` (line 22):
```python
BATCH_SIZE = 4  # Change from 8 to 4 or 2
```

---

### ⚠️ Warning: "Some weights of ProkBertModel were not initialized..."
**This is normal!** You can safely ignore this warning.

---

## Complete Directory Structure

After downloading everything, your project should look like this:

```
gut_microbiome_project/
├── generate_embeddings.py          # Main script
├── data_loading.py                 # Helper functions
├── README_EMBEDDINGS.md            # This file
│
├── data/                           # Downloaded model files
│   ├── checkpoint_epoch_0_final_epoch3_conf00.pt
│   ├── prokbert_embeddings.h5
│   └── samples-otus.97.metag.minfilter.minCov90.noMulticell.rod2025companion.biom
│
├── data_preprocessing/
│   ├── mapref_data/               # Downloaded parquet files
│   │   ├── samples-otus-97.parquet
│   │   └── otus_97_to_dna.parquet
│   │
│   └── datasets_preprocessing_scripts/
│       └── goldberg/              # Your dataset CSV files
│           ├── T1.csv
│           ├── T2.csv
│           └── T3.csv
│
└── generated_embeddings/          # Output (created automatically)
    ├── sequences/
    ├── dna_embeddings/
    └── microbiome_embeddings/
```

For more details, see `data_loading.py` and `generate_embeddings.py` source code.