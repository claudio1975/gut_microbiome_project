# ML-Based Prediction of Food Allergy Development From Gut Microbiome Data

## Project Overview

This project aims to develop a robust machine learning classifier that can predict the development of food allergies by analyzing an individual's gut microbiome data. By leveraging the latest advancements in microbiome foundation models, we seek to transform the landscape of food allergy prediction and prevention.

The core goal is to build a **binary classifier** that can distinguish between "healthy" and "allergic" subjects based on their gut microbiota composition. Future iterations will explore extending this model to predict the risk of developing specific types of food allergies.

## Motivation: The Microbiome-Allergy Link

Accumulating scientific evidence highlights the critical role of the gut microbiota in shaping immune tolerance to dietary antigens. Key findings that motivate this research include:

*   **Reduced Diversity:** Allergic children consistently exhibit reduced microbial diversity compared to healthy controls.
*   **Taxa Shifts:** There is a notable depletion of protective taxa (e.g., *Bifidobacterium*, *Faecalibacterium*, butyrate-producing *Clostridia*) and an enrichment of pro-inflammatory bacteria (e.g., *Escherichia-Shigella*, *Ruminococcus gnavus*).
*   **Predictive Potential:** These microbial shifts are often detectable months before the clinical manifestation of food allergies, suggesting a strong predictive potential for early intervention.

## Methodology

Our approach is built on a modern, two-stage machine learning architecture designed to handle the complexity and high dimensionality of microbiome data:

1.  **Microbiome Embedding Model:** We utilize a recently developed **foundation model** for microbiome data. This model serves as a backbone, extracting rich, low-dimensional, and meaningful representations (embeddings) from raw 16S rRNA or shotgun metagenomics data.
2.  **Classifier Head:** The extracted embeddings are then fed into a simpler machine learning model, such as **Logistic Regression** or a similar classifier, to perform the final binary prediction.

### Evaluation Strategy

To ensure the model is robust and generalizable, we employ a rigorous evaluation strategy:

*   **Standard Metrics:** Performance is assessed using standard classification metrics (e.g., AUC, F1-score, Accuracy) with cross-validation.
*   **Cross-Dataset Validation:** We implement a **leave-one-dataset-out** validation strategy to test the model's ability to generalize across different cohorts, sequencing protocols, and study conditions, which is crucial for real-world applicability.

## Repository Structure

The project follows a modular structure to separate concerns and facilitate collaboration:

| Directory/File | Purpose |
| :--- | :--- |
| `data_preprocessing/` | Contains datasets, preprocessing scripts, and generated embeddings. |
| `data_preprocessing/datasets/` | Raw dataset CSV files (sample IDs and labels). |
| `data_preprocessing/mapref_data/` | Parquet files mapping sample IDs to OTU IDs to DNA sequences. |
| `data_preprocessing/dna_sequences/` | Generated DNA sequence CSVs (one per sample). |
| `data_preprocessing/dna_embeddings/` | Generated ProkBERT embeddings (H5 format). |
| `data_preprocessing/microbiome_embeddings/` | Generated MicrobiomeTransformer embeddings (H5 format). |
| `huggingface_datasets/` | HuggingFace datasets with metadata and processed embeddings. |
| `modules/` | Core model classes: `MicrobiomeTransformer` and `SKClassifier`. |
| `utils/` | Helper utilities for data preparation and evaluation. |
| `data_loading.py` | Unified data loading pipeline with automatic artifact generation. |
| `generate_embeddings.py` | Standalone script for batch embedding generation. |
| `create_unified_embeddings.py` | Script to merge all embeddings into unified files per dataset. |
| `example_unified_embeddings.py` | Examples showing how to use unified embeddings. |
| `main.py` | Main execution script for training and evaluation. |
| `config.yaml` | Centralized configuration file for all parameters. |

## Custom Sample Subsets

You can create custom sample subsets without recomputing embeddings. Simply:

1. Create a CSV file with your desired samples (must have `sid` and `label` columns)
2. Place it in `huggingface_datasets/{DatasetName}/metadata/your_custom_file.csv`
3. Update `config.yaml` to point to your custom file:
   ```yaml
   data:
     use_unified_embeddings: true
     hugging_face:
       dataset_name: "Diabimmune"
       csv_filename: "your_custom_file.csv"
   ```
4. Run `python main.py` as usual

**Note:** If auto-downloading datasets from HuggingFace, unified embeddings are already included. Otherwise, run `python create_unified_embeddings.py --dataset all` once to merge embeddings.

## Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/AI-For-Food-Allergies/gut_microbiome_project.git
cd gut_microbiome_project

# Using pip
pip install -e .

# Or using uv (recommended)
uv sync
```

### 2. Data Setup

#### Download Required Files

You need three types of files to run the pipeline:

1.  **Datasets**: Sample metadata CSVs (sample IDs and labels)
    - Download from: [Google Drive - Datasets](https://drive.google.com/drive/folders/1-MM3xOOhaEgILnD-D9IiLBrSBQOlz6QP?usp=sharing)
    - Available datasets: Tanaka, Goldberg, Diabimmune, Gadir
    - Place in: `data_preprocessing/datasets/<dataset_name>/`

2.  **Model Checkpoint**: Pre-trained MicrobiomeTransformer model
    - Download from: [Google Drive - Model Checkpoint](https://drive.google.com/file/d/1hykscEI4CbQm5ZzPOy9-o4HYjdwHL4u0/view?usp=sharing)
    - File: `checkpoint_epoch_0_final_epoch3_conf00.pt`
    - Place in: `data/`

3.  **Parquet Mapping Files**: OTU-to-DNA sequence mappings
    - Download from: [Google Drive - Parquet Files](https://drive.google.com/drive/folders/1d33c5JtZREoDWRAu14o-fDXOpuriuyQC?usp=sharing)
    - Files: `samples-otus-97.parquet`, `otus_97_to_dna.parquet`
    - Place in: `data_preprocessing/mapref_data/`

#### Option: Download Pre-generated Embeddings (Skip Embedding Pipeline)

Just execute `python main.py` as suggested in the lower sections. The script is updated to automatically pull the datasets from HuggingFace and place them in appropriate folders.
When running the script, make sure to set the `csv_filename` argument to a valid csv file for that dataset (an error will be thrown otherwise).

**Note:** With preprocessed embeddings, you can skip directly to [Step 2: Train and Evaluate](#step-2-train-and-evaluate) without running the embedding generation pipeline.

#### Directory Structure

After setup, your directory should look like this:

```text
gut_microbiome_project/
├── data/
│   └── checkpoint_epoch_0_final_epoch3_conf00.pt
├── data_preprocessing/
│   ├── datasets/
│   │   ├── diabimmune/
│   │   │   ├── Month_1.csv
│   │   │   ├── Month_2.csv
│   │   │   └── ...
│   │   ├── goldberg/
│   │   │   ├── T1.csv
│   │   │   └── ...
│   │   └── ...
│   ├── mapref_data/
│   │   ├── samples-otus-97.parquet
│   │   └── otus_97_to_dna.parquet
│   ├── dna_sequences/       # Generated automatically
│   ├── dna_embeddings/      # Generated automatically
│   └── microbiome_embeddings/  # Generated automatically
```

### 3. Configuration

Edit `config.yaml` to configure your experiment:

```yaml
data:
  # Point to your dataset
  dataset_path: "data_preprocessing/datasets/diabimmune/Month_2.csv"

  # Model checkpoint (downloaded above)
  microbiome_transformer_checkpoint: "data/checkpoint_epoch_0_final_epoch3_conf00.pt"

  # Device: "cpu", "cuda", or "mps"
  device: "cpu"

model:
  # Classifier type: "logreg", "rf", "svm", or "mlp"
  classifier: "logreg"
  use_scaler: true

evaluation:
  cv_folds: 5
  results_output_dir: "eval_results"
```

See `config.yaml` for all available options including hyperparameter grids.

## Usage

### Step 1: Generate Embeddings

The pipeline can generate embeddings automatically or you can pre-generate them for better control and performance.

#### Option A: Automatic Download (Quick Start)

When you run `main.py`, the pipeline automatically downloads the embeddings for the selected dataset.

```bash
python main.py
```

#### Option B: Pre-generate Embeddings (Recommended)

For more control and to process multiple datasets efficiently, use the standalone embedding generation script.

**1. Activate your environment:**

```bash
# Using uv (recommended)
source .venv/bin/activate

# Or if using pip
source venv/bin/activate
```

**2. Configure the script:**

Edit `generate_embeddings.py` and update the dataset path:

```python
# Lines near the bottom of generate_embeddings.py
BASE_OUTPUT_DIR = Path("data_preprocessing")
DATASET_DIR = Path("data_preprocessing/datasets/diabimmune")  # Change this!
```

**3. Run the script:**

```bash
python generate_embeddings.py
```

**What gets generated:**

1. **DNA sequences** (`dna_sequences/`) - CSV files with OTU DNA sequences per sample
2. **DNA embeddings** (`dna_embeddings/`) - ProkBERT embeddings per OTU (H5 format)
3. **Microbiome embeddings** (`microbiome_embeddings/`) - Aggregated embeddings per sample (H5 format)

**Important Notes:**

- ⏸️ **Resume capability**: You can safely interrupt (Ctrl+C) and restart. Already generated files are kept.
- ⚠️ **Incomplete files**: If interrupted during embedding generation (while processing samples), delete the incomplete `.h5` file before restarting.
- 💡 **Performance tip**: Use GPU by setting `DEVICE = "cuda"` (or `"mps"` for Apple Silicon) in `generate_embeddings.py` for 5-10x speedup.
- 📖 **Detailed guide**: See `README_EMBEDDINGS.md` for complete instructions, troubleshooting, and file structure details.

### Step 2: Train and Evaluate

Since the `main.py` script is now equipped to download automatically the datasets from HuggingFace, here is what you need to know about the config:

```yaml
# Data path
data:
  # Other params like before
  device: "cpu"  # cpu, cuda, or mps
  hugging_face:
    download_path: "huggingface_datasets"
    dataset_name: "Tanaka" # e.g., "Diabimmune", "Goldberg", "Gadir"
    base_repo_url: "https://huggingface.co/datasets/hugging-science"
    pull_from_huggingface: true
    csv_filename: "month_2.csv" # e.g., "Month_2.csv" for Diabimmune, "T1.csv" for Goldberg, "gadir_all_months.csv" for Gadir

```
- `download_path` - Folder where the datasets will be downloaded.
- `dataset_name` - Dataset name you wish to pull from HF.
- `base_repo_url` - Base repo for the uploaded datasets.
- `pull_from_huggingface` - Whether you wish to use local files or pull from HF (recommended to pull from HF)
- `csv_filename` - The CSV you wish to work with. Please note that HF pull will download the entire dataset at once.

If the datasets are pulled once, they are skipped the next time. You can change the `csv_filename` if you wish to work with other months.
Once the datasets are pulled from HuggingFace, they should look like below:

```bash
./huggingface_datasets/
├── Diabimmune
│   ├── metadata
│   ├── processed
│   └── README.md
├── Gadir
│   ├── metadata
│   ├── processed
│   └── README.md
├── Goldberg
│   ├── metadata
│   ├── processed
│   └── README.md
└── Tanaka
    ├── metadata
    ├── processed
    └── README.md
```
- `metadata/` - Folder contains the labels CSV files.
- `processed/` - Folder contains the `dna_embeddings`, `dna_sequences` and `microbiome_embeddings` as usual.


## Running Experiments

Run the main pipeline using `main.py` with Hydra config management. All behavior is controlled via CLI arguments and config files - **no code editing required**.

### Basic Usage

#### Select Dataset
```bash
# Run with specific dataset (default: tanaka)
python main.py datasets=gadir

# Or use other datasets
python main.py datasets=diabimmune
python main.py datasets=goldberg
python main.py datasets=tanaka
```

#### Enable Experiment Tracking
```bash
# Enable tracking to Hugging Face spaces
python main.py datasets=gadir tracking.enabled=True

# Tracking project and space_id are auto-generated from dataset name
# Override if needed:
python main.py datasets=gadir tracking.enabled=True tracking.space_id="custom/space-id"
```

### Evaluation Modes

#### Mode 1: Simple Evaluation (No Hyperparameter Tuning)

Evaluate classifiers with default parameters from `configs/model/model.yaml`:
```bash
# Single classifier
python main.py datasets=gadir model.classifier=[logreg]

# Multiple classifiers
python main.py datasets=gadir model.classifier=[logreg,rf,svm,mlp]
```

#### Mode 2: Grid Search with Unbiased Evaluation (Recommended)

Perform hyperparameter tuning with proper two-stage evaluation:
```bash
# Run grid search for all classifiers
python main.py datasets=gadir model.classifier=[logreg,rf,svm,mlp]
```

**Two-stage approach:**
1. **Inner CV (Grid Search)**: Find best hyperparameters using 5-fold CV
2. **Outer CV (Final Eval)**: Evaluate best model on independent 5-fold CV

This prevents optimistic bias from hyperparameter leakage.

#### Mode 3: Evaluation Only (No Training)

Load existing models and evaluate:
```bash
python main.py datasets=gadir mode.eval_only=True
```

#### Mode 4: Train/Test Split Evaluation

Instead of evaluating on the full dataset with cross-validation, you can hold out a test set. Grid search (if any) runs on the training set only, and final performance is measured on the held-out test set.

**Auto split** — a single dataset is automatically split into train and test using a stratified split that preserves class balance:

```bash
# Default 80/20 split with grid search on train set
python main.py datasets=tanaka evaluation.split.enabled=true

# Custom split ratio and seed
python main.py datasets=tanaka evaluation.split.enabled=true \
    evaluation.split.auto_split.test_size=0.3 \
    evaluation.split.auto_split.random_state=42

# Eval only (no grid search) — useful when hyperparams are already known
python main.py datasets=tanaka evaluation.split.enabled=true mode.eval_only=true
```

**Pre-split** — provide two separate CSV files for train and test:

```bash
python main.py evaluation.split.enabled=true \
    evaluation.split.pre_split.train_dataset_path=data/train.csv \
    evaluation.split.pre_split.test_dataset_path=data/test.csv
```

When pre-split paths are provided they take priority over auto-split.

**Behavior summary:**

| `split.enabled` | `eval_only` | What happens |
| :--- | :--- | :--- |
| `false` | `false` | Default: GridSearchCV + final CV on full dataset |
| `false` | `true` | Simple k-fold CV on full dataset |
| `true` | `false` | GridSearchCV on train set + holdout test evaluation |
| `true` | `true` | Fit on train set + holdout test evaluation (no grid search) |

**Config reference** (`configs/base.yaml`):

```yaml
evaluation:
  split:
    enabled: false              # set to true to activate
    auto_split:
      test_size: 0.2            # fraction held out for testing
      random_state: 20260101    # seed for reproducible splits
    pre_split:
      train_dataset_path: null  # path to train CSV
      test_dataset_path: null   # path to test CSV
```

### Advanced Usage

#### Override Hyperparameter Grids

Modify search space via CLI:
```bash
# Override specific classifier params
python main.py datasets=gadir \
  model.param_grids.logreg.C=[0.01,0.1,1] \
  model.param_grids.rf.n_estimators=[100,200]
```

#### Customize Evaluation Settings
```bash
# Change CV folds
python main.py datasets=gadir evaluation.cv_folds=10

# Change scoring metric
python main.py datasets=gadir evaluation.grid_search_scoring="f1"

# Set random seeds for reproducibility
python main.py datasets=gadir \
  evaluation.grid_search_random_state=42 \
  evaluation.final_eval_random_state=123
```

#### View Current Config

Verify your config without running experiments:
```bash
# Print full resolved config
python main.py datasets=gadir --cfg job

# Print only specific sections
python main.py datasets=gadir --cfg job --package data
```

### Common Workflows

**Quick experiment on new dataset:**
```bash
python main.py datasets=gadir model.classifier=[logreg,rf] tracking.enabled=True
```

**Full evaluation with tracking:**
```bash
python main.py datasets=diabimmune \
  model.classifier=[logreg,rf,svm,mlp] \
  tracking.enabled=True \
  tracking.tags=[diabimmune,production,full-grid]
```

**Debug with single classifier:**
```bash
python main.py datasets=tanaka \
  model.classifier=[logreg] \
  evaluation.cv_folds=2
```

### Config Structure
```
configs/
├── base.yaml           # Base settings, tracking templates
├── data/
│   └── data.yaml      # Data paths, embedding settings
├── datasets/          # Dataset-specific configs
│   ├── diabimmune.yaml
│   ├── gadir.yaml
│   ├── goldberg.yaml
│   └── tanaka.yaml
└── model/
    └── model.yaml     # Classifier configs and param grids
```

**No code editing needed** - control everything via config files and CLI overrides.


### Step 3: View Results

Results are saved to `eval_results/<dataset_name>/<timestamp>/`:
Dataset names are automatically infered from the HF pull config setup.
```text
eval_results/
└── diabimmune/
    └── Month_2/
        └── 2024-12-15_14-30-45/
            ├── Logistic_Regression/
            │   ├── classification_report.txt
            │   ├── confusion_matrix.png
            │   ├── confusion_matrix_normalized.png
            │   ├── roc_curve.png
            │   └── predictions.csv
            ├── Random_Forest/
            │   └── ...
            ├── combined_report.txt        # Compare all classifiers
            ├── comparison_roc_curves.png  # ROC curves on same plot
            └── best_params_summary.json   # Best hyperparameters
```

Each classifier folder contains:
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: True vs predicted labels (raw and normalized)
- **ROC Curve**: AUC and true/false positive rates
- **Predictions**: Sample-level predictions and probabilities

## Advanced Usage

### Using the Data Loading Pipeline Programmatically

```python
from data_loading import load_dataset_df
from utils.data_utils import load_config, prepare_data

# Load config
config = load_config()

# Load dataset (auto-generates embeddings if needed)
dataset_df = load_dataset_df(config)
print(dataset_df.head())
# Output: DataFrame with columns [sid, label, embedding]

# Prepare for sklearn
X, y = prepare_data(dataset_df)
# X: numpy array of shape (n_samples, embedding_dim)
# y: numpy array of labels
```

### Using the Classifier Programmatically

```python
from modules.classifier import SKClassifier

# Initialize classifier
classifier = SKClassifier("logreg", config)

# Simple evaluation
metrics = classifier.evaluate_model(X, y, cv=5)
print(metrics.classification_report)

# Grid search
param_grid = {"C": [0.1, 1, 10], "penalty": ["l2"]}
metrics = classifier.grid_search_with_final_eval(
    X, y,
    param_grid=param_grid,
    grid_search_cv=5,
    final_eval_cv=5
)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size_embedding` in `config.yaml`
2. **Slow Embedding Generation**: First run is always slow. Consider using GPU by setting `device: "cuda"` or `device: "mps"`
3. **Missing Dependencies**: Run `pip install -e .` or `uv sync`
4. **Parquet File Errors**: Ensure you downloaded both parquet files from Figshare

### Performance Tips

- Use GPU/MPS for faster embedding generation
- Pre-generate embeddings for multiple datasets in batch
- Embeddings are cached - only generated once per dataset
- Start with a small dataset to verify pipeline works

## 🤝 Contributing

We welcome contributions from researchers, data scientists, and developers!

Please refer to our dedicated **[Contributing Guide](Contributing.md)** for detailed instructions on:

*   Setting up your development environment
*   Understanding the codebase structure
*   Guidelines for submitting pull requests and writing clean code

**Want to get involved?** Join the [huggingscience Discord server](https://discord.com/invite/VYkdEVjJ5J) and communicate your interest!

Thank you for helping us advance the prediction of food allergies!
