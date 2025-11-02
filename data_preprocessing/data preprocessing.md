# Data preprocessing

### SRS to DNA

### Scope

The `dna_from_srs.py` script extracts DNA sequences for all OTUs (Operational Taxonomic Units) associated with a given SRS (Sequence Read Sample) ID. 

It performs a two-step lookup process:
1. **SRS ID → OTU IDs**: Maps a sample ID to all its associated OTUs using `samples-otus-97.parquet`
2. **OTU IDs → DNA sequences**: Retrieves the DNA sequence for each OTU using `otus_97_to_dna.parquet`

### Prerequisites

Before running the script, you need to download the required parquet files:

1. Download the parquet files from this [public Google Drive folder](https://drive.google.com/drive/folders/1pB1tmbkAr_gIY-Cyec6lOuMWUgk8rLdq?usp=sharing)
   - `samples-otus-97.parquet` (2.22 GB)
   - `otus_97_to_dna.parquet` (75 MB)
2. Place both files in the `mapref_data/` folder

### Usage

```bash
python dna_from_srs.py --srs_id <SRS_ID>
```

### Inputs

- **SRS ID** (via command line argument): The sample identifier to extract sequences for
- **mapref_data/samples-otus-97.parquet**: Mapping between SRS IDs and OTU IDs
- **mapref_data/otus_97_to_dna.parquet**: Mapping between OTU IDs and DNA sequences

### Outputs

- **CSV file**: Saved to `dna_sequences/<SRS_ID>.csv` containing:
  - `otu_id`: The OTU identifier
  - `dna_sequence`: The corresponding DNA sequence

### Example

```bash
python dna_from_srs.py --srs_id SRS7011253
```

This will create `dna_sequences/SRS7011253.csv` with all OTU-DNA mappings for that sample.

### Implementation Details

The script uses PyArrow for efficient parquet file filtering and pandas for data manipulation. Progress is displayed via tqdm for each OTU being processed.

