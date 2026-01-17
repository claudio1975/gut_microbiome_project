# Gut Microbiome Experiments Tracking

This repository runs machine-learning evaluation and grid-search pipelines for gut microbiome datasets with Trackio-based experiment tracking.

Runs can be executed locally (safe, no publishing) or published to a Hugging Face Space with explicit user confirmation.



## IMPORTANT: CONFIG FILE IS REQUIRED

This project does NOT run safely with defaults.

You MUST pass a configuration file using the –config argument.
Running without a config may:
* load unintended defaults
* trigger Hugging Face dataset downloads
* fail during dataset resolution



## INSTALLATION
1.  Python 3.10+
2.	Install dependencies: pip install -e .
3.	Ensure trackio is installed and available on PATH


## BASIC USAGE

### Local run (no publishing)

Runs grid search + final evaluation and logs everything locally only.

**Example**

python main.py -–config configs/diabimmune.yaml


View local results:
trackio show –project "<project_id>"


### Publish results to Hugging Face Space

* Pushes metrics, tables, and media to the configured Space.

python main.py –-config configs/diabimmune.yaml –-publish_to_main

**Before publishing**:
* You will be prompted for confirmation
* Publishing is aborted unless you explicitly type “yes”


### Evaluation-only mode (no grid search)

* Skips hyperparameter tuning and runs evaluation only.

* python main.py -–config configs/diabimmune.yaml -–eval_only

**Useful for**:
* quick validation
* debugging datasets
* faster iteration



## COMMAND-LINE ARGUMENTS

* --config (REQUIRED)

Path to the YAML configuration file.

--config configs/diabimmune.yaml

**Controls**:
* dataset location
* evaluation parameters
* tracking configuration
* output directories



* --dataset_path (optional)

Override the dataset path defined in the config.

–-dataset_path data_preprocessing/datasets/diabimmune/Month_3.csv

**Use this to**:
* switch datasets without editing YAML
* run ad-hoc experiments



* –-project (optional)

Override the Trackio project name.

–-project diabimmune-month-3-test

**Useful for**:
* temporary experiments
* separating local test runs


* –-space_id (optional)

Override the Hugging Face Space ID defined in the config.

–-space_id username/space-name

**Note:** Only relevant when used with –publish_to_main.


* –-publish_to_main

Enable publishing to the Hugging Face Space.

–-publish_to_main

**Behavior**:
* prompts for explicit confirmation
* updates metrics, tables, and media in the Space
* uses stable run names (no duplication)


* –-eval_only

Run evaluation only (skip grid search and hyperparameter tuning).

–-eval_only

**Behavior**:
* faster execution
* no parameter optimization
* metrics, tables, and media are still logged


## DATASET CONFIG NOTES

The primary dataset is controlled by:

data:
dataset_path: data_preprocessing/datasets/diabimmune/Month_3.csv

If Hugging Face dataset fields are present, they MUST match:

hugging_face:
csv_filename: Month_3.csv

If dataset_path and csv_filename do not match, you may see:
* missing output folders
* incorrect dataset selection
* confusing or duplicated results




## TRACKING BEHAVIOR SUMMARY

### Local mode:
* Trackio enabled
* No Hugging Face publishing

### Publish mode:
* Trackio enabled
* Hugging Face Space updated

**Notes:**
* Media and tables are logged at a fixed step
* Metrics update without duplicating runs
* Run names are stable and dataset-scoped


**When in doubt: run locally first.**
