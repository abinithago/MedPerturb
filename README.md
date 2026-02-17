# MedPerturb
[🌐 Project Website](https://abinithago.github.io/MedPerturb/) • [🤗 Hugging Face Hub](https://huggingface.co/datasets/abinitha/MedPerturb)

MedPerturb is a toolkit for perturbing and evaluating clinical context datasets using large language models (LLMs). It supports various perturbations (gender, stylistic, viewpoint) and evaluation of multiple models (GPT-4, Llama-3-8B, Llama-3-70B, Palmyra-Med) on triage questions.

## File & Directory Overview

```
MedPerturb/
├── code/
│   ├── perturb_data.py         # Script for perturbing clinical contexts
│   ├── evaluate_models.py      # Script for evaluating models on triage questions
│   ├── utils.py                # Utility functions (token loading, validation, logging)
├── case_studies/
│   └── case_study1.ipynb       # Case study 1 in paper (example analysis)
│   └── case_study2.ipynb       # Case study 2 in paper (example analysis)
├── .env                        # Environment variables (tokens for HuggingFace/OpenAI)
├── data.csv                    # Dataset
├── clinician_responses.csv     # Clinician responses
├── model_responses.csv         # Model responses
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── ...                         # (Other files or directories)
```

## Features
- **Perturb clinical text** by gender, style, or viewpoint using LLMs
- **Evaluate LLMs** on triage questions (MANAGE, VISIT, RESOURCE)
- **Supports both HuggingFace and OpenAI models**

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/abinithago/MedPerturb.git
   cd MedPerturb
   ```
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   .\venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables
Create a `.env` file in the project root with the following:
```
# HuggingFace token
HF_TOKEN=your_huggingface_token_here

# OpenAI API token
OPENAI_API_KEY=your_openai_token_here
```

## data.csv

**Purpose**: Contains consensus predictions from clinicians and models, along with gold standard labels.

**Rows**: 6,587  
**Columns**: 28

### Column Structure

#### Base Columns (11)
- `dataset` - Source dataset (askdocs, MeDiSumQA, oncqa, sct, usmle_derm)
- `context_id` - Unique identifier for each clinical context
- `perturbation` - Type of perturbation applied:
  - `baseline` - Original unperturbed text
  - `colorful_tone` - Text with intensified/colorful language
  - `uncertain_tone` - Text with uncertain/hedging language
  - `gender_removal` - Gender references replaced with neutral terms
  - `gender_swap` - Gender references swapped (M→F, F→M)
  - `summary` - Summarized version of the clinical context
- `clinical_context` - The clinical scenario text
- `original_gender` - Original gender of the patient (M, F, or X)
- `age` - Patient age
- `gendered_condition` - Boolean indicating if condition is gender-specific (all False in this file)
- `gold_standard_manage` - Gold standard for manage decision (YES/NO)
- `gold_standard_visit` - Gold standard for visit decision (YES/NO)
- `gold_standard_resource` - Gold standard for resource decision (YES/NO)

#### Clinician Consensus Columns (3)
- `clinician_consensus_manage` - Mode across clinician_1, clinician_2, clinician_3 (0/1)
- `clinician_consensus_visit` - Mode across clinician_1, clinician_2, clinician_3 (0/1)
- `clinician_consensus_resource` - Mode across clinician_1, clinician_2, clinician_3 (0/1)

#### Model Consensus Columns (15)
For each of the 5 models, consensus across 3 seeds (seed0, seed1, seed2):
- `{model}_manage` - Mode across seeds (YES/NO)
- `{model}_visit` - Mode across seeds (YES/NO)
- `{model}_resource` - Mode across seeds (YES/NO)

**Models**: 
- `gpt-4o` - GPT-4O (OpenAI)
- `medgemma` - Google MedGemma-27B-Text-IT
- `deepseek` - DeepSeek-R1-Distill-Qwen-32B
- `qwen` - Qwen2.5-32B-Instruct
- `llama` - Llama-3.3-70B-Instruct

### Key Features
- All rows have `gendered_condition = False` (rows with True were filtered out)
- Clinician consensus available for 916 rows (rows with clinician annotations)
- Model consensus coverage varies by model (gpt-4o has highest coverage: ~6,585 rows)

### Usage
This file is ideal for:
- Comparing model performance against gold standard
- Analyzing consensus predictions
- Evaluating agreement between clinicians and models

---

## llm_data.csv

**Purpose**: Contains individual model predictions for each seed, allowing analysis of model variability across different random seeds.

**Rows**: 6,702  
**Columns**: 71

### Column Structure

#### Base Columns (11)
Same as data.csv base columns, plus:
- `gold_standard_resource_allocation` - Detailed resource allocation text from gold standard

#### Model Prediction Columns (60)
For each of the 5 models × 3 seeds × 4 column types:
- `{model}_{seed}_manage` - Individual seed prediction (YES/NO)
- `{model}_{seed}_visit` - Individual seed prediction (YES/NO)
- `{model}_{seed}_resource` - Individual seed prediction (YES/NO)
- `{model}_{seed}_resource_allocation` - Detailed resource allocation text

**Models**: 
- `gpt-4o` - GPT-4O (OpenAI)
- `medgemma` - Google MedGemma-27B-Text-IT
- `deepseek` - DeepSeek-R1-Distill-Qwen-32B
- `qwen` - Qwen2.5-32B-Instruct
- `llama` - Llama-3.3-70B-Instruct

**Seeds**: seed0, seed1, seed2

### Key Features
- Contains all rows from main.csv (before filtering gendered_condition=True)
- Individual seed predictions allow analysis of:
  - Model consistency across seeds
  - Variability in predictions
  - Seed-specific patterns

### Usage
This file is ideal for:
- Analyzing model variability across seeds
- Understanding prediction consistency
- Creating consensus predictions (mode across seeds)
- Detailed resource allocation analysis

---

## clinician_data.csv

**Purpose**: Contains individual clinician annotations for clinical decision tasks, with up to 3 clinician responses per context.

**Rows**: 943  
**Columns**: 26

### Column Structure

#### Base Columns (11)
Same as data.csv base columns, plus:
- `gold_standard_resource_allocation` - Detailed resource allocation text

#### Clinician Annotation Columns (15)
For each of the 3 clinicians:
- `clinician_{i}_manage` - Individual clinician decision (0/1)
- `clinician_{i}_visit` - Individual clinician decision (0/1)
- `clinician_{i}_resource` - Individual clinician decision (0/1)
- `clinician_{i}_resource_allocation` - Clinician's detailed resource allocation text
- `clinician_{i}_user_id` - Unique identifier for the clinician

**Note**: Not all rows have all 3 clinician annotations. Coverage:
- clinician_1: 943 rows (100%)
- clinician_2: 935 rows (99.2%)
- clinician_3: 921 rows (97.7%)

### Key Features
- Only includes rows that have at least one clinician annotation
- Rows filtered from main.csv (5,759 rows without annotations were dropped)
- Individual annotations allow analysis of:
  - Inter-clinician agreement
  - Clinician variability
  - Consensus calculation

### Usage
This file is ideal for:
- Analyzing inter-clinician agreement
- Understanding clinician decision patterns
- Creating consensus predictions (mode across clinicians)
- Comparing individual clinician responses

---

## Data Relationships

```
main.csv (base data)
    ↓
    ├─→ data.csv (consensus predictions)
    ├─→ model_responses.csv (individual model seeds)
    └─→ clinicial_responses.csv (individual clinician annotations)
```

### Key Differences

| Feature | data.csv | model_responses.csv | clinician_responses.csv |
|---------|----------|--------------|-------------------|
| **Focus** | Consensus | Individual seeds | Individual clinicians |
| **Rows** | 6,587 | 6,702 | 943 |
| **Clinician data** | Consensus only | None | Individual annotations |
| **Model data** | Consensus only | All seeds | None |
| **Filtering** | gendered_condition=False | None | Has clinician annotations |

---

## Tasks

All three files contain predictions for three binary clinical decision tasks:

1. **MANAGE** - Whether the patient should be managed (treated) based on the clinical context
2. **VISIT** - Whether the patient should have an in-person visit
3. **RESOURCE** - Whether additional resources (tests, referrals, etc.) are needed

### Value Formats

- **Gold Standard**: YES/NO (string)
- **Model Predictions**: YES/NO (string)
- **Clinician Annotations**: 0/1 (numeric, where 1=YES, 0=NO)
- **Consensus**: Mode calculation (0/1 for clinicians, YES/NO for models)

---

## Perturbations

All files include the following perturbation types:

- **baseline** - Original, unmodified clinical context
- **colorful_tone** - Text modified with intensified/colorful language
- **uncertain_tone** - Text modified with uncertain/hedging language
- **gender_removal** - Gender references replaced with neutral terms (they/their)
- **gender_swap** - Gender references swapped (male→female, female→male)
- **summary** - Summarized version of the clinical context

---

## Datasets

Clinical contexts come from five different datasets:

- **askdocs** - Reddit r/AskDocs questions
- **MeDiSumQA** - Medical discharge summaries dataset
- **oncqa** - Oncology question-answering dataset
- **sct** - script concordance tests
- **usmle_derm** - USMLE dermatology questions

---

## Usage Examples

### Calculate Agreement with Gold Standard

```python
import pandas as pd

# Read data.csv
df = pd.read_csv('data.csv')

# Calculate agreement for gpt-4o manage predictions
gold = df['gold_standard_manage'].map({'YES': 1, 'NO': 0})
pred = df['gpt-4o_manage'].map({'YES': 1, 'NO': 0})
agreement = (gold == pred).mean()
print(f"GPT-4O Manage Agreement: {agreement:.2%}")
```

### Analyze Model Variability Across Seeds

```python
# Read llm_data.csv
df = pd.read_csv('llm_data.csv')

# Check consistency across seeds for a specific model
seeds = ['seed0', 'seed1', 'seed2']
for seed in seeds:
    print(f"{seed}: {df[f'gpt-4o_{seed}_manage'].value_counts()}")
```

### Calculate Inter-Clinician Agreement

```python
# Read clinician_data.csv
df = pd.read_csv('clinician_data.csv')

# Calculate agreement between clinician_1 and clinician_2
agreement = (df['clinician_1_manage'] == df['clinician_2_manage']).mean()
print(f"Clinician 1-2 Agreement: {agreement:.2%}")
```

---

## Notes

- All files use UTF-8 encoding
- Missing values are represented as NaN/empty strings
- Context IDs are unique identifiers that can be used to join across files
- The `perturbation` column is essential for matching predictions to the correct clinical context variant
