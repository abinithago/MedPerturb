# MedPerturb

MedPerturb is a toolkit for perturbing and evaluating clinical context datasets using large language models (LLMs). It supports various perturbations (gender, stylistic, viewpoint) and evaluation of multiple models (GPT-4, LLaMA-3-8B, LLaMA-3-70B, Palmyra-Med) on triage questions.

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
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (tokens for HuggingFace/OpenAI)
├── README.md                   # Project documentation
└── ...                         # (Other files or directories)
```

## Features
- **Perturb clinical text** by gender, style, or viewpoint using LLMs
- **Evaluate LLMs** on triage questions (MANAGE, VISIT, RESOURCE)
- **Bootstrap-based mutual information analysis** for clinicians and models
- **Supports both HuggingFace and OpenAI models**

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MedPerturb.git
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

## Usage

### 1. Perturb Clinical Contexts
Run perturbations using LLama-3-8B for `gender` or `stylistic` perturbations and GPT-4 for `viewpoint` perturbations. 
```bash
python code/perturb_data.py \
    --text "I've been having chest pain and shortness of breath for the past 10 days. I'm 22 years old." \
    --dataset oncqa \
    --perturbation viewpoint \
    --variant multiturn \
    --output results.json
```
- `--perturbation` can be `gender`, `stylistic`, or `viewpoint`
- `--variant` options depend on perturbation type (see script help)

### 2. Evaluate Models on Triage Questions
```bash
python code/evaluate_models.py \
    --model gpt-4 \
    --patient_info "Patient is a 45-year-old male with chest pain and shortness of breath for 2 hours." \
    --output eval_results.json
```
- Supported models: `GPT-4`, `Llama-3-8B`, `Llama-3-70B`, `Palmyra-Med-20B`

## Contributing
Pull requests and issues are welcome!

## License

This project is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

[![CC BY 4.0][cc-by-shield]][cc-by]

[cc-by]: https://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://licensebuttons.net/l/by/4.0/88x31.png