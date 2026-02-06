# EO Policy Tracker - Evaluation Framework

A **modular, configurable benchmarking tool** for evaluating LLM performance on policy-EO compliance analysis.

## 🎯 Mission

> Turn the EO evaluation script into a reusable benchmarking tool that anyone can configure without touching code.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API access
cp .env.example .env
# Edit .env with your USAI API key

# 3. Place your dataset
cp ../path/to/golden_dataset.csv datasets/eo/golden_dataset.csv

# 4. Run evaluation
python run_eval.py --model claude_3_5_sonnet --phases 1,2
```

## 📁 Project Structure

```
Evaluation_Framework/
├── config/
│   ├── models.json         # Swap models without code changes
│   └── settings.json       # General settings
├── prompts/
│   ├── phase1_classification.txt
│   ├── phase2_reasoning.txt
│   └── phase3_justification.txt
├── datasets/
│   └── eo/golden_dataset.csv
├── src/
│   ├── ingestion/          # Load data + config
│   ├── orchestration/      # Pipeline + LLM client
│   ├── scoring/            # Metrics calculation
│   └── storage/            # CSV + JSON + SQLite persistence
├── results/                # Output (auto-created)
├── run_eval.py             # Main CLI
└── requirements.txt
```

## 🔧 CLI Usage

### Basic Run
```bash
python run_eval.py
```

### Specify Model & Phases
```bash
python run_eval.py --model gemini_2_0_flash_exp --phases 1,2,3
```

### Try New Prompt
```bash
python run_eval.py --prompt prompts/phase1_v2.txt
```

### Sample Run (5 records)
```bash
python run_eval.py --sample 5
```

### View Run History
```bash
python run_eval.py --history
```

### Compare Two Runs
```bash
python run_eval.py --compare run_20260203_123456_abc123 run_20260203_134567_def456
```

### Dry Run (validate config)
```bash
python run_eval.py --dry-run
```

## 📊 Output Formats

Every run saves to **three formats**:

| Format | File | Use Case |
|--------|------|----------|
| **CSV** | `results/{run_id}_results.csv` | Excel/spreadsheet analysis |
| **JSON** | `results/{run_id}_full.json` | API/programmatic access |
| **SQLite** | `results/benchmark.db` | Historical queries |

## 🔄 Three-Phase Evaluation

| Phase | Purpose | Output |
|-------|---------|--------|
| **Phase 1** | Classification | `Affected` / `Not Affected` |
| **Phase 2** | Final Reasoning | Full justification |
| **Phase 3** | Justification Comparison | Similarity score (0-100) |

## 📈 Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Justification Similarity**: Average score from Phase 3 comparison

## 🔄 Swapping Models

Edit `config/models.json`:
```json
{
  "models": [
    {"id": "claude_3_5_sonnet", "name": "Claude 3.5", "enabled": true},
    {"id": "gemini_2_0_flash_exp", "name": "Gemini 2.0", "enabled": true}
  ],
  "default_model": "claude_3_5_sonnet"
}
```

## 📝 Swapping Prompts

1. Create new prompt file: `prompts/phase1_v2.txt`
2. Run with: `--prompt prompts/phase1_v2.txt`

## 🗄️ Database Schema

```sql
-- runs: One row per evaluation run
runs (
  run_id, timestamp, model, prompt_version, dataset,
  accuracy, precision_score, recall, f1_score, ...
)

-- results: One row per record per run  
results (
  run_id, compliance_id, ground_truth, predicted,
  is_correct, similarity_score, phase2_justification
)
```
