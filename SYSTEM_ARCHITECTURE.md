# EO Evaluation Framework - System Architecture

> **Version:** 1.0  
> **Last Updated:** February 3, 2026  
> **Author:** AI COE  

---

## Executive Summary

The EO Evaluation Framework is a **standalone, modular benchmarking platform** designed for rapid iteration and testing of LLM prompts against a golden dataset of policy-EO compliance decisions. It was built separately from the existing EO Policy Tracker to avoid interference with ongoing production work while enabling fast experimentation.

---

## Why We Built This

After discussions with Matt and Mike, the team identified the need to:

1. **Break apart the monolithic system** into modular, testable components
2. **Enable rapid prompt iteration** without affecting production code
3. **Compare AI reasoning quality** against Subject Matter Expert (SME) justifications
4. **Track historical runs** to measure improvement over time

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Frontend** | Streamlit | Rapid prototyping; Python-native; minimal boilerplate |
| **Backend** | Python 3.10+ | Team expertise; LLM library ecosystem |
| **LLM Access** | US AI API | Government-approved; supports Gemini, Claude, GPT |
| **Persistence** | SQLite + JSON + CSV | Lightweight; portable; multi-format export |
| **Processing** | ThreadPoolExecutor | Parallel LLM calls for batch processing |

### Why Streamlit?

- **Speed to prototype:** From zero to working UI in hours, not days
- **Python-native:** No frontend/backend separation required
- **Interactive widgets:** Instant feedback during testing
- **Easy deployment:** Single file, simple requirements

---

## Directory Structure

```
Evaluation_Framework/
├── app.py                    # Streamlit frontend (main entry point)
├── run_eval.py               # CLI alternative for batch processing
├── requirements.txt          # Python dependencies
├── .env                      # API keys (US AI key required)
│
├── config/                   # Configuration files
│   ├── models.json          # Available models and settings
│   └── settings.json        # Batch size, timeouts, output formats
│
├── prompts/                  # Prompt templates (versioned)
│   ├── phase1_classification.txt
│   ├── phase2_reasoning.txt
│   ├── phase2_v2.txt         # Iterated version
│   └── phase3_justification.txt
│
├── datasets/                 # Golden datasets
│   └── eo/
│       └── golden_dataset.csv  # 152 SME-validated records
│
├── results/                  # Output directory
│   ├── benchmark.db         # SQLite database
│   ├── run_YYYYMMDD_*.csv   # CSV exports
│   └── run_YYYYMMDD_*.json  # JSON exports
│
└── src/                      # Source modules
    ├── ingestion/           # Data loading
    ├── orchestration/       # Pipeline execution
    ├── scoring/             # Metrics calculation
    └── storage/             # Persistence layer
```

---

## Module Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI                             │
│                         (app.py)                                 │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────┐  │
│  │Run Eval  │  │Edit Prompts│  │Analyze    │  │Results History│  │
│  │          │  │            │  │& Debug    │  │               │  │
│  └──────────┘  └───────────┘  └───────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION MODULE                             │
│                   src/ingestion/loader.py                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ConfigLoader  │  │PromptLoader  │  │DatasetLoader │           │
│  │- models.json │  │- .txt files  │  │- CSV parsing │           │
│  │- settings    │  │- versioning  │  │- validation  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION MODULE                           │
│                src/orchestration/pipeline.py                     │
│                                                                  │
│    ┌──────────┐      ┌──────────┐      ┌──────────────┐         │
│    │ PHASE 1  │ ──▶  │ PHASE 2  │ ──▶  │   PHASE 3    │         │
│    │Classify  │      │Reasoning │      │Justification │         │
│    │          │      │          │      │ Comparison   │         │
│    └──────────┘      └──────────┘      └──────────────┘         │
│         │                 │                   │                  │
│    Affected/         Final Flag +        Similarity              │
│    Not Affected      Justification       Score 0-100             │
│                                                                  │
│    ┌──────────────────────────────────────────────────┐         │
│    │              LLM Client Wrapper                   │         │
│    │         src/orchestration/llm_client.py          │         │
│    │  - US AI API integration                          │         │
│    │  - Retry logic                                    │         │
│    │  - Multi-model support (Gemini, Claude, GPT)      │         │
│    └──────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SCORING MODULE                              │
│                    src/scoring/metrics.py                        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  calculate_metrics()                     │    │
│  │  - Accuracy, Precision, Recall, F1 Score                │    │
│  │  - Confusion matrix (TP, FP, TN, FN)                    │    │
│  │  - Flag normalization                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │             calculate_justification_stats()              │    │
│  │  - Average similarity score                              │    │
│  │  - High/low quality counts                               │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      STORAGE MODULE                              │
│                   src/storage/database.py                        │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │    CSV       │  │    JSON      │  │   SQLite     │           │
│  │  - Overview  │  │  - Complete  │  │  - History   │           │
│  │  - Excel-    │  │  - Full      │  │  - Compare   │           │
│  │    friendly  │  │    records   │  │  - Query     │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Three-Phase Pipeline

### Phase 1: Classification
- **Input:** Policy text + Executive Order text
- **Output:** Binary classification (Affected / Not Affected)
- **Purpose:** Quick initial determination

### Phase 2: Reasoning
- **Input:** Phase 1 results + full record context
- **Output:** Final flag + full justification
- **Purpose:** Deep analysis with explanation
- **Key Feature:** Prompt is versioned (e.g., `phase2_v2.txt`)

### Phase 3: Justification Comparison
- **Input:** Model justification + SME justification
- **Output:** Similarity score (0-100) + key differences
- **Purpose:** Measure reasoning quality, not just final answer
- **Why Added:** Matt requested quality analysis of the "why"

---

## Key Design Principles

### 1. Modularity
Every component is independent and replaceable:
- Swap LLM providers by changing `llm_client.py`
- Add new metrics by extending `metrics.py`
- Change storage formats by modifying `database.py`

### 2. Prompt Versioning
Prompts are stored as plain text files with version suffixes:
- `phase2_reasoning.txt` (original)
- `phase2_v2.txt` (first iteration)
- `phase2_v3.txt` (second iteration)

The UI automatically detects new versions in the dropdown.

### 3. Data Isolation
The framework uses its own:
- **Golden Dataset:** 152 SME-validated records from May 2025
- **Results Directory:** Separate from production outputs
- **Database:** Standalone SQLite file

### 4. Rapid Iteration Workflow
1. Run evaluation → View results
2. Identify failures → Analyze disagreement
3. Generate improved prompt → Test on single record
4. Save as new version → Run full evaluation
5. Compare metrics → Repeat

---

## Data Flow

```
┌────────────────┐
│ Golden Dataset │  (152 policy-EO pairs with SME verdicts)
│    (CSV)       │
└───────┬────────┘
        │
        ▼
┌────────────────┐     ┌─────────────────┐
│   DatasetLoader │────▶│   DataFrame     │
└────────────────┘     └───────┬─────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│    Phase 1    │     │    Phase 2    │     │    Phase 3    │
│   (LLM Call)  │     │   (LLM Call)  │     │   (LLM Call)  │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                     ┌───────────────┐
                     │   Results[]   │  (enriched records)
                     └───────┬───────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌─────────┐    ┌─────────┐    ┌─────────┐
        │   CSV   │    │  JSON   │    │ SQLite  │
        └─────────┘    └─────────┘    └─────────┘
```

---

## Key Classes Reference

### ingestion/loader.py

| Class | Responsibility |
|-------|----------------|
| `ConfigLoader` | Load models.json, settings.json |
| `PromptLoader` | Load and cache prompt templates |
| `DatasetLoader` | Load and validate CSV datasets |

### orchestration/pipeline.py

| Method | Description |
|--------|-------------|
| `run_phase1()` | Binary classification |
| `run_phase2()` | Full reasoning with justification |
| `run_phase3()` | SME comparison analysis |
| `_extract_phase2_decision()` | Parse LLM response for flag/justification |

### scoring/metrics.py

| Function | Output |
|----------|--------|
| `calculate_metrics()` | Accuracy, Precision, Recall, F1, confusion matrix |
| `calculate_justification_stats()` | Avg/min/max similarity, quality counts |
| `normalize_flag()` | Standardize "Affected"/"Not Affected" variations |

### storage/database.py

| Method | Format |
|--------|--------|
| `_save_csv()` | Excel-friendly overview |
| `_save_json()` | Complete record with all fields |
| `_save_sqlite()` | Queryable historical database |
| `get_run_history()` | Recent runs for sidebar |
| `compare_runs()` | Diff metrics between runs |

---

## Configuration Files

### config/models.json
```json
{
  "models": [
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "enabled": true},
    {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "enabled": true},
    {"id": "gpt-4o-2024-11-20", "name": "GPT-4o", "enabled": true}
  ],
  "default_model": "gemini-2.5-pro"
}
```

### config/settings.json
```json
{
  "batch_size": 20,
  "max_workers": 5,
  "timeout_seconds": 120,
  "database_path": "results/benchmark.db",
  "output_formats": ["csv", "json", "sqlite"]
}
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `USAI_API_KEY` | US AI API authentication token | ✅ Yes |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export USAI_API_KEY="your-key-here"

# Run Streamlit UI
streamlit run app.py

# Or run CLI batch evaluation
python run_eval.py --model gemini-2.5-pro --sample 20
```

---

## Future Extensibility

| Feature | Implementation Path |
|---------|---------------------|
| New LLM provider | Add adapter in `llm_client.py` |
| New metric | Add function to `metrics.py` |
| New phase | Add `run_phase4()` to `pipeline.py` |
| New export format | Add `_save_parquet()` to `database.py` |
| New dataset | Drop CSV in `datasets/` folder |

---

## FAQ

**Q: Why not modify the existing EO Tracker?**  
A: To avoid interfering with Tim and Varsha's production work. This is a research/iteration environment.

**Q: Why SQLite instead of a real database?**  
A: Portability. The entire framework can be copied to any laptop and run immediately.

**Q: Why three phases instead of two?**  
A: Matt requested Phase 3 to analyze justification quality, not just flag accuracy.

**Q: How do I compare Gemini vs Claude?**  
A: Run the same dataset with each model, then compare runs in Results History.

---

## Contact

For questions or contributions, reach out to the AI COE team.
