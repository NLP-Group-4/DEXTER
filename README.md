# RAGs for Open Domain Complex Question Answering

<p align="center">
  <img src="dexter.png" />
</p>

## Overview

This project investigates the impact of context quality in **Retrieval Augmented Generation (RAG) systems** for complex question answering. We focus on compositional multi-hop questions from the 2WikiMultiHopQA dataset to understand how different types of contexts affect answer generation performance when using Large Language Models (LLMs).

> **Based on DEXTER Framework**: This work builds upon the [DEXTER](https://github.com/VenkteshV/DEXTER) toolkit for evaluating RAG pipelines on complex QA tasks.

---

## Research Questions

Our goal is to answer the following research questions:

- **RQ1**: How do negative contexts impact downstream answer generation performance?
- **RQ2**: Are negative contexts more important for answer generation than related contexts?
- **RQ3**: Does providing only gold contexts deteriorate performance compared to mixing with other negative or related contexts?

---

## Implemented Experiments

We conducted three main experiments on the first 1200 questions from the 2WikiMultiHopQA dev set:

### Experiment 2: Oracle Contexts (Upper Bound)
**Script**: [`evaluation/wikimultihop/run_oracle_gemma3_ollama.py`](evaluation/wikimultihop/run_oracle_gemma3_ollama.py)

Uses only ground-truth (oracle) contexts for each question to establish performance upper bound. This experiment helps us understand the best possible performance with perfect retrieval.

- **Model**: Gemma 3 4B (via Ollama)
- **Contexts**: Ground truth evidences from dev.json
- **Results**: [`results/Experiment_2/`](results/Experiment_2/)

### Experiment 3: Noise Injection
**Script**: [`evaluation/wikimultihop/run_noise_experiment.py`](evaluation/wikimultihop/run_noise_experiment.py)

Injects randomly sampled irrelevant documents as noise into the LLM input contexts to test robustness and analyze the surprising phenomenon of performance variation with noise.

- **Model**: Gemma 3 4B (via Ollama)  
- **Noise levels**: k=1, 3, 5 random documents
- **Method**: Combines oracle contexts with k randomly sampled documents, shuffled to prevent positional bias
- **Results**: [`results/Experiment_3/`](results/Experiment_3/)

### Experiment 4: Hard Negative Injection
**Script**: [`evaluation/wikimultihop/run_hard_negatives_experiment.py`](evaluation/wikimultihop/run_hard_negatives_experiment.py)

Evaluates RAG robustness when ground-truth evidence is combined with hard negatives retrieved via dense retrieval. Unlike random noise, hard negatives are retriever-based and semantically similar to the query, making this experiment a realistic test to understand if hard negatives improve RAG performance compared to random documents.

- **Model**: Gemma 3 4B (via Ollama)
- **Hard Negative levels**: k = 1, 3, 5 hard negatives
- **Method**: Bootstraps dense retrieval (Contriever) to build a high-recall reduced corpus, followed by final retrieval. Combines gold evidence with retrieved hard negatives, shuffled to prevent positional bias.
- **Results**: [`results/Experiment_4/`](results/Experiment_4/)

### Analysis: Extended Churn Analysis
**Script**: [`results/churn_analysis.py`](results/churn_analysis.py)

Comprehensive analysis of how noise affects answer quality:
- **Churn analysis**: Tracks questions lost/gained across noise levels
- **Question type breakdown**: Performance by question type (Who, What, When, Where, Which, Other)
- **Robustness metrics**: Identifies fragile vs. robust questions
- **Answer length analysis**: Compares verbosity across conditions

---

## Project Structure

```
DEXTER/
├── dexter/                              # Core library
│   ├── config/                          # Configuration
│   │   ├── constants.py                 # Dataset & split constants
│   │   └── __init__.py
│   ├── data/                            # Data structures & loaders
│   │   ├── datastructures/              # Core data classes
│   │   │   ├── answer.py                # Answer class
│   │   │   ├── dataset.py               # Dataset classes
│   │   │   ├── evidence.py              # Document/Evidence class
│   │   │   ├── question.py              # Question class
│   │   │   └── sample.py                # Sample (Q+A+Evidence)
│   │   └── loaders/                     # Dataset loaders
│   │       ├── BaseDataLoader.py        # Base class
│   │       ├── DataLoaderFactory.py     # Factory pattern
│   │       ├── RetrieverDataset.py      # Main loader
│   │       ├── WikiMultihopQADataLoader.py
│   │       ├── MusiqueQaDataLoader.py
│   │       └── Tokenizer.py
│   └── llms/                            # LLM engines
│       ├── gemma_ollama_engine.py       # Gemma via Ollama
│       └── __init__.py
│
├── evaluation/                          # Evaluation scripts
│   ├── config.ini                       # Data paths configuration
│   ├── data/                            # Dataset files
│   │   └── musiqueqa/                   # 2WikiMultiHopQA dataset
│   │       ├── dev.json                 # Questions & annotations
│   │       └── wiki_musique_corpus.json # Document corpus
│   └── wikimultihop/                    # Evaluation scripts
│       ├── run_oracle_gemma3_ollama.py  # Experiment 2
│       └── run_noise_experiment.py      # Experiment 3
│
├── results/                             # Experiment results
│   ├── Experiment_1/                    # RAG baseline experiments
│   ├── Experiment_2/                    # Oracle context results
│   ├── Experiment_3/                    # Noise injection results
│   └── extented_analysis.py             # Analysis script
│
├── setup.py                             # Package installation
├── LICENSE.md                           # Apache 2.0 license
└── README.md                            # This file
```

---

## Setup Guide

### Prerequisites

- **Python**: 3.11.13
- **Ollama**: Required for running Gemma 3 4B model
- **Operating System**: macOS (tested on Apple Silicon)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd DEXTER
   ```

2. **Create conda environment**:
   ```bash
   conda create -n bcqa python=3.11.13 -y
   conda activate bcqa
   ```

3. **Install PyTorch** (macOS with Apple Silicon):
   ```bash
   conda install pytorch torchvision torchaudio -c pytorch
   ```

4. **Install project dependencies**:
   ```bash
   pip install -e .
   ```

5. **Install Ollama** (for Gemma model):
   ```bash
   # macOS
   brew install ollama
   
   # Start Ollama service
   ollama serve
   
   # Pull Gemma 3 model (in another terminal)
   ollama pull gemma3:4b
   ```

6. **Download dataset**:
   - Questions & Corpus: [Google Drive Link](https://drive.google.com/drive/folders/1aQAfNLq6HB0w4_fVnKMBvKA6cXJGRTpH?usp=sharing)
   - Extract to `evaluation/data/musiqueqa/`

7. **Configure data paths** in `evaluation/config.ini`:
   ```ini
   [Data-Path]
   wikimultihopqa = evaluation/data/musiqueqa
   wiki-musiqueqa-corpus = evaluation/data/musiqueqa/wiki_musique_corpus.json
   ```

### Environment Variables

```bash
export PYTHONPATH="/path/to/DEXTER"
```

---

## Running Experiments

### Experiment 2: Oracle Contexts

```bash
cd /path/to/DEXTER
python evaluation/wikimultihop/run_oracle_gemma3_ollama.py
```

**Output**: `results/Experiment_2/gemma3_oracle_retrieved_v2_ollama.tsv`

### Experiment 3: Noise Injection

```bash
python evaluation/wikimultihop/run_noise_experiment.py
```

**Output**: 
- `results/Experiment_3/gemma3_noise_k1.tsv`
- `results/Experiment_3/gemma3_noise_k3.tsv`
- `results/Experiment_3/gemma3_noise_k5.tsv`

### Analysis

```bash
python results/churn_analysis.py
```

**Output**: Console output with:
- Churn statistics (Lost/Gained/Stable)
- Question type breakdown
- Answer length analysis
- Cross-level robustness metrics

---

## Citation

This project builds upon the DEXTER framework. If you use this work, please cite:

```bibtex
@misc{venky:2024:dexter,
      title={DEXTER: A Benchmark for open-domain Complex Question Answering using LLMs}, 
      author={Venktesh V. and Deepali Prabhu and Avishek Anand},
      year={2024},
      eprint={2406.17158},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.17158}, 
}
```

### DEXTER Repository

GitHub: [https://github.com/VenkteshV/DEXTER](https://github.com/VenkteshV/DEXTER)

arXiv: [https://arxiv.org/abs/2406.17158](https://arxiv.org/abs/2406.17158)

---

## License

Apache License 2.0 - See [LICENSE.md](LICENSE.md) for details.

---