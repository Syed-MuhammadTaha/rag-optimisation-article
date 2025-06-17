# RAG Optimization Experiments

This project implements and evaluates various RAG (Retrieval-Augmented Generation) optimization techniques using the BioASQ dataset.

## Dataset Structure

The project uses the [BioASQ mini dataset](https://huggingface.co/datasets/enelpol/rag-mini-bioasq) which consists of:

1. Text Corpus: Scientific/medical text passages with unique IDs
2. Question-Answer Pairs: Questions with ground truth answers and relevant passage IDs

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Qdrant vector database:
```bash
docker-compose up -d
```

## Running the Baseline

The baseline implementation includes:
- Basic passage embedding using SentenceTransformer
- Vector similarity search using Qdrant
- Evaluation metrics: Recall@k, MRR, and latency

To run the baseline evaluation:
```bash
python src/main.py
```

Results will be saved in `results/baseline_metrics.json`.

## Project Structure

```
.
├── docker-compose.yml    # Qdrant container configuration
├── requirements.txt      # Python dependencies
├── src/
│   ├── data_loader.py   # Dataset loading and preprocessing
│   ├── vector_store.py  # Vector database operations
│   ├── evaluator.py     # Retrieval evaluation metrics
│   └── main.py         # Main execution script
└── results/            # Evaluation results
```

## Optimization Areas

Future optimizations will explore:

1. Ingestion Improvements:
   - Batch processing
   - Parallel embedding generation
   - Caching strategies

2. Retrieval Optimization:
   - HNSW index parameters
   - Space clustering
   - Embedding dimension reduction
   - Metadata filtering 