# RAG Strategy & Token Analysis

A tool for picking the right chunking strategy for a RAG (retrieval-augmented
generation) pipeline over real support-ticket documents, instead of guessing.
It runs multiple chunking strategies against the same documents, embeds every
chunk, tests retrieval quality against a set of realistic customer-support
questions, and ranks the strategies by a composite score — so the choice of
chunk size/overlap is backed by a number, not intuition.

## What it does

1. **Extracts** text from `.docx` support-ticket exports (with a `mammoth`
   fallback if `python-docx` fails on a malformed file).
2. **Chunks** each document four ways — word-based, sentence-based,
   paragraph-based, and token-based (via `tiktoken`, model-aware) — each
   with configurable size/overlap.
3. **Embeds** every chunk (`sentence-transformers`) and, for a set of test
   questions, measures cosine similarity against each chunking strategy's
   chunks.
4. **Scores and ranks** each strategy per document on a composite of
   average similarity, token utilization, and efficiency, and reports
   token/cost implications per strategy against real model context limits
   (GPT-4, GPT-4-32k, GPT-4-turbo, GPT-3.5).

Sample data included is a synthetic support-ticket export and a generic
support FAQ, both used purely as realistic-shaped test documents for the
chunking comparison — not a customer dataset.

## Running it

```bash
pip install openai numpy tiktoken sentence-transformers scikit-learn matplotlib python-docx mammoth
python test5.py
```

Reads `OPENAI_API_KEY` from the environment, or prompts for it if unset.

## Stack

Python, OpenAI, `sentence-transformers`, `tiktoken`, scikit-learn (cosine
similarity), matplotlib.
