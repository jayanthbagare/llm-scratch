# llm-scratch

Building an LLM from scratch. The path starts with Word2Vec — a concrete, small
implementation of distributed word representations — as a foundation for later
topics (attention, transformers) before rolling our own language model.

## Status (what's done so far)

### 1. Corpus download — `data_download.ipynb`

Downloads classic Russian literature (English translations) from Project Gutenberg
and cleans each text by stripping the standard Gutenberg header/footer markers.

- **Tolstoy** → `data/tolstoy/` (10 works: *War and Peace*, *Anna Karenina*, *Resurrection*, …)
- **Dostoevsky** → `data/dostoevsky/` (9 works: *The Brothers Karamazov*, *Crime and Punishment*, *The Idiot*, …)

A 2-second delay between requests keeps the download polite to Gutenberg servers.

### 2. Word2Vec scaffold — `word2vec.ipynb`

Skeleton notebook marking the intended implementation; code is not written yet.

## Project layout

```
llm-scratch/
├── data/                     # downloaded corpora
│   ├── tolstoy/              #   10 .txt files
│   └── dostoevsky/           #   9  .txt files
├── data_download.ipynb       # Gutenberg corpus download + cleaning
├── word2vec.ipynb            # Word2Vec implementation (in progress)
└── README.md
```

## Running the notebooks

```bash
python -m venv .venv
source .venv/bin/activate
jupyter notebook
```

The download notebook uses only the Python standard library (`os`, `re`, `time`,
`urllib.request`) — no third-party dependencies required.

## Roadmap

- [x] Download and clean a text corpus
- [ ] Word2Vec (skip-gram / CBOW, negative sampling)
- [ ] Attention and transformers
- [ ] Train a language model from scratch
