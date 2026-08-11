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
Each author's books are then consolidated into a single file under
`data/combined/` (`tolstoy.txt`, `dostoevsky.txt`) so downstream processing
reads one file per author instead of every book.

### 2. Word2Vec implementation — `word2vec/word2vec.ipynb`

`preprocess()` runs over the consolidated corpus:

1. Lowercase.
2. Strip punctuation.
3. Tokenize into words.
4. Build vocabulary: count word frequencies, drop words with
   frequency < `min_count` (default 5).
5. Build `word2idx` map.
6. Build `idx2word` map.

Then implements the skip-gram training pipeline:

- **Subsampling of frequent words** (`P_discard = 1 − √(t/f_w)`, `t ≈ 1e-5`) to
  thin out high-frequency, low-information words.
- **Negative sampling distribution** — sampling table weighted by
  `f(w)^(3/4)` normalized over the vocabulary.
- **Window / pair generation** — for each retained center word, emit one
  `(center, context)` pair per context word inside window `k = 5` (currently
  75,900 pairs on the retained tokens).

Latest run (Tolstoy corpus only, `min_count=5`): **1,452,978 tokens**,
**11,851 vocabulary words**.

## Project layout

```
llm-scratch/
├── data/                       # downloaded corpora
│   ├── tolstoy/                #   10 per-book .txt files
│   ├── dostoevsky/             #   9  per-book .txt files
│   └── combined/               #   tolstoy.txt, dostoevsky.txt (one per author)
├── data_download.ipynb         # Gutenberg corpus download + cleaning
├── word2vec/
│   ├── word2vec.ipynb          # preprocessing + vocab implementation
│   └── word2vec_algo.md        # skip-gram + negative sampling reference
└── README.md
```

## Running the notebooks

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy jupyter
jupyter notebook
```

The download notebook (`data_download.ipynb`) uses only the Python standard
library; the Word2Vec notebook additionally requires `numpy`.

## Roadmap

- [x] Download and clean a text corpus
- [x] Preprocess corpus + build vocabulary (`word2vec/word2vec.ipynb`)
- [~] Word2Vec (skip-gram): subsampling, negative sampling table, pair generation
- [ ] Word2Vec: model weights, forward pass, training loop
- [ ] Attention and transformers
- [ ] Train a language model from scratch
