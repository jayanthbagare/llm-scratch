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

- **Subsampling of frequent words** (`P_discard = 1 − √(t/f_w)`, `t ≈ 1e-5`) applied
  per token occurrence, thinning out high-frequency, low-information words.
- **Negative sampling distribution** — sampling table over the full vocabulary,
  weighted by `f(w)^(3/4)` normalized (sums to 1).
- **Window / pair generation** — per document, one `(center, context)` pair per
  context word inside window `k = 5`; windows never cross document boundaries
  (currently **4,942,460 pairs** on the subsampled token stream).
- **Model weights** — input/center embeddings `W` and output/context embeddings
  `W'`, both `(V, N)` with `N = 128`, initialized uniformly in `±0.5/N`.
- **Forward pass, loss & gradients** — negative-sampling loss with a stable
  softplus objective, computed in a vectorized mini-batch update.
- **Training loop** — negative samples drawn from the precomputed table
  (`K = 5`), learning rate decaying linearly toward 0 over epochs.
- **Inspection** — cosine nearest-neighbors and the `a − b + c` analogy test
  over the trained center embeddings.

Latest run (both corpora, `min_count=5`, `seed=42`, 1M pairs, 5 epochs, `N=128`):
**2,659,120 tokens**, **15,839 vocabulary words**; subsampling keeps ~494k token
occurrences; training loss falls from ~4.16 to ~2.43.

## Project layout

```
llm-scratch/
├── data/                       # downloaded corpora
│   ├── tolstoy/                #   10 per-book .txt files
│   ├── dostoevsky/             #   9  per-book .txt files
│   └── combined/               #   tolstoy.txt, dostoevsky.txt (one per author)
├── data_download.ipynb         # Gutenberg corpus download + cleaning
├── word2vec/
│   ├── word2vec.ipynb          # preprocessing + vocab + skip-gram training
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
- [x] Word2Vec (skip-gram): subsampling, negative sampling table, pair generation
- [x] Word2Vec: model weights, forward pass, training loop, evaluation
- [ ] Attention and transformers
- [ ] Train a language model from scratch
