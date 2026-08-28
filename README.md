# llm-scratch

Building an LLM from scratch. The path starts with Word2Vec — a concrete, small
implementation of distributed word representations — then moves through modern
tokenization (BPE) and input embeddings (token + positional) before rolling our
own attention-based language model.

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

### 3. Tokenization — `embedding/embedding.ipynb`

Builds a vocabulary from the Tolstoy corpus using a simple regex split
(`re.split` on punctuation, quotes, em-dashes and whitespace) over the raw text
(8,252,126 chars → 40,889 unique tokens).

- **SimpleTokenizerV1** — `encode()`/`decode()` over the regex vocabulary.
- **SimpleTokenizerV2** — adds the special tokens `<|endoftext|>` (document
  separator) and `<|unk|>` (out-of-vocabulary fallback); 40,891 vocab entries.

This is the "hand-rolled" tokenizer used to motivate why we move to BPE next.

### 4. BPE tokenization + input embeddings — `embedding/bpe.ipynb`

Replaces the regex tokenizer with OpenAI's Byte-Pair-Encoding tokenizer
(`tiktoken`, GPT-2 encoding, 50,257 vocab):

- **Tokenization** — the full Tolstoy corpus encodes to **2,178,265 tokens**
  (vs. ~1.7M regex tokens).
- **GPTDatasetV1 / `create_dataloader_v1`** — chunks the token stream into
  `(input_ids, target_ids)` pairs of length `max_length` (default 256) with a
  `stride` (default 128) so windows overlap; returned as PyTorch tensors via a
  `DataLoader` (default `batch_size=4`).
- **Token embedding** — `torch.nn.Embedding(vocab_size=50257, output_dim=256)`;
  maps token ids → dense vectors, output shape `(batch, context, output_dim)`.
- **Positional embedding** — `torch.nn.Embedding(context_length, output_dim)`
  indexed by `torch.arange(context_length)`, shape `(context, output_dim)`.
- **Input embedding** — `input = token_embeddings + pos_embeddings`, giving the
  combined representation that feeds into an attention layer.

This completes the embedding pipeline: raw text → BPE tokens → token + positional
embeddings.

## Project layout

```
llm-scratch/
├── data/                       # downloaded corpora
│   ├── tolstoy/                #   10 per-book .txt files
│   ├── dostoevsky/             #   9  per-book .txt files
│   └── combined/               #   tolstoy.txt, dostoevsky.txt (one per author)
├── data_download.ipynb         # Gutenberg corpus download + cleaning
├── embedding.ipynb             # draft of the regex tokenization scratch work
├── word2vec/
│   ├── word2vec.ipynb          # preprocessing + vocab + skip-gram training
│   └── word2vec_algo.md        # skip-gram + negative sampling reference
├── embedding/
│   ├── embedding.ipynb         # SimpleTokenizerV1/V2 (regex vocabulary)
│   └── bpe.ipynb               # BPE tokenization + GPT dataset + embeddings
├── attention/                  # next step: attention + transformers
└── README.md
```

## Running the notebooks

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy jupyter tiktoken torch
jupyter notebook
```

The download notebook (`data_download.ipynb`) uses only the Python standard
library. The Word2Vec notebook additionally requires `numpy`; the embedding
notebooks require `tiktoken` (BPE) and `torch` (dataset + embedding layers).

## Roadmap

- [x] Download and clean a text corpus
- [x] Preprocess corpus + build vocabulary (`word2vec/word2vec.ipynb`)
- [x] Word2Vec (skip-gram): subsampling, negative sampling table, pair generation
- [x] Word2Vec: model weights, forward pass, training loop, evaluation
- [x] Tokenization: regex tokenizers with special tokens (`embedding/embedding.ipynb`)
- [x] BPE tokenization + GPT dataset/dataloader (`embedding/bpe.ipynb`)
- [x] Input embeddings: token + positional (`embedding/bpe.ipynb`)
- [ ] Attention and transformers
- [ ] Train a language model from scratch