# Word2Vec (Skip-gram + Negative Sampling) — Implementation Reference

> Status: step 1 (preprocessing + vocabulary) is implemented in
> `word2vec/word2vec.ipynb`. Steps 2–8 are design notes for upcoming work.

## 1. Preprocessing
- Lowercase, strip punctuation (careful with translated Russian prose — em-dashes, nested quotes).
- Tokenize into words.
- Build vocabulary: count word frequencies, drop words with frequency < min_count (typically 5).
- Build `word2idx` and `idx2word` mappings.
- Let `V` = vocabulary size.

## 2. Subsampling of Frequent Words
Applied *before* generating training windows, to thin out high-frequency, low-information words ("the", "is", "and").

For each word `w` with corpus frequency `f(w)` (fraction of total tokens):

$$
P_{discard}(w) = 1 - \sqrt{\frac{t}{f(w)}}
$$

- `t` ≈ 1e-5 (threshold, tunable).
- For each occurrence of `w` in the corpus, discard it with probability `P_discard(w)` before window generation.
- Words rarer than `t` get `P_discard ≈ 0` (never discarded); very frequent words get discarded often.

## 3. Negative Sampling Distribution
Precompute a sampling table over the vocabulary, weighted by:

$$
P(w) = \frac{f(w)^{3/4}}{\sum_{j=1}^{V} f(w_j)^{3/4}}
$$

- `f(w)` = raw frequency count (or unigram probability) of word `w`.
- The `3/4` power dampens dominance of very frequent words, boosts relative chance of rare words being sampled as negatives.
- Implementation trick: build a large array (e.g. size 1e8) where each word index appears proportional to `P(w)`, then sample by random indexing — O(1) per draw.

## 4. Window / Pair Generation (Skip-gram)
For each position `i` in the (subsampled) token sequence, with window size `k`:
- Center word: `token[i]`
- Context words: `token[i-k] ... token[i-1], token[i+1] ... token[i+k]` (respecting sentence/document boundaries — don't let windows cross documents)
- Emit one **(center, context)** pair per context word — NOT one pair per window.

Example, k=2, center="greener": pairs = (greener,grass), (greener,is), (greener,on), (greener,the)

## 5. Model Parameters
Two weight matrices, both randomly initialized (small values, e.g. uniform in `[-0.5/N, 0.5/N]`):

- `W` — shape `(V, N)` — input/center embeddings. Row `c` = `v_c`.
- `W'` — shape `(V, N)` — output/context embeddings. Row `j` = `v'_j`.
- `N` = embedding dimension (e.g. 100–300; use smaller, like 50, for a small Tolstoy/Dostoevsky corpus).

`N` is a hyperparameter you choose based on corpus size — too large relative to corpus = overfitting/noise.

## 6. Forward Pass, Loss, Gradients (Negative Sampling)
For one training instance: center `c`, true context `o` (positive), and `k` sampled negatives `n_1...n_k`.

**Score (dot product):**
$$
u_j = v_c \cdot v'_j
$$

**Loss for this instance:**
$$
L = -\log\sigma(u_o) - \sum_{i=1}^{k} \log\sigma(-u_{n_i})
$$

where $\sigma(x) = \dfrac{1}{1+e^{-x}}$

**Gradients** (derived by differentiating L):

$$
\frac{\partial L}{\partial v_c} = (\sigma(u_o) - 1)\, v'_o \;+\; \sum_{i=1}^{k} \sigma(u_{n_i})\, v'_{n_i}
$$

$$
\frac{\partial L}{\partial v'_o} = (\sigma(u_o) - 1)\, v_c
$$

$$
\frac{\partial L}{\partial v'_{n_i}} = \sigma(u_{n_i})\, v_c \quad \text{(for each negative } n_i \text{)}
$$

**Update rule** (gradient descent, learning rate `η`):
$$
v_c \leftarrow v_c - \eta \frac{\partial L}{\partial v_c}, \quad
v'_o \leftarrow v'_o - \eta \frac{\partial L}{\partial v'_o}, \quad
v'_{n_i} \leftarrow v'_{n_i} - \eta \frac{\partial L}{\partial v'_{n_i}}
$$

Note: compute all gradients first using the *current* vectors, then apply all updates — don't update `v_c` mid-calculation and use the new value for the `v'_o` or `v'_{n_i}` gradients in the same step.

## 7. Training Loop
```
initialize W, W' randomly
build negative sampling table
for epoch in 1..num_epochs:
    for each document/sentence in corpus (optionally shuffle order):
        subsample tokens (step 2)
        for each (center, context) pair from sliding window (step 4):
            sample k negatives (step 3)
            compute loss, gradients (step 6)
            update v_c, v'_context, v'_negatives
    log average loss for the epoch
    optionally: decay learning rate linearly toward 0
```
- Typical: 5–15 epochs, η starting ~0.025, k = 5 (small corpus) to 20 (large corpus), window k=5, N=50–100 for a small corpus.

## 8. Inspection / Evaluation
- **Nearest neighbors**: for a word `w`, compute cosine similarity between `v_w` and every other row of `W`, sort descending.
$$
\text{sim}(a,b) = \frac{v_a \cdot v_b}{\|v_a\|\,\|v_b\|}
$$
- **Analogy test**: `king - man + woman ≈ queen` → compute `v_king - v_man + v_woman`, find nearest neighbor(s) to that resulting vector (excluding the three input words).
- Sanity checks for your corpus: character names should cluster with associated traits/other characters; moral/emotional vocabulary should cluster together.