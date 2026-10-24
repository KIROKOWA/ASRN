# ASRN — Adaptive Sparse Recurrence Network

[![tests](https://github.com/KIROKOWA/ASRN/actions/workflows/tests.yml/badge.svg)](https://github.com/KIROKOWA/ASRN/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Version 1.0.0](https://img.shields.io/badge/version-1.0.0-informational.svg)](docs/RELEASE_NOTES.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23607506.svg)](https://doi.org/10.5281/zenodo.23607506)

**Exact long-range copying for language models, with bounded work per token: learned hash tables find where the
current context appeared before, and a gated layer copies what came next.**

[Versión en español](README.es.md) · [Architecture](docs/ARCHITECTURE.md) ·
[Technical report](docs/TECHNICAL_REPORT.md) · [Release notes](docs/RELEASE_NOTES.md) · [Results](results/README.md) ·
Version 1.0.0 · Apache License 2.0

### Install and use

```bash
pip install git+https://github.com/KIROKOWA/ASRN      # Python >= 3.11, PyTorch >= 2.9; runs on CPU or GPU
```

```python
import torch, asrn

tokens = torch.tensor(list(open(asrn.nucleo.__file__, 'rb').read()))  # a text file's bytes as token ids (V = 256)
model = asrn.create_model(V=256, d=64, n_capas=4, semilla=0)          # local blocks + ASRN layers (stable layer)
asrn.train(model, tokens, largo=256, lote=8, pasos=200)               # AdamW, warmup + cosine; prints the loss
sequences, logits = asrn.generate(model, tokens[:512], n_nuevos=200)  # 200 new tokens, O(1) work per new token
print(bytes(sequences[0, 512:].tolist()).decode('utf-8', 'replace'))
```

Tested with Python 3.13 and PyTorch 2.11.0 and 2.14.1 (CPU and an NVIDIA A100); the
[CI workflow](.github/workflows/tests.yml) runs both test suites on CPU with Python 3.11 and 3.13. Next: the
[Quick start](#quick-start), the [100M training kit](examples/train_100m/README.md) for Colab, or
[reproducing the A100 benchmark](#reproducing-the-benchmark) in Colab (about 4 h 50 min).

---

**ASRN** is a sequence-mixing layer for language models, shipped as a PyTorch package that builds, trains, saves and
runs complete models. It is specialized in one job: finding where the current context occurred earlier in the text and
copying the token that followed. Attention compares every token with every earlier token, so its work per token grows
with the context; fixed-state models such as Mamba-2 and RWKV-7 have constant cost per token but compress the past into
a state of fixed size, which limits exact recall and copying. ASRN keeps exact access to earlier occurrences without
the growing comparison. Learned sign hashes (8 tables of 14 bits) put every position in a bucket, and each position
scores only its 32 most recent bucket mates per table. The score combines hash similarity, a learned confirmation term,
exact suffix matches of up to 4 tokens, and a learned, data-dependent *proper-time clock*. The layer copies the
embedding of the token that followed each match. A gate decides how much to copy. It reads Hawkes intensities over
repeated n-grams (1 to 4 tokens) and a repetition streak, the streak through a learned non-monotone response. All
recurrences run as a parallel associative scan, with fused Triton kernels on GPU. Activation memory is O(N), and in
training each token scores at most 8 × 32 candidates per layer whatever the sequence length (the bucket and n-gram sorts
are radix sorts on GPU). Step-by-step generation costs O(1) time per new token, however long the text: each table keeps
a ring with the 32 most recent positions of each bucket, and the Hawkes intensity is carried per n-gram as a running
state. Its memory still grows with the text, because what is copied must be stored.

The full model stacks causal local blocks (a short convolution and an MLP) and places one ASRN layer after every
second block. It has no positional table. The diagram shows the model and the inside of an ASRN layer; the
[architecture document](docs/ARCHITECTURE.md) has the detailed diagrams, and the
[technical report](docs/TECHNICAL_REPORT.md) has every formula.

<p align="center"><img src="docs/asrn_architecture.svg" width="100%"
alt="ASRN 1.0 architecture. Left: the language model, causal local blocks with an ASRN layer after every second
block, a final LayerNorm and a tied output head. Right: inside an ASRN layer. Where to copy: learned sign hashes put
each position in a bucket of each of 8 tables, and each token scores its 32 most recent bucket mates and copies the
token that followed each match. A learned proper-time clock sets the time decay. Whether to copy: seen flags of 1- to
4-token n-grams, Hawkes intensities and a repetition streak feed a gate. Below: training cost, O(1) writing per new
token and the numerical protections of the stable layer."></p>

> Code identifiers and docstrings are in Spanish, the author's language. The package also exports English aliases for
> its main names (`asrn.ASRNLayer`, `asrn.create_model`, `asrn.train`, ...; see [Quick start](#quick-start)). A short
> glossary: *pieza* = token, *casillero* = bucket, *reloj* = clock, *racha* = repetition streak, *puerta* = gate,
> *entrenar* = train, *escribir* = generate, *guardar*/*cargar* = save/load, *largo* = sequence length, *lote* = batch.

### Where to start

1. **Install** the package: [Installation](#installation) (`pip install .` from a clone, or straight from GitHub).
2. **Try it** on CPU in seconds: [Quick start](#quick-start) builds a layer and a full model, and the
   [command line](#command-line) trains one.
3. **Check it**: [Running the tests](#running-the-tests) (76 tests on CPU, 7 checks on GPU).
4. **Reproduce the benchmark** on CPU or in Colab on an A100: [Reproducing the benchmark](#reproducing-the-benchmark).
   The shipped results are in [`results/`](results/README.md).
5. **Read how it works**: the [architecture](docs/ARCHITECTURE.md) (diagrams), the
   [technical report](docs/TECHNICAL_REPORT.md) (every formula, every figure with its caveats) and the
   [release notes](docs/RELEASE_NOTES.md).
6. **Find your way around**: the [repository layout](#repository-layout) says what each folder holds.

### What 1.0 contains

- **An installable package** (`pip install .`). `import asrn` builds models whose ASRN layers are the
  *stable layer* (`CapaASRNEstable`), with numerical protection group A switched on. It has the same parameters as
  the v89 core layer and computes the same function up to rounding (1.3·10⁻⁷ in the logits). Checkpoints load in
  either layer, and `asrn.cargar` reads models saved by v83 through v89.
- **Generation in O(1) time per new token** (`asrn.generate`). It matches the previous O(t) writer and the whole-text
  forward pass up to 2.2·10⁻⁷; see [O(1) generation](#o1-generation-new-in-10).
- **A training command line**: `python -m asrn` (also installed as `asrn-train`), with FSDP for several processes.
- **A benchmark on an NVIDIA A100** against the official Mamba-2 (`mamba_ssm`) and RWKV-7
  (`flash-linear-attention`) and against our Transformer++. The benchmark records what favours ASRN and what does not.
- **The development history** of v85–v89 in [`history/`](history/README.md). It is not part of the package; it is
  kept because the documents cite it as evidence.

See the [release notes](docs/RELEASE_NOTES.md) for the full list of changes.

---

## Key results (A100 benchmark)

The models were trained and timed on one NVIDIA A100-SXM4-80GB in Colab on October 17, 2026, with PyTorch 2.14.1, in
bf16. Only the two compiled-training rows are compiled; every other row ran uncompiled. For the quality rows, each
model used the better of two learning rates (10⁻³, 3·10⁻³) on seed 0, and the figures are the mean ± standard deviation
of 3 seeds unless the cell says otherwise. ASRN is the 1.0 stable layer (group A). The forward pass and gradients of
1.0.0 are bit-identical to the code of this run, so every number applies unchanged to 1.0.0 except the generation
timing, which was measured with the previous O(t) writer (footnote ⁴). Every number in the table is in
[`results/banco_a100/`](results/banco_a100/RESUMEN.md), and the p-values in footnote ³ are paired t-tests on its
per-seed values. The previous A100 run (October 9 and 11, PyTorch 2.11, uncompiled) is kept as a replication in
[`history/results/banco_a100_pt211/`](history/results/banco_a100_pt211/RESUMEN.md), and the earlier development run
cited in footnote ³ is in [`history/results/`](history/results/README.md).

| Measurement | ASRN 1.0 | Transformer++ | Mamba-2 | RWKV-7 |
|---|---|---|---|---|
| One layer, forward + backward, 1,024 tokens (ms) | 12.1 | **3.3** | 7.7 | 11.6 |
| One layer, forward + backward, 262,144 tokens (ms) | **68.7** | 1,486.0 | 174.3 | 519.4 |
| One layer, peak memory at 262,144 tokens (GB) | 10.9 | **5.5** | 8.5 | 21.1 |
| ~50M model, training throughput at 1,024 tokens/sequence (tokens/s) | 172,065 | **182,116** | 118,002 | 50,439 |
| ~50M model, training throughput at 32,768 tokens/sequence (tokens/s) | **163,023** | 78,400 | 114,544 | 45,895 |
| ~50M model, compiled, training throughput at 1,024 tokens/sequence (tokens/s) ⁶ | 210,413 | **262,909** | — | — |
| ~50M model, compiled, training throughput at 32,768 tokens/sequence (tokens/s) ⁶ | **198,826** | 90,540 | — | — |
| ~50M model, generation with a 65,536-token context (ms/token) ⁴ | 43.1 | **15.2** | 24.5 | 39.5 |
| Copy, accuracy at 4× the training length (%) | **99.99, 99.98** (2 of 3 seeds) ⁷ | 0.2 ± 0.0 | 0.2 ± 0.0 ¹ | 0.2 ± 0.0 ¹ |
| MQAR, accuracy with 4× the training pairs (%) | **99.7 ± 0.4** | 0.3 ± 0.1 ² | 26.1 ± 25.6 | 16.5 ± 9.3 |
| WikiText-103 perplexity, 1,024-token context (~45M params) | **67.91 ± 1.00** | 93.27 ± 1.07 ⁵ | 74.25 ± 0.92 | 75.81 ± 6.39 ³ |
| WikiText-103 perplexity, 4,096-token context | **63.08 ± 0.97** | 128.87 ± 6.56 | 73.09 ± 0.88 | 73.96 ± 6.71 |

¹ Mamba-2 did not learn the copy task even at the training length (0.2%), and RWKV-7 had not finished learning it
within the budget (25.8% ± 20.0 at the training length), so neither row tests length extrapolation.
² Transformer++ did not learn MQAR with either learning rate in this budget. That reflects the budget or learning rate
of this protocol, not a limit of attention.
³ ASRN is ahead of RWKV-7 in each seed, but the difference is not significant (paired p = 0.13, n = 3; 0.084 at 4,096
tokens). In an earlier development run with another protocol and a reimplemented RWKV-7, RWKV-7 was the best on text
(47.87 vs 57.02 for ASRN).
⁴ ASRN's generation figures (time per token, prefix and memory) were measured with the O(t) writer of this run's code.
1.0.0 writes in O(1) per new token, and that writer has not been measured on the A100 yet (on CPU its time per token
does not grow with the context: see [O(1) generation](#o1-generation-new-in-10)).
Per-token time at batch 1, measured after the prefix. The prefix pass now computes logits for the last position only
(the previous run computed them for all 65,536 prefix positions of Transformer++). Re-measured at 65,536 tokens, the
prefix takes 0.387 s for Transformer++, 0.140 s for ASRN and 0.139 s for Mamba-2, and the peak memory of the whole
generation, prefix included, is 2.53, 3.82 and 1.26 GB. RWKV-7's prefix (111–131 s at every length) is dominated by
the compilation and autotuning of its fla kernels.
⁵ On WikiText, Transformer++ chose the lower edge of the two-rate grid (10⁻³; with 3·10⁻³, seed 0 reached 159.3) and
the other three models the upper edge (3·10⁻³).
⁶ ASRN compiled by parts (`compilar_partes`) and Transformer++ with `torch.compile`; the first step, which includes the
compilation, took 47.0 s and 26.2 s. Mamba-2 and RWKV-7 were not compiled.
⁷ Copy learning is bimodal. Seeds 0 and 2 left the initial plateau (loss < 1 from step 1,000); seed 1 stayed on it for
all 2,000 steps (final loss 6.232) and scored 0.15% at 4×, so the 3-seed mean is 66.71 ± 57.64. In the previous A100
run all 3 seeds learned (99.87 ± 0.18).

**Caveats that change the reading.** Compilation appears only in the compiled-training rows: the layer, generation,
quality and ablation measurements ran uncompiled. Transformer++ is our own Llama-recipe implementation, with
FlashAttention-2 through SDPA and with RMSNorm, RoPE and SwiGLU unfused. Mamba-2 ran without `causal-conv1d` and
uncompiled, so its speed is pessimistic and the ASRN-vs-Mamba-2 speedups are upper bounds; RWKV-7 ran with the fla
kernels. The single-layer rows compare different units: Mamba-2's whole block has 2.6× the parameters of the ASRN
layer. At 262,144 tokens the layer ratio is 2.5×, but the ratio between whole ~50M models is 1.4–1.5×. Quality was
measured at small scale: 2.5–46M parameters, and WikiText-103 training saw 20M positions (≈0.44 per parameter), far
from convergence. The protocol used two learning rates, three seeds and one text corpus that rewards copying. With the
same seeds, protocol and rival libraries but a different PyTorch version, the rivals' MQAR scores moved a lot between
this run and the previous one (Mamba-2 with 4× the pairs: 54.4 → 26.1; RWKV-7 at the training length: 99.9 → 65.4),
most likely because of non-deterministic GPU kernels and unconverged training, while ASRN stayed ≥ 96.45% in every
seed of both runs. ASRN also has exact token-id matching built in, an inductive bias aligned with copy and MQAR. The
[technical report](docs/TECHNICAL_REPORT.md) analyses every figure, with its caveats.

<p align="center">
  <img src="results/banco_a100/capa_tiempo.png" width="49%" alt="Time of one layer (forward and backward) against sequence length, 1,024 to 262,144 tokens: ASRN stays near 12 ms up to 16k tokens and reaches 69 ms at 262k; Transformer++ reaches 1,486 ms, Mamba-2 174 ms and RWKV-7 519 ms">
  <img src="results/banco_a100/modelo_velocidad.png" width="49%" alt="Training throughput of ~50M models against sequence length: ASRN nearly flat at 163-172k tokens/s, and 199-210k compiled; Transformer++ falls from 182k to 78k, and from 263k to 91k compiled; Mamba-2 115-118k; RWKV-7 46-50k">
</p>
<p align="center">
  <img src="results/banco_a100/calidad_copia.png" width="49%" alt="Copy accuracy against evaluation length, trained at 1,024 tokens, mean and range across seeds: ASRN mean 66.7% at 1x and 4x and 44.5% at 16x, with a range from chance to 100% (67% at 16x) because one of three seeds stayed on the initial plateau; Transformer++ 100% at its training length and chance beyond; Mamba-2 at chance; RWKV-7 25.8% at the training length">
  <img src="results/banco_a100/calidad_wikitext.png" width="49%" alt="WikiText-103 perplexity at 1,024 and 4,096 tokens of context: ASRN 67.9 and 63.1, Mamba-2 74.3 and 73.1, RWKV-7 75.8 and 74.0, Transformer++ 93.3 and 128.9">
</p>

*Chart labels are in Spanish: largo = sequence length, piezas por segundo = tokens per second, compilado = compiled,
exactitud = accuracy, perplejidad = perplexity, semillas = seeds. The same folder also has charts for MQAR, layer
memory, generation and the ablation.*

### Where ASRN leads and where it does not

**Leads, in this benchmark:**

- **Training throughput is nearly flat with sequence length** (−5.3% from 1,024 to 32,768 tokens). A ~50M model
  trains at 163–172k tokens/s from 1,024 to 32,768 tokens per sequence. ASRN overtakes Transformer++ between 1k and 4k
  tokens (~2.4k interpolated) and is 2.08× faster at 32k. It is 1.4–1.5× faster than the unfused Mamba-2 at every
  length and 3.4–3.6× faster than RWKV-7. Compiled, ASRN trains at 199–210k tokens/s (1.21–1.23× its uncompiled
  speed), overtakes the compiled Transformer++ at ~5.4k tokens and is 2.20× faster at 32k.
- **Single layers on very long inputs.** From ~20k tokens the ASRN layer is the fastest layer measured. At 262,144
  tokens it is 21.6× faster than attention, 2.5× faster than a Mamba-2 block and 7.6× faster than RWKV-7. In exchange
  it uses 2.0× the memory of attention and 1.3× that of Mamba-2, and about half that of RWKV-7.
- **Exact retrieval beyond the training length.** When ASRN leaves the initial plateau of the copy task, it copies at
  4× the training length with ≥ 99.98% (2 of 3 seeds here; 3 of 3, ≥ 99.67%, in the previous run), where Transformer++
  falls to chance. With 2× and 4× the MQAR pairs it scores 99.99% and 99.75%, against 57.17% and 26.09% for Mamba-2
  and 34.16% and 16.51% for RWKV-7 (paired p at 2× and 4×: 0.090, not significant, and 0.039 against Mamba-2; 0.025
  and 0.0041 against RWKV-7). These are the tasks that match ASRN's built-in exact matching.
- **WikiText-103 perplexity at this small budget**, with all the caveats above. ASRN is the best in every seed,
  significantly against Transformer++ and Mamba-2 (paired p ≤ 0.008) but not against RWKV-7 (p = 0.13 and 0.084).

**Does not lead:**

- **Short inputs at batch 1.** Up to 16k tokens one ASRN layer takes about 12 ms regardless of length, against
  3.3–10.0 ms for attention and 7.7–11.1 ms for Mamba-2. The floor is probably the cost of launching many small kernels
  without compilation (it was not profiled in this run). In development, CUDA Graphs brought one layer at 1,024 tokens
  from 11.4 to 1.6 ms. In the ~50M model at 1,024 tokens, Transformer++ is 6% faster, and 25% faster when both are
  compiled.
- **Memory per layer.** One ASRN layer needs about 40 KiB of peak training memory per token, against 20 KiB for
  attention and 31 KiB for a Mamba-2 block (RWKV-7: 79 KiB).
- **Generation on the A100.** It was the slowest of the four at batch 1 (43 ms/token), measured with the O(t) writer of
  this run's code. The GPU time was flat with context (slope 3.0 ns per context token, 95% CI −25 to 31 ns), probably
  because kernel launches dominate (not profiled), so the O(t) term did not show. 1.0.0 writes in O(1) per new token,
  but that writer has not been measured on the A100 yet; on CPU it removes the growth with the context (see
  [O(1) generation](#o1-generation-new-in-10)).
- **Learning speed within the training length.** Transformer++ learns the copy task in ≤ 200 steps. At the same
  learning rate ASRN needs 1,000 steps when it leaves the initial plateau (1,000–1,200 in the previous run), and
  leaving it within 2,000 steps is not guaranteed (it did in 5 of 6 base runs across both A100 runs). No comparison
  was made at equal compute.

### What the ablation shows

The ASRN layer separates two design responsibilities. **Retrieval** decides *where to copy from*: the bucketed sparse
attention (learned hash tables, candidate windows, and scores with exact suffix, confirmation and the time term)
performs exact retrieval. **Regulation** decides *whether to copy and when to stop*: the gate, the proper-time clock and
the Hawkes and streak signals regulate copying in open-ended generation. On the A100, each component was removed in turn
on copy and MQAR, with 3 seeds and the base's learning rates (10⁻³ for copy, 3·10⁻³ for MQAR); the base is the quality
run itself.

1. **The retrieval core solves copy and MQAR.** Without the ASRN layer (`solo_local`) both tasks stay at chance in 3 of
   3 seeds. This holds by construction, since the local blocks see only the last 13 tokens.
2. **The regulation mechanisms are non-intrusive on pure-copy tasks.** In copy and MQAR, 100% of the evaluated
   positions require copying, so the retrieval core is sufficient on its own. The presence of the regulation mechanisms
   does not degrade exact retrieval (≥ 99.98% on copy at 4× the training length in the seeds that learned, 99.75% on
   MQAR with 4× the pairs), and removing them does not change it either. Every copy run of the base and the regulation
   variants that left the initial plateau scores ≥ 99.89% at 4×, and the MQAR means with 4× the pairs are 99.75
   (base), 99.94 (`sin_puerta`), 99.50 (`puerta_sin_senales`), 99.95 (`sin_reloj`), 99.53 (`sin_hawkes_racha`), 99.69
   (`sin_sufijo`) and 99.88 (`sin_confirmacion`). Copy learning is bimodal, and seed 1 is the hard seed: it stayed on
   the plateau for all 2,000 steps in the base itself and without the gate, without the clock and with group B
   (`sin_sufijo` failed on seed 0 instead). Because the base itself failed on seed 1, one failed seed in a variant
   cannot be attributed to the removed component. Across both A100 runs (6 runs per variant), the copy was learned in
   5 of 6 base runs, 4 of 6 without the gate, the clock or the suffix, and 6 of 6 with a signal-blind gate, without the
   Hawkes and streak signals and without the confirmation.
3. **Their active role is validated in controlled open-ended generation.** In a controlled synthetic experiment with
   bounded quotations ([`benchmarks/experimento_citas.py`](benchmarks/experimento_citas.py): random tokens with
   occasional quotations of 6–12 tokens; 2 local blocks and 1 ASRN layer, d = 64, 6,000 steps, 3 seeds;
   [results](results/citas/RESUMEN.md)), the full 1.0 layer was compared with the same layer without the Hawkes and
   streak signals, with a gate that sees only the hidden state, with no gate, and without the proper-time clock. From
   the language-modelling loss alone, without explicit supervision, the full layer's gate learned to push copying when a
   quotation starts (+0.6 to +0.8 at a streak of one token) and to close hard once a streak outgrows the quotations in
   the data (−7.7 to −9.2 from 12 tokens on). This brakes *state aliasing*: without the streak, a 5-token and a 50-token
   quotation are the same state. With default sampling (T = 0.8, top-k 40), the longest written repetition streak is 10
   tokens with the full layer (real text: 10), against 21 without the Hawkes and streak signals, 27 with a signal-blind
   gate and 23 with no gate; streaks longer than 9 tokens are 0.3% of all streaks (real text: 0.9%), against 7.2%, 9.3%
   and 11.8%. Removing only the Hawkes and streak signals leaves the validation loss practically unchanged (3.703–3.706
   against 3.707–3.708), so the effect shows in what the model writes, not in its loss. Without the clock the layer
   writes like the full one (longest streak 10, 0.2% of streaks above 9, same loss): this task has no topic shifts,
   which is what the clock is for, so its role is still to be tested. A known limit: how often quotations start is not
   calibrated (quoted fraction 20.9% at T = 0.8 and 9.3% at T = 1, against 15.6% in the real text). The same experiment
   on the A100 ([results](results/citas_a100/RESUMEN.md)) replicates the result. The gate learned the same response
   (+0.6 to +0.7 at a streak of one token, −7.6 to −8.4 from 12 tokens on); with default sampling the longest streak is
   10 tokens with the full layer, against 19, 39 and 21, and streaks longer than 9 are 0.3%, against 7.9%, 9.2% and
   12.0%. Across the four sampling settings, the longest streak per seed is 9–12 with the full layer and 9–10 without
   the clock, against 14–24 without the Hawkes and streak signals, 15–39 with a signal-blind gate and 18–31 with no
   gate. The validation loss is again practically unchanged without those signals (3.703–3.705 against 3.706–3.709).
4. **Retrieval refinements.** The exact suffix, the confirmation and the number of tables refine retrieval; their
   individual contribution is not separable on these saturated tasks. With only two tables, copy was learned in 1 of 3
   seeds in each A100 run, which is suggestive. Mixing the tables by log-sum-exp (`mezcla_lse`) left the copy plateau
   by step 400 in 3 of 3 seeds in the previous run, but this did not reproduce: here it learned the copy in 1 of 3
   seeds. On MQAR, with the implementation error that made it diverge in the previous run fixed, it no longer diverges
   but does not learn (≤ 3.61% with 4× the pairs). It costs +61–62% per run (PyTorch only) and is not adopted in 1.0.
   Protection group B, off by default, showed no consistent cost or effect.

The large-scale ablation of the regulation mechanisms on natural corpora with distractors and topic shifts is planned
for v1.1 (see the [roadmap](#roadmap--future-work-v11)). Section 7 of the [technical report](docs/TECHNICAL_REPORT.md)
audits each component.

### Numerical stability (new in 1.0)

A review of the v89 core layer reproduced the following failures. The stable layer closes each one.

| Failure of the core layer | Measured (A100 unless noted) | Stable layer |
|---|---|---|
| The causal mask is a score of −10⁹, not −∞. With an extreme time kernel, the softmax moves its mass to invalid keys, including future tokens | The future changes the past by 0.016 with γ = 10⁹ (Triton kernel) | 0 exactly (`mascara_exacta`) |
| The 32-bit clock B_t = Σβ, computed with CUDA `cumsum` | The error of B_t − B_{t−1} is 2.2% at 2²⁰ tokens, 8.6% at 2²² and 50% at 2²⁴. The clock stops advancing on 1.1% of steps at ~4M tokens and on 16% at ~16M | 64-bit clock, split into two 32-bit halves in the kernel (`reloj_64`) |
| The score cancels large clock values | With the clock offset by 10⁶, 1.6·10⁷ and 10⁹, the output drifts by 9.4·10⁻⁵, 2.8·10⁻³ and 8.8·10⁻² | 5.5·10⁻⁸, the same at every offset (TF32 off) |
| TF32 in the clock projection | B_t moves 2.0 units over 65,536 tokens | 1.2·10⁻¹⁰ (`_ProyeccionIEEE`) |
| β has no upper bound | B overflows and the layer outputs NaN (reproduced in a CPU test) | β ≤ 10⁴ (`beta_max`) |

Group A also includes two preventive bounds, `log_alfa_min` and `mejor_max`. The failures they guard against were
deduced from the code and were not measured. On the A100, group A as a whole costs +7.5% time and +5.2% memory at
262,144 tokens (+4–5% memory at every length), measured with `_ProyeccionIEEE` included. The protection is advisable
from ~1M tokens per sequence and necessary from ~16M. Group B (`tope_puerta`, `kappa_min`) changes the function, so it
is off by default and intended for training from scratch on long inputs. `RecorteDinamico` clips the gradient per
parameter group (clock, gate, rest) with an adaptive threshold.

### O(1) generation (new in 1.0)

Writing step by step (`asrn.generate`, i.e. `escribir`) costs O(1) time per new token, independent of how much text
there is; before 1.0.0 it was O(t). Each table keeps, per bucket, a ring with the 32 most recent positions, exactly the
keys the previous writer chose among the whole memory, so the new token scores only its 8 × 32 candidates. The Hawkes
intensity is carried per n-gram key (the clock and the intensity at the key's last occurrence), the same recurrence
that the whole-text scan uses, and the repetition streak is one number. On CPU, the O(1) writer equals the previous
O(t) writer and the whole-text forward pass up to 2.2·10⁻⁷, and `escribir()` gives the same token sequences as the
previous implementation in 192 of 192 combinations of sampling options.

Time per new token on CPU ([`results/escritura_cpu/`](results/escritura_cpu/RESUMEN.md),
[`benchmarks/medir_escritura.py`](benchmarks/medir_escritura.py)): 4 threads, PyTorch 2.14.1, an ASRN model of 700,484
parameters, batch 1, no sampling, median of 9 batches of 32 tokens, the two writers interleaved at each context.

| Context (tokens) | O(1) writer (ms/token) | O(t) writer (ms/token) | O(1) writer, prompt (s) | O(t) writer, prompt (s) |
|---|---|---|---|---|
| 512 | **12.45** | 15.80 | 0.10 | 0.06 |
| 2,048 | **12.47** | 16.73 | 0.26 | 0.20 |
| 8,192 | **10.68** | 19.02 | 0.77 | 0.70 |
| 32,768 | **11.26** | 37.12 | 3.60 | 2.73 |
| 131,072 | **9.49** | 91.67 | 29.04 | 20.44 |

The slope of the time per token against the context is −19.21 ns per context token for the O(1) writer (95% CI −41.54
to 3.12, indistinguishable from zero) and 583.31 ns for the O(t) writer (95% CI 537.13 to 629.49). At 131,072 tokens
the O(1) writer is 9.7× faster per token, and 1.27× at 512. Its prompt pass also fills the rings and the Hawkes state,
profiled at about 0.3 s at 32,768 tokens and 1.3 s at 131,072; the differences in the table are larger because the
prompt time varies from run to run on this shared 4-core machine (the same prompt pass took 14.8–23.5 s at 131,072
tokens in four repetitions). Memory still grows with the text (each position's cached features), plus fixed rings of
17 MiB per ASRN layer per sequence at the default sizes, about 120 MiB per sequence for the ~50M benchmark model. The
O(1) writer has not been measured on the A100 yet: the A100 generation figures above come from the O(t) writer, and
the notebook's generation section now runs the O(1) writer, so the next Colab run measures it.

---

## Installation

```bash
pip install git+https://github.com/KIROKOWA/ASRN   # the package only, straight from GitHub
# or from a clone, with the tests, the benchmark and the examples:
git clone https://github.com/KIROKOWA/ASRN.git
cd ASRN
pip install .                  # Python >= 3.11, PyTorch >= 2.9, NumPy
pip install ".[bench]"         # optional, for the benchmark: flash-linear-attention, datasets, tiktoken, einops, matplotlib
```

`requirements.txt` lists the same runtime dependencies (`torch>=2.9`, `numpy`) for running from the clone without
installing. `mamba-ssm` is not part of the `bench` extra and is installed separately (the A100 notebook installs it).

- **GPU and Triton.** On CUDA, the bucket attention and the associative scan run as Triton kernels when Triton is
  importable. Triton is optional: without it, or on CPU, every kernel has a PyTorch path that computes the same values
  up to rounding. `ASRN_TRITON=0` disables the kernels, and `TRITON_INTERPRET=1` runs them in the Triton interpreter on
  CPU, which checks the arithmetic but not the speed.
- **Precision.** Keep the weights in float32 and get bf16 speed through autocast (`entrenar(..., precision='bf16')`, or
  `torch.autocast` at inference). The layer mixes 32- and 64-bit arithmetic, so converting a model with
  `.to(torch.bfloat16)` or `.half()` is rejected with an explicit error.
- **Compilation.** `torch.compile` of this model requires PyTorch ≥ 2.14. Compiling by parts (`compilar='partes'`,
  `--compilar` or `asrn.compilar_partes(model)`) compiles the local blocks and the small parts of each ASRN layer and
  keeps the Triton kernels. During development it was the fastest GPU setting for the core model (+26% training
  throughput on an A100). In the 1.0 benchmark (`results/banco_a100/`, PyTorch 2.14.1) it raises the training
  throughput of the ~50M model 1.21–1.23× at every length; the other benchmark sections ran uncompiled.
- **Colab.** [`benchmarks/asrn_colab.ipynb`](benchmarks/asrn_colab.ipynb)
  ([open in Colab](https://colab.research.google.com/github/KIROKOWA/ASRN/blob/main/benchmarks/asrn_colab.ipynb))
  runs every test, the full A100 benchmark and the bounded-quotation experiment with PyTorch 2.14, and
  [`examples/train_100m/`](examples/train_100m/README.md)
  ([open in Colab](https://colab.research.google.com/github/KIROKOWA/ASRN/blob/main/examples/train_100m/entrenar_asrn_100m.ipynb))
  is a ready-to-run Colab kit to train a 103.5M-parameter model on FineWeb (its first run is on the
  [roadmap](#roadmap--future-work-v11)).

## Quick start

The package exports English aliases for its main names. They are the same objects (`asrn.train is asrn.entrenar`):
`ASRNLayer` (= `CapaASRNEstable`, the default layer), `ASRNCoreLayer` (= `CapaASRN`, the unprotected core layer),
`ASRNModel`, `ASRNConfig`, `LocalBlock`, `Protections`, `DynamicClip`, `create_model`, `create_from_config`, `convert`,
`train`, `evaluate`, `generate`, `save` and `load`. Keyword arguments and methods keep their Spanish names (`V`, `d`,
`n_capas`, `semilla`, `largo`, `lote`, `pasos`, ...).

**A standalone layer and a full model** (runs on CPU in seconds):

```python
import torch
import asrn

# One ASRN layer for a model of width d = 64: asrn.ASRNLayer(d, protecciones=None, **layer options).
layer = asrn.ASRNLayer(64)                       # the stable layer, protection group A
E = torch.randn(2, 256, 64)                      # token embeddings  (batch, length, d)
h = torch.randn(2, 256, 64)                      # hidden state      (batch, length, d)
ids = torch.randint(0, 32000, (2, 256))          # token ids         (batch, length), for exact matching
out = layer(E, h, ids)                           # (batch, length, d): h plus what the layer copied

# A complete language model: local blocks, with an ASRN layer after every second block.
model = asrn.create_model(V=32000, d=64, n_capas=4, semilla=0)
logits = model(ids)                              # (batch, length, V)
```

Inside the model, `E` is the token embedding and `h` the output of the preceding local block.

**Size, train, load, evaluate and generate.** `tokens` is a 1-D tensor of token ids from any tokenizer.

```python
import asrn

# ASRNConfig.para_parametros finds a shape for any target size (exact count, no memory used).
cfg = asrn.ASRNConfig.para_parametros('50m', V=32000)
print(cfg.d, cfg.n_capas, cfg.contar_parametros())      # 600 9 48623832
model = asrn.create_from_config(cfg, semilla=0).to('cuda')
# or by hand: model = asrn.create_model(V=32000, d=512, n_capas=12, semilla=0)

# Train: AdamW, warmup + cosine, bf16 autocast on GPU, NaN steps skipped, Ctrl+C saves.
# Writes modelo.pt (weights, optimizer and step, for resuming) and modelo.mejor.pt (best validation, weights only).
history = asrn.train(model, tokens, validacion=tokens_val, largo=1024, lote=16, pasos=20000, lr=1e-3,
                     guardar_en='modelo.pt',
                     recorte=asrn.DynamicClip(model))   # optional; the default is a global clip at 1.0
# Resume later with asrn.train(model, tokens, ..., reanudar='modelo.pt').

# Load, evaluate and generate.
model = asrn.load('modelo.mejor.pt', dispositivo='cuda')
print(asrn.evaluate(model, tokens_val, largo=1024))      # {'perdida': ..., 'perplejidad': ..., 'piezas': ...}
sequences, logits = asrn.generate(model, prompt, n_nuevos=200, temperatura=0.8, top_k=40)
# prompt: (length,) or (batch, length). generate returns (prompt + new tokens, per-step logits) on the CPU;
# pass devolver_logits=False to get (sequences, None). asrn.save(model, 'file.pt') saves config + weights.
```

**Training from scratch on long inputs.** Add protection group B, which bounds the gate logit and sets a floor on the
Hawkes and streak decay rates so that intensities stay in the range seen in training:

```python
protection = asrn.Protections.textos_largos(largo_entrenamiento=4096)
model = asrn.create_model(V=32000, d=512, n_capas=12, semilla=0, protecciones=protection)
```

**The unprotected core.** `asrn.nucleo` is the v89 code with the original `CapaASRN` (`asrn.ASRNCoreLayer`), and
`Protections.ninguna()` reproduces it bit for bit. `asrn.convert` swaps the ASRN layers of a model for stable ones in
place, keeping the weights:

```python
import asrn.nucleo
core = asrn.nucleo.crear_modelo(V=32000, d=512, n_capas=12, semilla=0)   # CapaASRN, no protections
asrn.convert(core)                                                        # now CapaASRNEstable, group A, same weights
```

**Choosing a size.** `ConfigASRN.para_parametros(target, V, **fixed)` takes a number or a string ('3k', '50m', '1.5b',
7e9). It searches width and depth with an exact closed-form parameter count, and keeps any field you fix (for example
`d=512` or `n_capas=12`). Among the shapes within 3% of the target, it picks the one whose width-to-depth ratio is
closest to 64. That is a reasonable shape rule, not a measured optimum. If no shape comes within 50%, it raises an error
that explains why. `asrn.TALLAS` holds example configurations from `'nano-3k'` to `'27b'`. Being able to build and
shard a model of a given size says nothing about how well it learns. With the 1.0 recipe, the largest model trained
for quality has ~45M parameters (WikiText-103; the ~51M models of the speed benchmark were only timed). Earlier
versions, with a different shape, trained one of ~300M. The `'1b'` configurations and larger have never
been trained.

Other entry points: `checkpoint='asrn'` or `'todo'` recomputes layers in the backward pass to save memory.
`crear_desde_config(cfg, dispositivo='meta')`, followed by `repartir_fsdp` and `materializar`, builds a sharded model
that never exists whole in one process. `PasoGrafado` records a whole training step as a CUDA Graph.

## Command line

`python -m asrn` (or `asrn-train` after `pip install .`) trains a model with the stable layer.
`python -m asrn.nucleo` takes the same flags and trains the unprotected core. (For this form Python prints a
`RuntimeWarning` from `runpy`, because the package has already imported `asrn.nucleo`; training runs normally.)

```bash
# one GPU: choose a size with --parametros (needs --V), or take an example with --talla
python -m asrn --datos tokens.bin --V 32000 --parametros 125m --guardar modelo.pt
asrn-train --datos tokens.bin --talla 125m --largo 1024 --lote 16 --pasos 20000 --checkpoint todo --guardar modelo.pt

# several GPUs with FSDP (launch with torchrun; use --ancho rather than --d)
torchrun --nproc_per_node=8 -m asrn --fsdp --parametros 1b --V 32000 --datos tokens.bin --guardar modelo.pt

# resume from the --guardar file (it holds the optimizer state and the step); pass the original --pasos
python -m asrn --datos tokens.bin --reanudar modelo.pt --guardar modelo.pt --pasos 20000
```

| Flag | Meaning |
|---|---|
| `--datos`, `--validacion` | Token ids: `.pt` (1-D tensor), `.bin` / `.u16` (raw uint16), `.i32` / `.u32` (raw int32). Without `--validacion`, the last `--fraccion_val` of the data (default 0.01) is held out |
| `--V`, `--parametros`, `--talla` | Vocabulary size; target parameter count (`3k`, `50m`, `1.5b`); or an example from `TALLAS` |
| `--ancho` / `--d`, `--capas`, `--largo_cada` | Width, number of local blocks, one ASRN layer every *n* blocks (default 2) |
| `--mezcla`, `--memoria_max`, `--dropout` | Local block: `conv` (the release recipe) or `memoria` (experimental); with `memoria`, tokens it remembers (1024); dropout (0) |
| `--largo`, `--lote`, `--acumular`, `--pasos` | Tokens per sequence (512), sequences per batch (16), accumulated batches (1), steps (20000) |
| `--lr`, `--calentamiento`, `--decaimiento` | Learning rate (1e-3), warmup steps (default 2% of the steps, at most 2,000), weight decay (0.1) |
| `--precision` | `auto` (bf16 on GPUs that support it), `fp32`, `bf16`, `fp16` |
| `--checkpoint [asrn\|todo]` | Recompute the ASRN layers, or everything, in the backward pass |
| `--compilar [partes\|todo]` | `torch.compile` by parts (no value; PyTorch ≥ 2.14) or the whole model |
| `--fsdp` | Shard the model across the torchrun processes |
| `--evaluar_cada`, `--guardar`, `--guardar_cada`, `--reanudar`, `--semilla` | Validation interval (500), output file (`modelo.pt`), save interval (1000), resume file, seed (0) |

`python -m asrn --help` lists every flag. The `.mejor.pt` file holds only the weights, so load it with `asrn.cargar()`.

## Running the tests

```bash
python tests/test_nucleo.py          # 57 tests of the core (a few minutes on CPU)
python tests/test_nucleo.py fsdp     # only the tests whose name contains "fsdp"
python tests/test_estable.py         # 19 tests of the stable layer, the package API and the English aliases
python tests/verificar_gpu.py        # 7 GPU checks: bf16, checkpoint memory, FSDP with NCCL, torch.compile,
                                     #   GPU = CPU, causality, step-by-step generation = full sequence
```

The core tests cover the Triton kernels against their PyTorch paths, the O(1) writer against the O(t) writer and the
whole-text pass, gradient checkpointing, saving, loading and resuming, meta-device construction, FSDP2 with two CPU
processes, causality and edge cases. On CPU, the Triton kernels run in the interpreter. For 1.0.0 both suites pass in
full on CPU with Python 3.13.16, PyTorch 2.14.1 and Triton 3.8.0: 57 / 57 and 19 / 19 (76 tests). The A100 run of
October 17, 2026 (PyTorch 2.14.1) passed the suites of that time, 55 / 55 and 18 / 18, and the GPU checks 7 / 7,
including `torch.compile` (see [`results/banco_a100/`](results/banco_a100/RESUMEN.md)). With PyTorch 2.11.0, 55 core
tests pass and the two `torch.compile` tests are skipped (they need 2.14), and the stable suite passes 19 / 19; on that
version, Triton 3.6 runs the kernel tests in its interpreter only with NumPy < 2.4. The [CI
workflow](.github/workflows/tests.yml) runs both suites on CPU with Python 3.11 and 3.13; without Triton, the
kernel-versus-PyTorch tests are skipped. See the [release notes](docs/RELEASE_NOTES.md#verification).

## Reproducing the benchmark

**On CPU**, to get the exact cost per layer (FLOPs and bytes saved for the backward pass), latency, generation and
small-scale quality:

```bash
python benchmarks/banco_asrn.py costo latencia escritura            # ~5 min
TRITON_INTERPRET=1 python benchmarks/banco_asrn.py costo_kernel     # saved bytes with the Triton kernel
python benchmarks/banco_asrn.py calidad ablacion --tareas copia,mqar   # hours on CPU
python benchmarks/banco_asrn.py resumir                             # tables in results/banco_cpu/RESUMEN.md
python benchmarks/medir_escritura.py                                # O(1) and O(t) writers: results/escritura_cpu/
```

These commands write to `results/banco_cpu/` (`medir_escritura.py` to `results/escritura_cpu/`) and replace the
shipped results; add `--salida /tmp/banco_cpu` (or another folder) to each command to keep them.

The CPU text task reads the Vim 9.1 documentation from `/usr/share/vim/vim91/doc` (package `vim-runtime`).

**On an A100**, open [`benchmarks/asrn_colab.ipynb`](benchmarks/asrn_colab.ipynb) in Colab with an A100 runtime and
choose *Run all*:

[![Open in Colab: reproduce the A100 benchmark](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KIROKOWA/ASRN/blob/main/benchmarks/asrn_colab.ipynb)
**Reproduce the A100 benchmark in Colab** (A100 runtime; the full run takes about 4 h 50 min).

This one notebook runs everything on the GPU with PyTorch 2.14: it installs PyTorch 2.14.1 (the session restarts once;
then *Run all* again), the test suites and the 7 GPU checks, the GPU stability checks, one layer, the ~50M model (also
compiled), generation (now with the O(1) writer), copy, MQAR, WikiText-103, the ablation and the bounded-quotation
experiment. It installs the official rivals; if an original rival does not train on that GPU, it falls back to the
reimplementation in `rivales_lineales.py` and records this in the `implementacion` column. It saves every section to
Google Drive as soon as the section finishes and resumes after a disconnection. The full run of October 17, 2026 took
~4.7 h of sections on an A100 (the notebook's first cell gives the time of each section), and its results are in
`results/banco_a100/` and `results/citas_a100/`; a quick mode is estimated at ~1 h 45 min.

**The bounded-quotation experiment** (CPU; the Colab notebook also runs it on the GPU, into `results/citas_a100/`)
trains the full 1.0 stable layer and four variants (`sin_hawkes_racha`, `puerta_sin_senales`, `sin_puerta`, `sin_reloj`)
on synthetic text with bounded quotations, and compares what each one writes with the real continuation:

```bash
python benchmarks/experimento_citas.py --semillas 0 1 2   # one run per seed -> results/citas/semilla_<s>.json
python benchmarks/experimento_citas.py --resumir          # joins the seeds: results/citas/resultados.json, RESUMEN.md
```

Seeds can run in separate processes, and an interrupted run resumes from the last finished model. The development
probes stay in `history/` (for example `python history/probes/reloj.py`).

## Repository layout

```text
asrn/                     the package (pip install .)
├── __init__.py           public API: crear_modelo, crear_desde_config, cargar (stable layer, group A), entrenar,
│                         evaluar, escribir, guardar, ConfigASRN, TALLAS, Protecciones, RecorteDinamico, convertir, ...
│                         and their English aliases (ASRNLayer, create_model, train, generate, ...)
├── __main__.py           python -m asrn / asrn-train: the training command line with the stable layer
├── estable.py            CapaASRNEstable, Protecciones, convertir(), RecorteDinamico
└── nucleo.py             the v89 core: CapaASRN and its Triton kernels (incl. the ESTABLE mode), BloqueLocal,
                          ModeloASRN, ConfigASRN/TALLAS, entrenar, evaluar, escribir (the O(1) writer),
                          guardar/cargar, FSDP, CLI
tests/                    the test suites: test_nucleo.py (57 tests of the core), test_estable.py (19 tests of the
                          stable layer and the API), verificar_gpu.py (7 GPU checks)
benchmarks/               the 1.0 benchmark and the experiments, with the Colab notebooks that run them
├── asrn_colab.ipynb      the Colab notebook: every test, the A100 benchmark and the quotation experiment
│                         (PyTorch 2.14)
├── banco_asrn.py         the benchmark entry point (CPU and A100 profiles); banco_gpu.py (GPU-only measurements),
│                         banco_modelos.py (ASRN, its ablation variants and the reference layers)
├── modelos_oficiales.py  Transformer++ and the official Mamba-2 and RWKV-7; rivales_lineales.py (reimplementations)
├── experimento_citas.py  the bounded-quotation experiment with the 1.0 stable layer (CPU or GPU)
├── medir_escritura.py    time per new token against the context, O(1) and O(t) writers (CPU or GPU)
└── builders/             scripts that generate the Colab notebooks (asrn_colab.ipynb and the 100M kit's)
results/                  measured results, as they came out
├── banco_a100/           the A100 run, Oct 17, 2026 (PyTorch 2.14.1): JSON, summary, charts
├── banco_cpu/            CPU runs (generation measured with the O(t) writer): JSON, summary
├── citas/                the bounded-quotation experiment: per-seed JSON and logs, then the joined summary
├── citas_a100/           the same experiment on the A100 (GPU replication): per-seed JSON and the joined summary
└── escritura_cpu/        the O(1) and O(t) writers on CPU, Oct 23, 2026: JSON, summary
examples/train_100m/      ready-to-run Colab kit to train a 103.5M-parameter ASRN on FineWeb
docs/                     ARCHITECTURE.md, TECHNICAL_REPORT.md, RELEASE_NOTES.md (each also in Spanish, *_es.md);
                          asrn_architecture.svg and asrn_arquitectura.svg (the diagram above, English and Spanish),
                          drawn by hacer_diagrama.py
paper/                    the preprint: LaTeX sources (English and Spanish), figures, bibliography, Zenodo metadata
history/                  development history of v85-v89 (docs, experiments, results, probes) and the previous A100 run
                          (results/banco_a100_pt211/, PyTorch 2.11, kept as a replication); not part of the package,
                          kept because the documents cite it as evidence
.github/                  the CI workflow (workflows/tests.yml: both test suites on CPU, Python 3.11 and 3.13) and the
                          issue templates
pyproject.toml            package metadata; also requirements.txt, LICENSE, README.md, README.es.md
CHANGELOG.md              the changes of each version; also CITATION.cff (citation metadata) and SECURITY.md (security
                          policy: loading checkpoints)
```

Folder guides, each with what is there, how to run it and where its outputs go: [benchmarks/](benchmarks/README.md) ·
[results/](results/README.md) · [examples/train_100m/](examples/train_100m/README.md) · [paper/](paper/README.md) ·
[history/](history/README.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [español](docs/ARCHITECTURE_es.md) | The architecture diagram, package layout, model and layer data flow (Mermaid diagrams), the clock pipeline, training and generation paths |
| [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) · [español](docs/TECHNICAL_REPORT_es.md) | Full method and equations, numerical stability, every benchmark figure with its caveats, the ablation and a per-component evidence audit |
| [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md) · [español](docs/RELEASE_NOTES_es.md) | Changes in 1.0 and compatibility |
| [CHANGELOG.md](CHANGELOG.md) | The short list of changes of each version |
| [results/README.md](results/README.md) | Index of the raw results |
| [benchmarks/README.md](benchmarks/README.md) · [examples/train_100m/README.md](examples/train_100m/README.md) | The benchmark files and the 100M training kit |
| [history/README.md](history/README.md) | Development history, with a map from pre-1.0 paths to the 1.0 layout |
| [paper/README.md](paper/README.md) | Preprint drafts in LaTeX (English and Spanish) and their bibliography |

## Limitations

- **Scale.** Quality was measured with 2.5–46M parameters and speed with ~50M, and WikiText-103 training was far from
  convergence. Behaviour at larger scale has not been measured yet; the first 100M-parameter run is on the roadmap.
- **Protocol.** The study used two learning rates, three seeds and one text corpus, and the learning rates were chosen
  on the same evaluation that is reported. With the same learning rate of 10⁻³ (one seed), Mamba-2 was ahead of ASRN on
  WikiText (86.1 vs 89.6). No comparison was made at equal compute. The rivals' MQAR scores varied strongly between the
  two A100 runs (most likely non-deterministic GPU kernels and unconverged training; see the caveats above).
- **Text quality against RWKV-7 is unresolved.** In this benchmark ASRN scored 67.9 and RWKV-7 75.8 (not significant,
  p = 0.13). In an earlier run, with another protocol and a reimplemented RWKV-7, RWKV-7 was better (47.87 vs 57.02).
- **Inductive bias.** ASRN receives token ids and has exact matching built in. The MQAR variant used here is at
  ceiling for ASRN.
- **Robustness of copy learning.** Leaving the initial plateau within 2,000 steps at 10⁻³ is not guaranteed: one of
  the 3 base seeds stayed on it in this run (5 of 6 base runs learned across both A100 runs), and so did 5 of the 24
  copy runs of the base and its seven light ablation variants (group B added, or the gate, the gate's signals, the
  clock, the Hawkes and streak signals, the suffix or the confirmation removed). With 3·10⁻³, seed 0 left it by step
  400 (both rates tied at 100% at the training length, so the rule chose 10⁻³). This is optimization, not capacity:
  when ASRN leaves the plateau it copies at 4× with ≥ 99.98% (≥ 99.67% in the previous run), and generalization to 16×
  depends on the seed (66.8% and 66.5% here, 43% to 97% in the previous run). The same sensitivity made one core test
  (`test_entrenar_aprende_a_copiar_y_reanuda`) fail on the Colab A100 machine during development: at a learning rate
  of 3e-3, whether a small model leaves the plateau in 600 steps depends on the order of floating-point sums (thread
  count, MKL instruction set, PyTorch version). The test now trains at 6e-3, which left the plateau in every seed,
  thread count and PyTorch version tried, and it passes on the Colab A100 machine (October 17 run); the plateau itself
  remains a property of copy learning.
- **Speed and memory.** There is a ~12 ms floor per layer on short inputs. The layer uses about twice attention's memory
  per token. On the A100, generation was the slowest of the four models (measured with the O(t) writer; most likely
  bound by kernel launches). With the O(1) writer of 1.0.0 the cost per new token is constant in the context (CPU
  measurement), but the A100 figure is pending. Memory per sequence still grows with the text, plus the fixed rings
  (17 MiB per ASRN layer per sequence at the default sizes).
- **Regulation mechanisms on natural text.** The gate, the clock and the Hawkes and streak signals are non-intrusive on
  copy and MQAR. The active role of the gate and of the Hawkes and streak signals is validated in one controlled
  synthetic experiment (3 seeds, d = 64, on CPU and replicated on the A100); the clock had no measurable effect there,
  and its intended role (topic shifts) is untested. Their ablation on natural corpora with distractors and topic
  shifts is planned for v1.1. The suffix and the confirmation are retrieval refinements whose individual contribution
  is not separable on these saturated tasks.
- **Engineering.** FSDP has been tested with two CPU processes and with one GPU process (NCCL), but not yet across
  several GPUs. `RecorteDinamico` is untested with FSDP and re-warms after a resume. fp16 with FSDP is rejected (use
  bf16). Saving a sharded model gathers the full weights and AdamW state in the RAM of process 0, about 12 bytes per
  parameter. `log_alfa_min` and `mejor_max` have no dedicated tests.

## Roadmap / Future work (v1.1+)

ASRN 1.0 trains a ~50M-parameter model at a nearly flat 163–172k tokens/s (199–210k compiled by parts) from 1,024 to
32,768 tokens per sequence on a single A100. To scale further, the release includes FSDP2 sharding (`repartir_fsdp`),
covered by a two-process FSDP test on CPU in `tests/test_nucleo.py` and a one-process NCCL check on the A100, and
[`examples/train_100m/`](examples/train_100m/README.md), a ready-to-run Colab kit for a 103.5M-parameter model on
FineWeb. The next versions build on this:

- **Kernels.** A Triton kernel for log-normalizer table mixing (`mezcla_lse`) only if more seeds show that it leaves
  the copy plateau earlier, and the kernel proposals of the [technical report](docs/TECHNICAL_REPORT.md): a persistent
  sliding window, a shared n-gram sort, a 14-bit radix sort and a grid-parallel scan.
- **Generation on the GPU.** Measure the O(1) writer on the A100 and cut the per-token kernel launches of generation
  (for example with CUDA Graphs), which dominate on the GPU.
- **Scale.** The first 100M-parameter run on FineWeb with `examples/train_100m/`.
- **Regulation at scale.** A large-scale ablation of the gate, the clock and the Hawkes and streak signals on natural
  corpora with distractors and topic shifts.
- **Protocol.** Wider learning-rate grids (including Transformer++ at 3·10⁻⁴ and 6·10⁻⁴) and more seeds, which copy
  learning and the rivals' run-to-run variance call for.
- **Toolchain.** Compiled runs of the sections that are still uncompiled (layer, generation, quality), and Mamba-2
  with `causal-conv1d`.

## Citation

If you use ASRN, please cite the software ([`CITATION.cff`](CITATION.cff) has the same data for GitHub and Zenodo):

```bibtex
@software{calvario2026asrn,
  author  = {Calvario, Alexis},
  title   = {{ASRN}: Adaptive Sparse Recurrence Network},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/KIROKOWA/ASRN}
}
```

and the accompanying preprint, *ASRN: Adaptive Sparse Recurrence Networks — Linear-memory long-range copying with a
learned proper-time clock and a Hawkes-gated copy gate* (Alexis Calvario, 2026), published on Zenodo with DOI
[10.5281/zenodo.23607506](https://doi.org/10.5281/zenodo.23607506); its LaTeX sources are in [`paper/`](paper/README.md):

```bibtex
@misc{calvario2026asrnpreprint,
  author    = {Calvario, Alexis},
  title     = {{ASRN}: Adaptive Sparse Recurrence Networks --- Linear-memory long-range copying with a learned
               proper-time clock and a {Hawkes}-gated copy gate},
  year      = {2026},
  publisher = {Zenodo},
  note      = {Preprint},
  doi       = {10.5281/zenodo.23607506},
  url       = {https://doi.org/10.5281/zenodo.23607506}
}
```

## License

The code is released under the [Apache License 2.0](LICENSE). The paper sources and figures in
[`paper/`](paper/README.md) are released under CC BY 4.0 ([`paper/LICENSE.md`](paper/LICENSE.md)). Everything else in
the repository (tests, benchmarks, results, examples, documentation and history) follows the code license.

**Author:** Alexis Calvario ([ORCID 0009-0004-0854-6398](https://orcid.org/0009-0004-0854-6398)) — conception,
architecture, implementation, experiments and direction of the project. Every measured figure in this repository is
traceable to runnable code or to the published results; what was deduced rather than measured is marked as such. If
something does not hold, please open an issue.
