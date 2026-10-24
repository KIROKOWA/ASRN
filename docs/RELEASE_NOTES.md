# ASRN 1.0.0 — Release notes

*October 2026 · Alexis Calvario · Apache 2.0 · [Versión en español](RELEASE_NOTES_es.md)*

ASRN (Adaptive Sparse Recurrence Network) is a language-model layer that finds earlier occurrences of the current
context with learned hash tables (8 tables, 14-bit buckets, at most the 32 most recent bucket-mates per table) and
copies what came next. A data-dependent proper-time clock, per-n-gram Hawkes intensities and a repetition-run counter
feed a copy gate. Work per token in the bucketed attention is bounded (at most 8 × 32 candidates), and the activations
saved for the backward pass grow linearly with length. Writing step by step costs O(1) time per new token, whatever
the length of the text; its memory still grows with the text.

1.0.0 is the first packaged release. It turns the v89 research code into an installable package, makes the
numerically protected layer the default, makes generation O(1) per new token, ships the benchmark against Mamba-2,
RWKV-7 and a Transformer++ together with its results, and moves the development history to `history/`.

Every number below comes from the repository: mainly [`results/banco_a100/`](../results/banco_a100/RESUMEN.md) (the
A100 run of 17 October 2026, PyTorch 2.14.1), [`results/citas_a100/`](../results/citas_a100/RESUMEN.md) and
[`results/citas/`](../results/citas/RESUMEN.md) (the quotation experiment on the A100 and on CPU),
[`results/escritura_cpu/`](../results/escritura_cpu/RESUMEN.md) (the O(1) and O(t) writers on CPU, 23 October 2026)
and [`results/banco_cpu/`](../results/banco_cpu/RESUMEN.md) (raw results); `history/docs/ESTABILIDAD_Y_BANCO.md` (the
stability and benchmark report; its paths use the pre-1.0 layout, see [`history/README.md`](../history/README.md));
[`history/results/banco_a100_pt211/`](../history/results/banco_a100_pt211/RESUMEN.md) (the previous A100 run, PyTorch
2.11); and `history/docs/` (v89 whitepaper and performance log). Code identifiers, comments and most of the
development documents are in Spanish.

---

## Highlights

- **An installable package.** `pip install .` gives `import asrn` (version `1.0.0`), `python -m asrn` and the
  `asrn-train` command. Python ≥ 3.11, PyTorch ≥ 2.9, NumPy. English aliases (`asrn.ASRNLayer`, `asrn.ASRNModel`,
  `asrn.create_model`, `asrn.train`, `asrn.generate`, `asrn.load`, …) name the same objects as the Spanish API.
- **The full test suite passes.** The current suites, 57 core tests and 19 tests of the stable layer and the package
  API (76 in total), pass 57 / 57 and 19 / 19 on CPU with PyTorch 2.14.1 (23 October 2026). The suites of the A100 run
  of 17 October (55 and 18 tests) passed 55 / 55 and 18 / 18 both on CPU and on the Colab A100 machine, where the 7 / 7
  GPU checks also passed, `torch.compile` included. With PyTorch 2.11.0 on CPU, the two `torch.compile`
  tests of the core are skipped (55 pass) and the stable suite passes 19 / 19. A GitHub Actions
  workflow runs both suites on CPU with Python 3.11 and 3.13 (see [Verification](#verification)).
- **The stable layer is the default.** `asrn.crear_modelo`, `asrn.crear_desde_config` and `asrn.cargar` return models
  whose ASRN layers are `CapaASRNEstable` with protection group A. They have the same parameters as the v89 layer and
  give the same numbers up to rounding (1.3·10⁻⁷ in the logits). Saved models load in either layer.
- **Generation in O(1) time per new token.** `escribir()` / `asrn.generate` no longer rescan the text: each table keeps
  a ring with the 32 most recent positions of each bucket (exactly the keys the previous writer chose among the whole
  memory), and the Hawkes intensity is carried per n-gram key with the same recurrence as the whole-text scan. The
  O(1) writer matches the previous O(t) writer and the whole-text forward pass (max difference 2.2·10⁻⁷), and
  `escribir()` gives the same token sequences in 192 of 192 combinations of sampling options. On CPU
  ([`results/escritura_cpu/`](../results/escritura_cpu/RESUMEN.md), a 700,484-parameter model) it takes 9.49–12.47 ms
  per new token from 512 to 131,072 tokens of context, with a slope indistinguishable from zero, while the O(t) writer
  goes from 15.80 to 91.67 ms (9.7× slower at 131,072). Memory per sequence still grows with the text, and the O(1)
  writer has not been measured on the A100 yet (see [O(1) generation](#o1-generation)).
- **Six v89 numerical problems fixed, each one reproduced or measured first.** Causal leak through the −10⁹ mask
  (0.016 → 0 exactly on an A100). 32-bit clock that loses precision from ~1 M tokens and stops advancing on 16% of
  steps at 16 M. Score cancellation (output drift up to 8.8·10⁻² with the clock shifted by 10⁹ → 5.5·10⁻⁸, constant).
  Unbounded β leading to NaN. Clock projection in 16 bits under autocast. TF32 on the clock projection (2.0 units of
  clock error → 1.2·10⁻¹⁰). Measured cost on an A100, with the TF32-proof clock projection: +7.5% time and +5.2%
  memory at 262,144 tokens.
- **The Triton kernels carried over from v89** (bucketed attention without atomics, associative scan) ship unchanged;
  the bucket-attention kernels now have an `ESTABLE` compile-time mode with a two-part (hi/lo) clock and an exact mask.
  With the kernel, the layer saves 20 KiB per token for the backward pass (d = 512, fp32), against 44 KiB on the PyTorch
  path.
- **A new benchmark against the official rivals** on an NVIDIA A100-SXM4-80GB: official Mamba-2 (`mamba_ssm`),
  official RWKV-7 (`flash-linear-attention`) and our own Transformer++ (Llama recipe with FlashAttention-2), with
  PyTorch 2.14.1. Everything ran uncompiled except two separate training rows: ASRN compiled by parts and Transformer++
  with `torch.compile`.
  - **Where ASRN came out ahead:** throughput on long sequences (a ~50 M model trains at 163–172k tokens/s from 1k to
    32k tokens, which is 2.08× Transformer++ at 32k, 1.4–1.5× Mamba-2 run unfused and 3.4–3.6× RWKV-7; compiled, 2.20×
    the compiled Transformer++ at 32k). Long single layers (21.6× attention and 2.5× a Mamba-2 block at 262k tokens).
    Exact recall beyond the training length (copy at 4× in the 2 of 3 seeds that left the initial plateau; MQAR with
    2× and 4× the pairs, significant against Mamba-2 only at 4×). WikiText-103 perplexity at this small budget (not
    significant against RWKV-7).
  - **Where it did not:** short inputs (a ~12 ms floor per layer), memory per layer, generation (slowest of the four,
    measured with the previous O(t) writer; the O(1) writer is not measured on the A100 yet), learning copy quickly
    within the training length, and how reliably it learns copy at all (2 of 3 seeds here).
- **Nearly flat throughput with length, and the tooling to scale.** A ~50 M-parameter model trains at 163–172k tokens/s
  uncompiled (−5.3%) and 199–210k compiled by parts, from 1,024 to 32,768 tokens per sequence on a single A100. To go
  beyond 100 M parameters, the release includes `examples/train_100m/`, a ready-to-run Colab kit for a 103.5 M-parameter
  model on FineWeb, and FSDP2 sharding (`repartir_fsdp`), covered by a two-process FSDP test on CPU and a one-process
  NCCL check on the A100.
- **A clear map of the evidence.** The layer separates *where to copy* (bucketed sparse attention: learned hash tables,
  candidate windows, and scores with exact suffix, confirmation and the time term) from *whether to copy and when to
  stop* (the gate, the proper-time clock and the Hawkes/streak signals). On copy and MQAR, where every evaluated
  position requires copying, the retrieval core is sufficient on its own, and without the ASRN layer both tasks stay at
  chance. The regulation mechanisms are non-intrusive there: with them ASRN reaches ≥ 99.98% on copy at 4× in the 2 of 3
  seeds that learned and 99.75% on MQAR with 4× the pairs, and removing them one at a time leaves the accuracy of the
  runs that learned within seed variation. Their active role is validated in a controlled synthetic bounded-quotation
  experiment run with the 1.0 layer (3 seeds), where, without explicit supervision, they keep the longest written
  repetition streak at 10 tokens, as in the real text, against 21–27 on CPU (19–39 on the A100) when the Hawkes and
  streak signals or the gate's signals are removed. Log-sum-exp table mixing (`mezcla_lse`) is not adopted in 1.0. See
  [Known issues & limitations](#known-issues--limitations).

---

## What's in 1.0

### Layout

| Path | Contents |
|---|---|
| `asrn/nucleo.py` | The core: the former `asrn_v89.py`. Includes `CapaASRN` and its Triton kernels (including the `ESTABLE` mode), `BloqueLocal`, `ModeloASRN`, `ConfigASRN` / `TALLAS` / `para_parametros`, `entrenar`, `evaluar`, `escribir` (with the O(1) writer `_Escritor`), `guardar` / `cargar`, FSDP (`repartir_fsdp`, `materializar`), `compilar_partes`, `PasoGrafado` and the command line |
| `asrn/estable.py` | The former `asrn_estable.py`: `CapaASRNEstable`, `Protecciones`, `_ProyeccionIEEE`, `_TiempoRelativo`, `convertir()`, `RecorteDinamico` |
| `asrn/__init__.py` | The public API (stable layer by default), with English aliases of the same objects (`ASRNLayer`, `ASRNCoreLayer`, `ASRNModel`, `ASRNConfig`, `LocalBlock`, `Protections`, `DynamicClip`, `create_model`, `create_from_config`, `convert`, `train`, `evaluate`, `generate`, `save`, `load`); `__version__ = '1.0.0'` |
| `asrn/__main__.py` | `python -m asrn` and the `asrn-train` console script: the core command line with the stable layer |
| `pyproject.toml`, `requirements.txt` | `pip install .`; Python ≥ 3.11, `torch>=2.9`, `numpy`; extra `[bench]` |
| `tests/` | `test_nucleo.py` (57 tests of the core), `test_estable.py` (19 tests of the stable layer, the package API and the English aliases), `verificar_gpu.py` (7 GPU checks) |
| [`benchmarks/`](../benchmarks/README.md) | The benchmark: `asrn_colab.ipynb` (the one Colab notebook: every test, the A100 benchmark and the quotation experiment, with PyTorch 2.14; its run of 17 October 2026 produced `results/banco_a100/` and `results/citas_a100/`; its generation section now runs the O(1) writer), `banco_asrn.py`, `banco_gpu.py`, `banco_modelos.py`, `modelos_oficiales.py` (the rivals), `rivales_lineales.py` (fallback reimplementations), `builders/` (notebook builders). Also `experimento_citas.py`: the bounded-quotation experiment ported to the 1.0 stable layer (full layer against `sin_hawkes_racha`, `puerta_sin_senales`, `sin_puerta` and `sin_reloj`; 3 seeds, CPU or GPU); and `medir_escritura.py`: ms per new token against the context for the O(1) and O(t) writers |
| [`results/`](../results/README.md) | `banco_a100/` (A100 run of 17 October 2026 with PyTorch 2.14.1: `resultados.json`, `RESUMEN.md`, charts), `citas_a100/` (the quotation experiment on the A100, same run), `escritura_cpu/` (the O(1) and O(t) writers on CPU, from `benchmarks/medir_escritura.py`), `banco_cpu/` (exact cost, CPU latency, generation with the O(t) writer, small-scale copy) and `citas/` (per-seed logs and results of `benchmarks/experimento_citas.py` on CPU) |
| [`examples/train_100m/`](../examples/train_100m/README.md) | A ready-to-run kit to train a 103.5 M-parameter ASRN on FineWeb in Colab with `import asrn`; its first run is on the [roadmap](#roadmap--future-work-v11) |
| [`paper/`](../paper/README.md) | LaTeX paper drafts (English and Spanish); the figures are drawn from `results/` by `paper/hacer_figuras.py`, and the paper text is being updated to the 1.0 results |
| [`history/`](../history/README.md) | Development history of the v89 cycle (`docs/`, `experiments/`, `probes/`, `results/`, including `results/banco_a100_pt211/`, the previous A100 run of 9 and 11 October 2026 with PyTorch 2.11); not part of the package, kept because the documents cite it as evidence (see [Breaking changes](#breaking-changes--migration)) |
| [`docs/`](README.md) | These release notes, [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md), each with its Spanish version (`_es.md`); the architecture diagram, [`asrn_architecture.svg`](asrn_architecture.svg) (Spanish: [`asrn_arquitectura.svg`](asrn_arquitectura.svg)), drawn by `hacer_diagrama.py`: the model and the inside of an ASRN layer |
| [`.github/`](../.github/workflows/tests.yml) | The CI workflow `workflows/tests.yml` (both test suites on CPU, Python 3.11 and 3.13) and the issue templates |
| [`CHANGELOG.md`](../CHANGELOG.md), [`CITATION.cff`](../CITATION.cff), [`SECURITY.md`](../SECURITY.md) | The changelog, the citation metadata and the security policy (checkpoint loading) |

[`ARCHITECTURE.md`](ARCHITECTURE.md) / [`ARCHITECTURE_es.md`](ARCHITECTURE_es.md) describe how the layer works, and
[`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) gives the full analysis.

### Install and use

```bash
pip install .                 # installs the asrn package only; tests and benchmarks run from the checkout
pip install ".[bench]"        # + flash-linear-attention, datasets, tiktoken, einops, matplotlib (mamba-ssm separately)
```

```python
import asrn
modelo = asrn.crear_modelo(V=8000, d=256, n_capas=8)            # ASRN layers are CapaASRNEstable, group A
asrn.entrenar(modelo, piezas, validacion=piezas_val, guardar_en='modelo.pt',
              recorte=asrn.RecorteDinamico(modelo))             # optional per-group gradient clipping
secuencias, _ = asrn.escribir(asrn.cargar('modelo.pt'), inicio, n_nuevos=200, devolver_logits=False)
```

```bash
python -m asrn --datos piezas.pt --V 8000 --guardar modelo.pt     # same as: asrn-train ...
python -m asrn --datos piezas.pt --parametros 50m --V 32000       # any size, as in v89
torchrun --nproc_per_node=8 -m asrn --fsdp --talla 1b ...         # several GPUs
python -m asrn.nucleo ...                                          # same command line, core layer without protections
```

### The stable layer by default

| Call | Result |
|---|---|
| `asrn.crear_modelo(...)`, `asrn.crear_desde_config(...)` (also on `'meta'`), `asrn.cargar(...)` | `ModeloASRN` with `CapaASRNEstable` and `Protecciones()` (group A). For the same seed, the weights are identical to `asrn.nucleo.crear_modelo` |
| `protecciones=asrn.Protecciones.textos_largos(L)` | Group A + group B, meant for training from scratch on long texts |
| `protecciones=asrn.Protecciones.ninguna()` | No protections: bit-identical to the core layer (CPU test, logits and gradients) |
| `asrn.convertir(modelo, protecciones)` | Converts the ASRN layers of any `ModeloASRN` in place, keeping its weights |
| `asrn.RecorteDinamico(modelo)` | Opt-in per-group gradient clipping, passed as `recorte=` to `entrenar()`. Neither the default `entrenar()` (global clip at 1.0) nor the command line uses it |

The package also exports `entrenar`, `evaluar`, `escribir`, `guardar`, `ConfigASRN`, `TALLAS`, `CapaASRN`,
`CapaASRNEstable`, `BloqueLocal`, `ModeloASRN`, `PasoGrafado`, `inicializar`, `materializar`, `repartir_fsdp`,
`esta_repartido`, `pesos_completos`, `compilar_partes`, `crear_optimizador`, `tasa_aprendizaje`, `leer_piezas`,
`leer_cantidad`, `firmas_de_piezas`, `escaneo`, and the modules `nucleo` and `estable`.

### O(1) generation

Writing step by step (`_Escritor` in `asrn/nucleo.py`, used by `escribir()` / `asrn.generate`) costs O(1) time per new
token, independent of how much text there is. Before 1.0.0 it was O(t) per token.
- **Where to copy.** Each table keeps, per bucket, a ring with the C = 32 most recent positions: exactly the keys the
  previous `_atender` chose among the whole memory. With prompt-only copy memory (`memoria_de_lo_ajeno`, the default),
  only prompt positions can be keys, so the rings are filled from the prompt and never change. The new token scores
  only its T × C candidates (`_atender_cand`); the stable layer keeps the exact mask and the 64-bit clock.
- **Hawkes intensity.** It is carried per n-gram key (`_MemoriaHawkes`): the clock of the key's last occurrence B_u and
  A = λ + α at that moment; then λ_n(t) = e^{−κ_n (B_t − B_u)} · A, and afterwards A ← λ_n(t) + α_t and B_u ← B_t. This
  is the same sum as the definition, carried as the recurrence that the whole-text associative scan uses. Keys are
  looked up in Python dictionaries, computed on the CPU from the sampled tokens (no GPU wait); the states stay on the
  device. The streak was already one number.
- **Sampling helpers.** The quotation-preserving temperature and `no_repetir_ngramas` use n-gram dictionaries updated
  per token (`_Ngramas`) instead of rescanning the text, and the generated text lives in a preallocated buffer.
- **Fallback.** The previous O(t) writer stays available as `_Escritor(modelo, o1=False)`. It is used automatically
  only if the rings would exceed 2 GiB (very large `n_bits` or batch).
- **Memory.** It still grows with the text (each position's cached features: what is copied must be stored), plus
  fixed rings: T · 2^n_bits · C positions (int32) and counters, 17 MiB per ASRN layer per sequence at the default sizes
  (8 tables, 14 bits, C = 32), about 120 MiB per sequence for the ~50 M benchmark model (7 ASRN layers).
- **Verified on CPU (PyTorch 2.14.1).** O(1) writer = O(t) writer = whole-text forward pass, max difference
  2.2·10⁻⁷: v89 and stable layer, rings that overflow, batches of 1–3, prompts of 1, 2 and 40 tokens, with and without
  prompt-only copy memory (new tests `test_escritura_o1_igual_que_o_t`, in both suites, and
  `test_anillo_y_ngramas_como_su_definicion`). `escribir()` gives the same token sequences, and logits within 10⁻⁴, as
  the previous O(t) implementation in **192 of 192** combinations of sampling options (temperature, top-k, top-p,
  prompt-only copy memory, quotation-preserving temperature, repetition penalty, no-repeat n-grams), v89 and stable
  layer. The model's forward pass and gradients are bit-identical to the code of the A100 run (see
  [Provenance of the results](#provenance-of-the-results)).

**Measured on CPU** ([`results/escritura_cpu/`](../results/escritura_cpu/RESUMEN.md), from
`benchmarks/medir_escritura.py`; 4 threads, PyTorch 2.14.1; an ASRN model of 700,484 parameters with V = 256, d = 128,
4 local blocks and 2 ASRN layers; batch 1, no sampling; median of 9 batches of 32 tokens; the two writers interleaved at
each context, after warming up both and after one unmeasured prompt of each length):

| Context (tokens) | O(1): ms per token | O(t): ms per token | O(1): prompt (s) | O(t): prompt (s) |
|---|---|---|---|---|
| 512 | 12.45 | 15.80 | 0.10 | 0.06 |
| 2,048 | 12.47 | 16.73 | 0.26 | 0.20 |
| 8,192 | 10.68 | 19.02 | 0.77 | 0.70 |
| 32,768 | 11.26 | 37.12 | 3.60 | 2.73 |
| 131,072 | 9.49 | 91.67 | 29.04 | 20.44 |

- **Time per token.** Against the context (least squares, 95% CI), the O(1) slope is −19.21 ns per context token
  (−41.54 to 3.12), indistinguishable from zero; the O(t) slope is 583.31 ns (537.13 to 629.49). At 131,072 tokens the
  O(1) writer is 9.7× faster per token (91.67 / 9.49), and 1.27× at 512.
- **Prompt time.** The O(1) bookkeeping (filling the rings and the Hawkes state) was profiled at about 0.3 s at 32,768
  tokens and 1.3 s at 131,072 (rings ~0.3 s, Hawkes state 0.8–1.1 s). The differences in the table are larger because
  the prompt time varies from run to run on this shared 4-core machine: the same prompt pass took 14.8–23.5 s at
  131,072 tokens in four repetitions.
- **Not measured on the A100 yet.** The A100 generation figures of this release were measured with the O(t) writer
  (see [Known issues & limitations](#known-issues--limitations)). The notebook's generation section
  (`benchmarks/asrn_colab.ipynb`, section 6) now runs the O(1) writer, with the new tokens passed from the CPU as
  `escribir()` does, so the next Colab run measures it.

---

## Kernel improvements carried from v89

The core layer's compute is the same as in v89. The improvements listed here were made during the v89 cycle and ship
in 1.0. The `ESTABLE` mode is the new part.

**Bucketed attention in Triton** (`_casilleros_ida` / `_casilleros_vuelta`).
- One program runs per (batch row × table, block of 32 queries in that table's bucket order). It reads its queries
  and 64 keys directly in text order and builds the scores in SRAM with two `tl.dot`, the suffix weight, the time
  kernel and the mask. It applies a 32-bit softmax and writes each output to its position in the text.
- The backward pass recomputes the scores, as FlashAttention does, and saves only the inputs.
- The backward pass uses no atomics. Each key gradient is written to its own slot for each of its two windows and
  summed afterwards, so the result is deterministic. In the v89 cycle, removing the atomics took the backward kernel
  from 6.4 to 0.99 ms at 16,384 tokens (A100 40 GB).
- `ASRN_TRITON_CASILLEROS=0` turns this kernel off, and `ASRN_TRITON=0` turns off all the kernels. The PyTorch
  implementation remains the reference and the fallback on CPU or without Triton.

**Associative scan in Triton** (`_escaneo_kernel`).
- This scan computes Hawkes and the run counter: S_i = a_i S_{i−1} + b_i with a_i ∈ [0, 1], composed as
  (a₁, b₁)∘(a₂, b₂) = (a₁a₂, a₂b₁ + b₂). The kernel runs one program per row, in blocks of 1,024 with
  `tl.associative_scan` and a carry between blocks. The backward pass is the same kernel run right to left.
- About 290 PyTorch operations per layer became about 6.
- Also from v89: `_Tomar` replaced an atomic `index_add_` that took half of the layer's GPU time at 16,384 tokens
  (18.4 of 36.6 ms).

**`ESTABLE` mode (new for the stable layer).** `ESTABLE` is a `tl.constexpr` parameter of the two bucket-attention
kernels (`_casilleros_ida`, `_casilleros_vuelta`); the scan kernel has no such mode. With it off, the compiled kernel is
the one v89 had.
- **Two-part clock.** B arrives as the high and low 32-bit halves of a 64-bit clock. The time term uses
  dt = (B_k − B_q)_hi + (B_k − B_q)_lo, taken *before* multiplying by the rate. Between nearby tokens the high
  difference is exact (Sterbenz lemma), and the low part corrects the rounding. Cost: 4 more bytes read per query and
  key.
- **Exact mask.** The softmax maximum is taken over valid keys only, and invalid keys get weight exactly 0.
- **Verified with the Triton interpreter on CPU.** The kernel matches the stable PyTorch path in forward and backward,
  in fp32 and bf16. It shows no leak, and its output changes by at most 8·10⁻⁸ with the clock shifted by up to 10⁹.
- **Compiled on the A100, TF32 off.** It matches PyTorch to 4.9·10⁻⁸ in the output and 4.5·10⁻⁶ in the worst gradient
  (fp32). In bf16 it is 0.096 from the fp32 answer, against 0.094 for PyTorch in bf16: the same order of error.

**Memory.**
- With the kernel, the layer saves 20.1 KiB per token for the backward pass (20.2 KiB for the stable layer), against
  43.8 KiB on the PyTorch path (43.9 KiB). These figures are for d = 512 and fp32, are constant with length, and were
  measured with autograd hooks.
- Since v86 the layer's tensor flow avoids three avoidable copies. Saved activations dropped by **44% versus v85**
  (55.3 → 30.9 floats per token in units of d, d = 256) with the same math, so saved memory stays O(N).

**Compute per token is constant** (`FlopCounterMode`, d = 512, forward and backward). The ASRN layer costs 5.19, 5.14
and 5.13 MFLOP per token at 1k, 4k and 16k tokens. An attention layer costs 9.44, 18.87 and 56.62.

**Measured speed on the A100** (PyTorch 2.14.1, CUDA 13.0, bf16; uncompiled except the two compiled rows):

| One layer, d = 512, batch 1, forward + backward (ms) | 1,024 | 4,096 | 16,384 | 65,536 | 262,144 |
|---|---|---|---|---|---|
| ASRN (stable layer) | 12.1 | 12.2 | 12.5 | 19.3 | 68.7 |
| Transformer++ attention (FlashAttention-2) | 3.3 | 3.4 | 10.0 | 102.4 | 1,486.0 |
| Mamba-2 block (`mamba_ssm`, without `causal-conv1d`) | 7.7 | 9.9 | 11.1 | 34.8 | 174.3 |
| RWKV-7 (`flash-linear-attention`) | 11.6 | 12.0 | 35.0 | 131.1 | 519.4 |
| Peak memory, ASRN / T++ / Mamba-2 / RWKV-7 (GB) | 0.04 / 0.02 / 0.05 / 0.08 | 0.17 / 0.09 / 0.16 / 0.33 | 0.68 / 0.34 / 0.55 / 1.32 | 2.7 / 1.4 / 2.1 / 5.3 | 10.9 / 5.5 / 8.5 / 21.1 |

| ~50 M model, 32,768 tokens per step (tokens/s; peak GB) | 1,024 | 4,096 | 16,384 | 32,768 |
|---|---|---|---|---|
| ASRN (50.7 M) | 172,065 (23.5) | 171,275 (23.5) | 163,503 (23.5) | 163,023 (23.5) |
| ASRN, compiled by parts (`compilar_partes`) | 210,413 (22.3) | 207,945 (22.3) | 200,800 (22.3) | 198,826 (22.3) |
| Transformer++ (51.0 M) | 182,116 (25.5) | 160,772 (25.5) | 110,342 (25.5) | 78,400 (25.6) |
| Transformer++, compiled (`torch.compile`) | 262,909 (22.9) | 220,468 (22.9) | 135,614 (22.9) | 90,540 (22.9) |
| Mamba-2 (51.2 M) | 118,002 (26.4) | 118,314 (26.4) | 116,935 (26.4) | 114,544 (26.4) |
| RWKV-7 (51.1 M) | 50,439 (30.5) | 50,082 (30.5) | 48,397 (30.5) | 45,895 (30.5) |

What the tables show:
- **Single layer.** Up to 16k tokens ASRN sits on a flat ~12 ms floor. It is among the slowest there, and
  FlashAttention-2 is 3.7× faster at 1–4k. It overtakes RWKV-7 at about 4.2k tokens, attention at about 17.8k and the
  Mamba-2 block at about 20.2k (linear interpolation in L). At 262,144 tokens it is 21.6× faster than attention, 2.5×
  faster than the Mamba-2 block and 7.6× faster than RWKV-7, using 2.0× and 1.3× the memory of the first two and about
  half of RWKV-7's (~40 KiB per token, against 20, 31 and 79).
- **Whole model, uncompiled.** ASRN loses 5.3% of its throughput from 1,024 to 32,768 tokens, against 57.0% for
  Transformer++. It overtakes Transformer++ at about 2.4k tokens (Transformer++ is 6% faster at 1,024, ASRN is 2.08×
  faster at 32,768). It is 1.4–1.5× faster than Mamba-2 at every length and 3.4–3.6× faster than RWKV-7, and it needs
  7.7%, 10.9% and 22.9% less peak memory than Transformer++, Mamba-2 and RWKV-7.
- **Whole model, compiled.** Compiling ASRN by parts raises its throughput 1.21–1.23× at every length; `torch.compile`
  raises Transformer++'s 1.44× at 1,024 and 1.15× at 32,768. Compiled against compiled, ASRN has 0.80× the throughput
  of Transformer++ at 1,024 and 0.94× at 4,096, overtakes it at about 5.4k tokens and is 2.20× faster at 32,768. The
  first step, compilation included, took 47.0 s for ASRN and 26.2 s for Transformer++.

Caveats that change how to read these numbers:
- **Only the two compiled rows are compiled.** ASRN compiled by parts and Transformer++ with `torch.compile` are
  separate rows of the model table. Every other measurement (single layer, the uncompiled training rows, generation,
  quality, ablation, quotation experiment) ran uncompiled, and Mamba-2 and RWKV-7 were never compiled.
- **Mamba-2 ran without `causal-conv1d` and uncompiled.** It used its unfused path, so its numbers are pessimistic and
  every ASRN-versus-Mamba-2 speedup is an upper bound. RWKV-7 ran with its `flash-linear-attention` kernels.
- **The single-layer rows are different units.** The ASRN row is only its mixing layer (0.65 M parameters). The
  Transformer++ row is attention with RMSNorm (1.05 M, no SwiGLU). The Mamba-2 row is its whole block (1.72 M,
  2.6× the ASRN layer). With whole models the ASRN-to-Mamba-2 ratio is 1.4–1.5×.
- **Transformer++ is our implementation.** It follows the Llama recipe with FlashAttention-2 through SDPA, but
  RMSNorm, RoPE and SwiGLU are separate PyTorch operations, not fused.
- **One run per point.** Each time is the median of 2–5 repetitions within that run.

The ~12 ms floor is very probably the cost of launching many small operations (this run was not profiled). Earlier in
the project, recording the step as a CUDA Graph (`PasoGrafado`) took one layer at 1,024 tokens from 11.4 to 1.6 ms
(A100 40 GB). The same-session speed log of the v89 cycle (A100 40 GB, PyTorch 2.14.1, core 50 M model, 1,024 × 32
tokens) shows the whole speed-up: 44,879 tokens/s before the speed work, 147,548 with the kernels (one ASRN layer per
block), 171,235 with the current recipe (one every two blocks) and 214,536 with `compilar_partes`. The compiled
Transformer reached 271,582 in that session.

---

## Stability

The stable layer (`asrn/estable.py`) subclasses the core layer with the same parameters and names. It rewrites the
forward pass with the protections below. In the normal regime it gives the same numbers as v89 up to rounding
(1.3·10⁻⁷ in the logits).

### Each fixed v89 problem, with before and after measurements

| Problem in v89 | Before (v89) | After (stable layer) | Protection |
|---|---|---|---|
| **Causal leak through the −10⁹ mask.** The mask is a score of −10⁹, not −∞. If every valid score falls below it (a huge time kernel), the softmax sends its mass to invalid keys, including the query itself (which copies the *next* token) and later tokens | The future changes the past: 0.016 with γ = 10⁹ and 0.075 with γ = 10¹¹ (CPU, PyTorch and kernel alike); 0.016 with the compiled kernel on the A100 | 0 exactly, on CPU and on the A100. Also in step-by-step generation | `mascara_exacta` |
| **32-bit clock: precision loss and stall.** B_t = Σβ is a 32-bit running sum, and its mantissa limits the step (0.06 at B = 2²⁰, 1 at 2²⁴) | Error of B_t − B_{t−1} with CUDA `cumsum` on the A100: 2.2% at 2²⁰ (1 M), 8.6% at 2²² (4 M), 50% at 2²⁴ (16 M), 100% at 2²⁵. The clock stops on 0.03%, 1.1%, 16% and 37% of the steps | Accumulated in 64 bits. In the 2²⁵-token test every step advances by β (relative error < 10⁻⁶), while the 32-bit sum stalls on more than 20% of the last 2²² steps | `reloj_64` |
| **Score cancellation.** v89 computes (log α_k + rate·B_k) − rate·B_q: two large numbers subtracted after the score is added. Only clock differences should matter | Change in the output with the clock shifted (A100, TF32 off): 9.4·10⁻⁵ at 10⁶, 2.8·10⁻³ at 1.6·10⁷, 8.8·10⁻² at 10⁹. CPU: 1.2·10⁻⁴, 2.8·10⁻³, 7.8·10⁻² | 5.5·10⁻⁸, identical for all three shifts, on CPU and on the A100 with TF32 off. A constant fixed rounding difference, not drift. With TF32 on it was 3.55·10⁻⁶, also constant: TF32 in the products after the time term | `reloj_64` (the difference is taken in 64 bits before multiplying, `_TiempoRelativo`; hi/lo in the kernel) |
| **Unbounded β → NaN.** β = softplus(·) has no ceiling | With broken weights B passes 3.4·10³⁸ within a few tokens, and the time kernel gives ∞ − ∞ = NaN across the layer | Output and gradient stay finite | `beta_max = 10⁴` |
| **Clock projection in 16 bits.** To save a read, v89 computes w·ĥ in the same matmul as Qc/Kc, under autocast | In bf16, β has about 3 significant digits and the error accumulates in B: 0.39, 0.65, 0.93 and 1.27 units after 512, 1,024, 2,048 and 4,096 tokens (d = 64); 3.3 after 4,096 (d = 512). In fp16 the sum can overflow (deduced, not measured). This was listed as open in v89 | The projection runs in 32 bits. The test requires B's error to be at least 10× smaller than on the 16-bit path | `reloj_64` |
| **TF32 on the clock projection.** `entrenar()` and the notebook turn on `allow_tf32`, so an "fp32" matmul uses 10 mantissa bits | B_t in fp32 changes by up to 2.0 units over 65,536 tokens (d = 512, A100) when TF32 is switched on | 1.2·10⁻¹⁰: on GPU the clock is projected with `_ProyeccionIEEE` (elementwise multiply-add, ~2d operations per token) | `reloj_64` |

When this matters: the clock degrades from about 1 M tokens and stops advancing from about 16 M. How much that
changes the output depends on γ. On a 2²⁰-token layer (forward only, d = 64, bf16) the v89 and stable outputs stayed
within 8.2·10⁻⁴ (under 0.1%). That figure mixes the clock, v89's bf16 projection, the mask and rounding, and does not
isolate the drift.

### Protection groups

`mascara_exacta`, `reloj_64` and `beta_max` respond to failures that were measured. `log_alfa_min`, `mejor_max`,
`tope_puerta` and `kappa_min` are **preventive**: the failure they avoid is deduced from the code and was not
measured. `log_alfa_min` and `mejor_max` have no dedicated test.

| Protection | Group | Failure it prevents | Cost |
|---|---|---|---|
| `mascara_exacta` | A (on) | Causal leak (PyTorch, kernel and generation) | None: the same bits in the normal regime |
| `reloj_64` | A (on) | Clock precision, cancellation, stall, 16-bit and TF32 projection | 8 bytes per token for B instead of 4; +4 B read per query and key in the kernel; 2 columns of a true 32-bit product |
| `beta_max = 10⁴` | A (on) | Infinite clock → NaN | None in practice (far above normal values) |
| `log_alfa_min = −30` | A (on) | log α → −∞, another way to fall below −10⁹ (preventive) | None |
| `mejor_max = 10⁴` | A (on) | A −10⁹ entering the gate through the `mejor` signal (preventive) | None |
| `tope_puerta = 10` | B (off) | Gate saturation from signals outside the training range (preventive) | Changes the gate: logit → c·tanh(logit / c) |
| `kappa_min = 1/(4·L_train)` | B (off) | With κ → 0, λ grows with text length (preventive; in the gradient probe, log(1 + λ) stayed flat from 256 to 65,536 tokens) | Changes Hawkes: bounded maximum memory |

Group A is on by default and does not change the function. Group B changes the function. It is off by default, so
saved models give the same results, and is meant for training from scratch on long texts
(`Protecciones.textos_largos(L)`).

**What was deliberately not added.** The per-table softmax is already stable, because it subtracts the maximum in
32 bits. LayerNorm runs in 32 bits with ε = 10⁻⁵. Gradient protection for the clock was not needed either: the
hypothesis that the clock gradient grows with length was tested and rejected (flat norms from 256 to 65,536 tokens).
There is no `nan_to_num` inside the layer, so that `entrenar()` can still detect a NaN and skip the step.

### `RecorteDinamico`

`RecorteDinamico` clips gradients per group, with an adaptive threshold.
- **Groups:** the clock and rates, the gate, and the rest. Each group keeps an exponential moving average of its norm
  and of its square.
- **Threshold:** min(`tope`, μ + z·σ) after a warm-up (defaults `tope=1.0`, `z=3.0`, `beta=0.98`, `calentar=50`). Only
  the group that exceeds it is scaled.
- **Averages:** they update with the clipped norm, so a spike does not raise the next threshold, and never with a NaN.
  Nothing is synchronized with the GPU.
- **Why:** with one global clip, a spike in a few numbers (the clock, γ) shrinks the whole model's gradient for that
  step.
- **Tested:** with a 10⁴ spike in the clock, only the clock group is clipped and the "rest" group is unchanged bit for
  bit. The gate group is not checked.
- **Not tested:** with FSDP. Its history is not saved in checkpoints, so it warms up again after resuming.

### Costs

- **Time and memory (A100, group A, with `_ProyeccionIEEE`).** Time is the same as v89 at 1,024 tokens (12.1 against
  12.0 ms), +4–7% from 4,096 to 65,536 tokens and +7.5% at 262,144 (68.7 against 64.0 ms). Memory is +4–5% at every
  length (+5.2% at 262,144), because ĥ is kept in 32 bits for the clock.
- **Saved bytes per token.** 43.9 against 43.8 KiB on the PyTorch path, and 20.2 against 20.1 KiB with the kernel.
- **One layer with 2²⁰ tokens** (forward only, d = 64, bf16). v89: 41.7 ms and 2.185 GB. Stable layer: 33.3 ms and
  2.198 GB, which is +12 bytes per token. These are single measurements, and the time differences are noise (the
  previous runs gave 35.2 ms for v89 and 32.4–36.3 ms for the stable layer).
- **Group B** showed no consistent cost or effect in the ablation (copy and MQAR, 3 seeds): −3% and +5% per run, copy
  learned in 2 of 3 seeds as in the base, 99.29% on MQAR with 4× the pairs.

---

## Problems resolved from v89 and the development cycle

**In the layer (v89 → 1.0).** The six numerical problems in the Stability table: causal leak, 32-bit clock precision
and stall, score cancellation, unbounded β, 16-bit clock projection (listed as "still open" in the v89 review), and
TF32 on the clock projection.

**Found and fixed while building the 1.0 benchmark:**

| Problem | Effect | Fix |
|---|---|---|
| `FlopCounterMode` miscounts the backward pass of a grouped convolution (Mamba-2's depthwise conv, 1,152 groups) | It assigned 10.7 MFLOP per token to that convolution, about 1,000× the real cost. The first cost table gave Mamba-2 22.1 MFLOP/token | `banco_asrn.py` uses a formula that accounts for groups (it matches PyTorch on a dense convolution). Mamba-2: **22.1 → 11.48 MFLOP/token**; the other rows are unchanged |
| The RWKV-7 wrapper built a one-layer model, and `flash-linear-attention` divides by (layers − 1) at initialization | `ZeroDivisionError` stopped RWKV-7 in every section of the 9 October run | RWKV-7 is built with at least 2 layers (the single-layer figure is the first layer of a two-layer model). The 11 October run added RWKV-7 to all six sections |
| `mezcla_lse` ablation variant (experimental, not part of 1.0) | Loss became NaN on MQAR from step 400 in the previous run, because of an implementation error in that variant | The mask is applied before `exp`, and there is no softmax over rows without candidates. The previous run's MQAR row (0.0% in `history/results/banco_a100_pt211/`) does not count. Re-run with the fix on 17 October, it no longer diverges but does not learn MQAR (3.61 / 1.08 / 3.15% with 4× the pairs; final loss 8.06–8.28) |
| Transformer++ prefix in the generation benchmark | In the previous run the wrapper computed logits for every prefix position (65,536 × 32,768 logits, 4.3 GB in bf16), while the other models computed only the last. This inflated its prefix time and generation memory; its per-token time was unaffected | Only the last position is computed now. Re-measured on 17 October at 65,536 tokens: prefix 0.387 s, generation peak 2.53 GB |
| `test_convertir_guardar_cargar` on the A100 | Failed: the loaded model went to the GPU and was compared on the CPU (a bug in the test, not the library) | The test loads on `'cpu'` |
| The stable layer's clock projection could go through TF32 (first A100 run) | The protection was weaker than intended on GPU | `_ProyeccionIEEE` (1.2·10⁻¹⁰ from the second run on) |
| The stable layer's constant 3.55·10⁻⁶ under a clock shift had no confirmed cause | It could have been mistaken for clock drift | The numerical tests now run with TF32 off. The second run and the 17 October run gave 5.5·10⁻⁸, constant, which confirms TF32 in the matmuls after the time term |
| `test_entrenar_aprende_a_copiar_y_reanuda` on the Colab A100 machine (PyTorch 2.11) | Failed from the fifth GPU run (core code `d9927a1bba79`; it passed with 2.11 in runs 2–4): copy loss 4.19 after 600 steps, against a threshold of 1.5. The test trains on the CPU, and at its learning rate of 3e-3, leaving the initial copy plateau (ln 63 ≈ 4.14) within 600 steps depended on the order of floating-point sums (thread count, MKL instruction set, PyTorch version). Reproduced off Colab with PyTorch 2.11 on 6 and 12 CPU threads (4.19–4.20); with 2.14.1, model seed 6 also stayed on the plateau (4.22–4.23), even with 1,000 steps (4.28) | The test trains at 6e-3, with the same steps, batch, threshold and resume check. It left the plateau by step 250 in every case tried: 22 runs, model seeds 0–7 on 4 threads and 0–3 on 12 threads with PyTorch 2.14.1; seeds 0–3 on 12 threads, 0–3 on 6 threads with MKL set to AVX-512 and 0–1 on 12 threads with ATen and MKL set to AVX2 with PyTorch 2.11.0; final copy loss 0.118–0.159. The fixed test passes on the Colab A100 machine (17 October run, PyTorch 2.14.1) |

**Already fixed during the v89 cycle (part of the core in 1.0):**
- NumPy is now a declared dependency, needed for sharded saving. The minimum PyTorch is 2.9: with 2.6 FSDP fails, and
  with 2.7–2.8 the sharded gradient norm comes out doubled under gloo.
- Resuming in fp16 from bf16 or fp32 training failed when loading the gradient scaler. It is fixed.
- `escribir()` edge cases are fixed: `top_p=0`, `no_repetir_ngramas` banning every token, and tokens outside the
  vocabulary.
- Several silent misconfigurations now warn or fail clearly, including `torchrun` without `--fsdp` and impossible
  `para_parametros` targets.
- FSDP builds its mesh on the CPU under gloo.
- The `torch.compile` checks are skipped before PyTorch 2.14.
- `_respuesta_racha` turned a NaN into an out-of-range index. The NaN now propagates, so `entrenar()` can skip the step.
- `found_inf` had the wrong shape in `PasoGrafado`. It is fixed.

---

## Breaking changes & migration

**The modules moved into a package, and the folders were renamed.** The archived documents in `history/` use the old
paths, and [`history/README.md`](../history/README.md) has the full mapping.

| Before (v89 and pre-release layout) | 1.0 |
|---|---|
| `import asrn_v89 as asrn` | `import asrn` (stable layer by default) or `import asrn.nucleo as asrn` (exact v89 core) |
| `import asrn_estable as estable` | `import asrn.estable as estable`, or `from asrn import CapaASRNEstable, Protecciones, RecorteDinamico, convertir` |
| `estable.convertir(asrn_v89.crear_modelo(...))` | `asrn.crear_modelo(...)` (already converted) |
| `python asrn_v89.py --datos ...` | `python -m asrn --datos ...` or `asrn-train --datos ...` (stable layer); `python -m asrn.nucleo ...` (core layer) |
| `torchrun --nproc_per_node=N asrn_v89.py --fsdp ...` | `torchrun --nproc_per_node=N -m asrn --fsdp ...` |
| `asrn_v89.py`, `asrn_estable.py` | `asrn/nucleo.py`, `asrn/estable.py` |
| `pruebas/test_asrn_v89.py`, `test_asrn_estable.py`, `verificar_gpu_v89.py` | `tests/test_nucleo.py`, `tests/test_estable.py`, `tests/verificar_gpu.py` |
| `pruebas/versiones_anteriores/` (`asrn_v85.py`, `asrn_v87.py`) | Removed: the 1.0 tests no longer compare with v85/v87 |
| `experimentos/` (benchmark scripts and notebooks), `experimentos/fuentes/`, `resultados/banco_a100/`, `resultados/banco_cpu/` | `benchmarks/`, `benchmarks/builders/`, `results/banco_a100/`, `results/banco_cpu/` |
| `experimentos/experimento_cita_v87.py` | Removed; the experiment is ported to the 1.0 stable layer in `benchmarks/experimento_citas.py` |
| `experimentos/sondas/`, `medir_velocidad_*` | `history/probes/`, `history/experiments/` |
| `resultados/corrida_*`, `bloques_1`, `velocidad` | `history/results/` |
| `docs/*` (whitepaper, RENDIMIENTO, ESTABILIDAD_Y_BANCO) | `history/docs/` |
| `archive_v89/`, `entrenar_100m/`, `articulo/` | `history/`, `examples/train_100m/`, `paper/` |
| `ARCHITECTURE*.md`, `ASRN_Technical_Report*.md`, `RELEASE_NOTES*.md` at the root | `docs/ARCHITECTURE*.md`, `docs/TECHNICAL_REPORT*.md`, `docs/RELEASE_NOTES*.md` |

**Default numerics changed slightly.** `import asrn` now gives the stable layer. Its outputs differ from v89 by
rounding only (1.3·10⁻⁷ in the logits). To get v89's exact behaviour, use either of these:
- `import asrn.nucleo as asrn`, or `python -m asrn.nucleo` on the command line (for this form Python prints a
  `RuntimeWarning` from `runpy`, because the package has already imported `asrn.nucleo`; training runs normally).
- `asrn.crear_modelo(..., protecciones=asrn.Protecciones.ninguna())`, or `asrn.convertir(modelo,
  asrn.Protecciones.ninguna())`. This gives bit-identical logits and gradients (CPU test).

**Checkpoints load unchanged.** `guardar()` writes a plain dictionary, `{'config': dict, 'pesos': state_dict}`, so no
class or module path is stored. The core and stable layers share parameter names, so files cross in both directions
(tested: saved by the core, loaded by the package, and vice versa). `cargar()` still accepts files from v83 onwards;
loading files from earlier versions is not covered by the 1.0 tests.

**Other changes.**
- Step-by-step generation uses the O(1) writer (see [O(1) generation](#o1-generation)): the same token sequences as
  the previous O(t) writer and logits within 10⁻⁴, plus fixed rings in memory (17 MiB per ASRN layer per sequence at
  the default sizes).
- In step-by-step generation, the stable layer keeps the clock in 64 bits (8 bytes per token per layer instead of 4).
- The options removed in the October 2026 cleanup of the v89 cycle (`partir_valores`, the per-channel convolution
  without memory, the gated dense layer) are not in 1.0.
- `largo_cada=2` (one ASRN layer every two blocks) has been the default since that cycle. `largo_cada=1` gives the
  previous shape, and saved models load with their own shape.

---

## Known issues & limitations

**Generation on the A100: the O(1) writer is not measured there yet.**
- On the A100 (~50 M model, batch 1), generation was the slowest of the four models: flat at 42.82–43.45 ms per token
  from 1k to 64k context (slope 3.0 ns per context token, 95% CI −25 to 31 ns), 2.9× Transformer++ (14.44–15.23 ms),
  1.7× Mamba-2 (24.47–25.28 ms) and 1.2× RWKV-7 (34.00–39.51 ms). These figures were measured with the previous O(t)
  writer. The time per token was already flat there (most likely kernel-launch bound; not profiled), so the O(t) term
  was hidden.
- With the O(1) writer the cost per token is constant in the context on CPU (see [O(1) generation](#o1-generation)),
  but the A100 figure is pending: no A100 speed-up of the O(1) writer is claimed. The notebook's generation section
  now runs the O(1) writer, so the next Colab run measures it.
- Memory per sequence still grows with the text, plus the fixed rings (17 MiB per ASRN layer per sequence at the
  default sizes). With the O(t) writer, the A100 peak at 65,536 tokens was 3.82 GB, against 2.53 GB for Transformer++,
  1.26 GB for Mamba-2 and 3.69 GB for RWKV-7. These figures include the prefix pass.

**Short inputs are launch-bound.**
- A single layer has a ~12 ms floor up to 16k tokens.
- In the whole model at 1,024 tokens, Transformer++ is 6% faster, and 25% faster when both are compiled.
- CUDA Graphs (`PasoGrafado`) cut one layer at 1,024 tokens from 11.4 to 1.6 ms. `PasoGrafado` is not part of
  `entrenar()`. Its limits: one GPU, bf16 or fp32, no checkpointing, no FSDP or `torch.compile`, no accumulation, and a
  fixed batch shape.

**`torch.compile` needs PyTorch ≥ 2.14.** Inductor fails on this model with earlier versions: invalid C++ on CPU, and
a Triton internal error on an A100 with 2.11 (the previous run). `entrenar()` warns, and with `compilar='partes'` it
continues uncompiled. With 2.14.1 on the A100 the `torch.compile` GPU check passes (logits within 4.8·10⁻⁴, gradients
1.2·10⁻⁷), and the stable layer compiled by parts trains 1.21–1.23× faster than uncompiled. Only the two compiled rows
of the training table are compiled; every other A100 number in `results/banco_a100/` is uncompiled.

**Copy learning starts on a plateau, and leaving it is not guaranteed.** A model first sits on the initial copy plateau
(ln 511 ≈ 6.24 in the benchmark's copy task, ln 63 ≈ 4.14 in the core test) and leaves it at a step that depends on the
seed, the learning rate and even the order of floating-point sums. On the A100 at 10⁻³, ASRN left it in 2 of 3 seeds:
seed 1 stayed on it for all 2,000 steps (final loss 6.232; 0.15% at 4×), while seeds 0 and 2 copied at 4× with 99.99%
and 99.98% (with 3·10⁻³, seed 0 left it at step 400). All 3 seeds learned in the previous run, so the base left the
plateau in 5 of 6 runs across both. In the ablation, 5 of the 24 runs of the light variants (the base, `estable_AB`,
`sin_puerta`, `puerta_sin_senales`, `sin_reloj`, `sin_hawkes_racha`, `sin_sufijo` and `sin_confirmacion`, 3 seeds each)
stayed on it (3 of 24 in the previous run), and the runs that learned score from 4.88% to 97.35% at 16× the training
length. This is optimization (leaving the plateau), not capacity. The same sensitivity made the core copy test fail on
the Colab machine (see Problems resolved).

**Kernel tests on CPU with Triton 3.6 need NumPy < 2.4.** On CPU the tests run the Triton kernels in Triton's
interpreter. The interpreter of Triton 3.6 (the one paired with PyTorch 2.11) converts one-element arrays to Python
integers, which NumPy 2.4 turned into an error, so three kernel tests fail there with NumPy ≥ 2.4. Triton 3.8 (PyTorch
2.14) and GPU runs are not affected.

**Methodological notes on the benchmark.** These describe what was measured:
- **Transformer++ generation prefix: fixed and re-measured.** In the previous run the wrapper computed logits for all
  65,536 prefix positions (4.3 GB in bf16), against only the last position for the other models. With the corrected
  wrapper, the prefix at 65,536 tokens takes 0.387 s and the generation peaks at 2.53 GB (ASRN 0.140 s and 3.82 GB,
  Mamba-2 0.139 s and 1.26 GB). Its per-token time was never affected. RWKV-7's prefix (110.9–130.9 s at every length)
  is dominated by the compilation and autotuning of its fla kernels.
- **`mezcla_lse` on MQAR.** In the previous run the experimental variant diverged (NaN from step 400) because of an
  implementation error in that variant. With the corrected implementation it no longer diverges, but it does not learn
  MQAR either (≤ 3.61% with 4× the pairs). It is not part of 1.0.
- **Learning rates.** Each model's rate is the better of a two-rate grid (10⁻³ and 3·10⁻³) on seed 0, with ties going
  to the lower rate. On WikiText-103, Transformer++ chose the lower edge of the grid (10⁻³) and the other three models
  the upper edge (3·10⁻³). ASRN's MQAR rate was 3·10⁻³ in this run and 10⁻³ in the previous one.

**The RWKV-7 text result is unresolved.** Here RWKV-7 (official `RWKV7Model`) reached a WikiText-103 perplexity of 75.8
against 67.9 for ASRN (72.4 against 67.8 in the previous run). ASRN was ahead in every seed (by 6.26, 3.43 and 14.02),
but the difference is not significant (paired p = 0.13 with 1,024 tokens, 0.084 with 4,096). RWKV-7 varies a lot
across seeds and its chosen learning rate is at the edge of the grid. In the fifth v89 run, with another protocol (30 M
tokens) and a reimplemented RWKV-7, RWKV-7 was best at 47.87 against 57.02. Neither result shows which reflects RWKV-7
better, so **ASRN cannot be claimed to beat RWKV-7 on text in general**.

**Scope of the benchmark.** These conditions limit every quality claim:
- **Small scale.** Models have 2.5–46 M parameters. WikiText-103 training used 20 M positions, about 0.44 per
  parameter, far from convergence. The curves were still falling, and the ASRN–Mamba-2 loss gap shrank during training.
- **Small search.** Only two learning rates (10⁻³ and 3·10⁻³) and 3 seeds; there is no separate test set for choosing
  the rate. With 10⁻³, Mamba-2 is ahead of ASRN on WikiText (86.1 against 89.6, one seed).
- **One text corpus,** and it rewards copying. Perplexities are per token on the validation set, so they are not
  comparable with published per-word numbers.
- **Every model ties its output layer to the token embeddings,** including RWKV-7, whose official recipe does not.
- **ASRN has exact token-id matching built in** (suffix, seen-n-grams, word tables). That inductive bias is aligned with
  copy and MQAR, and this MQAR is at the ceiling for ASRN.
- **Some rivals did not learn some tasks within this budget.** Mamba-2 did not learn copy, RWKV-7 had not finished
  learning it, Transformer++ did not learn MQAR, and RWKV-7 did not converge on MQAR in 2 of 3 seeds. These are budget
  or protocol effects, not evidence against those architectures.
- **The rivals' MQAR varies between runs.** With the same seeds, protocol and rival libraries but a different PyTorch
  version, Mamba-2 with 4× the pairs went from 54.39 ± 5.89 in the previous run to 26.09 ± 25.55, and RWKV-7 at the
  training length from 99.90 ± 0.17 to 65.43 ± 31.76 (most likely non-deterministic GPU kernels plus unconverged
  training). ASRN's MQAR was stable across runs. Against Mamba-2, ASRN's MQAR lead is significant only with 4× the
  pairs (paired p = 0.039; 0.070 and 0.090 at 1× and 2×).
- **Learning speed in-length favours Transformer++.** It learns copy in ≤ 200 steps against 1,000 for ASRN at the same
  rate when ASRN leaves the plateau (1,000–1,200 in the previous run). ASRN runs take 1.2–1.9× as long. Nothing was
  compared at equal compute.

**What the evidence says about each component.** The layer separates two design responsibilities. *Retrieval*
decides where to copy: the bucketed sparse attention (learned hash tables, candidate windows, and scores with the exact
suffix, the confirmation and the time term) performs exact retrieval. *Regulation* decides whether to copy and when to
stop: the gate and its signals, the proper-time clock and the Hawkes and streak signals regulate copying in open-ended
generation. The ablation ran on the A100 with 3 seeds at the base's rates (10⁻³ for copy, 3·10⁻³ for MQAR; the
previous run used 10⁻³ for both), and its base row is the quality run itself.
- **Retrieval core: validated on copy and MQAR.** Without the ASRN layer (`solo_local`) both tasks stay at chance in 3
  of 3 seeds (by construction: the local blocks see only the last 13 tokens). With it, ASRN reaches 99.99% and 99.98%
  on copy at 4× the training length (2 of 3 seeds: seed 1 stayed on the plateau, 0.15%; 3-seed mean 66.71 ± 57.64;
  previous run 99.87 ± 0.18, 3 of 3) and 99.75% ± 0.43 on MQAR with 4× the pairs.
- **Regulation mechanisms: non-intrusive on copy and MQAR.** In these tasks 100% of the evaluated positions require
  copying, so the retrieval core is sufficient on its own. The presence of the regulation mechanisms does not degrade
  exact retrieval, and removing them one at a time does not lower it beyond seed variation in the runs that learned:
  among the copy runs that left the plateau, the lowest at 4× is 99.89% for the base and the regulation variants
  (`estable_AB`, seed 0). Seed 1 is the hard seed: it stayed on the plateau for all 2,000 steps in the base itself, and
  also without the gate (`sin_puerta`), without the clock (`sin_reloj`) and with group B (`estable_AB`). Because the
  base itself failed on seed 1, one failed seed in a variant cannot be attributed to the removed component. Across both
  runs (6 per variant), copy was learned in 5 of 6 base runs, 4 of 6 without the gate or the clock, and 6 of 6 with the
  gate but without its signals and without Hawkes and the streak. The MQAR means with 4× the pairs are 99.75% for the
  base, 99.94% without the gate, 99.50% with the gate but without its signals (`puerta_sin_senales`), 99.95% without
  the clock and 99.53% without Hawkes and the streak (`sin_hawkes_racha`); every variant except `solo_local` and
  `mezcla_lse` scores ≥ 97.92% in every seed.
- **Regulation mechanisms: validated in controlled open-ended generation.** In the controlled synthetic
  bounded-quotation experiment (`benchmarks/experimento_citas.py`: random tokens with occasional quotations of 6–12
  tokens; d = 64, 2 local blocks and 1 ASRN layer, 6,000 steps, seeds 0–2; [results](../results/citas/RESUMEN.md)), the
  full 1.0 layer is compared with the same layer without the Hawkes and streak signals (`sin_hawkes_racha`), with a gate
  that sees only the hidden state (`puerta_sin_senales`), with no gate (`sin_puerta`) and without the clock
  (`sin_reloj`). From the language-modelling loss alone, without explicit supervision, the full layer's gate learns a
  "cliff": a push to copy when a quotation starts (+0.6 to +0.8 at a streak of one token) and a hard brake once the
  streak outgrows the data's quotations (−7.7 to −9.2 from 12 tokens on). This brakes *state aliasing* (without the
  streak, a 5-token and a 50-token quotation are the same state). With default sampling (T = 0.8, top-k 40), the longest
  written streak is 10 tokens (real text: 10) against 21, 27 and 23 for the first three variants, and streaks longer
  than 9 tokens are 0.3% of all streaks (real text: 0.9%) against 7.2%, 9.3% and 11.8%. Removing only the Hawkes and
  streak signals leaves the validation loss practically unchanged (3.703–3.706 against 3.707–3.708); the variants
  without the gate's signals or without a gate lose 0.03–0.05. The same experiment on the A100
  ([results](../results/citas_a100/RESUMEN.md)) replicates it: the gate learns the same cliff (+0.6 to +0.7 at a streak
  of one token, −7.6 to −8.4 from 12 tokens on); with default sampling the longest streak is 10 tokens against 19, 39
  and 21, and streaks longer than 9 are 0.3% against 7.9%, 9.2% and 12.0%; across the four sampling settings the longest
  streak per seed is 9–12 for the full layer and 9–10 without the clock, against 14–24, 15–39 and 18–31 for the first
  three variants; and the validation loss is 3.703–3.705 for the full layer against 3.706–3.709 without the Hawkes and
  streak signals, 3.735–3.748 without the gate's signals, 3.728–3.731 without a gate and 3.704–3.705 without the clock.
  Known limits: without the clock the layer writes like the full one (longest streak 10; 0.2% of streaks above 9 on CPU,
  0.4% on the A100; same loss), so the clock's own role, which concerns topic shifts that this task does not have, is
  untested; how often a quotation starts is not calibrated (quoted fraction 20.9% at T = 0.8 and 9.3% at T = 1 on CPU,
  21.8% and 10.1% on the A100, against 15.6% in the real text); and it is one synthetic task at d = 64.
- **Retrieval refinements: not separable on these saturated tasks.** The exact suffix (`sin_sufijo`, MQAR 99.69%), the
  confirmation (`sin_confirmacion`, 99.88%) and the number of tables refine retrieval; their individual contribution
  cannot be separated on copy and MQAR. Without the suffix, copy failed on seed 0, and seed 2 learned at 1× (87.02%) but
  gave 39.77% at 4× and 4.88% at 16×, the lowest learned run of the ablation. `dos_tablas` (1 + 1 tables) learned copy
  in 1 of 3 seeds (2 of 6 across both runs), which is suggestive, not conclusive.
- **Not adopted: `mezcla_lse`.** In the previous run, mixing tables by log-sum-exp left the copy plateau by step 400 in
  3 of 3 seeds. That did not reproduce here: only seed 0 learned (loss < 1 from step 600), so faster copy learning is
  not a robust effect. With the corrected implementation it does not learn MQAR (≤ 3.61% with 4× the pairs), and it
  costs +62% per run on copy and +61% on MQAR (PyTorch only).
- **Group B: off by default.** It showed no consistent cost or effect in the ablation.
- **Next.** The large-scale ablation of the regulation mechanisms on natural corpora with distractors and topic shifts
  is planned for v1.1 (see the [roadmap](#roadmap--future-work-v11)).

**Untested combinations and other limits.**
- FSDP, `PasoGrafado` and the GPU checks (the `torch.compile` check included) are tested with the core layer only, not
  with the stable layer. `compilar_partes` with the stable layer has been timed on the A100, but its numerical agreement
  is checked only with the core layer. `RecorteDinamico` is untested with FSDP.
- FSDP2 is covered by a two-process test on CPU (gloo) and a one-process NCCL check on the A100; it has not yet been
  run across several GPUs.
- Sharded saving gathers full weights and AdamW state on process 0 (~12 bytes per parameter).
- Sharded `evaluar()` repeats the same computation on every process.
- Behaviour beyond ~51 M parameters has not been measured yet; the 103.5 M kit (`examples/train_100m/`) is ready, and
  its first run is on the roadmap.
- The layer uses ~40 KiB per token per layer on the A100 (twice attention).

---

## Roadmap / Future work (v1.1+)

ASRN 1.0 trains a ~50 M-parameter model at a nearly flat 163–172k tokens/s (199–210k compiled by parts) from 1,024 to
32,768 tokens per sequence on a single A100. To scale beyond 100 M parameters, the release includes
[`examples/train_100m/`](../examples/train_100m/README.md), a ready-to-run Colab kit for a 103.5 M-parameter model on
FineWeb, and FSDP2 sharding (`repartir_fsdp`), covered by a two-process FSDP test on CPU in `tests/test_nucleo.py` and a
one-process NCCL check on the A100. The next versions build on this:

- **More seeds for log-normalizer table mixing (`mezcla_lse`)**, to settle whether it leaves the copy plateau earlier
  (3 of 3 seeds by step 400 in the previous run; here only 1 of 3, from step 600), and a Triton kernel for it only if
  it does.
- **Generation on the A100:** measure the O(1) writer there, and cut the per-token kernel launches of generation (for
  example with CUDA Graphs), which dominate on the GPU.
- **The first 100 M-parameter run** on FineWeb with `examples/train_100m/`.
- **A large-scale ablation of the regulation mechanisms** (the gate and its signals, the clock, Hawkes and the streak)
  on corpora with distractors and topic shifts.
- **Wider learning-rate grids** (including Transformer++ at 3·10⁻⁴ and 6·10⁻⁴). The re-runs of `mezcla_lse` on MQAR
  and of the Transformer++ prefix with the corrected wrapper are done (17 October run).
- **Compiled runs beyond training throughput** (single layer, generation, quality), and Mamba-2 compiled and with
  `causal-conv1d`.
- **Kernel proposals** from the technical report (estimated from the code, not measured): a persistent sliding window
  (−34% kernel bytes), sharing the n-gram sort across layers, a 14-bit radix sort, a grid-parallel scan, packed
  per-position records and autotuning.

---

## Verification

| Where | What | Result |
|---|---|---|
| CPU (4 cores, Python 3.13, PyTorch 2.14.1, Triton 3.8 in interpreter mode), from a clean export of the release commit, 15 October 2026 | `python pruebas/test_nucleo.py`; `python pruebas/test_estable.py` (now in `tests/`; 62-test and 17-test suites at that time) | **62 / 62** pass (~4.5 min) and **17 / 17** pass (~50 s) |
| CPU (4 cores, Python 3.13, PyTorch 2.14.1, Triton 3.8, NumPy 2.5.3), 16 October 2026, after the copy-test fix and the English aliases | `python pruebas/test_nucleo.py`; `python pruebas/test_estable.py` (now in `tests/`; 62-test core suite at that time) | **62 / 62** pass (~3 min) and **18 / 18** pass (~40 s) |
| Same CPU, PyTorch 2.11.0 (the version of the previous Colab runs), Triton 3.6, NumPy 2.3.5, 16 October 2026 | Same two suites (62-test core suite at that time) | 60 pass and 2 skipped (`torch.compile`, needs 2.14); **18 / 18** pass. With NumPy ≥ 2.4, Triton 3.6's interpreter fails in the three kernel-versus-PyTorch tests (`TypeError: only 0-dimensional arrays can be converted to Python scalars`), a Triton issue that Triton 3.8 does not have |
| Same CPU, PyTorch 2.11.0 and 2.14.1, 12 threads (`torch.set_num_threads(12)`), 16 October 2026 | `test_entrenar_aprende_a_copiar_y_reanuda` | Passes on both. The previous version of the test (lr 3e-3) fails on 2.11 with copy loss 4.199, the value seen on Colab |
| Same CPU, one thread (`OMP_NUM_THREADS=1`), after the reorganization of 1.0 (folder names in English, v85/v87 comparisons removed), 16 October 2026 | `python tests/test_nucleo.py`; `python tests/test_estable.py` | PyTorch 2.14.1 (Triton 3.8, NumPy 2.5.3): **55 / 55** and **18 / 18** pass. PyTorch 2.11.0 (Triton 3.6, NumPy 2.3.5): 53 pass and 2 skipped (`torch.compile`, needs 2.14), **18 / 18** pass |
| Fresh virtual environment (Python 3.13, PyTorch 2.14.1), 15 October 2026 | `pip install .` (no build isolation, no dependency resolution) | Installs `asrn 1.0.0` and the `asrn-train` entry point |
| CPU (4 cores, Python 3.13, PyTorch 2.14.1, Triton 3.8), 23 October 2026, with the O(1) writer | `python tests/test_nucleo.py`; `python tests/test_estable.py` (the current suites: 57 and 19 tests, 76 in total) | **57 / 57** and **19 / 19** pass |
| Same CPU, PyTorch 2.11.0 (Triton 3.6.0, NumPy 2.3.5), 23 October 2026, with the O(1) writer | Same two suites | 55 pass and 2 skipped (`torch.compile`, needs 2.14); **19 / 19** pass |
| Same CPU, PyTorch 2.14.1, 23 October 2026 | O(1) writer against the O(t) writer and the whole-text forward pass (`test_escritura_o1_igual_que_o_t` in both suites, `test_anillo_y_ngramas_como_su_definicion`): v89 and stable layer, rings that overflow, batches of 1–3, prompts of 1, 2 and 40 tokens, with and without prompt-only copy memory | Equal: max difference 2.2·10⁻⁷ |
| Same CPU, PyTorch 2.14.1, 23 October 2026 | `escribir()` against the previous O(t) implementation in 192 combinations of sampling options (temperature, top-k, top-p, prompt-only copy memory, quotation-preserving temperature, repetition penalty, no-repeat n-grams), v89 and stable layer | Same token sequences and logits within 10⁻⁴ in **192 / 192** |
| Same CPU (4 threads, PyTorch 2.14.1), 23 October 2026 | `benchmarks/medir_escritura.py`: ms per new token and prompt time against the context (512 to 131,072 tokens), O(1) against O(t) writer | [`results/escritura_cpu/`](../results/escritura_cpu/RESUMEN.md): O(1) 9.49–12.47 ms per token, slope −19.21 ns per context token (95% CI −41.54 to 3.12); O(t) 15.80–91.67 ms, slope 583.31 ns (537.13 to 629.49) |
| CPU, PyTorch 2.14.1 | Exact cost (FLOPs, saved bytes), latency, generation (O(t) writer), small-scale copy; probes for the clock, leak, shift and gradients | `results/banco_cpu/`, `history/probes/` |
| NVIDIA A100-SXM4-80GB (Colab; PyTorch 2.14.1+cu130, CUDA 13.0, Python 3.13.15; `mamba_ssm` 2.3.2.post1, `flash-linear-attention` 0.5.2; code fingerprint `5164e1f32513`), run of 17 October 2026 with `benchmarks/asrn_colab.ipynb` | `tests/test_nucleo.py`; `tests/test_estable.py` | **55 / 55** pass (5.3 min) and **18 / 18** pass (1.0 min), including the copy test at its new learning rate and the three package tests (package API, package command line and English aliases) |
| Same A100, same run | The 7 GPU checks of `tests/verificar_gpu.py` | **7 / 7** pass, `torch.compile` included (logits within 4.8·10⁻⁴, gradients 1.2·10⁻⁷): bf16 training of a 30 M model (loss 9.806 → 0.203, 46,435 tokens/s); checkpointing levels 2,009 / 1,708 / 1,099 MB; FSDP with NCCL in one process identical to unsharded (difference 0); GPU = CPU (max difference 6.1·10⁻⁷, gradient 4.5·10⁻⁷ relative); causality (1.8·10⁻⁷); step-by-step generation = full text over 80 steps (2.3·10⁻⁷) |
| Same A100, same run | GPU stability, single layer, ~50 M model (uncompiled and compiled), generation, copy, MQAR, WikiText-103, ablation; bounded-quotation experiment | [`results/banco_a100/`](../results/banco_a100/RESUMEN.md), [`results/citas_a100/`](../results/citas_a100/RESUMEN.md) |
| Previous run: NVIDIA A100-SXM4-80GB (Colab; PyTorch 2.11.0+cu130, CUDA 13.0, Python 3.13.15; `mamba_ssm` 2.3.2.post1, `flash-linear-attention` 0.5.2; uncompiled), runs of 9 October 2026 (from a ZIP, without a code fingerprint) and 11 October 2026 (code hash `176cfe250773`) | GPU stability, single layer, ~50 M model, generation, copy, MQAR, WikiText-103, ablation | [`history/results/banco_a100_pt211/`](../history/results/banco_a100_pt211/RESUMEN.md) |
| Previous run (`history/results/banco_a100_pt211/`), pre-1.0 file names | `test_asrn_v89.py` (now `tests/test_nucleo.py`; 62 tests at that time) | 59 pass, 2 skipped (`torch.compile`, needs 2.14), 1 fails (`test_entrenar_aprende_a_copiar_y_reanuda`; cause found and fixed since, see Problems resolved) |
| Previous run (`history/results/banco_a100_pt211/`), pre-1.0 file names | `test_asrn_estable.py` (now `tests/test_estable.py`; 14 tests at that time) | 13 pass, 1 failed because of a bug in the test (device), since fixed |
| Previous run (`history/results/banco_a100_pt211/`) | The 7 GPU checks of `verificar_gpu_v89.py` (now `tests/verificar_gpu.py`) | 6 pass, `torch.compile` skipped: bf16 training of a 30 M model (loss 9.806 → 0.203); checkpointing levels 2,009 / 1,708 / 1,099 MB; FSDP with NCCL in one process identical to unsharded; GPU = CPU (max difference 6.1·10⁻⁷, gradient 4.4·10⁻⁷ relative); causality (1.8·10⁻⁷); step-by-step generation = full text (2.3·10⁻⁷) |

The test suites run from the repository checkout: they import `asrn` from the root, and the core suite also uses
`benchmarks/rivales_lineales.py`. To reproduce the A100 benchmark, run `benchmarks/asrn_colab.ipynb` in Colab (PyTorch
2.14; it also runs every test and the bounded-quotation experiment). It saves to Drive and resumes where it left off.
Its sections took 284 minutes in total on the 17 October run. A GitHub Actions workflow
([`.github/workflows/tests.yml`](../.github/workflows/tests.yml)) runs both suites on CPU with Python 3.11 and 3.13;
without Triton, the kernel-versus-PyTorch tests are skipped there. Tested environments: Python 3.13 with PyTorch
2.11.0 and 2.14.1 (CPU, A100); CI on Python 3.11 and 3.13 (CPU).

### Provenance of the results

`results/banco_a100/` and `results/citas_a100/` come from code fingerprint `5164e1f32513`: the first 12 hex digits of
the sha256 of `asrn/*.py`, the root `*.py` and `benchmarks/*.py` concatenated in sorted order (the notebook prints
it). The 1.0.0 code differs from that code only in the O(1) writer (`asrn/nucleo.py`, `asrn/estable.py`), the
generation benchmark passing new tokens from the CPU (`benchmarks/banco_gpu.py`), docstrings, and the new
`benchmarks/medir_escritura.py`. The model's forward pass and gradients are bit-identical (verified), so every A100
number except the generation timing applies unchanged to 1.0.0.
