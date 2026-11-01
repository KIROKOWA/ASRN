# Changelog

All notable changes to ASRN are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/).
The measurements and caveats behind each entry are in the release notes.

## [1.1] — 2026-11-01

A training change: the suffix weight and the gate signals of every ASRN layer learn faster. No architecture change, no
extra cost, and checkpoints load unchanged.

### Changed

- `crear_optimizador()` puts the suffix weight and the gate signals of each ASRN layer (`w_sufijo`, `g.bias`,
  `g_conf.weight`, `g_conf.bias`; listed in `asrn.PESOS_SUFIJO_Y_PUERTA`) in their own AdamW
  groups, with the learning rate multiplied by `mult_sufijo_y_puerta` (default `asrn.MULT_SUFIJO_Y_PUERTA = 3`) and the
  same weight decay they had. With AdamW each weight moves at most about `lr` per step, and these few numbers, which
  decide what to copy, did not grow in time with the 1.0 rate.
- `entrenar()` multiplies each group's scheduled rate by its `mult`, saves `mult_sufijo_y_puerta` in the checkpoint
  settings, and on resume uses the file's value. Checkpoints from 1.0 (which lack it) resume with 1, exactly as they
  started; asking for a different value on resume is a clear error.
- `PasoGrafado` keeps one learning-rate tensor per group, so the faster groups also work inside a CUDA Graph.
- New option `--mult_sufijo_y_puerta` in `python -m asrn`, `examples/train_100m/entrenar.py` and
  `benchmarks/banco_asrn.py` (1 reproduces the 1.0 recipe and the published 1.0 tables).

### Why (measured with `benchmarks/banco_asrn.py`, `a100` profile, Colab L4)

| Task | 1.0 (×1) | 1.1 (×3 suffix and gate) |
| --- | --- | --- |
| Copy, accuracy at L = 1,024 (3 seeds) | 66.7% ± 57.6 (2 of 3 seeds) | 100% ± 0.0 (3 of 3) |
| Copy, accuracy at L = 16,384 | 45.4% ± 39.7 | 82.9% ± 14.5 |
| MQAR, accuracy at L = 2,048 (3 seeds) | 99.58% ± 0.34 | 99.93% ± 0.09 |
| WikiText-103 (45 M), perplexity at L = 4,096 (2 seeds) | 63.34 ± 0.68 | 63.00 ± 0.26 |

Higher doses (×5, ×10) learned in every seed but lost long-context accuracy (copy at 16,384: 51.9% and 29.6%); ×2
still left one seed stuck. The good zone corresponds to a learned suffix advantage of about 2 to 4.

### Tests

- New: the optimizer groups and rates (×3 and ×1 = the 1.0 groups), saving and resuming with ×3, resuming a
  1.0-style checkpoint with ×1, and the per-group rate in `PasoGrafado`. The `PasoGrafado` test's reference step now
  applies each group's `mult`.

## [1.0] — 2026-10-24

The first packaged release, relative to the v89 development code. Details:
[docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md) (Spanish: [docs/RELEASE_NOTES_es.md](docs/RELEASE_NOTES_es.md)).

### Added

- The `asrn` package: `pip install .` gives `import asrn`, `python -m asrn` and the `asrn-train` command (Python ≥ 3.11,
  PyTorch ≥ 2.9, NumPy).
- English aliases of the Spanish API (`asrn.ASRNLayer`, `asrn.ASRNModel`, `asrn.create_model`, `asrn.train`,
  `asrn.generate`, `asrn.load`, …).
- The stable layer `CapaASRNEstable` (`asrn/estable.py`) with numerical protections (`Protecciones`): group A
  (`mascara_exacta`, `reloj_64`, `beta_max`, `log_alfa_min`, `mejor_max`) on by default, group B
  (`Protecciones.textos_largos(L)`) opt-in; `asrn.convertir()`; opt-in per-group gradient clipping
  (`RecorteDinamico`); and an `ESTABLE` compile-time mode of the bucket-attention Triton kernels (two-part clock, exact
  mask).
- O(1) generation per new token (`escribir()` / `asrn.generate`): a ring of the most recent positions per bucket and
  the Hawkes state carried per n-gram key, with the same token sequences as the previous O(t) writer.
- The benchmark against official Mamba-2, official RWKV-7 and a Transformer++: the Colab notebook
  `benchmarks/asrn_colab.ipynb` (every test, the A100 benchmark and the quotation experiment) and its results in
  `results/banco_a100/`, plus `results/banco_cpu/` and the CPU writing benchmark (`benchmarks/medir_escritura.py`,
  `results/escritura_cpu/`).
- The bounded-quotation experiment ported to the 1.0 stable layer (`benchmarks/experimento_citas.py`), with results on
  CPU (`results/citas/`) and on the A100 (`results/citas_a100/`).
- `examples/train_100m/`: a Colab kit to train a 103.5 M-parameter model on FineWeb.
- A GitHub Actions workflow that runs both test suites on CPU (Python 3.11 and 3.13), issue templates, `SECURITY.md`,
  `CITATION.cff` and an architecture diagram (`docs/asrn_architecture.svg`, `docs/asrn_arquitectura.svg`).
- Tests for the stable layer, the package API, the English aliases and the O(1) writer.

### Changed

- `import asrn` gives the stable layer by default; its outputs differ from v89 by rounding only. `import asrn.nucleo`
  or `Protecciones.ninguna()` gives the exact v89 behaviour. Checkpoints load unchanged in either layer.
- The modules moved into a package (`asrn_v89.py` → `asrn/nucleo.py`, `asrn_estable.py` → `asrn/estable.py`) and the
  folders were renamed (for example `pruebas/` → `tests/` and `experimentos/` → `benchmarks/`; the full mapping is in
  the release notes). The documents moved to `docs/`, and the development history to `history/` (not part of the
  package; kept because the documents cite it as evidence).
- Step-by-step generation costs O(1) per new token instead of O(t). Its memory still grows with the text, plus fixed
  rings.

### Fixed

- Six numerical problems of the v89 layer: the causal leak through the −10⁹ mask, the precision loss and stall of the
  32-bit clock, score cancellation under a clock shift, unbounded β leading to NaN, the 16-bit clock projection under
  autocast, and TF32 on the clock projection.
- Benchmark problems: `FlopCounterMode` miscounting Mamba-2's grouped convolution, the `ZeroDivisionError` of the
  one-layer RWKV-7 wrapper, the Transformer++ prefix computing logits for every position, the NaN of the experimental
  `mezcla_lse` variant on MQAR, and a device bug in `test_convertir_guardar_cargar`.
- `test_entrenar_aprende_a_copiar_y_reanuda`, which failed on the Colab A100 machine with PyTorch 2.11 because leaving
  the copy plateau depended on the order of floating-point sums; the test now trains at 6e-3.

### Removed

- The v85 and v87 reference implementations (`pruebas/versiones_anteriores/asrn_v85.py`, `asrn_v87.py`) and the tests
  that compared with them.
- `experimentos/experimento_cita_v87.py`, replaced by `benchmarks/experimento_citas.py`.
