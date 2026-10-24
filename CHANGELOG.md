# Changelog

All notable changes to ASRN are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/).
The measurements and caveats behind each entry are in the release notes.

## [1.0.0] — 2026-10-24

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
