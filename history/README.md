# History: the development of ASRN v89

Reference material from the development of ASRN v89, the layer that became the core of ASRN 1.0 (`asrn/nucleo.py`).
It is kept for provenance and reproducibility. **None of it is needed to use ASRN 1.0**: the package (`asrn/`) does not
import anything from here, and the 1.0 measurements are in [`results/`](../results/README.md). It also keeps the
previous A100 run of the 1.0 benchmark (`results/banco_a100_pt211/`), which the 1.0 documents cite as a replication. The
material is not maintained; the documents are in Spanish.

| Folder | Contents |
|---|---|
| [`docs/`](docs/) | The development reports: `ASRN_v89_Whitepaper.md` (technical record of v86 → v89: the design, the synthetic quotation experiment, the GPU runs), `ESTABILIDAD_Y_BANCO.md` (the October 2026 review: numerical failure modes, the protections of the stable layer, and the analysis of the 1.0 benchmark on the CPU and in its previous A100 run, with PyTorch 2.11) and `RENDIMIENTO.md` (where the training time goes, the speed work and the kernel plan) |
| [`experiments/`](experiments/) | The per-version speed measurement: `medir_velocidad_v89.py` (same session: same numbers with and without Triton, one layer, the ~50 M model with and without `torch.compile`, where the time goes); the Colab notebook that ran it was retired in 1.0 |
| [`results/`](results/README.md) | The development runs on A100 GPUs: runs 1–5 of the v89 GPU verification notebook (`pruebas_gpu_a100.ipynb`, retired in 1.0; `benchmarks/asrn_colab.ipynb` replaces it) (`corrida_1/` … `corrida_5/`; run 5 is against reimplementations of Mamba-2 and RWKV-7), the local-block variants (`bloques_1/`), the compiled speed (`velocidad/`). Also [`banco_a100_pt211/`](results/banco_a100_pt211/RESUMEN.md): the previous A100 runs of the 1.0 benchmark (9 and 11 October 2026, notebook `banco_a100.ipynb`, PyTorch 2.11.0, nothing compiled), formerly `results/banco_a100/`, now replaced by the run of 17 October 2026 with PyTorch 2.14.1 |
| [`probes/`](probes/) | The numerical-failure probes that justify the protections of the stable layer (`asrn/estable.py`): `reloj.py` (precision of the 32-bit clock), `fuga.py` and `fuga_estable.py` (causal leak with a huge time kernel), `corrimiento.py` (shifting the clock far from the start of the text), `gradientes.py` and `gradientes_raro.py` (gradients and gate signals against length) |

Run the scripts from the repository root, for example `python history/probes/reloj.py`. They were updated only so they find the package and the other files at
their current paths; their logic is unchanged. The speed measurement needs a GPU and measures the v89 layer, not the
stable layer of 1.0. Development charts, executed notebooks and the superseded architecture notes are not part of the
published repository; every number they showed is in the JSON files kept here.

**Names in the archived documents.** Paths in these documents were updated to the current layout, but they still use
the file names of the time, and their test counts are those of the suites of the time:

| Then | Now |
|---|---|
| `asrn_v89.py` | `asrn/nucleo.py` (`import asrn.nucleo`) |
| `asrn_estable.py` | `asrn/estable.py` (`import asrn` uses it by default) |
| `pruebas/test_asrn_v89.py`, `test_asrn_estable.py`, `verificar_gpu_v89.py` | `tests/test_nucleo.py`, `tests/test_estable.py`, `tests/verificar_gpu.py` |
| `asrn_v85.py`, `asrn_v87.py` (reference implementations) | removed in 1.0 |
| `experimento_cita_v87.py` (quotation experiment, v85 vs v87) | removed in 1.0; the experiment with the 1.0 layer is `benchmarks/experimento_citas.py` |

The v85 and v87 reference implementations were removed in 1.0, together with the tests that compared v89 against them;
mentions of v85, v86 and v87 in these documents are historical.

## Español

Material de referencia del desarrollo de ASRN v89, la capa que se volvió el núcleo de ASRN 1.0 (`asrn/nucleo.py`). Se
conserva para poder rastrear y reproducir el trabajo. **Nada de esto hace falta para usar ASRN 1.0**: el paquete
(`asrn/`) no importa nada de aquí, y las mediciones de 1.0 están en [`results/`](../results/README.md). También guarda
la corrida anterior del banco de 1.0 en A100 (`results/banco_a100_pt211/`), que los documentos de 1.0 citan como
réplica. Este material no se mantiene; los documentos están en español.

| Carpeta | Qué hay |
|---|---|
| [`docs/`](docs/) | Los informes de desarrollo: `ASRN_v89_Whitepaper.md` (registro técnico de v86 a v89: el diseño, el experimento sintético de citas, las corridas en GPU), `ESTABILIDAD_Y_BANCO.md` (la revisión de octubre de 2026: modos de falla numérica, las protecciones de la capa estable y el análisis del banco de 1.0 en CPU y en su corrida anterior en A100, con PyTorch 2.11) y `RENDIMIENTO.md` (dónde se va el tiempo de entrenamiento, el trabajo de velocidad y el plan de kernels) |
| [`experiments/`](experiments/) | La medición de velocidad por versión: `medir_velocidad_v89.py` (en una misma sesión: mismas cuentas con y sin Triton, una capa, el modelo de ~50 M con y sin `torch.compile`, dónde se va el tiempo); el cuaderno de Colab que la corrió se retiró en 1.0 |
| [`results/`](results/README.md) | Las corridas de desarrollo en GPUs A100: las corridas 1 a 5 del cuaderno de verificación en GPU de v89 (`pruebas_gpu_a100.ipynb`, retirado en 1.0; lo reemplaza `benchmarks/asrn_colab.ipynb`) (`corrida_1/` … `corrida_5/`; la 5 es contra reimplementaciones de Mamba-2 y RWKV-7), las variantes del bloque local (`bloques_1/`), la velocidad compilada (`velocidad/`). Además, [`banco_a100_pt211/`](results/banco_a100_pt211/RESUMEN.md): las corridas anteriores del banco de 1.0 en A100 (9 y 11 de octubre de 2026, cuaderno `banco_a100.ipynb`, PyTorch 2.11.0, nada compilado), antes `results/banco_a100/`, hoy reemplazadas por la corrida del 17 de octubre de 2026 con PyTorch 2.14.1 |
| [`probes/`](probes/) | Las sondas de falla numérica que justifican las protecciones de la capa estable (`asrn/estable.py`): `reloj.py` (precisión del reloj en 32 bits), `fuga.py` y `fuga_estable.py` (fuga causal con un núcleo de tiempo enorme), `corrimiento.py` (correr el reloj lejos del inicio del texto), `gradientes.py` y `gradientes_raro.py` (gradientes y señales de la puerta según el largo) |

Los programas se corren desde la raíz del repositorio, por ejemplo `python history/probes/reloj.py`. Solo se cambiaron para que encuentren el paquete y los demás archivos en sus
rutas actuales; su lógica es la misma. La medición de velocidad necesita una GPU y mide la capa de v89, no la capa
estable de 1.0. Las gráficas de desarrollo, los cuadernos ejecutados y las notas de arquitectura anteriores no forman
parte del repositorio publicado; cada número que mostraban está en los JSON que se guardan aquí.

**Nombres en los documentos archivados.** Las rutas de estos documentos se actualizaron a la organización actual, pero
siguen usando los nombres de archivo de entonces, y sus cuentas de pruebas son las de las pruebas de entonces:

| Entonces | Ahora |
|---|---|
| `asrn_v89.py` | `asrn/nucleo.py` (`import asrn.nucleo`) |
| `asrn_estable.py` | `asrn/estable.py` (`import asrn` la usa por omisión) |
| `pruebas/test_asrn_v89.py`, `test_asrn_estable.py`, `verificar_gpu_v89.py` | `tests/test_nucleo.py`, `tests/test_estable.py`, `tests/verificar_gpu.py` |
| `asrn_v85.py`, `asrn_v87.py` (implementaciones de referencia) | eliminados en 1.0 |
| `experimento_cita_v87.py` (experimento de citas, v85 contra v87) | eliminado en 1.0; el experimento con la capa de 1.0 es `benchmarks/experimento_citas.py` |

Las implementaciones de referencia v85 y v87 se eliminaron en 1.0, junto con las pruebas que comparaban v89 con ellas;
las menciones de v85, v86 y v87 en estos documentos son históricas.
