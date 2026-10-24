# Benchmarks

The ASRN 1.0 benchmark, the bounded-quotation experiment and the Google Colab notebooks that run them on an A100. The
measured results are in [`results/`](../results/README.md); the analysis is in the
[technical report](../docs/TECHNICAL_REPORT.md). Run every command from the repository root.

| File | What it does | Writes to |
|---|---|---|
| `banco_asrn.py` | The benchmark entry point, with a `cpu` and an `a100` profile. Sections: `costo` (exact forward + backward FLOPs and bytes saved for the backward pass, per layer), `latencia` (one layer's forward + backward time against length), `escritura` (time per generated token against context, with the writer `escribir()` uses: O(1) per token in 1.0; the committed `results/banco_cpu/` figures were measured with the earlier O(t) writer), `calidad` (copy, MQAR and text, ASRN against Transformer, Mamba-2 and RWKV-7), `ablacion` (ASRN removing one piece at a time), `resumir` (Markdown tables from the JSON) | `results/banco_cpu/` (change with `--salida`) |
| `banco_gpu.py` | The GPU-only sections, called by `asrn_colab.ipynb`: stability on the GPU (32-bit clock with CUDA `cumsum`, `ESTABLE` kernel, causal leak, shifted clock, TF32), one layer alone, a ~50 M model, generation (ASRN with the O(1) writer of 1.0, the new tokens passed from the CPU as `escribir()` does), and the charts | the notebook's output folder |
| `banco_modelos.py` | ASRN and its ablation variants, and the reference layers used for the exact cost | — |
| `modelos_oficiales.py` | The rivals in their original versions: Transformer++ (Llama recipe with FlashAttention-2), Mamba-2 (`mamba_ssm`) and RWKV-7 (`flash-linear-attention`), with step-by-step generation | — |
| `rivales_lineales.py` | Reimplementations of Mamba-2 and RWKV-7 (recurrence with the flash-linear-attention kernels on a GPU, plain PyTorch otherwise). Used when an original cannot be installed, and by the core tests | — |
| `experimento_citas.py` | The bounded-quotation experiment with the ASRN 1.0 stable layer (CPU, or GPU with `--dispositivo cuda`), see below | `results/citas/` (the notebook's GPU run: `results/citas_a100/`) |
| `medir_escritura.py` | Time per new token when writing, against the context, for the O(1) writer of 1.0 and the previous O(t) one (`_Escritor(modelo, o1=False)`), which give the same outputs: an ASRN model of 700,484 parameters (d = 128, 2 ASRN layers), batch 1, no sampling, the median of `--repeticiones` (9) batches of `--nuevos` (32) tokens at each of `--contextos` (512 to 131,072 tokens), the two writers interleaved; also the slope of ms per token against the context with its 95% interval (needs SciPy). CPU by default, GPU with `--dispositivo cuda` | `results/escritura_cpu/` (on a GPU: `results/escritura_gpu/`; change with `--salida`) |
| `asrn_colab.ipynb` | **The one Colab notebook** (A100, PyTorch 2.14): the test suites and the 7 GPU checks, GPU stability, one layer, the ~50 M model (also compiled), generation (the O(1) writer), copy, MQAR, WikiText-103, the ablation and the bounded-quotation experiment against the original rivals. It saves every finished section to Google Drive and resumes after a disconnection | Drive folder `ASRN/colab_pt214`, then `asrn_colab.zip` |
| `builders/` | The scripts that generate the Colab notebooks: `asrn_colab.ipynb` and the 100M kit's (see below) | the `.ipynb` files |

## Running on CPU

```bash
python benchmarks/banco_asrn.py costo latencia escritura            # ~5 min
TRITON_INTERPRET=1 python benchmarks/banco_asrn.py costo_kernel     # saved bytes with the Triton kernel
python benchmarks/banco_asrn.py calidad ablacion --tareas copia,mqar   # hours on CPU
python benchmarks/banco_asrn.py resumir                             # tables in results/banco_cpu/RESUMEN.md
python benchmarks/medir_escritura.py                                # O(1) against O(t) writing: results/escritura_cpu/
```

Each section replaces its part of `results/banco_cpu/resultados.json` (the committed results); to keep them, write
elsewhere with `--salida DIR`. The CPU text task reads the Vim 9.1 documentation from `/usr/share/vim/vim91/doc`
(package `vim-runtime`); without it, use `--tareas copia,mqar`.

## Running on an A100 (Colab)

Open `asrn_colab.ipynb` in Colab, choose an A100 runtime and run all cells. Its first section installs PyTorch 2.14.1
(the first version whose `torch.compile` handles the ASRN layer) and uninstalls Colab's `torchvision` and `torchaudio`,
which the notebook does not use; the session then restarts once, and *Run all* again continues. The configuration cell
sets `MODO` (`'completo'` or `'rapido'`), `SECCIONES`, `PYTORCH`, `REPO` and `RAMA`; the notebook clones the repository
(or asks for a ZIP of the branch if it cannot clone). At the end it downloads `asrn_colab.zip` with `resultados.json`,
`RESUMEN.md`, the charts and the `citas/` folder. In the `completo` mode, the run of 17 October 2026 took about
4 h 50 min (~62 Colab compute units, at ~13 per hour on an A100); the table in the notebook's first cell gives the
measured time of each section (the `rapido` mode is estimated at about 1 h 45 min).

**Generation on the A100.** The notebook's generation section (section 6) now measures ASRN with the O(1) writer of
1.0, with the new tokens passed from the CPU as `escribir()` does. The generation figures of the 17 October 2026 run
(`results/banco_a100/`) were measured with the previous O(t) writer, so the O(1) writer has not been measured on the
A100 yet: the next run measures it.

**How the rivals get installed.** After PyTorch 2.14, the notebook installs, in this order:
`pip install -r requirements.txt`; `pip install flash-linear-attention datasets tiktoken einops ninja packaging wheel
matplotlib` (RWKV-7 comes from `flash-linear-attention`); then `mamba-ssm` with `--no-build-isolation` and
`MAMBA_SKIP_CUDA_BUILD=TRUE`, which skips compiling the Mamba-1 CUDA extension (Mamba-2 uses the authors' Triton SSD
kernels). With `MAMBA_COMPILADO = True` it also tries a compiled `causal-conv1d` (the fused path of Mamba-2); if there
is no ready wheel and compiling takes more than 8 minutes, Mamba-2 uses PyTorch's conv1d. Setting `ASRN_SIN_MAMBA=1`
skips `mamba-ssm`. The notebook first checks that each original trains and generates on that GPU; if one cannot train,
it uses the reimplementation from `rivales_lineales.py` and records it in the `implementacion` field of each row.
Outside Colab, `pip install ".[bench]"` installs the same Python extras except `mamba-ssm`, which is installed
separately.

## The bounded-quotation experiment (`experimento_citas.py`)

Does the regulation machinery of the layer (the copy gate, the Hawkes intensity and the streak) teach the model when to
stop copying? The training text is random tokens (V = 64) with quotations of 6 to 12 tokens copied from the last 200, so
real quotations always end. Five models with the ASRN 1.0 stable layer (protection group A, 2 local blocks and 1 ASRN
layer, d = 64) are trained with the same seed, data and recipe (lr 3e-3, batch 16, length 256, 6,000 steps by default):

| Variant | What changes |
|---|---|
| `ASRN 1.0` (`estable_A`) | the full layer |
| `sin Hawkes ni racha` (`sin_hawkes_racha`) | `u_hawkes`, `d_racha` and their two kappas are frozen at their neutral initial values: the gate does not see repetition |
| `puerta sin senales` (`puerta_sin_senales`) | the gate sees only the state: σ(w_g · h + b) |
| `sin puerta` (`sin_puerta`) | the gate is fixed at 1 |
| `sin reloj` (`sin_reloj`) | the proper-time clock frozen at its initial value (β = 1, α = 1, γ = 0): time counts tokens |

Each model then continues 64 validation texts (128-token prompt, 128 new tokens) at temperature 1 and at 0.8 with
top-k 40, with and without counting the prompt as already seen (ASRN 1.0 also with `conservar_cita=False`), and the
script measures the fraction of quoted tokens and the length of quotation streaks against the real continuation. It also
records the validation loss and the gate's response to the streak.

```bash
python benchmarks/experimento_citas.py --semillas 0 1 2     # one run per seed -> results/citas/semilla_<s>.json
python benchmarks/experimento_citas.py --resumir            # joins the seeds -> results/citas/resultados.json, RESUMEN.md
```

Each model took 39–87 minutes on one CPU thread at 6,000 steps in the runs of `results/citas/` (4 cores shared by three
seeds and other work); on the A100, in the notebook's quotation section, each model trained in 1.6–2.1 minutes and the
three seeds took 36.9 minutes with the writing ([`results/citas_a100/`](../results/citas_a100/RESUMEN.md)). Seeds can
run as separate processes, for example `python benchmarks/experimento_citas.py --semillas 0 --hilos 1 >
results/citas/semilla_0.log 2>&1` (likewise for 1 and 2). The per-seed JSON is saved after each model; relaunching an
interrupted run resumes from the last finished model of that seed. Other options: `--pasos`, `--hilos`, `--salida`.

## Changing a notebook

Do not edit the `.ipynb` files by hand. Change the builder in `builders/` and regenerate (the notebooks are written
without outputs):

```bash
python benchmarks/builders/construir_cuaderno.py benchmarks/asrn_colab.ipynb
python benchmarks/builders/construir_entrenar_100m.py examples/train_100m/entrenar_asrn_100m.ipynb
```

The A100 results in `results/banco_a100/` and `results/citas_a100/` come from one execution of `asrn_colab.ipynb` on
17 October 2026 (PyTorch 2.14.1). The previous A100 runs (9 and 11 October 2026), made by the notebook of that time
(`banco_a100.ipynb`, PyTorch 2.11, nothing compiled), which `asrn_colab.ipynb` replaces, are kept in
[`history/results/banco_a100_pt211/`](../history/results/banco_a100_pt211/RESUMEN.md). The tools used to develop v89
(per-version speed measurement, numerical probes) are in [`history/`](../history/README.md).

## Español

El banco de pruebas de ASRN 1.0, el experimento de citas acotadas y los cuadernos de Google Colab que los corren en una
A100. Los resultados medidos están en [`results/`](../results/README.md); el análisis, en el
[informe técnico](../docs/TECHNICAL_REPORT_es.md). Todos los comandos se corren desde la raíz del repositorio.

| Archivo | Qué hace | Dónde escribe |
|---|---|---|
| `banco_asrn.py` | El banco en sí, con perfiles `cpu` y `a100`. Secciones: `costo` (FLOPs exactos de ida y vuelta y bytes guardados para la vuelta, por capa), `latencia` (ida y vuelta de una capa según el largo), `escritura` (tiempo por pieza nueva según el contexto, con la escritura que usa `escribir()`: O(1) por pieza en 1.0; las cifras de `results/banco_cpu/` del repositorio se midieron con la escritura O(t) de antes), `calidad` (copia, MQAR y texto, ASRN contra Transformer, Mamba-2 y RWKV-7), `ablacion` (ASRN quitando una pieza a la vez), `resumir` (tablas en Markdown desde el JSON) | `results/banco_cpu/` (se cambia con `--salida`) |
| `banco_gpu.py` | Lo que solo se mide en GPU, desde `asrn_colab.ipynb`: estabilidad en la GPU (reloj de 32 bits con `cumsum` de CUDA, kernel `ESTABLE`, fuga causal, reloj corrido, TF32), una capa sola, un modelo de ~50 M, escritura (ASRN con la escritura O(1) de 1.0, con las piezas nuevas pasadas desde la CPU como hace `escribir()`) y las gráficas | la carpeta de salida del cuaderno |
| `banco_modelos.py` | ASRN y sus variantes de ablación, y las capas de referencia para el costo exacto | — |
| `modelos_oficiales.py` | Los rivales en su versión original: Transformer++ (receta de Llama con FlashAttention-2), Mamba-2 (`mamba_ssm`) y RWKV-7 (`flash-linear-attention`), con escritura paso a paso | — |
| `rivales_lineales.py` | Reimplementaciones de Mamba-2 y RWKV-7 (la recurrencia con los kernels de flash-linear-attention en GPU; PyTorch normal si no). Se usan si un original no se puede instalar, y en las pruebas del núcleo | — |
| `experimento_citas.py` | El experimento de citas acotadas con la capa estable de ASRN 1.0 (CPU, o GPU con `--dispositivo cuda`), ver abajo | `results/citas/` (la corrida del cuaderno en GPU: `results/citas_a100/`) |
| `medir_escritura.py` | Tiempo por pieza nueva al escribir, según el contexto, con la escritura O(1) de 1.0 y la O(t) de antes (`_Escritor(modelo, o1=False)`), que dan las mismas salidas: un modelo ASRN de 700 484 parámetros (d = 128, 2 capas ASRN), lote 1, sin muestrear, la mediana de `--repeticiones` (9) tandas de `--nuevos` (32) piezas en cada uno de los `--contextos` (de 512 a 131 072 piezas), las dos escrituras alternadas; también la pendiente de los ms por pieza según el contexto con su intervalo de 95% (necesita SciPy). En CPU por omisión, en GPU con `--dispositivo cuda` | `results/escritura_cpu/` (en una GPU: `results/escritura_gpu/`; se cambia con `--salida`) |
| `asrn_colab.ipynb` | **El único cuaderno de Colab** (A100, PyTorch 2.14): las baterías de pruebas y las 7 comprobaciones en GPU, estabilidad en la GPU, una capa, el modelo de ~50 M (también compilado), escritura (la escritura O(1)), copia, MQAR, WikiText-103, la ablación y el experimento de citas acotadas contra los rivales originales. Guarda cada sección terminada en Google Drive y retoma si Colab se desconecta | carpeta `ASRN/colab_pt214` de Drive y después `asrn_colab.zip` |
| `builders/` | Los programas que arman los cuadernos de Colab: `asrn_colab.ipynb` y el del kit de 100M (ver abajo) | los `.ipynb` |

### Correr en CPU

```bash
python benchmarks/banco_asrn.py costo latencia escritura            # ~5 min
TRITON_INTERPRET=1 python benchmarks/banco_asrn.py costo_kernel     # bytes guardados con el kernel Triton
python benchmarks/banco_asrn.py calidad ablacion --tareas copia,mqar   # horas en CPU
python benchmarks/banco_asrn.py resumir                             # tablas en results/banco_cpu/RESUMEN.md
python benchmarks/medir_escritura.py                                # escritura O(1) contra O(t): results/escritura_cpu/
```

Cada sección reemplaza su parte de `results/banco_cpu/resultados.json` (los resultados del repositorio); para
conservarlos, escribe en otra carpeta con `--salida DIR`. La tarea de texto en CPU lee la documentación de Vim 9.1
de `/usr/share/vim/vim91/doc` (paquete `vim-runtime`); sin ella, usa `--tareas copia,mqar`.

### Correr en una A100 (Colab)

Abre `asrn_colab.ipynb` en Colab, elige un entorno con A100 y ejecuta todas las celdas. Su primera sección instala
PyTorch 2.14.1 (la primera versión cuyo `torch.compile` maneja la capa ASRN) y desinstala `torchvision` y `torchaudio`
de Colab, que el cuaderno no usa; después la sesión se reinicia una vez, y *Ejecutar todas* otra vez sigue. La celda de
configuración fija `MODO` (`'completo'` o `'rapido'`), `SECCIONES`, `PYTORCH`, `REPO` y `RAMA`; el cuaderno clona el
repositorio (o pide el ZIP de la rama si no puede clonarlo). Al final descarga `asrn_colab.zip` con `resultados.json`,
`RESUMEN.md`, las gráficas y la carpeta `citas/`. En el modo `completo`, la corrida del 17 de octubre de 2026 tardó
unas 4 h 50 min (~62 unidades de Colab, a ~13 por hora con una A100); la tabla de la primera celda del cuaderno da el
tiempo medido de cada sección (el modo `rapido` se estima en unas 1 h 45 min).

**La escritura en la A100.** La sección de escritura del cuaderno (la sección 6) ahora mide ASRN con la escritura O(1)
de 1.0, con las piezas nuevas pasadas desde la CPU como hace `escribir()`. Las cifras de escritura de la corrida del
17 de octubre de 2026 (`results/banco_a100/`) se midieron con la escritura O(t) anterior, así que la escritura O(1)
todavía no se midió en la A100: la próxima corrida la mide.

**Cómo se instalan los rivales.** Después de PyTorch 2.14, el cuaderno instala, en este orden:
`pip install -r requirements.txt`; `pip install flash-linear-attention datasets tiktoken einops ninja packaging wheel
matplotlib` (RWKV-7 sale de `flash-linear-attention`); después `mamba-ssm` con `--no-build-isolation` y
`MAMBA_SKIP_CUDA_BUILD=TRUE`, que evita compilar la extensión CUDA de Mamba-1 (Mamba-2 usa los kernels SSD de Triton de
sus autores). Con `MAMBA_COMPILADO = True` intenta también `causal-conv1d` compilado (el camino fusionado de Mamba-2);
si no hay rueda lista y compilarlo tarda más de 8 minutos, Mamba-2 usa el conv1d de PyTorch. Con `ASRN_SIN_MAMBA=1` no
se instala `mamba-ssm`. El cuaderno comprueba primero que cada original entrena y escribe en esa GPU; si uno no entrena,
usa la reimplementación de `rivales_lineales.py` y lo anota en el campo `implementacion` de cada fila. Fuera de Colab,
`pip install ".[bench]"` instala los mismos extras de Python salvo `mamba-ssm`, que se instala aparte.

### El experimento de citas acotadas (`experimento_citas.py`)

¿Los mecanismos de regulación de la capa (la puerta de copia, la intensidad de Hawkes y la racha) le enseñan al modelo
cuándo dejar de copiar? El texto de entrenamiento son piezas al azar (V = 64) con citas de 6 a 12 piezas copiadas de las
últimas 200, así que las citas reales siempre terminan. Se entrenan cinco modelos con la capa estable de ASRN 1.0 (grupo
A de protecciones, 2 bloques locales y 1 capa ASRN, d = 64), con la misma semilla, los mismos datos y la misma receta
(lr 3e-3, lote 16, largo 256, 6 000 pasos por omisión):

| Variante | Qué cambia |
|---|---|
| `ASRN 1.0` (`estable_A`) | la capa completa |
| `sin Hawkes ni racha` (`sin_hawkes_racha`) | `u_hawkes`, `d_racha` y sus dos kappas quedan congelados en su valor inicial neutro: la puerta no ve lo repetido |
| `puerta sin senales` (`puerta_sin_senales`) | la puerta solo ve el estado: σ(w_g · h + b) |
| `sin puerta` (`sin_puerta`) | la puerta fija en 1 |
| `sin reloj` (`sin_reloj`) | el reloj de tiempo propio congelado en su valor inicial (β = 1, α = 1, γ = 0): el tiempo cuenta piezas |

Después cada modelo continúa 64 textos de validación (inicio de 128 piezas, 128 piezas nuevas) con temperatura 1 y con
0.8 y top-k 40, contando o no el inicio como ya visto (ASRN 1.0 también con `conservar_cita=False`), y se mide la
fracción de piezas citadas y el largo de las rachas de cita contra la continuación real. También se guarda la pérdida de
validación y la respuesta de la puerta a la racha.

```bash
python benchmarks/experimento_citas.py --semillas 0 1 2     # una corrida por semilla -> results/citas/semilla_<s>.json
python benchmarks/experimento_citas.py --resumir            # junta las semillas -> results/citas/resultados.json, RESUMEN.md
```

Cada modelo tardó entre 39 y 87 minutos en un hilo de CPU con 6 000 pasos en las corridas de `results/citas/` (4 núcleos
compartidos por tres semillas y otro trabajo); en la A100, en la sección de citas del cuaderno, cada modelo entrenó en
1.6–2.1 minutos y las tres semillas tardaron 36.9 minutos con la escritura
([`results/citas_a100/`](../results/citas_a100/RESUMEN.md)). Las semillas se pueden correr en procesos separados, por
ejemplo `python benchmarks/experimento_citas.py --semillas 0 --hilos 1 > results/citas/semilla_0.log 2>&1` (igual con
1 y 2). El JSON de cada semilla se guarda después de cada modelo; si una corrida se corta, volver a lanzarla retoma
desde el último modelo terminado de esa semilla. Otras opciones: `--pasos`, `--hilos`, `--salida`.

### Cambiar un cuaderno

No edites los `.ipynb` a mano. Cambia su programa en `builders/` y vuelve a generarlo (los cuadernos se escriben sin
salidas):

```bash
python benchmarks/builders/construir_cuaderno.py benchmarks/asrn_colab.ipynb
python benchmarks/builders/construir_entrenar_100m.py examples/train_100m/entrenar_asrn_100m.ipynb
```

Los resultados de la A100 en `results/banco_a100/` y `results/citas_a100/` salen de una sola ejecución de
`asrn_colab.ipynb` el 17 de octubre de 2026 (PyTorch 2.14.1). Las corridas anteriores en A100 (9 y 11 de octubre de
2026), hechas con el cuaderno de entonces (`banco_a100.ipynb`, PyTorch 2.11, nada compilado), al que reemplaza
`asrn_colab.ipynb`, se guardan en [`history/results/banco_a100_pt211/`](../history/results/banco_a100_pt211/RESUMEN.md).
Lo que se usó para desarrollar v89 (la medición de velocidad por versión, las sondas numéricas) está en
[`history/`](../history/README.md).
