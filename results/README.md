# Results

The measured results of ASRN 1.0, as they came out of the scripts (including what does not favour ASRN). Each folder
has a `RESUMEN.md` with the tables: read the numbers there or in the JSON, not in this index. The analysis, with the
caveats of each figure, is in the [technical report](../docs/TECHNICAL_REPORT.md).

| Folder | Produced by | Contents |
|---|---|---|
| [`banco_a100/`](banco_a100/RESUMEN.md) | [`benchmarks/asrn_colab.ipynb`](../benchmarks/README.md): `banco_asrn.py` with the `a100` profile, plus `banco_gpu.py` | The A100 benchmark of ASRN 1.0: the run of 17 October 2026 with PyTorch 2.14.1 |
| [`banco_cpu/`](banco_cpu/RESUMEN.md) | `python benchmarks/banco_asrn.py ...` | The CPU benchmark: exact cost, latency, generation (O(t) writer), small-scale copy |
| [`citas/`](citas/RESUMEN.md) | `python benchmarks/experimento_citas.py` | The bounded-quotation experiment of ASRN 1.0 (CPU) |
| [`citas_a100/`](citas_a100/RESUMEN.md) | the quotation section of `asrn_colab.ipynb` (`experimento_citas.py` on the GPU), in the same run as `banco_a100/` | The same experiment on the A100: a GPU replication of `citas/` |
| [`escritura_cpu/`](escritura_cpu/RESUMEN.md) | `python benchmarks/medir_escritura.py` | Time per new token when writing, against the context, for the O(1) writer of 1.0 and the previous O(t) one (CPU) |

## `banco_a100/`

One NVIDIA A100-SXM4-80GB in Google Colab (PyTorch 2.14.1+cu130, CUDA 13.0, Python 3.13.15), against Transformer++,
Mamba-2 (`mamba_ssm` 2.3.2.post1 with PyTorch's conv1d, without `causal-conv1d`, uncompiled) and RWKV-7
(`flash-linear-attention` 0.5.2): the test suites and the GPU checks, GPU stability, one layer alone, a ~50 M model,
generation, copy, MQAR, WikiText-103 and the ablation. The ~50 M training table adds two compiled rows (ASRN with
`compilar_partes`, Transformer++ with `torch.compile`); every other section ran uncompiled.

| File | Contents |
|---|---|
| `resultados.json` | Every number: each training run and seed (with its learning curve), each timing row, the test and GPU-check results, the environment, and the quotation section (`citas`, the same as `citas_a100/resultados.json`) |
| `RESUMEN.md` | The tables, generated from `resultados.json` by the notebook's last cell |
| `*.png` | The charts: `capa_tiempo`, `capa_memoria` (one layer), `modelo_velocidad` (~50 M model), `escritura` (generation), `calidad_copia`, `calidad_mqar`, `calidad_wikitext`, `ablacion_copia`, `ablacion_mqar` |

**One execution.** Every row, RWKV-7 included, comes from a single execution started on **17 October 2026** at
02:12 UTC from a ZIP of the repository (code fingerprint `5164e1f32513`, see *Provenance* below). In the JSON,
`entorno` is that environment (with the versions of the rivals and `compila`), `estado` gives each section's status
and minutes (284 in total), `pruebas` the results of the suites of that time: `tests/test_nucleo.py` (55 of 55
passed), `tests/test_estable.py` (18 of 18) and `tests/verificar_gpu.py` (7 of 7, `torch.compile` included),
`prueba_originales` the check that each original rival trains and generates, and each row's `implementacion` says which
implementation of the model was used. The quotation section of the same run is in `citas_a100/`. The previous A100
runs, formerly in this folder, are in
[`history/results/banco_a100_pt211/`](../history/results/banco_a100_pt211/RESUMEN.md) (see *Earlier results*).

**Provenance.** `banco_a100/` and `citas_a100/` come from code fingerprint `5164e1f32513`: the first 12 hex digits of
the sha256 of `asrn/*.py`, the root `*.py` and `benchmarks/*.py` concatenated in sorted order (the notebook prints it,
and `entorno.codigo` records it). The 1.0.0 code differs from that code only in the O(1) writer (`asrn/nucleo.py`,
`asrn/estable.py`), the generation benchmark passing new tokens from the CPU (`benchmarks/banco_gpu.py`), docstrings,
and the new `benchmarks/medir_escritura.py`; the forward pass and gradients are bit-identical (verified). Every A100
number except the generation timing therefore applies unchanged to 1.0.0; the generation timing (`escritura_gpu`) was
measured with the previous O(t) writer, and the O(1) writer has not been measured on the A100 yet. The published
repository does not contain the development commits, so the code is identified by this fingerprint, not by a commit.

## `banco_cpu/`

A 4-thread CPU without GPU, PyTorch 2.14.1 (the `entorno` field); `costo_kernel` ran the Triton kernel in Triton's
interpreter.

| File | Contents |
|---|---|
| `resultados.json` | One key per section: `costo` (exact FLOPs and bytes saved for the backward pass, per layer), `costo_kernel` (saved bytes with the Triton kernel), `latencia`, `escritura` (time per generated token, measured with the O(t) writer in an earlier session; the O(1) writer is in `escritura_cpu/`), `calidad` (each run with its curve and evaluation) |
| `RESUMEN.md` | The tables, from `python benchmarks/banco_asrn.py resumir` |

The quality section covers only the copy task: as its `nota` field says, the run was stopped midway through it (RWKV-7
with one seed and one learning rate); MQAR, text and the ablation were left to the A100 notebook.

## `citas/`

The bounded-quotation experiment with the ASRN 1.0 stable layer (full layer, `sin_hawkes_racha`,
`puerta_sin_senales`, `sin_puerta`, `sin_reloj`; 3 seeds, 6,000 steps, CPU). What it measures and how to run it:
[`benchmarks/README.md`](../benchmarks/README.md#the-bounded-quotation-experiment-experimento_citaspy). Its GPU
replication is in `citas_a100/`.

| File | Produced by | Contents |
|---|---|---|
| `semilla_<s>.json` | `experimento_citas.py --semillas <s>` | One seed: `receta`, `entorno` (date, PyTorch, threads, commit), `texto_real` (quoted fraction and streak-length counts of the real continuation) and, per model, the validation loss, training seconds, `respuesta_racha` (the gate's learned response to the streak, per ASRN layer) and `escritura` (quoted fraction and streak-length counts for each sampling mode). Saved after each model, so it can be partial while a seed runs |
| `semilla_<s>.log` | the console output of that seed's process | Progress lines and warnings |
| `resultados.json` | `experimento_citas.py --resumir` | The seeds joined: validation loss and gate response per seed, and per sampling mode the quoted percentage, number, mean and maximum of streaks, maximum per seed, and the share of streaks longer than 9 |
| `RESUMEN.md` | `experimento_citas.py --resumir` | The same as tables |

`resultados.json` and `RESUMEN.md` appear only once the three seeds have finished and `--resumir` has been run. The
`commit` field in `entorno` is kept as recorded: it names a development commit, which the published repository does not
contain.

## `citas_a100/`

The same experiment (same 5 variants, 3 seeds, 6,000 steps) on the A100, run by the notebook's quotation section on
17 October 2026 (36.9 min), in the same Colab run as `banco_a100/`. The files are those of `citas/` without the logs:
`semilla_<s>.json`, `resultados.json` and `RESUMEN.md`. In `entorno`, `dispositivo` is `cuda` and `gpu` names the A100;
`commit` is empty because the notebook ran from a ZIP (the code fingerprint is in `banco_a100/resultados.json`).

## `escritura_cpu/`

Time per new token when writing, against how much text there already is, for the O(1) writer of ASRN 1.0 and the
previous O(t) one (`_Escritor(modelo, o1=False)`), which give the same outputs (`test_escritura_o1_igual_que_o_t`).
Produced by `python benchmarks/medir_escritura.py` on 23 October 2026: CPU, 4 threads, PyTorch 2.14.1; an ASRN model
of 700,484 parameters (V = 256, d = 128, 4 local blocks, 2 ASRN layers); batch 1, no sampling; at each context the
median of 9 batches of 32 new tokens (contexts of 512, 2,048, 8,192, 32,768 and 131,072 tokens), the two writers
interleaved, after warming up both and after one unmeasured prompt of each length.

| File | Contents |
|---|---|
| `resultados.json` | `entorno`, `modelo` (the sizes above), `nuevos` (tokens per batch), `repeticiones` (batches), `filas` (per writer and context: `prefijo_s`, seconds to read the prompt; `ms_por_pieza`, the median ms per new token; `tandas_ms`, each batch) and `pendiente_ns_por_pieza_de_contexto` (least-squares slope of ms per token against the context, with its 95% interval `ic95`) |
| `RESUMEN.md` | The same as a table, written by the script |

The O(1) writer takes 12.45 ms per token at 512 tokens of context and 9.49 at 131,072, with a slope of −19.21 ns per
context token (95% CI −41.54 to 3.12): indistinguishable from zero. The O(t) writer grows from 15.80 to 91.67 ms, a
slope of 583.31 ns (537.13 to 629.49). In the prompt column the O(1) writer is slower (29.04 against 20.44 s at
131,072 tokens) by more than its bookkeeping (filling the rings and the Hawkes state, profiled at about 0.3 s at 32,768
tokens and 1.3 s at 131,072), because the prompt time varies run to run on this shared 4-core machine: the same prompt
pass took 14.8–23.5 s at 131,072 tokens in four repetitions. With `--dispositivo cuda` the script writes the same
measurement for a GPU to `escritura_gpu/`, which does not exist yet.

## Earlier results

The previous A100 runs of the 1.0 benchmark (9 and 11 October 2026, notebook `banco_a100.ipynb`, PyTorch 2.11.0,
nothing compiled), formerly `banco_a100/`, are in
[`history/results/banco_a100_pt211/`](../history/results/banco_a100_pt211/RESUMEN.md) and are cited as a replication.
The 11 October execution (fingerprint `176cfe250773`) added RWKV-7 and repeated the stability probes with TF32
disabled; every other row comes from 9 October. Its `entornos_anteriores` lists the earlier sessions, and `pruebas` uses
the test names of the time (`pruebas/test_asrn_v89.py`, `pruebas/test_asrn_estable.py`).

The five v89 development runs on the GPU (against reimplementations of Mamba-2 and RWKV-7), the local-block variants
and the compiled speed are in [`history/results/`](../history/results/README.md), with their numbers in the JSON files
kept there. They used an earlier protocol and are kept as development history, because the documents cite them as
evidence.

## Spanish keys

The scripts write their JSON keys and `RESUMEN.md` tables in Spanish (*pieza* = token). The keys that appear in these
files:

| Keys | Meaning |
|---|---|
| `entorno`, `entornos`, `fecha`, `hilos`, `dispositivo`, `commit` | environment (of each session): date, CPU threads, device, development commit |
| `codigo` | in `entorno`: where the code came from and its fingerprint; in `pruebas`: each suite's exit code |
| `estado`, `ok`, `minutos` | each section's status: finished without error, and its minutes |
| `pruebas`, `resumen`, `fallas`, `prueba`, `resultado`, `detalle` | test results: summary line and failed tests; for the GPU checks, each check, its result and detail |
| `prueba_originales`, `originales`, `implementacion` | the check that each original rival trains and generates; the implementation each row used |
| `estabilidad_gpu`, `reloj_cumsum`, `parado`, `reloj_corrido`, `fuga_gamma_1e9` | GPU stability probes: the 32-bit clock (`parado` = share of steps where it stopped), the shifted clock, the causal leak |
| `capa_gpu`, `modelo_gpu`, `escritura_gpu` | A100 sections: one layer, the ~50 M model, generation |
| `costo`, `costo_kernel`, `latencia` | CPU sections: exact cost, saved bytes with the Triton kernel, one layer's latency |
| `filas`, `modelo`, `largo`, `contexto`, `lote`, `parametros` | table rows: model, sequence length, context length (generation), batch, parameters |
| `segundos`, `memoria_gb`, `piezas_por_s`, `segundos_paso`, `primer_paso_s` | time in seconds, peak memory in GB, tokens per second, seconds per step, seconds of the first step |
| `flops_por_pieza`, `bytes_guardados_por_pieza`, `exponente` | FLOPs and bytes saved for the backward pass per token; exponent of time against length (1 = linear, 2 = quadratic) |
| `escritura`, `nuevos`, `n_nuevos`, `prefijo_s`, `ms_por_pieza`, `segundos_por_pieza`, `tandas_ms` | writing (generation): new tokens, seconds to read the prompt, time per new token, each batch's ms per token; in `escritura_cpu/` `escritura` names the writer, in the quotation files it holds each sampling mode |
| `repeticiones`, `pendiente`, `ic95` | number of batches; slope and its 95% interval |
| `calidad`, `ablacion`, `tareas`, `tarea`, `corridas` | quality and ablation sections: the tasks and every training run |
| `semilla`, `semillas`, `pasos`, `lr`, `mejor_lr` | seed(s), training steps, learning rate, the best learning rate per task and model |
| `curva`, `evaluacion`, `evaluar`, `exactitud`, `perplejidad` | learning curve (step, training loss), evaluation at each length, the lengths evaluated, accuracy, perplexity |
| `citas`, `receta`, `modelos`, `variante`, `segundos_entrenando` | quotation experiment: recipe, models, variant, training seconds |
| `perdida_validacion`, `respuesta_racha` | validation loss; the gate's learned response to the streak |
| `texto_real`, `fraccion`, `rachas` | the real continuation; quoted fraction; streaks (counts by length per seed, their number in `resultados.json`) |
| `citado_pct`, `media`, `max`, `max_por_semilla`, `mayores_de_9_pct` | quoted %, mean and maximum streak, maximum per seed, % of streaks longer than 9 |
| `nota` | a note left by the script |

In the `RESUMEN.md` tables: *largo* = length, *contexto* = context, *semilla* = seed, *exactitud* = accuracy,
*perplejidad* = perplexity, *s/corrida* = seconds per run, *piezas/s* = tokens per second, *ida y vuelta* = forward
and backward, *prefijo* = prompt, *Perdida de validacion* = validation loss, *Citado* = quoted, *Rachas* = streaks,
*Media* = mean, *reloj parado* = stopped clock.

## Español

Los resultados medidos de ASRN 1.0, tal como salieron de los programas (incluido lo que no favorece a ASRN). Cada
carpeta tiene un `RESUMEN.md` con las tablas: los números se leen ahí o en el JSON, no en este índice. El análisis, con
las salvedades de cada cifra, está en el [informe técnico](../docs/TECHNICAL_REPORT_es.md).

| Carpeta | Lo produjo | Qué hay |
|---|---|---|
| [`banco_a100/`](banco_a100/RESUMEN.md) | [`benchmarks/asrn_colab.ipynb`](../benchmarks/README.md): `banco_asrn.py` con el perfil `a100`, más `banco_gpu.py` | El banco de ASRN 1.0 en A100: la corrida del 17 de octubre de 2026 con PyTorch 2.14.1 |
| [`banco_cpu/`](banco_cpu/RESUMEN.md) | `python benchmarks/banco_asrn.py ...` | El banco en CPU: costo exacto, latencia, escritura (O(t)), copia a escala chica |
| [`citas/`](citas/RESUMEN.md) | `python benchmarks/experimento_citas.py` | El experimento de citas acotadas de ASRN 1.0 (CPU) |
| [`citas_a100/`](citas_a100/RESUMEN.md) | la sección de citas de `asrn_colab.ipynb` (`experimento_citas.py` en la GPU), en la misma corrida que `banco_a100/` | El mismo experimento en la A100: una réplica en GPU de `citas/` |
| [`escritura_cpu/`](escritura_cpu/RESUMEN.md) | `python benchmarks/medir_escritura.py` | Tiempo por pieza nueva al escribir, según el contexto, con la escritura O(1) de 1.0 y la O(t) de antes (CPU) |

### `banco_a100/`

Una NVIDIA A100-SXM4-80GB en Google Colab (PyTorch 2.14.1+cu130, CUDA 13.0, Python 3.13.15), contra Transformer++,
Mamba-2 (`mamba_ssm` 2.3.2.post1 con el conv1d de PyTorch, sin `causal-conv1d` ni compilar) y RWKV-7
(`flash-linear-attention` 0.5.2): las baterías de pruebas y las comprobaciones en GPU, estabilidad en la GPU, una capa
sola, un modelo de ~50 M, escritura, copia, MQAR, WikiText-103 y la ablación. La tabla de entrenamiento del modelo de
~50 M suma dos filas compiladas (ASRN con `compilar_partes`, Transformer++ con `torch.compile`); el resto de las
secciones corrió sin `torch.compile`.

| Archivo | Qué hay |
|---|---|
| `resultados.json` | Todos los números: cada entrenamiento y semilla (con su curva), cada fila de tiempos, el resultado de las pruebas y de las comprobaciones en GPU, el entorno y la sección de citas (`citas`, lo mismo que `citas_a100/resultados.json`) |
| `RESUMEN.md` | Las tablas, generadas desde `resultados.json` por la última celda del cuaderno |
| `*.png` | Las gráficas: `capa_tiempo`, `capa_memoria` (una capa), `modelo_velocidad` (modelo de ~50 M), `escritura`, `calidad_copia`, `calidad_mqar`, `calidad_wikitext`, `ablacion_copia`, `ablacion_mqar` |

**Una sola ejecución.** Todas las filas, RWKV-7 incluido, salen de una sola ejecución que empezó el **17 de octubre de
2026** a las 02:12 UTC desde un ZIP del repositorio (huella del código `5164e1f32513`, ver *Procedencia* abajo). En el
JSON, `entorno` es ese entorno (con las versiones de los rivales y `compila`), `estado` da el estado y los minutos de
cada sección (284 en total), `pruebas` el resultado de las baterías de entonces: `tests/test_nucleo.py` (pasaron 55 de
55), `tests/test_estable.py` (18 de 18) y `tests/verificar_gpu.py` (7 de 7, `torch.compile` incluido),
`prueba_originales` la comprobación de que cada rival original entrena y escribe, y el campo `implementacion` de cada
fila dice qué implementación del modelo se usó. La sección de citas de la misma corrida está en `citas_a100/`. Las
corridas anteriores en A100, que antes estaban en esta carpeta, están en
[`history/results/banco_a100_pt211/`](../history/results/banco_a100_pt211/RESUMEN.md) (ver *Resultados anteriores*).

**Procedencia.** `banco_a100/` y `citas_a100/` salen del código de huella `5164e1f32513`: los primeros 12 dígitos
hexadecimales del sha256 de `asrn/*.py`, los `*.py` de la raíz y `benchmarks/*.py` concatenados en orden (el cuaderno
la imprime y `entorno.codigo` la guarda). El código de 1.0.0 difiere de ese solo en la escritura O(1) (`asrn/nucleo.py`,
`asrn/estable.py`), en que el banco de escritura pasa las piezas nuevas desde la CPU (`benchmarks/banco_gpu.py`), en
docstrings y en el nuevo `benchmarks/medir_escritura.py`; la ida y los gradientes son bit a bit iguales (comprobado).
Por eso cada número de la A100, salvo el tiempo de escritura, vale sin cambios para 1.0.0; el tiempo de escritura
(`escritura_gpu`) se midió con la escritura O(t) anterior, y la escritura O(1) todavía no se midió en la A100. El
repositorio publicado no contiene los commits del desarrollo, así que el código se identifica por esta huella y no por
un commit.

### `banco_cpu/`

Una CPU de 4 hilos sin GPU, PyTorch 2.14.1 (el campo `entorno`); `costo_kernel` corrió el kernel Triton en el
intérprete de Triton.

| Archivo | Qué hay |
|---|---|
| `resultados.json` | Una clave por sección: `costo` (FLOPs exactos y bytes guardados para la vuelta, por capa), `costo_kernel` (bytes guardados con el kernel Triton), `latencia`, `escritura` (tiempo por pieza escrita, medido con la escritura O(t) en una sesión anterior; la escritura O(1) está en `escritura_cpu/`), `calidad` (cada corrida con su curva y su evaluación) |
| `RESUMEN.md` | Las tablas, de `python benchmarks/banco_asrn.py resumir` |

La sección de calidad cubre solo la copia: como dice su campo `nota`, la corrida se detuvo a mitad de esa tarea
(RWKV-7 con una semilla y una tasa); MQAR, texto y la ablación quedaron para el cuaderno de la A100.

### `citas/`

El experimento de citas acotadas con la capa estable de ASRN 1.0 (capa completa, `sin_hawkes_racha`,
`puerta_sin_senales`, `sin_puerta`, `sin_reloj`; 3 semillas, 6 000 pasos, CPU). Qué mide y cómo se corre:
[`benchmarks/README.md`](../benchmarks/README.md#el-experimento-de-citas-acotadas-experimento_citaspy). Su réplica en
GPU está en `citas_a100/`.

| Archivo | Lo produce | Qué hay |
|---|---|---|
| `semilla_<s>.json` | `experimento_citas.py --semillas <s>` | Una semilla: `receta`, `entorno` (fecha, PyTorch, hilos, commit), `texto_real` (fracción citada y conteo de largos de racha de la continuación real) y, por modelo, la pérdida de validación, los segundos de entrenamiento, `respuesta_racha` (la respuesta aprendida de la puerta a la racha, por capa ASRN) y `escritura` (fracción citada y conteo de largos de racha en cada modo de escritura). Se guarda después de cada modelo, así que puede estar a medias mientras corre una semilla |
| `semilla_<s>.log` | la salida en consola del proceso de esa semilla | Líneas de avance y avisos |
| `resultados.json` | `experimento_citas.py --resumir` | Las semillas juntas: pérdida de validación y respuesta de la puerta por semilla, y por modo de escritura el porcentaje citado, número, media y máximo de las rachas, el máximo por semilla y la fracción de rachas de más de 9 |
| `RESUMEN.md` | `experimento_citas.py --resumir` | Lo mismo en tablas |

`resultados.json` y `RESUMEN.md` aparecen solo cuando terminan las tres semillas y se corre `--resumir`. El campo
`commit` de `entorno` queda como se registró: nombra un commit del desarrollo, que el repositorio publicado no contiene.

### `citas_a100/`

El mismo experimento (las mismas 5 variantes, 3 semillas, 6 000 pasos) en la A100, corrido por la sección de citas del
cuaderno el 17 de octubre de 2026 (36.9 min), en la misma corrida de Colab que `banco_a100/`. Los archivos son los de
`citas/` sin los registros de consola: `semilla_<s>.json`, `resultados.json` y `RESUMEN.md`. En `entorno`,
`dispositivo` es `cuda` y `gpu` nombra la A100; `commit` está vacío porque el cuaderno corrió desde un ZIP (la huella
del código está en `banco_a100/resultados.json`).

### `escritura_cpu/`

Tiempo por pieza nueva al escribir, según cuánto texto ya hay, con la escritura O(1) de ASRN 1.0 y la O(t) de antes
(`_Escritor(modelo, o1=False)`), que dan las mismas salidas (`test_escritura_o1_igual_que_o_t`). Lo produjo
`python benchmarks/medir_escritura.py` el 23 de octubre de 2026: CPU, 4 hilos, PyTorch 2.14.1; un modelo ASRN de
700 484 parámetros (V = 256, d = 128, 4 bloques locales, 2 capas ASRN); lote 1, sin muestrear; en cada contexto, la
mediana de 9 tandas de 32 piezas nuevas (contextos de 512, 2 048, 8 192, 32 768 y 131 072 piezas), las dos escrituras
alternadas, después de calentar las dos y de leer sin medir un inicio de cada largo.

| Archivo | Qué hay |
|---|---|
| `resultados.json` | `entorno`, `modelo` (los tamaños de arriba), `nuevos` (piezas por tanda), `repeticiones` (tandas), `filas` (por escritura y contexto: `prefijo_s`, los segundos de leer el inicio; `ms_por_pieza`, la mediana de ms por pieza nueva; `tandas_ms`, cada tanda) y `pendiente_ns_por_pieza_de_contexto` (pendiente por mínimos cuadrados de los ms por pieza según el contexto, con su intervalo de 95% `ic95`) |
| `RESUMEN.md` | Lo mismo en una tabla, escrita por el programa |

La escritura O(1) tarda 12.45 ms por pieza con 512 piezas de contexto y 9.49 con 131 072, con una pendiente de
−19.21 ns por pieza de contexto (IC 95% −41.54 a 3.12): no se distingue de cero. La O(t) crece de 15.80 a 91.67 ms,
una pendiente de 583.31 ns (537.13 a 629.49). En la columna del inicio la escritura O(1) es más lenta (29.04 contra
20.44 s con 131 072 piezas) por más que sus cuentas propias (llenar los anillos y el estado de Hawkes, que perfiladas
dan unos 0.3 s con 32 768 piezas y 1.3 s con 131 072), porque el tiempo de leer el inicio varía de una corrida a otra
en esta máquina compartida de 4 núcleos: la misma lectura del inicio tardó 14.8–23.5 s con 131 072 piezas en cuatro
repeticiones. Con `--dispositivo cuda` el programa escribe la misma medición en una GPU en `escritura_gpu/`, que todavía
no existe.

### Resultados anteriores

Las corridas anteriores del banco de 1.0 en A100 (9 y 11 de octubre de 2026, cuaderno `banco_a100.ipynb`, PyTorch
2.11.0, nada compilado), antes en `banco_a100/`, están en
[`history/results/banco_a100_pt211/`](../history/results/banco_a100_pt211/RESUMEN.md) y se citan como réplica. La
ejecución del 11 de octubre (huella `176cfe250773`) agregó RWKV-7 y repitió las sondas de estabilidad con TF32
apagado; las demás filas son del 9 de octubre. Su `entornos_anteriores` lista las sesiones anteriores, y `pruebas` usa
los nombres de pruebas de entonces (`pruebas/test_asrn_v89.py`, `pruebas/test_asrn_estable.py`).

Las cinco corridas de desarrollo de v89 en GPU (contra reimplementaciones de Mamba-2 y RWKV-7), las variantes del
bloque local y la velocidad compilada están en [`history/results/`](../history/results/README.md), con sus números en
los JSON que se guardan ahí. Usaron un protocolo anterior y se conservan como historia del desarrollo, porque los
documentos las citan como evidencia.

### Claves en español

Los programas escriben las claves de sus JSON y las tablas de `RESUMEN.md` en español (*pieza* = token). Las claves que
aparecen en estos archivos:

| Claves | Qué son |
|---|---|
| `entorno`, `entornos`, `fecha`, `hilos`, `dispositivo`, `commit` | el entorno (de cada sesión): fecha, hilos de CPU, dispositivo, commit del desarrollo |
| `codigo` | en `entorno`: de dónde salió el código y su huella; en `pruebas`: el código de salida de cada batería |
| `estado`, `ok`, `minutos` | el estado de cada sección: si terminó sin error, y sus minutos |
| `pruebas`, `resumen`, `fallas`, `prueba`, `resultado`, `detalle` | el resultado de las pruebas: la línea de resumen y las que fallaron; en las comprobaciones en GPU, cada comprobación, su resultado y su detalle |
| `prueba_originales`, `originales`, `implementacion` | la comprobación de que cada rival original entrena y escribe; la implementación que usó cada fila |
| `estabilidad_gpu`, `reloj_cumsum`, `parado`, `reloj_corrido`, `fuga_gamma_1e9` | las sondas de estabilidad en GPU: el reloj de 32 bits (`parado` = fracción de pasos en que se detuvo), el reloj corrido, la fuga causal |
| `capa_gpu`, `modelo_gpu`, `escritura_gpu` | secciones de la A100: una capa, el modelo de ~50 M, la escritura |
| `costo`, `costo_kernel`, `latencia` | secciones de la CPU: costo exacto, bytes guardados con el kernel Triton, latencia de una capa |
| `filas`, `modelo`, `largo`, `contexto`, `lote`, `parametros` | las filas de las tablas: modelo, largo del texto, contexto (al escribir), lote, parámetros |
| `segundos`, `memoria_gb`, `piezas_por_s`, `segundos_paso`, `primer_paso_s` | tiempo en segundos, memoria pico en GB, piezas por segundo, segundos por paso, segundos del primer paso |
| `flops_por_pieza`, `bytes_guardados_por_pieza`, `exponente` | FLOPs y bytes guardados para la vuelta por pieza; exponente del tiempo según el largo (1 = lineal, 2 = cuadrático) |
| `escritura`, `nuevos`, `n_nuevos`, `prefijo_s`, `ms_por_pieza`, `segundos_por_pieza`, `tandas_ms` | la escritura: piezas nuevas, segundos de leer el inicio, tiempo por pieza nueva, ms por pieza de cada tanda; en `escritura_cpu/` `escritura` nombra la escritura usada, en los archivos de citas guarda cada modo de muestreo |
| `repeticiones`, `pendiente`, `ic95` | número de tandas; pendiente y su intervalo de 95% |
| `calidad`, `ablacion`, `tareas`, `tarea`, `corridas` | las secciones de calidad y de ablación: las tareas y cada entrenamiento |
| `semilla`, `semillas`, `pasos`, `lr`, `mejor_lr` | semilla(s), pasos de entrenamiento, tasa de aprendizaje, la mejor tasa por tarea y modelo |
| `curva`, `evaluacion`, `evaluar`, `exactitud`, `perplejidad` | curva de aprendizaje (paso, pérdida de entrenamiento), evaluación en cada largo, los largos evaluados, exactitud, perplejidad |
| `citas`, `receta`, `modelos`, `variante`, `segundos_entrenando` | el experimento de citas: receta, modelos, variante, segundos de entrenamiento |
| `perdida_validacion`, `respuesta_racha` | pérdida de validación; la respuesta aprendida de la puerta a la racha |
| `texto_real`, `fraccion`, `rachas` | la continuación real; fracción citada; rachas (conteo por largo en cada semilla, su número en `resultados.json`) |
| `citado_pct`, `media`, `max`, `max_por_semilla`, `mayores_de_9_pct` | % citado, racha media y máxima, máxima por semilla, % de rachas de más de 9 |
| `nota` | una nota que dejó el programa |

En las tablas de `RESUMEN.md`: *largo* = largo del texto, *contexto* = piezas ya escritas, *s/corrida* = segundos por
corrida, *piezas/s* = piezas por segundo, *ida y vuelta* = ida (forward) y vuelta (backward), *prefijo* = el inicio,
*Rachas > 9* = rachas de más de 9 piezas, *reloj parado* = el reloj que se detiene.
