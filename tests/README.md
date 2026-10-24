# Tests

The test suites of ASRN 1.0. They are plain Python scripts (no pytest needed): each one runs every function whose name
starts with `test_`, prints `ok`, `salta` (skipped) or `FALLA` (failed) per test and a final count, and exits with
code 1 if any test failed.

| File | What it covers |
|---|---|
| `test_nucleo.py` | **57 tests of the core** (`asrn/nucleo.py`, the v89 layer): the layer against its definitions (Hawkes intensity, streak, scan, the quotation options of generation, the faster paths that move fewer bytes), the Triton kernels against their PyTorch paths, the model, configuration and sizes, building on the `meta` device, level-wise gradient checkpointing, mixed precision, FSDP with 2 CPU processes, causality, edge cases, clear error messages when saving, resuming and on the command line, step-by-step generation equal to the full sequence, the O(1) writer equal to the O(t) writer and to the whole text, the rings and n-gram dictionaries of the O(1) writer against their definitions, O(N) saved memory, and training that learns to copy and resumes. It also checks the linear-cost rivals in `benchmarks/rivales_lineales.py` |
| `test_estable.py` | **19 tests of the stable layer** (`asrn/estable.py`) and of the package: without protections it equals v89 bit for bit, with group A it equals v89 up to rounding; step-by-step writing, O(1) and O(t), equal to the whole text; each protection fixes its failure (causal leak, shifted clock, stopped clock, huge β, 16-bit clock projection); the Triton kernel in `ESTABLE` mode against PyTorch; dynamic gradient clipping; the package API (`import asrn`), the `python -m asrn` command line and the English aliases |
| `verificar_gpu.py` | **7 GPU checks** that the CPU cannot do: bf16 training of a real size, checkpoint memory, FSDP with NCCL in one process, `torch.compile`, GPU = CPU, causality on the GPU, step-by-step generation = full sequence. Prints PASA / SALTADA / FALLA |

## How to run

From the repository root, with the package requirements installed (`pip install .` or
`pip install -r requirements.txt`):

```bash
python tests/test_nucleo.py              # all 57 core tests
python tests/test_nucleo.py fsdp copia   # only the tests whose name contains "fsdp" or "copia"
python tests/test_estable.py             # all 19 stable-layer tests (same name filter)
python tests/verificar_gpu.py [--rapido] [--json resultados.json]   # on a GPU (e.g. Colab)
```

Without a GPU, `verificar_gpu.py` runs on the CPU with small sizes, only to check the script itself. The Colab notebook
`benchmarks/asrn_colab.ipynb` runs these suites and the GPU checks on an A100 with PyTorch 2.14 (where the two
`torch.compile` tests of the core also run).

**Continuous integration.** [`.github/workflows/tests.yml`](../.github/workflows/tests.yml) runs both suites on CPU
with Python 3.11 and 3.13 on every push to `main` and every pull request: it installs the CPU build of PyTorch and the
package, then runs `tests/test_estable.py` and `tests/test_nucleo.py`. Without Triton the kernel-versus-PyTorch tests
are skipped and every other test runs the PyTorch path; the GPU checks run only in the notebook.

**Runtime on CPU:** the core suite takes a few minutes, the stable suite under a minute (more on a busy machine).

**Notes**

- On CPU the Triton kernels run in Triton's interpreter (`TRITON_INTERPRET=1`, set by the tests themselves), which
  checks the arithmetic, not the speed. Without Triton installed, the kernel tests are reported as skipped.
- On CPU with Python 3.13.16, PyTorch 2.14.1 and Triton 3.8.0 both suites pass in full: 57 / 57 and 19 / 19. The Colab
  A100 run (PyTorch 2.14.1, 17 October 2026) passed the suites of that time in full, 55 / 55 and 18 / 18, and
  `verificar_gpu.py` 7 / 7, `torch.compile` included (field `pruebas` of
  [`results/banco_a100/resultados.json`](../results/banco_a100/resultados.json)). With PyTorch 2.11.0 (Triton 3.6.0,
  NumPy 2.3.5) the two `torch.compile` tests of the core are skipped (the compiler fails with this model before 2.14):
  55 pass and 2 are skipped, and the stable suite passes 19 / 19 (23 October 2026). Tested environments: Python 3.13
  with PyTorch 2.11.0 and 2.14.1 (CPU, A100); CI on Python 3.11 and 3.13 (CPU).
- With Triton 3.6 (the version paired with PyTorch 2.11), the interpreter needs **NumPy < 2.4**: with NumPy 2.4 or
  newer the kernel-versus-PyTorch tests fail inside Triton (`TypeError: only 0-dimensional arrays can be converted to
  Python scalars`). Triton 3.8 (PyTorch 2.14) and GPU runs are not affected.

## Español

Las pruebas de ASRN 1.0. Son programas de Python normales (no hace falta pytest): cada uno corre todas las funciones
cuyo nombre empieza con `test_`, imprime `ok`, `salta` o `FALLA` por prueba y la cuenta final, y termina con código 1 si
alguna falló.

| Archivo | Qué cubre |
|---|---|
| `test_nucleo.py` | **57 pruebas del núcleo** (`asrn/nucleo.py`, la capa de v89): la capa contra sus definiciones (intensidad de Hawkes, racha, escaneo, las opciones de cita al escribir, las versiones que mueven menos bytes), los kernels Triton contra su camino de PyTorch, el modelo, la configuración y las tallas, construir en el dispositivo `meta`, el checkpoint por niveles, la precisión mixta, FSDP con 2 procesos en CPU, la causalidad, los casos límite, los mensajes de error claros al guardar, reanudar y en la línea de comandos, escribir paso a paso igual que el texto entero, la escritura O(1) igual que la O(t) y que el texto entero, los anillos y los diccionarios de grupos de la escritura O(1) contra sus definiciones, la memoria guardada O(N), y entrenar que aprende a copiar y reanuda. También prueba los rivales de costo lineal de `benchmarks/rivales_lineales.py` |
| `test_estable.py` | **19 pruebas de la capa estable** (`asrn/estable.py`) y del paquete: sin protecciones es v89 bit a bit, con el grupo A es v89 salvo el redondeo; escribir paso a paso, O(1) y O(t), igual que el texto entero; cada protección arregla su falla (fuga causal, reloj corrido, reloj detenido, β enorme, proyección del reloj en 16 bits); el kernel Triton en modo `ESTABLE` contra PyTorch; el recorte dinámico del gradiente; la API del paquete (`import asrn`), la línea de comandos `python -m asrn` y los alias en inglés |
| `verificar_gpu.py` | **7 comprobaciones en GPU** que la CPU no puede hacer: entrenar en bf16 una talla real, la memoria del checkpoint, FSDP con NCCL en un proceso, `torch.compile`, GPU = CPU, causalidad en la GPU, escribir paso a paso = texto entero. Imprime PASA / SALTADA / FALLA |

### Cómo correrlas

Desde la raíz del repositorio, con lo que pide el paquete instalado (`pip install .` o
`pip install -r requirements.txt`):

```bash
python tests/test_nucleo.py              # las 57 pruebas del núcleo
python tests/test_nucleo.py fsdp copia   # solo las que tienen "fsdp" o "copia" en el nombre
python tests/test_estable.py             # las 19 pruebas de la capa estable (mismo filtro por nombre)
python tests/verificar_gpu.py [--rapido] [--json resultados.json]   # en una GPU (por ejemplo, Colab)
```

Sin GPU, `verificar_gpu.py` corre en la CPU con tamaños chicos, solo para revisar el programa. El cuaderno de Colab
`benchmarks/asrn_colab.ipynb` corre estas pruebas y las comprobaciones en GPU en una A100 con PyTorch 2.14 (donde
también corren las dos pruebas de `torch.compile` del núcleo).

**Integración continua.** [`.github/workflows/tests.yml`](../.github/workflows/tests.yml) corre las dos baterías en CPU
con Python 3.11 y 3.13 en cada push a `main` y en cada pull request: instala PyTorch para CPU y el paquete, y después
corre `tests/test_estable.py` y `tests/test_nucleo.py`. Sin Triton se saltan las pruebas de los kernels contra PyTorch
y todas las demás usan el camino de PyTorch; las comprobaciones en GPU solo corren en el cuaderno.

**Tiempo en CPU:** el núcleo tarda unos minutos y la capa estable menos de un minuto (más en una máquina ocupada).

**Notas**

- En CPU los kernels Triton corren en el intérprete de Triton (`TRITON_INTERPRET=1`, que ponen las mismas pruebas): se
  comprueban las cuentas, no la velocidad. Sin Triton instalado, las pruebas de los kernels se informan como saltadas.
- En CPU con Python 3.13.16, PyTorch 2.14.1 y Triton 3.8.0 las dos baterías pasan completas: 57 / 57 y 19 / 19. La
  corrida en la A100 de Colab (PyTorch 2.14.1, 17 de octubre de 2026) pasó completas las baterías de entonces, 55 / 55 y
  18 / 18, y `verificar_gpu.py` 7 / 7, `torch.compile` incluido (campo `pruebas` de
  [`results/banco_a100/resultados.json`](../results/banco_a100/resultados.json)). Con PyTorch 2.11.0 (Triton 3.6.0,
  NumPy 2.3.5) se saltan las dos pruebas de `torch.compile` del núcleo (el compilador falla con este modelo antes de
  2.14): pasan 55 y se saltan 2, y la capa estable pasa 19 / 19 (23 de octubre de 2026). Entornos probados: Python 3.13
  con PyTorch 2.11.0 y 2.14.1 (CPU, A100); la integración continua, Python 3.11 y 3.13 (CPU).
- Con Triton 3.6 (el de PyTorch 2.11), el intérprete necesita **NumPy < 2.4**: con NumPy 2.4 o más nuevo, las pruebas
  de los kernels contra PyTorch fallan dentro de Triton (`TypeError: only 0-dimensional arrays can be converted to
  Python scalars`). Triton 3.8 (PyTorch 2.14) y la GPU no tienen ese problema.
