# Resultados en GPU (A100, octubre 2026)

Lo que se midió de ASRN en GPUs A100 de Google Colab durante el desarrollo de v89, y la corrida anterior del banco 1.0
(`banco_a100_pt211/`), tal como salió, incluido lo que no favorece a ASRN. El banco de ASRN 1.0 está en
[`results/`](../../results/README.md).

Los números de cada corrida están en su `resultados.json` (y, cuando lo hay, en su `RESUMEN.md`). Las gráficas de
desarrollo y los cuadernos ejecutados no forman parte de la versión publicada: se pueden rehacer desde estos JSON.

| Carpeta | Qué hay |
|---|---|
| `corrida_1/` | Primera corrida completa de `pruebas_gpu_a100.ipynb`, antes de acelerar ASRN: `RESUMEN.md` (las tablas del cuaderno), `resultados.json`, `verificacion_gpu.json` y las gráficas de entonces |
| `corrida_2/` | Con el kernel Triton del escaneo |
| `corrida_3/` | Con los kernels de la atención por casilleros |
| `corrida_4/` | Agrega "ASRN cada 2" (una capa ASRN cada dos bloques) |
| `corrida_5/` | Contra Mamba-2 y RWKV-7 (reimplementados), 3 semillas: la comparación principal del desarrollo |
| `bloques_1/` | Variantes del bloque local (una semilla) |
| `velocidad/` | La velocidad de cada versión de ASRN en una misma sesión, compilado y sin compilar (`../experiments/`) |
| `banco_a100_pt211/` | Las corridas anteriores del banco de ASRN 1.0 (9 y 11 de octubre de 2026, cuaderno `banco_a100.ipynb`, PyTorch 2.11.0, nada compilado), antes `results/banco_a100/`: `RESUMEN.md` y `resultados.json`. Las reemplaza la corrida del 17 de octubre de 2026 con PyTorch 2.14.1 ([`results/banco_a100/`](../../results/banco_a100/RESUMEN.md)); los documentos de 1.0 las citan como réplica |

**Lo principal** (quinta corrida, A100 de 80 GB; "ASRN (cada 2)" es la forma por omisión):

- RWKV-7 predice mejor el texto (perplejidad 47.9 contra 57.0); ASRN queda por delante de Mamba-2 (59.5) y del
  Transformer (64.3).
- Solo ASRN copia desde lejos, también con textos 4 veces más largos que los de entrenamiento (99.3%).
- Con textos largos, ASRN es el más rápido: la capa sola y el modelo entero.

Abajo, cada corrida en orden.

## Primera corrida: antes de acelerar (`corrida_1/`)

Una corrida completa de `pruebas_gpu_a100.ipynb` en Google Colab: NVIDIA A100-SXM4-40GB, PyTorch 2.11.0 (CUDA 13.0),
Python 3.13, cuentas en bf16.

| Archivo | Contenido |
|---|---|
| `RESUMEN.md` | Todas las tablas que generó el cuaderno, sin editar |
| `resultados.json` | Todos los números (cada corrida, cada medición, cada curva de aprendizaje) |
| `verificacion_gpu.json` | Las 7 comprobaciones de `verificar_gpu_v89.py` |
| `*.png` | Las gráficas de esa corrida |

### Qué se comprobó

- **ASRN da lo mismo en GPU que en CPU:** misma salida y mismo gradiente (diferencia máxima 6·10⁻⁷, el redondeo normal
  de 32 bits). Causalidad: el futuro no cambia el pasado. Escribir paso a paso da lo mismo que procesar el texto entero.
- **Entrena en bf16** (un modelo de 31 millones aprende: pérdida de 9.7 a 0.2), el checkpoint por niveles baja la
  memoria de 3 390 MB a 1 756 MB y a 1 147 MB, y FSDP con NCCL da exactamente lo mismo repartido que sin repartir.
- **Lo que falló en esta corrida y ya está arreglado:**
  - `torch.compile` falla con PyTorch 2.11 en la A100 (un error interno del compilador Triton, no de ASRN; con
    PyTorch 2.14 funciona en CPU). Ahora la comprobación se marca *saltada* con PyTorch anterior a 2.14.
  - 2 pruebas automáticas de FSDP fallaron: en una máquina con GPU, FSDP armaba su "malla" (cómo se ordenan los
    procesos) en la GPU aunque la prueba usara comunicación por CPU. Arreglado en `repartir_fsdp`: con gloo la malla
    va en la CPU. Esto no afecta al uso normal con GPU (NCCL), que pasó la comprobación 3.

### Una capa sola: tiempo y memoria según el largo del texto

*Pendiente* = cómo crece el costo: 1 es lineal (el doble de texto, el doble de costo), 2 es cuadrático (el doble de
texto, cuatro veces el costo).

- **Tiempo:** ASRN crece de forma lineal (pendiente 0.88); la atención con FlashAttention, de forma cuadrática (1.88).
  FlashAttention es más rápida hasta 65 mil piezas; **ASRN es más rápida desde 131 mil** (284 ms contra 367 ms) y
  **2.6 veces más rápida con 262 mil** (562 ms contra 1 449 ms).
- **Memoria:** las dos crecen de forma lineal, pero **ASRN usa unas 4 veces más memoria** que FlashAttention a
  cualquier largo (16.8 GB contra 4.1 GB con 262 mil piezas). La atención ingenua (la que arma la tabla de N×N) se
  queda sin memoria desde 16 mil piezas.

### Un modelo entero de ~50 millones de parámetros

- **El Transformer entrena más rápido:** 5.9 veces más piezas por segundo con textos de 1 024 (207 mil contra 35 mil)
  y 2.1 veces con textos de 32 mil (92 mil contra 44 mil). La velocidad de ASRN casi no cambia con el largo; la del
  Transformer baja, pero en este rango sigue arriba.
- **Memoria:** ASRN 28.7 GB, ASRN con checkpoint 17.1 GB, Transformer 22.2 GB.

### Copiar desde lejos (memoria a larga distancia)

Cada modelo (~2.5 millones de parámetros) aprende a repetir un pedazo de texto que vio mucho antes. Se entrena con
textos de 1 024 piezas y se prueba también con textos 4 y 16 veces más largos. Al azar se acertaría 0.2%.

| Modelo | Con 1 024 | Con 4 096 | Con 16 384 |
|---|---|---|---|
| ASRN | 99.9% ± 0.0 | 99.8% ± 0.3 | 83.4% ± 25.9 |
| Transformer | 67.7% ± 43.6 | 0.2% | 0.2% |
| Solo bloques locales (ASRN sin su capa de copia) | 0.2% | 0.2% | 0.2% |

- ASRN aprendió la tarea con las 3 semillas y **la sigue haciendo con textos más largos de los que vio**. Con 16 veces
  el largo, una de las 3 semillas baja a 54% (las otras: 97% y 99.8%).
- El Transformer la aprendió bien con 1 de 3 semillas (99.7%), a medias con otra (85%) y casi nada con la tercera
  (18%); con textos más largos no acierta nunca.
- Sin la capa de copia no se aprende: lo que hace el trabajo es la capa ASRN.
- Ojo: ASRN solo aprendió con la tasa de aprendizaje más alta (0.003); con 0.001 no aprendió. Y cada corrida de ASRN
  tardó unos 9.5 minutos contra 36 segundos del Transformer.

### Texto real: WikiText-103

Dos modelos de ~44 millones de parámetros ven las mismas 30 millones de piezas (tokenizador de GPT-2), con textos de
1 024. *Perplejidad*: qué tan sorprendido está el modelo por el texto que sigue; menos es mejor.

| Modelo | Perplejidad con 1 024 | Perplejidad con 4 096 | Tiempo por corrida |
|---|---|---|---|
| ASRN | **56.75 ± 0.61** | **52.76 ± 0.72** | 12.7 min |
| Transformer | 63.76 ± 0.48 | 128.17 ± 2.77 | 3.1 min |

- **Con los mismos datos, ASRN predice mejor** (11% menos perplejidad) y la diferencia es mucho mayor que lo que varía
  entre semillas.
- **Con textos 4 veces más largos de los que vio, ASRN mejora** (usa el contexto extra) y el Transformer empeora
  mucho (no sabe usar posiciones que nunca vio).
- **Pero ASRN tardó 4 veces más.** Esta comparación es a igual cantidad de datos, no a igual tiempo de cómputo. Con
  los mismos 12.7 minutos el Transformer habría visto unas 4 veces más texto; eso no se ha medido.

### Lo que esa corrida no decía

- Son modelos chicos (hasta 50 millones de parámetros) y entrenamientos cortos (30 millones de piezas, unos minutos).
  No dicen qué pasa con modelos de miles de millones.
- Solo se probaron dos tasas de aprendizaje por modelo (0.001 y 0.003); con otras, cualquiera de los dos podría mejorar.
- No hay comparación a igual tiempo de cómputo, ni contra otras arquitecturas lineales (Mamba, RWKV y similares;
  la quinta corrida agrega Mamba-2 y RWKV-7).
- Hoy la capa ASRN es lenta para entrenar textos cortos y medianos: su ventaja de velocidad aparece por encima de
  ~100 mil piezas, y su memoria es mayor que la de FlashAttention.

## Segunda corrida: después de acelerar la capa (`corrida_2/`)

A100 de **80 GB**, PyTorch 2.11, bf16, con el código que incluye los arreglos de velocidad y el kernel Triton del
escaneo (`asrn_v89.py` sha256 b7529340f6f9). Solo está `resultados.json` (todos los números).

- **Pruebas:** 56 pasan y 1 se salta (`torch.compile` en CPU, que necesita PyTorch 2.14). Verificación en GPU: 6 pasan
  y `torch.compile` se salta por la misma razón. El kernel Triton corrió en la GPU sin problemas.
- **Velocidad del modelo de 50 M** (piezas por segundo; el Transformer se volvió a medir en la misma GPU):

| Largo × lote | ASRN, 1ª corrida | ASRN, 2ª corrida | Transformer, 2ª corrida |
|---|---|---|---|
| 1 024 × 32 | 35 136 | 63 107 | 230 907 |
| 4 096 × 8 | 44 033 | 97 627 | 201 759 |
| 16 384 × 2 | 44 503 | 100 918 | 137 567 |
| 32 768 × 1 | 44 410 | **100 658** | 96 542 |

  Con textos de 32 mil piezas ASRN ya entrena igual de rápido que el Transformer (un poco más). Con textos cortos
  sigue más lento.
- **Una capa:** 15.0 ms con 1 024 piezas (antes 23.5), 26.9 ms con 32 768 (FlashAttention 26.2) y 182 ms con 262 144
  (FlashAttention 1 470): ahora empata desde unas 32 mil piezas y es 8 veces más rápida con 262 mil. La memoria no
  cambió: unas 3.4 veces la de FlashAttention.
- **WikiText-103:** el mismo resultado que la primera corrida. ASRN 57.19 ± 0.13 contra 63.41 ± 0.78 del Transformer,
  y con textos 4 veces más largos 53.15 ± 0.28 contra 121.0 ± 4.8. Cada corrida de ASRN tardó 7.4 minutos (antes
  12.7) y la del Transformer 2.8: ASRN tarda 2.6 veces más (antes 4).
- **Copiar desde lejos:** ASRN 99.96% ± 0.02 con 1 024, 95.5% ± 7.7 con 4 096 y 65.6% ± 33.6 con 16 384. La semilla 0
  generalizó peor que en la primera corrida (86.6% y 30.5% contra 100% y 96.9%): con estas pocas semillas, la
  generalización a textos mucho más largos varía bastante de una corrida a otra. Lo que se repite en las dos: ASRN
  aprende la tarea con la tasa 0.003 en las 6 semillas y la sigue haciendo bien con 4 veces el largo; el Transformer
  (80.7% ± 21 con 1 024) no acierta nada con textos más largos, y sin la capa ASRN no se aprende.

## Tercera corrida: con los kernels de la atención por casilleros (`corrida_3/`)

A100 de 80 GB, PyTorch 2.11, bf16, `asrn_v89.py` sha256 0bf7ee1175c5. El objetivo era confirmar que, con los
gradientes calculados por los kernels nuevos, ASRN aprende igual.

- **Pruebas:** 57 pasan y 1 se salta (`torch.compile`, que necesita PyTorch 2.14). Verificación en GPU: 6 pasan y
  `torch.compile` se salta. GPU = CPU, causalidad y escritura paso a paso dan lo mismo que antes.
- **WikiText-103: aprende igual.** ASRN 56.97 ± 0.18 contra 63.44 ± 0.32 del Transformer (primera corrida 56.75 y
  segunda 57.19), y con textos 4 veces más largos 52.99 ± 0.30 contra 125.3 ± 7.6. Cada corrida de ASRN tardó **4.0
  minutos** (12.7 en la primera, 7.4 en la segunda); la del Transformer, 2.8.
- **Copiar desde lejos:** ASRN 100% con 1 024, 99.9% ± 0.2 con 4 096 y 90.9% ± 9.4 con 16 384 (semillas: 80.9%,
  99.6% y 92.3%). El Transformer eligió esta vez la tasa 0.003: 64.0% ± 55.4 con 1 024 (una semilla 100%, otra 0.2%)
  y nada con textos más largos. Sin la capa ASRN, nada.
- **Velocidad del modelo de 50 M** (piezas por segundo, misma GPU):

| Largo × lote | ASRN, 1ª corrida | ASRN, 3ª corrida | Transformer |
|---|---|---|---|
| 1 024 × 32 | 35 136 | 163 612 | 230 639 |
| 4 096 × 8 | 44 033 | 163 713 | 202 482 |
| 16 384 × 2 | 44 503 | **161 090** | 137 053 |
| 32 768 × 1 | 44 410 | **162 136** | 96 251 |

  Memoria total: 23.8 GB (28.7 en la primera corrida; el Transformer, 22.2).
- **Una capa:** 10.8 ms con 1 024 piezas (FlashAttention 1.5), 13.7 ms con 32 768 (FlashAttention 25.9) y 73 ms con
  262 144 (FlashAttention 1 485): **20 veces más rápida** con textos muy largos. Memoria: 10.8 GB con 262 144 piezas
  (FlashAttention 4.0).

## Cuarta corrida: una capa ASRN cada dos bloques (`corrida_4/`)

A100 de **40 GB**, PyTorch 2.11 (sin compilar), bf16, `asrn_v89.py` sha256 249b32f9af64. Agrega "ASRN cada 2"
(`largo_cada=2`): casi los mismos parámetros, más bloques locales y la mitad de capas ASRN. Detalle en la sección 5f de
[`docs/RENDIMIENTO.md`](../docs/RENDIMIENTO.md). Está `resultados.json`.

- **Pruebas:** 58 pasan y 2 se saltan (las de `torch.compile`, que necesitan PyTorch 2.14).
- **Verificación en GPU:** 6 pasan y `torch.compile` se salta (PyTorch 2.11).
- **Aprende igual.** WikiText-103: ASRN cada 2 57.06 ± 0.47, ASRN 56.97 ± 0.18, Transformer 64.06 ± 0.83; con textos 4
  veces más largos, 53.15, 52.99 y 125.3. Copia: 100%, 99.3% ± 1.3 y 89.6% ± 16.6 con 1, 4 y 16 veces el largo (ASRN:
  100%, 99.9% y 90.9%; Transformer: 97.8%, 0.2% y 0.2%).
- **Entrena 16–18% más rápido** (piezas por segundo, modelo de 50 M, misma GPU):

| Largo × lote | ASRN | ASRN cada 2 | Transformer |
|---|---|---|---|
| 1 024 × 32 | 148 007 | 172 394 | 207 538 |
| 4 096 × 8 | 145 312 | 171 502 | 184 781 |
| 16 384 × 2 | 145 969 | **170 410** | 129 159 |
| 32 768 × 1 | 146 205 | **169 984** | 92 877 |

  Cada corrida de WikiText tardó 3.86 minutos (ASRN 4.34, Transformer 3.06) y cada una de copia 68 segundos (ASRN 89).
- En esta GPU de 40 GB ASRN normal fue algo más lento que en la de 80 GB de la tercera corrida (148 mil contra 164 mil
  piezas por segundo); por eso las comparaciones de arriba son siempre dentro de la misma corrida.

## Quinta corrida: contra Mamba-2 y RWKV-7 (`corrida_5/`)

A100 de **80 GB**, PyTorch 2.11 (sin compilar), bf16, `asrn_v89.py` sha256 d9927a1bba79 (con menos conversiones de
tipo). Mamba-2 y RWKV-7 son las reimplementaciones de `benchmarks/rivales_lineales.py`, con los kernels Triton de
flash-linear-attention para su recurrencia; el resto de su código es PyTorch sin fusionar.

- **Verificación:** 6 pasan y `torch.compile` se salta. Los kernels de Mamba-2 y RWKV-7 dan lo mismo que su
  recurrencia paso a paso (diferencia relativa 1.9·10⁻³ y 1.5·10⁻³, redondeo de bf16). Pruebas automáticas: 59 pasan,
  2 se saltan y **1 falla**: `test_entrenar_aprende_a_copiar_y_reanuda` (en la A100 la pérdida de copia queda en
  4.19; pasaba con la misma versión de PyTorch, 2.11, en las corridas 2 a 4, y pasa en CPU con 2.14). Sigue sin
  explicar: no está aislado si la causa es el cambio de código, la versión o la máquina.
- **Texto real (WikiText-103), mejor tasa, 3 semillas, los mismos 30 millones de piezas:**

| Modelo | Perplejidad con 1 024 | Con 4 096 | Minutos por corrida |
|---|---|---|---|
| **RWKV-7** | **47.87 ± 0.15** | **45.83 ± 0.11** | 7.5 |
| ASRN | 56.48 ± 0.86 | 52.46 ± 0.91 | 3.9 |
| ASRN cada 2 | 57.02 ± 0.32 | 53.06 ± 0.36 | 3.5 |
| Mamba-2 | 59.46 ± 0.50 | 58.34 ± 0.60 | 4.8 |
| Transformer | 64.26 ± 1.23 | 122.6 ± 2.5 | 2.8 |

  **RWKV-7 predice el texto bastante mejor que ASRN** (15% menos perplejidad, mucho más que lo que varía entre
  semillas). ASRN queda segundo, por delante de Mamba-2 y del Transformer. Los cuatro modelos de costo lineal mejoran
  con textos más largos; el Transformer no.

- **Copiar desde lejos** (exactitud; al azar 0.2%):

| Modelo | Con 1 024 | Con 4 096 | Con 16 384 |
|---|---|---|---|
| ASRN | **100%** | **99.6% ± 0.7** | **79.9% ± 33.4** |
| ASRN cada 2 | 100% | 99.3% ± 0.8 | 76.9% ± 22.9 |
| Transformer | 70.8% ± 47.3 | 0.2% | 0.2% |
| RWKV-7 | 2.4% ± 2.1 | 0.2% | 0.2% |
| Mamba-2 | 0.2% | 0.2% | 0.2% |

  **Solo ASRN copia desde lejos.** Mamba-2 y RWKV-7 guardan el pasado en un estado de tamaño fijo y no logran
  repetir 512 piezas al pie de la letra en 2 000 pasos.

- **Una capa sola** (ida y vuelta, d = 512): con 1 024 piezas FlashAttention 1.5 ms, Mamba-2 5.1, RWKV-7 9.1, ASRN 11.1;
  con 262 144 piezas **ASRN 59.7 ms**, Mamba-2 216, RWKV-7 547, FlashAttention 1 485. Pendiente del tiempo: ASRN 0.60,
  RWKV-7 0.97, Mamba-2 1.15, FlashAttention 1.90. Memoria con 262 144: ASRN 10.3 GB, Mamba-2 12.9, RWKV-7 20.8.
- **Modelo de 50 M** (piezas por segundo, sin compilar): ASRN 176 423, ASRN cada 2 200 258, Transformer 231 545 con
  1 024 (96 204 con 32 768), Mamba-2 94 806, RWKV-7 53 048. La velocidad de Mamba-2 y RWKV-7 no representa a sus
  versiones oficiales: aquí solo su recurrencia usa kernels optimizados.
- ASRN entrenó a 176 mil piezas por segundo con 1 024; en la tercera corrida, en otra A100 de 80 GB, a 164 mil. Parte
  de esa diferencia puede ser el cambio de las conversiones de tipo, pero son sesiones distintas.
- Después de esta corrida, "cada 2" pasó a ser la forma por omisión de ASRN (`largo_cada=2`): aprende igual, dentro de
  la variación entre semillas, y entrena 13% más rápido en esta GPU.

## Bloques locales (`bloques_1/`)

A100 de 80 GB, PyTorch 2.11, bf16, una semilla y la tasa de ASRN (3e-3), `pruebas_gpu_a100.ipynb` con
`BLOQUES = True`. Cambia el bloque local que va entre capas ASRN, siempre con los mismos parámetros (modelos de ~44
millones). Detalle en la sección 5i de [`docs/RENDIMIENTO.md`](../docs/RENDIMIENTO.md).

| Bloque local | WikiText 1 024 | WikiText 4 096 | Copia 1× / 4× / 16× | Piezas/s (50 M, 1 024) |
|---|---|---|---|---|
| Convolución (por omisión) | 56.24 | 52.32 | 100% / 100% / 99.2% | 175 145 |
| Convolución por canal (quitada) | 56.03 | 52.02 | 100% / 99.6% / 76.9% | 172 663 |
| Por canal + puerta (quitada) | 58.74 | 54.65 | 99.9% / 86.8% / 23.8% | 165 334 |
| **Memoria** (`mezcla='memoria'`) | **53.06** | **49.21** | 100% / **35.4%** / **2.7%** | 145 478 |
| Memoria + puerta (quitada) | 53.96 | 50.06 | 100% / 62.6% / 20.4% | 142 371 |

- La memoria con olvido baja la perplejidad 5.7%, pero rompe la copia con textos más largos que los de entrenamiento y
  cuesta 17% de velocidad. La convolución sigue por omisión.
- Lo que no ganó se quitó del código. La siguiente prueba (lista en el cuaderno, sin correr) es la memoria con escalas
  más cortas: `memoria_max` de 128 y de 32 piezas.

## Velocidad de cada versión en una misma sesión (`velocidad/`)

`history/experiments/medir_velocidad_v89.py` (desde un cuaderno de Colab de entonces, retirado en 1.0) en una A100 de
**40 GB** con PyTorch 2.14.1, para poder compilar (`torch.compile` necesita
2.14). Modelo de 50 millones, bf16. Todo en la misma sesión, así que las comparaciones son justas entre sí (en la
quinta corrida, otra GPU, los números absolutos son distintos).

| Piezas por segundo | 1 024 × 32 | 4 096 × 8 | 16 384 × 2 | 32 768 × 1 |
|---|---|---|---|---|
| ASRN antes de acelerar | 44 879 | 45 752 | 45 332 | 44 877 |
| ASRN cada 1 | 147 548 | 145 844 | 144 302 | 142 897 |
| ASRN cada 1, compilado | 184 982 | 184 599 | 182 030 | 179 315 |
| ASRN cada 2 | 171 235 | 171 118 | 169 762 | 169 189 |
| **ASRN cada 2, compilado** | **214 536** | **214 550** | **211 390** | **212 413** |
| Transformer | 207 573 | 184 860 | 129 271 | 92 922 |
| Transformer, compilado | 271 582 | 233 917 | 151 717 | 103 942 |

- Compilando los dos, el Transformer es 1.27 veces más rápido con 1 024 piezas y 1.09 con 4 096; ASRN es 1.39 veces más
  rápido con 16 384 y 2.04 con 32 768.
- `velocidad_compilada.json` también trae las mismas cuentas antes y después (`mismas_cuentas`), dónde se va el tiempo
  dentro de una capa y el paso entero por partes.
