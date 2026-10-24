# ASRN 1.0 — Notas de la versión

*Octubre de 2026 · Alexis Calvario · Apache 2.0 · [English version](RELEASE_NOTES.md)*

ASRN (Adaptive Sparse Recurrence Network) es una capa para modelos de lenguaje que encuentra apariciones anteriores
del contexto actual con tablas de hash aprendidas (8 tablas, casilleros de 14 bits, a lo sumo los 32 compañeros de
casillero más recientes por tabla) y copia lo que vino después. Un reloj de tiempo propio que depende de los datos,
intensidades de Hawkes por grupo de piezas y una racha de repetición alimentan una puerta de copia. El trabajo por
pieza de la atención por casilleros está acotado (a lo sumo 8 × 32 candidatos), y lo que se guarda para la vuelta
crece lineal con el largo. Escribir paso a paso cuesta un tiempo O(1) por pieza nueva, sea cual sea el largo del
texto; su memoria sigue creciendo con el texto.

1.0 es el primer lanzamiento empaquetado. Convierte el código de investigación de v89 en un paquete instalable, deja
como capa por omisión la que tiene las protecciones numéricas, hace que escribir cueste O(1) por pieza nueva, trae el
banco de pruebas contra Mamba-2, RWKV-7 y un Transformer++ junto con sus resultados, y pasa el historial de desarrollo
a `history/`.

Cada número de estas notas sale del repositorio. Las fuentes principales son
[`results/banco_a100/`](../results/banco_a100/RESUMEN.md) (la ejecución en la A100 del 17 de octubre de 2026, PyTorch
2.14.1), [`results/citas_a100/`](../results/citas_a100/RESUMEN.md) y [`results/citas/`](../results/citas/RESUMEN.md)
(el experimento de citas en la A100 y en CPU), [`results/escritura_cpu/`](../results/escritura_cpu/RESUMEN.md) (la
escritura O(1) y la O(t) en CPU, 23 de octubre de 2026) y [`results/banco_cpu/`](../results/banco_cpu/RESUMEN.md) (los
resultados tal cual); `history/docs/ESTABILIDAD_Y_BANCO.md` (el informe de estabilidad y del banco; sus rutas son las
de antes de 1.0, ver [`history/README.md`](../history/README.md));
[`history/results/banco_a100_pt211/`](../history/results/banco_a100_pt211/RESUMEN.md) (la ejecución anterior en la
A100, PyTorch 2.11), y `history/docs/` (el whitepaper de v89 y el registro de velocidad).

---

## Lo principal

- **Un paquete instalable.** `pip install .` da `import asrn` (versión `1.0`), `python -m asrn` y el comando
  `asrn-train`. Python ≥ 3.11, PyTorch ≥ 2.9, NumPy. Los alias en inglés (`asrn.ASRNLayer`, `asrn.ASRNModel`,
  `asrn.create_model`, `asrn.train`, `asrn.generate`, `asrn.load`, …) nombran los mismos objetos que la API en español.
- **La batería completa de pruebas pasa.** Las baterías actuales, 57 pruebas del núcleo y 19 de la capa estable y de la
  API del paquete (76 en total), pasan 57 / 57 y 19 / 19 en CPU con PyTorch 2.14.1 (23 de octubre de 2026). Las
  baterías de la ejecución en la A100 del 17 de octubre (55 y 18 pruebas) pasaron 55 / 55 y 18 / 18 tanto en CPU como
  en la máquina de la A100 de Colab, donde también pasaron las 7 / 7 comprobaciones en GPU, `torch.compile` incluido.
  Con PyTorch 2.11.0 en CPU se saltan las dos pruebas de `torch.compile` del núcleo (pasan 55) y la batería
  de la capa estable pasa 19 / 19. Un flujo de GitHub Actions corre las dos baterías en CPU
  con Python 3.11 y 3.13 (ver [Verificación](#verificación)).
- **La capa estable es la de por omisión.** `asrn.crear_modelo`, `asrn.crear_desde_config` y `asrn.cargar` devuelven
  modelos cuyas capas ASRN son `CapaASRNEstable` con el grupo A de protecciones. Tienen los mismos parámetros que la
  capa de v89 y dan los mismos números salvo el redondeo (1.3·10⁻⁷ en los logits). Los modelos guardados cargan en las
  dos capas.
- **Escribir cuesta un tiempo O(1) por pieza nueva.** `escribir()` / `asrn.generate` ya no recorren el texto otra vez:
  cada tabla lleva un anillo con las 32 posiciones más recientes de cada casillero (justo las llaves que la escritura
  anterior elegía entre toda la memoria), y la intensidad de Hawkes se lleva por clave de grupo de piezas con la misma
  recurrencia que el escaneo sobre el texto entero. La escritura O(1) da lo mismo que la escritura O(t) anterior y que
  la ida sobre el texto entero (diferencia máxima 2.2·10⁻⁷), y `escribir()` da las mismas secuencias de piezas en 192
  de 192 combinaciones de opciones de muestreo. En CPU ([`results/escritura_cpu/`](../results/escritura_cpu/RESUMEN.md),
  un modelo de 700 484 parámetros) tarda 9.49–12.47 ms por pieza nueva de 512 a 131 072 piezas de contexto, con una
  pendiente que no se distingue de cero, mientras que la escritura O(t) pasa de 15.80 a 91.67 ms (9.7 veces más lenta
  con 131 072). La memoria por secuencia sigue creciendo con el texto, y la escritura O(1) todavía no se midió en la
  A100 (ver [Escritura en O(1)](#escritura-en-o1)).
- **Seis problemas numéricos de v89 corregidos, cada uno reproducido o medido antes.** La fuga causal por la máscara de
  −10⁹ (0.016 → 0 exacto en una A100). El reloj de 32 bits, que pierde precisión desde ~1 M de piezas y se detiene en
  el 16% de los pasos a 16 M. La cancelación en el puntaje (la salida se movía hasta 8.8·10⁻² con el reloj corrido 10⁹
  → 5.5·10⁻⁸, constante). β sin techo, que termina en NaN. La proyección del reloj en 16 bits con autocast. TF32 en la
  proyección del reloj (2.0 unidades de error del reloj → 1.2·10⁻¹⁰). Costo medido en una A100, ya con la proyección
  del reloj inmune a TF32: +7.5% de tiempo y +5.2% de memoria con 262 144 piezas.
- **Los kernels Triton que vienen de v89** (atención por casilleros sin atómicas, escaneo asociativo) siguen igual;
  los de la atención por casilleros tienen ahora un modo `ESTABLE`, fijado al compilar, con un reloj en dos partes (alta
  y baja) y una máscara exacta. Con el kernel, la capa guarda 20 KiB por pieza para la vuelta (d = 512, fp32), contra
  44 KiB por el camino de PyTorch.
- **Un banco de pruebas nuevo contra los rivales oficiales** en una NVIDIA A100-SXM4-80GB: Mamba-2 original
  (`mamba_ssm`), RWKV-7 original (`flash-linear-attention`) y nuestro Transformer++ (la receta de Llama con
  FlashAttention-2), con PyTorch 2.14.1. Todo corrió sin compilar salvo dos filas aparte de entrenamiento: ASRN
  compilado por partes y el Transformer++ con `torch.compile`.
  - **Dónde quedó por delante ASRN:** velocidad con textos largos (un modelo de ~50 M entrena a 163–172 mil piezas/s
    de 1 024 a 32 768 piezas: 2.08× el Transformer++ con 32 768, 1.4–1.5× Mamba-2 sin fusionar y 3.4–3.6× RWKV-7;
    compilado, 2.20× el Transformer++ compilado con 32 768). Una capa sola con textos largos (21.6× la atención y 2.5×
    un bloque de Mamba-2 con 262 144 piezas). Recuperar algo exacto más allá del largo entrenado (copia con 4× en las
    2 de 3 semillas que salieron de la meseta inicial; MQAR con 2× y 4× los pares, significativo frente a Mamba-2 solo
    con 4×). La perplejidad en WikiText-103 con este presupuesto chico (no significativa frente a RWKV-7).
  - **Dónde no:** textos cortos (un piso de ~12 ms por capa), memoria por capa, escritura (la más lenta de las cuatro,
    medida con la escritura O(t) anterior; la O(1) todavía no se midió en la A100), aprender la copia rápido dentro del
    largo entrenado y la seguridad con que la aprende (2 de 3 semillas aquí).
- **Velocidad casi plana con el largo, y las herramientas para escalar.** Un modelo de ~50 M de parámetros entrena a
  163–172 mil piezas/s sin compilar (−5.3%) y a 199–210 mil compilado por partes, de 1 024 a 32 768 piezas por texto,
  en una sola A100. Para pasar de 100 M de parámetros, el lanzamiento trae `examples/train_100m/`, un kit de Colab
  listo para correr para un modelo de 103.5 M de parámetros con FineWeb, y el reparto con FSDP2 (`repartir_fsdp`),
  cubierto por una prueba de FSDP con dos procesos en CPU y una comprobación con un proceso NCCL en la A100.
- **Un mapa claro de la evidencia.** La capa separa *dónde copiar* (la atención dispersa por casilleros: tablas de hash
  aprendidas, ventanas de candidatos y puntajes con sufijo exacto, confirmación y el término de tiempo) de *si copiar y
  cuándo parar* (la puerta, el reloj de tiempo propio y las señales de Hawkes y de racha). En copia y MQAR, donde cada
  posición evaluada pide copiar, el núcleo de recuperación basta por sí solo, y sin la capa ASRN las dos tareas quedan
  en el azar. Ahí los mecanismos de regulación no interfieren: con ellos ASRN llega a ≥ 99.98% en copia con 4× en las
  2 de 3 semillas que aprendieron y a 99.75% en MQAR con 4× los pares, y quitarlos de a uno deja la exactitud de las
  corridas que aprendieron dentro de la variación entre semillas. Su papel activo se valida en un experimento sintético
  controlado de citas de largo acotado corrido con la capa de 1.0 (3 semillas), donde, sin supervisión explícita,
  mantienen la racha de repetición escrita más larga en 10 piezas, como en el texto real, contra 21–27 en CPU (19–39 en
  la A100) cuando se quitan las señales de Hawkes y de racha o las señales de la puerta. Mezclar las tablas por
  log-suma-exp (`mezcla_lse`) no se adopta en 1.0. Ver [Problemas conocidos y
  limitaciones](#problemas-conocidos-y-limitaciones).

---

## Qué trae 1.0

### Organización

| Ruta | Contenido |
|---|---|
| `asrn/nucleo.py` | El núcleo: lo que antes era `asrn_v89.py`. Trae `CapaASRN` y sus kernels Triton (con el modo `ESTABLE`), `BloqueLocal`, `ModeloASRN`, `ConfigASRN` / `TALLAS` / `para_parametros`, `entrenar`, `evaluar`, `escribir` (con la escritura O(1), `_Escritor`), `guardar` / `cargar`, FSDP (`repartir_fsdp`, `materializar`), `compilar_partes`, `PasoGrafado` y la línea de comandos |
| `asrn/estable.py` | Lo que antes era `asrn_estable.py`: `CapaASRNEstable`, `Protecciones`, `_ProyeccionIEEE`, `_TiempoRelativo`, `convertir()`, `RecorteDinamico` |
| `asrn/__init__.py` | La API pública (capa estable por omisión), con alias en inglés de los mismos objetos (`ASRNLayer`, `ASRNCoreLayer`, `ASRNModel`, `ASRNConfig`, `LocalBlock`, `Protections`, `DynamicClip`, `create_model`, `create_from_config`, `convert`, `train`, `evaluate`, `generate`, `save`, `load`); `__version__ = '1.0'` |
| `asrn/__main__.py` | `python -m asrn` y el comando `asrn-train`: la línea de comandos del núcleo con la capa estable |
| `pyproject.toml`, `requirements.txt` | `pip install .`; Python ≥ 3.11, `torch>=2.9`, `numpy`; extra `[bench]` |
| `tests/` | `test_nucleo.py` (57 pruebas del núcleo), `test_estable.py` (19 pruebas de la capa estable, la API del paquete y los alias en inglés), `verificar_gpu.py` (7 comprobaciones en GPU) |
| [`benchmarks/`](../benchmarks/README.md) | El banco de pruebas: `asrn_colab.ipynb` (el único cuaderno de Colab: todas las pruebas, el banco en A100 y el experimento de citas, con PyTorch 2.14; su ejecución del 17 de octubre de 2026 produjo `results/banco_a100/` y `results/citas_a100/`; su sección de escritura corre ahora la escritura O(1)), `banco_asrn.py`, `banco_gpu.py`, `banco_modelos.py`, `modelos_oficiales.py` (los rivales), `rivales_lineales.py` (reimplementaciones de respaldo), `builders/` (los programas que arman los cuadernos). También `experimento_citas.py`: el experimento de citas de largo acotado llevado a la capa estable de 1.0 (la capa completa contra `sin_hawkes_racha`, `puerta_sin_senales`, `sin_puerta` y `sin_reloj`; 3 semillas, CPU o GPU); y `medir_escritura.py`: ms por pieza nueva según el contexto, con la escritura O(1) y la O(t) |
| [`results/`](../results/README.md) | `banco_a100/` (la ejecución en la A100 del 17 de octubre de 2026 con PyTorch 2.14.1: `resultados.json`, `RESUMEN.md`, gráficas), `citas_a100/` (el experimento de citas en la A100, en la misma ejecución), `escritura_cpu/` (la escritura O(1) y la O(t) en CPU, de `benchmarks/medir_escritura.py`), `banco_cpu/` (costo exacto, latencia en CPU, escritura con la versión O(t), copia a escala chica) y `citas/` (registros y resultados por semilla de `benchmarks/experimento_citas.py` en CPU) |
| [`examples/train_100m/`](../examples/train_100m/README.md) | Un kit listo para correr que entrena un ASRN de 103.5 M de parámetros con FineWeb en Colab, con `import asrn`; su primera corrida está en la [hoja de ruta](#hoja-de-ruta--trabajo-futuro-v11) |
| [`paper/`](../paper/README.md) | Borradores del artículo en LaTeX (inglés y español); las figuras salen de `results/` con `paper/hacer_figuras.py`, y el texto del artículo se está actualizando a los resultados de 1.0 |
| [`history/`](../history/README.md) | El historial de desarrollo del ciclo de v89 (`docs/`, `experiments/`, `probes/`, `results/`, con `results/banco_a100_pt211/`, la ejecución anterior en la A100, del 9 y el 11 de octubre de 2026 con PyTorch 2.11); no forma parte del paquete, y se conserva porque los documentos lo citan como evidencia (ver [Cambios incompatibles](#cambios-incompatibles-y-migración)) |
| [`docs/`](README.md) | Estas notas, [`ARCHITECTURE_es.md`](ARCHITECTURE_es.md) y [`TECHNICAL_REPORT_es.md`](TECHNICAL_REPORT_es.md), cada uno con su versión en inglés (sin `_es`); el diagrama de la arquitectura, [`asrn_arquitectura.svg`](asrn_arquitectura.svg) (en inglés: [`asrn_architecture.svg`](asrn_architecture.svg)), dibujado por `hacer_diagrama.py`: el modelo y el interior de una capa ASRN |
| [`.github/`](../.github/workflows/tests.yml) | El flujo de integración continua `workflows/tests.yml` (las dos baterías de pruebas en CPU, Python 3.11 y 3.13) y las plantillas de issues |
| [`CHANGELOG.md`](../CHANGELOG.md), [`CITATION.cff`](../CITATION.cff), [`SECURITY.md`](../SECURITY.md) | El registro de cambios (en inglés), los datos para citar y la política de seguridad (carga de archivos guardados) |

[`ARCHITECTURE_es.md`](ARCHITECTURE_es.md) / [`ARCHITECTURE.md`](ARCHITECTURE.md) explican cómo funciona la capa, y
[`TECHNICAL_REPORT_es.md`](TECHNICAL_REPORT_es.md) ([`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) en inglés) trae el
análisis completo.

### Instalar y usar

```bash
pip install .                 # instala solo el paquete asrn; las pruebas y el banco corren desde el repositorio
pip install ".[bench]"        # + flash-linear-attention, datasets, tiktoken, einops, matplotlib (mamba-ssm aparte)
```

```python
import asrn
modelo = asrn.crear_modelo(V=8000, d=256, n_capas=8)            # capas ASRN: CapaASRNEstable, grupo A
asrn.entrenar(modelo, piezas, validacion=piezas_val, guardar_en='modelo.pt',
              recorte=asrn.RecorteDinamico(modelo))             # recorte del gradiente por grupos (opcional)
secuencias, _ = asrn.escribir(asrn.cargar('modelo.pt'), inicio, n_nuevos=200, devolver_logits=False)
```

```bash
python -m asrn --datos piezas.pt --V 8000 --guardar modelo.pt     # lo mismo que: asrn-train ...
python -m asrn --datos piezas.pt --parametros 50m --V 32000       # el tamaño que quieras, como en v89
torchrun --nproc_per_node=8 -m asrn --fsdp --talla 1b ...         # varias GPUs
python -m asrn.nucleo ...                                          # la misma línea de comandos, capa sin protecciones
```

### La capa estable por omisión

| Llamada | Resultado |
|---|---|
| `asrn.crear_modelo(...)`, `asrn.crear_desde_config(...)` (también en `'meta'`), `asrn.cargar(...)` | `ModeloASRN` con `CapaASRNEstable` y `Protecciones()` (grupo A). Con la misma semilla, los pesos son idénticos a los de `asrn.nucleo.crear_modelo` |
| `protecciones=asrn.Protecciones.textos_largos(L)` | Grupo A + grupo B, pensado para entrenar desde cero con textos largos |
| `protecciones=asrn.Protecciones.ninguna()` | Sin protecciones: igual bit a bit a la capa del núcleo (prueba en CPU, logits y gradientes) |
| `asrn.convertir(modelo, protecciones)` | Cambia en el lugar las capas ASRN de cualquier `ModeloASRN`, con sus mismos pesos |
| `asrn.RecorteDinamico(modelo)` | Recorte del gradiente por grupos, opcional, que se pasa como `recorte=` a `entrenar()`. No lo usan ni `entrenar()` por omisión (recorte global en 1.0) ni la línea de comandos |

El paquete exporta además `entrenar`, `evaluar`, `escribir`, `guardar`, `ConfigASRN`, `TALLAS`, `CapaASRN`,
`CapaASRNEstable`, `BloqueLocal`, `ModeloASRN`, `PasoGrafado`, `inicializar`, `materializar`, `repartir_fsdp`,
`esta_repartido`, `pesos_completos`, `compilar_partes`, `crear_optimizador`, `tasa_aprendizaje`, `leer_piezas`,
`leer_cantidad`, `firmas_de_piezas`, `escaneo` y los módulos `nucleo` y `estable`.

### Escritura en O(1)

Escribir paso a paso (`_Escritor` en `asrn/nucleo.py`, que usan `escribir()` / `asrn.generate`) cuesta un tiempo O(1)
por pieza nueva, sin importar cuánto texto haya. Antes de 1.0 costaba O(t) por pieza.
- **Dónde copiar.** Cada tabla lleva, por casillero, un anillo con las C = 32 posiciones más recientes: justo las
  llaves que el `_atender` anterior elegía entre toda la memoria. Con la memoria de lo ajeno (`memoria_de_lo_ajeno`, por
  omisión), solo las posiciones del inicio pueden ser llaves, así que los anillos se llenan con el inicio y no cambian
  más. La pieza nueva puntúa solo sus T × C candidatos (`_atender_cand`); la capa estable conserva la máscara exacta y
  el reloj de 64 bits.
- **Intensidad de Hawkes.** Se lleva por clave de grupo de piezas (`_MemoriaHawkes`): el reloj de la última aparición
  de la clave, B_u, y A = λ + α en ese momento; luego λ_n(t) = e^{−κ_n (B_t − B_u)} · A, y después A ← λ_n(t) + α_t y
  B_u ← B_t. Es la misma suma de la definición, llevada como la recurrencia que usa el escaneo asociativo sobre el texto
  entero. Las claves se buscan en diccionarios de Python, calculadas en la CPU a partir de las piezas muestreadas (sin
  esperar a la GPU); los estados quedan en el dispositivo. La racha ya era un solo número.
- **Ayudas del muestreo.** La temperatura que conserva la cita y `no_repetir_ngramas` usan diccionarios de grupos de
  piezas que se actualizan en cada pieza (`_Ngramas`) en vez de recorrer el texto otra vez, y el texto escrito vive en
  un búfer reservado de antemano.
- **Respaldo.** La escritura O(t) anterior sigue disponible como `_Escritor(modelo, o1=False)`. Se usa automáticamente
  solo si los anillos pasaran de 2 GiB (un `n_bits` o un lote muy grandes).
- **Memoria.** Sigue creciendo con el texto (lo que se guarda de cada posición: lo que se copia hay que guardarlo), más
  los anillos fijos: T · 2^n_bits · C posiciones (int32) y contadores, 17 MiB por capa ASRN y por secuencia con los
  tamaños por omisión (8 tablas, 14 bits, C = 32), unos 120 MiB por secuencia para el modelo de ~50 M del banco (7
  capas ASRN).
- **Comprobado en CPU (PyTorch 2.14.1).** Escritura O(1) = escritura O(t) = ida sobre el texto entero, diferencia
  máxima 2.2·10⁻⁷: capa de v89 y estable, anillos que se desbordan, lotes de 1 a 3, inicios de 1, 2 y 40 piezas, con y
  sin la memoria de lo ajeno (pruebas nuevas `test_escritura_o1_igual_que_o_t`, en las dos baterías, y
  `test_anillo_y_ngramas_como_su_definicion`). `escribir()` da las mismas secuencias de piezas, y logits a menos de
  10⁻⁴, que la implementación O(t) anterior en **192 de 192** combinaciones de opciones de muestreo (temperatura, top-k,
  top-p, memoria de lo ajeno, temperatura que conserva la cita, pena de repetición, prohibición de n-gramas), con la
  capa de v89 y la estable. La ida y los gradientes del modelo son iguales bit a bit a los del código de la ejecución en
  la A100 (ver [Procedencia de los resultados](#procedencia-de-los-resultados)).

**Medido en CPU** ([`results/escritura_cpu/`](../results/escritura_cpu/RESUMEN.md), con
`benchmarks/medir_escritura.py`; 4 hilos, PyTorch 2.14.1; un modelo ASRN de 700 484 parámetros con V = 256, d = 128,
4 bloques locales y 2 capas ASRN; lote 1, sin muestrear; mediana de 9 tandas de 32 piezas; las dos escrituras
intercaladas en cada contexto, después de calentar las dos y de un inicio sin medir de cada largo):

| Contexto (piezas) | O(1): ms por pieza | O(t): ms por pieza | O(1): inicio (s) | O(t): inicio (s) |
|---|---|---|---|---|
| 512 | 12.45 | 15.80 | 0.10 | 0.06 |
| 2 048 | 12.47 | 16.73 | 0.26 | 0.20 |
| 8 192 | 10.68 | 19.02 | 0.77 | 0.70 |
| 32 768 | 11.26 | 37.12 | 3.60 | 2.73 |
| 131 072 | 9.49 | 91.67 | 29.04 | 20.44 |

- **Tiempo por pieza.** Contra el contexto (mínimos cuadrados, IC del 95%), la pendiente de la escritura O(1) es
  −19.21 ns por pieza de contexto (−41.54 a 3.12), que no se distingue de cero; la de la O(t) es 583.31 ns (537.13 a
  629.49). Con 131 072 piezas la escritura O(1) es 9.7 veces más rápida por pieza (91.67 / 9.49), y 1.27 veces con 512.
- **Tiempo del inicio.** Lo que agrega la escritura O(1) (llenar los anillos y el estado de Hawkes) se perfiló en unos
  0.3 s con 32 768 piezas y 1.3 s con 131 072 (anillos ~0.3 s, estado de Hawkes 0.8–1.1 s). Las diferencias de la tabla
  son mayores porque el tiempo del inicio varía de una corrida a otra en esta máquina compartida de 4 núcleos: la misma
  pasada del inicio tardó 14.8–23.5 s con 131 072 piezas en cuatro repeticiones.
- **Todavía sin medir en la A100.** Las cifras de escritura en la A100 de este lanzamiento se midieron con la escritura
  O(t) (ver [Problemas conocidos y limitaciones](#problemas-conocidos-y-limitaciones)). La sección de escritura del
  cuaderno (`benchmarks/asrn_colab.ipynb`, sección 6) corre ahora la escritura O(1), con las piezas nuevas pasadas
  desde la CPU como hace `escribir()`, así que la próxima ejecución en Colab la mide.

---

## Mejoras del kernel que vienen de v89

Las cuentas de la capa del núcleo son las mismas que en v89. Estas mejoras se hicieron durante el ciclo de v89 y
llegan en 1.0. Lo nuevo es el modo `ESTABLE`.

**Atención por casilleros en Triton** (`_casilleros_ida` / `_casilleros_vuelta`).
- Corre un programa por (fila del lote × tabla, bloque de 32 preguntas en el orden por casillero de esa tabla). Lee sus
  preguntas y 64 llaves directamente del orden del texto y arma los puntajes en la SRAM con dos `tl.dot`, el peso del
  sufijo, el núcleo de tiempo y la máscara. Hace un softmax en 32 bits y escribe cada salida en su posición del texto.
- La vuelta recalcula los puntajes, como FlashAttention, y guarda solo las entradas.
- La vuelta no usa atómicas. El gradiente de cada llave se escribe en su propio lugar para cada una de sus dos ventanas
  y se suma después, así que el resultado es siempre el mismo. En el ciclo de v89, quitar las atómicas bajó la vuelta
  del kernel de 6.4 a 0.99 ms con 16 384 piezas (A100 de 40 GB).
- `ASRN_TRITON_CASILLEROS=0` apaga este kernel y `ASRN_TRITON=0` apaga todos. La versión de PyTorch sigue siendo la
  referencia y el respaldo en CPU o sin Triton.

**Escaneo asociativo en Triton** (`_escaneo_kernel`).
- Este escaneo calcula Hawkes y la racha: S_i = a_i S_{i−1} + b_i con a_i ∈ [0, 1], compuesto como
  (a₁, b₁)∘(a₂, b₂) = (a₁a₂, a₂b₁ + b₂). El kernel corre un programa por fila, en bloques de 1 024 con
  `tl.associative_scan` y un acarreo entre bloques. La vuelta es el mismo kernel corriendo de derecha a izquierda.
- Unas 290 operaciones de PyTorch por capa pasaron a unas 6.
- También de v89: `_Tomar` reemplazó un `index_add_` atómico que se llevaba la mitad del tiempo de GPU de la capa con
  16 384 piezas (18.4 de 36.6 ms).

**Modo `ESTABLE` (nuevo, para la capa estable).** `ESTABLE` es un parámetro `tl.constexpr` de los dos kernels de la
atención por casilleros (`_casilleros_ida`, `_casilleros_vuelta`); el escaneo no lo tiene. Apagado, el kernel compilado
es el que tenía v89.
- **Reloj en dos partes.** B llega como las mitades alta y baja, de 32 bits cada una, de un reloj de 64 bits. El
  término de tiempo usa dt = (B_k − B_q)_alta + (B_k − B_q)_baja, tomada *antes* de multiplicar por la tasa. Entre
  piezas cercanas la resta de la parte alta es exacta (lema de Sterbenz) y la baja corrige el redondeo. Costo: 4 bytes
  más leídos por pregunta y llave.
- **Máscara exacta.** El máximo del softmax se toma solo entre las llaves válidas, y las no válidas pesan 0 exacto.
- **Comprobado con el intérprete de Triton en CPU.** El kernel da lo mismo que el camino de PyTorch de la capa estable
  en ida y vuelta, en fp32 y bf16. No tiene la fuga, y su salida cambia como mucho 8·10⁻⁸ con el reloj corrido hasta
  10⁹.
- **Compilado en la A100, sin TF32.** Coincide con PyTorch a 4.9·10⁻⁸ en la salida y a 4.5·10⁻⁶ en el peor gradiente
  (fp32). En bf16 queda a 0.096 de la respuesta en fp32, contra 0.094 de PyTorch en bf16: el mismo orden de error.

**Memoria.**
- Con el kernel, la capa guarda 20.1 KiB por pieza para la vuelta (20.2 KiB la estable), contra 43.8 KiB por el camino
  de PyTorch (43.9 KiB). Son cifras con d = 512 y fp32, constantes con el largo, medidas con ganchos de autograd.
- Desde v86, el flujo de tensores de la capa evita tres copias innecesarias. Lo guardado bajó **44% respecto de v85**
  (55.3 → 30.9 floats por pieza, en unidades de d, con d = 256) con la misma matemática, así que la memoria guardada
  sigue siendo O(N).

**Las cuentas por pieza son constantes** (`FlopCounterMode`, d = 512, ida y vuelta). La capa ASRN cuesta 5.19, 5.14 y
5.13 MFLOP por pieza con 1 024, 4 096 y 16 384 piezas. Una capa de atención cuesta 9.44, 18.87 y 56.62.

**Velocidad medida en la A100** (PyTorch 2.14.1, CUDA 13.0, bf16; sin compilar salvo las dos filas compiladas):

| Una capa, d = 512, lote 1, ida y vuelta (ms) | 1 024 | 4 096 | 16 384 | 65 536 | 262 144 |
|---|---|---|---|---|---|
| ASRN (capa estable) | 12.1 | 12.2 | 12.5 | 19.3 | 68.7 |
| Atención del Transformer++ (FlashAttention-2) | 3.3 | 3.4 | 10.0 | 102.4 | 1 486.0 |
| Bloque de Mamba-2 (`mamba_ssm`, sin `causal-conv1d`) | 7.7 | 9.9 | 11.1 | 34.8 | 174.3 |
| RWKV-7 (`flash-linear-attention`) | 11.6 | 12.0 | 35.0 | 131.1 | 519.4 |
| Memoria pico, ASRN / T++ / Mamba-2 / RWKV-7 (GB) | 0.04 / 0.02 / 0.05 / 0.08 | 0.17 / 0.09 / 0.16 / 0.33 | 0.68 / 0.34 / 0.55 / 1.32 | 2.7 / 1.4 / 2.1 / 5.3 | 10.9 / 5.5 / 8.5 / 21.1 |

| Modelo de ~50 M, 32 768 piezas por paso (piezas/s; GB pico) | 1 024 | 4 096 | 16 384 | 32 768 |
|---|---|---|---|---|
| ASRN (50.7 M) | 172 065 (23.5) | 171 275 (23.5) | 163 503 (23.5) | 163 023 (23.5) |
| ASRN, compilado por partes (`compilar_partes`) | 210 413 (22.3) | 207 945 (22.3) | 200 800 (22.3) | 198 826 (22.3) |
| Transformer++ (51.0 M) | 182 116 (25.5) | 160 772 (25.5) | 110 342 (25.5) | 78 400 (25.6) |
| Transformer++, compilado (`torch.compile`) | 262 909 (22.9) | 220 468 (22.9) | 135 614 (22.9) | 90 540 (22.9) |
| Mamba-2 (51.2 M) | 118 002 (26.4) | 118 314 (26.4) | 116 935 (26.4) | 114 544 (26.4) |
| RWKV-7 (51.1 M) | 50 439 (30.5) | 50 082 (30.5) | 48 397 (30.5) | 45 895 (30.5) |

Lo que muestran las tablas:
- **Capa sola.** Hasta 16 mil piezas ASRN queda en un piso plano de ~12 ms. Ahí es de las más lentas, y
  FlashAttention-2 es 3.7 veces más rápida con 1 y 4 mil piezas. Pasa a RWKV-7 desde unas 4.2 mil piezas, a la atención
  desde unas 17.8 mil y al bloque de Mamba-2 desde unas 20.2 mil (interpolación lineal en L). Con 262 144 piezas es 21.6
  veces más rápida que la atención, 2.5 veces que el bloque de Mamba-2 y 7.6 veces que RWKV-7. Usa 2.0 y 1.3 veces la
  memoria de los dos primeros y cerca de la mitad que RWKV-7 (~40 KiB por pieza, contra 20, 31 y 79).
- **Modelo entero, sin compilar.** ASRN pierde 5.3% de velocidad de 1 024 a 32 768 piezas, contra 57.0% del
  Transformer++. Lo pasa desde unas 2.4 mil piezas (el Transformer++ es 6% más rápido con 1 024 y ASRN 2.08 veces más
  rápido con 32 768). Es 1.4–1.5 veces más rápido que Mamba-2 a todo largo y 3.4–3.6 veces más rápido que RWKV-7, y
  necesita 7.7%, 10.9% y 22.9% menos memoria pico que el Transformer++, Mamba-2 y RWKV-7.
- **Modelo entero, compilado.** Compilar ASRN por partes sube su velocidad 1.21–1.23 veces a todo largo; `torch.compile`
  sube la del Transformer++ 1.44 veces con 1 024 y 1.15 veces con 32 768. Compilado contra compilado, ASRN tiene 0.80
  veces la velocidad del Transformer++ con 1 024 y 0.94 veces con 4 096, lo pasa desde unas 5.4 mil piezas y es 2.20
  veces más rápido con 32 768. El primer paso, con la compilación incluida, tardó 47.0 s en ASRN y 26.2 s en el
  Transformer++.

Salvedades que cambian la lectura de estos números:
- **Solo están compiladas las dos filas compiladas.** ASRN compilado por partes y el Transformer++ con `torch.compile`
  son filas aparte de la tabla del modelo. Todo lo demás (capa sola, las filas de entrenamiento sin compilar, escritura,
  calidad, ablación, experimento de citas) corrió sin compilar, y Mamba-2 y RWKV-7 nunca se compilaron.
- **Mamba-2 corrió sin `causal-conv1d` y sin compilar.** Usó su camino sin fusionar, así que sus cifras son pesimistas
  y cada ventaja de ASRN sobre Mamba-2 es una cota superior. RWKV-7 corrió con sus kernels de
  `flash-linear-attention`.
- **Las filas de capa sola son unidades distintas.** La de ASRN es solo su capa de mezcla (0.65 M de parámetros). La
  del Transformer++ es la atención con su RMSNorm (1.05 M, sin la SwiGLU). La de Mamba-2 es su bloque entero (1.72 M,
  2.6 veces la capa ASRN). Con modelos enteros, la razón entre ASRN y Mamba-2 es 1.4–1.5.
- **El Transformer++ es nuestra implementación.** Sigue la receta de Llama con FlashAttention-2 por SDPA, pero RMSNorm,
  RoPE y SwiGLU son operaciones sueltas de PyTorch, sin fusionar.
- **Una corrida por punto.** Cada tiempo es la mediana de 2 a 5 repeticiones dentro de esa corrida.

El piso de ~12 ms es, muy probablemente, el costo de lanzar muchas operaciones chicas (esta corrida no se perfiló).
Antes en el proyecto, grabar el paso como CUDA Graph (`PasoGrafado`) bajó una capa con 1 024 piezas de 11.4 a 1.6 ms
(A100 de 40 GB). La medición de velocidad del ciclo de v89 en una misma sesión (A100 de 40 GB, PyTorch 2.14.1, modelo
del núcleo de 50 M, 1 024 × 32 piezas) muestra la aceleración completa: 44 879 piezas/s antes del trabajo de
velocidad, 147 548 con los kernels (una capa ASRN por bloque), 171 235 con la receta actual (una cada dos bloques) y
214 536 con `compilar_partes`. El Transformer compilado llegó a 271 582 en esa sesión.

---

## Estabilidad

La capa estable (`asrn/estable.py`) hereda de la capa del núcleo, con los mismos parámetros y los mismos nombres.
Reescribe la ida con las protecciones de abajo. En el régimen normal da los mismos números que v89 salvo el redondeo
(1.3·10⁻⁷ en los logits).

### Cada problema de v89 corregido, con lo medido antes y después

| Problema en v89 | Antes (v89) | Después (capa estable) | Protección |
|---|---|---|---|
| **Fuga causal por la máscara de −10⁹.** La máscara es un puntaje de −10⁹, no −∞. Si todos los puntajes válidos quedan debajo (un núcleo de tiempo enorme), el softmax manda su masa a llaves no válidas, que incluyen la propia pregunta (que copia la pieza *siguiente*) y piezas posteriores | El futuro cambia el pasado: 0.016 con γ = 10⁹ y 0.075 con γ = 10¹¹ (CPU, igual en PyTorch y en el kernel); 0.016 con el kernel compilado en la A100 | 0 exacto, en CPU y en la A100. También en la escritura paso a paso | `mascara_exacta` |
| **Reloj de 32 bits: pierde precisión y se detiene.** B_t = Σβ es una suma acumulada en 32 bits, y su mantisa limita el paso (0.06 con B = 2²⁰, 1 con 2²⁴) | Error de B_t − B_{t−1} con el `cumsum` de CUDA en la A100: 2.2% con 2²⁰ (1 M), 8.6% con 2²² (4 M), 50% con 2²⁴ (16 M) y 100% con 2²⁵. El reloj se detiene en el 0.03%, 1.1%, 16% y 37% de los pasos | Acumulado en 64 bits. En la prueba con 2²⁵ piezas cada paso avanza β (error relativo < 10⁻⁶), mientras que la suma en 32 bits se detiene en más del 20% de los últimos 2²² pasos | `reloj_64` |
| **Cancelación en el puntaje.** v89 calcula (log α_k + tasa·B_k) − tasa·B_q: dos números grandes que se restan después de sumarles el puntaje. Solo deberían importar las diferencias del reloj | Cambio de la salida con el reloj corrido (A100, sin TF32): 9.4·10⁻⁵ con 10⁶, 2.8·10⁻³ con 1.6·10⁷, 8.8·10⁻² con 10⁹. En CPU: 1.2·10⁻⁴, 2.8·10⁻³, 7.8·10⁻² | 5.5·10⁻⁸, idéntico con los tres corrimientos, en CPU y en la A100 sin TF32. Es una diferencia de redondeo fija, no una deriva. Con TF32 encendido daba 3.55·10⁻⁶, también constante: TF32 en las multiplicaciones que siguen al término de tiempo | `reloj_64` (la diferencia se toma en 64 bits antes de multiplicar, `_TiempoRelativo`; alta y baja en el kernel) |
| **β sin techo → NaN.** β = softplus(·) no tiene tope | Con pesos rotos, B pasa de 3.4·10³⁸ en pocas piezas y el núcleo de tiempo da ∞ − ∞ = NaN en toda la capa | La salida y el gradiente quedan finitos | `beta_max = 10⁴` |
| **Proyección del reloj en 16 bits.** Para ahorrar una lectura, v89 calcula w·ĥ en la misma multiplicación que Qc/Kc, con autocast | En bf16, β tiene unas 3 cifras y el error se acumula en B: 0.39, 0.65, 0.93 y 1.27 unidades tras 512, 1 024, 2 048 y 4 096 piezas (d = 64); 3.3 tras 4 096 (d = 512). En fp16 la suma puede desbordar (deducido, no medido). En v89 figuraba como pendiente | La proyección va en 32 bits. La prueba exige que el error de B sea al menos 10 veces menor que por el camino de 16 bits | `reloj_64` |
| **TF32 en la proyección del reloj.** `entrenar()` y el cuaderno encienden `allow_tf32`, y entonces una multiplicación "de 32 bits" usa 10 bits de mantisa | B_t en fp32 cambia hasta 2.0 unidades en 65 536 piezas (d = 512, A100) al encender TF32 | 1.2·10⁻¹⁰: en GPU el reloj se proyecta con `_ProyeccionIEEE` (multiplicar y sumar elemento a elemento, ~2d cuentas por pieza) | `reloj_64` |

Cuándo importa: el reloj se degrada desde ~1 millón de piezas y se detiene desde ~16 millones. Cuánto cambia eso la
salida depende de γ. En una capa con 2²⁰ piezas (solo ida, d = 64, bf16), las salidas de v89 y de la estable quedaron a
8.2·10⁻⁴ (menos de 0.1%). Esa cifra mezcla el reloj, la proyección en bf16 de v89, la máscara y el redondeo, y no
aísla la deriva.

### Grupos de protecciones

`mascara_exacta`, `reloj_64` y `beta_max` responden a fallas medidas. `log_alfa_min`, `mejor_max`, `tope_puerta` y
`kappa_min` son **preventivas**: la falla que evitan se deduce del código y no se midió. `log_alfa_min` y `mejor_max`
no tienen prueba propia.

| Protección | Grupo | Falla que evita | Costo |
|---|---|---|---|
| `mascara_exacta` | A (encendida) | Fuga causal (PyTorch, kernel y escritura) | Ninguno: los mismos bits en el régimen normal |
| `reloj_64` | A (encendida) | Precisión del reloj, cancelación, reloj detenido, proyección en 16 bits y en TF32 | 8 bytes por pieza para B en vez de 4; +4 B leídos por pregunta y llave en el kernel; 2 columnas de un producto en 32 bits de verdad |
| `beta_max = 10⁴` | A (encendida) | Reloj infinito → NaN | En la práctica, ninguno (muy por encima de lo normal) |
| `log_alfa_min = −30` | A (encendida) | log α → −∞, otra forma de bajar de −10⁹ (preventiva) | Ninguno |
| `mejor_max = 10⁴` | A (encendida) | Un −10⁹ que entra a la puerta por la señal `mejor` (preventiva) | Ninguno |
| `tope_puerta = 10` | B (apagada) | Puerta saturada por señales fuera de lo visto al entrenar (preventiva) | Cambia la puerta: logit → c·tanh(logit / c) |
| `kappa_min = 1/(4·L_entr)` | B (apagada) | Con κ → 0, λ crece con el largo del texto (preventiva; en la sonda de gradientes, log(1 + λ) quedó plano de 256 a 65 536 piezas) | Cambia Hawkes: memoria máxima acotada |

El grupo A está encendido por omisión y no cambia la función. El grupo B sí la cambia. Está apagado por omisión, así
que los modelos guardados dan lo mismo, y está pensado para entrenar desde cero con textos largos
(`Protecciones.textos_largos(L)`).

**Lo que no se agregó a propósito.** El softmax por tabla ya es estable, porque resta el máximo en 32 bits. LayerNorm
corre en 32 bits con ε = 10⁻⁵. Tampoco hizo falta proteger el gradiente del reloj: la hipótesis de que crece con el
largo se puso a prueba y se descartó (normas planas de 256 a 65 536 piezas). No hay ningún `nan_to_num` dentro de la
capa, para que `entrenar()` pueda seguir detectando un NaN y saltar el paso.

### `RecorteDinamico`

`RecorteDinamico` recorta el gradiente por grupos, con un umbral que se adapta.
- **Grupos:** el reloj y las tasas, la puerta, y el resto. Cada grupo lleva una media móvil exponencial de su norma y
  de su cuadrado.
- **Umbral:** min(`tope`, μ + z·σ) después de un calentamiento (por omisión `tope=1.0`, `z=3.0`, `beta=0.98`,
  `calentar=50`). Solo se escala el grupo que lo pasa.
- **Medias:** se actualizan con la norma ya recortada, para que un pico no suba el umbral siguiente, y nunca con un
  NaN. No se sincroniza nada con la GPU.
- **Por qué:** con un solo recorte global, un pico en unos pocos números (el reloj, γ) achica en ese paso el gradiente
  de todo el modelo.
- **Probado:** con un pico de 10⁴ en el reloj, solo se recorta el grupo del reloj, y el grupo "resto" queda igual bit a
  bit. El grupo de la puerta no se comprueba.
- **Sin probar:** con FSDP. Su historia no se guarda en los archivos, así que vuelve a calentar al reanudar.

### Costos

- **Tiempo y memoria (A100, grupo A, con `_ProyeccionIEEE`).** El tiempo es el de v89 con 1 024 piezas (12.1 contra
  12.0 ms), +4–7% de 4 096 a 65 536 piezas y +7.5% con 262 144 (68.7 contra 64.0 ms). La memoria es +4–5% a todo largo
  (+5.2% con 262 144), porque ĥ se guarda en 32 bits para el reloj.
- **Bytes guardados por pieza.** 43.9 contra 43.8 KiB por el camino de PyTorch, y 20.2 contra 20.1 KiB con el kernel.
- **Una capa con 2²⁰ piezas** (solo ida, d = 64, bf16). v89: 41.7 ms y 2.185 GB. La estable: 33.3 ms y 2.198 GB, es
  decir +12 bytes por pieza. Son mediciones únicas, y las diferencias de tiempo son ruido (las ejecuciones anteriores
  dieron 35.2 ms para v89 y 32.4–36.3 ms para la estable).
- **El grupo B** no tuvo un costo ni un efecto consistentes en la ablación (copia y MQAR, 3 semillas): −3% y +5% por
  corrida, la copia aprendida en 2 de 3 semillas como en la base, 99.29% en MQAR con 4× los pares.

---

## Problemas resueltos de v89 y del ciclo de desarrollo

**En la capa (de v89 a 1.0).** Los seis problemas numéricos de la tabla de Estabilidad: la fuga causal, la precisión
del reloj de 32 bits y su detención, la cancelación en el puntaje, β sin techo, la proyección del reloj en 16 bits
(que la revisión de v89 dejó como "sigue abierto") y TF32 en la proyección del reloj.

**Encontrados y corregidos al armar el banco de 1.0:**

| Problema | Efecto | Arreglo |
|---|---|---|
| `FlopCounterMode` cuenta mal la vuelta de una convolución por grupos (la de Mamba-2 es por canales, 1 152 grupos) | Le asignaba 10.7 MFLOP por pieza a esa convolución, unas 1 000 veces lo real. La primera tabla de costo le daba a Mamba-2 22.1 MFLOP por pieza | `banco_asrn.py` usa una fórmula que tiene en cuenta los grupos (con una convolución densa da lo mismo que PyTorch). Mamba-2: **22.1 → 11.48 MFLOP por pieza**; las demás filas no cambian |
| El envoltorio de RWKV-7 armaba un modelo de una capa, y `flash-linear-attention` divide por (capas − 1) al inicializar | Un `ZeroDivisionError` detuvo a RWKV-7 en todas las secciones de la ejecución del 9 de octubre | RWKV-7 se arma con al menos 2 capas (la cifra de capa sola es la primera de un modelo de dos). La ejecución del 11 de octubre agregó RWKV-7 en las seis secciones |
| La variante `mezcla_lse` de la ablación (experimental, no forma parte de 1.0) | La pérdida se volvió NaN en MQAR desde el paso 400 en la ejecución anterior, por un error de implementación en esa variante | La máscara se aplica antes de `exp` y no hay softmax sobre filas sin candidatos. La fila de MQAR de la ejecución anterior (0.0% en `history/results/banco_a100_pt211/`) no cuenta. Repetida con el arreglo el 17 de octubre, ya no diverge pero no aprende MQAR (3.61 / 1.08 / 3.15% con 4× los pares; pérdida final 8.06–8.28) |
| El prefijo del Transformer++ en la medición de escritura | En la ejecución anterior el envoltorio calculaba los logits de todas las posiciones del prefijo (65 536 × 32 768 logits, 4.3 GB en bf16), y los otros modelos solo los de la última. Eso inflaba su tiempo de prefijo y su memoria al escribir; su tiempo por pieza no cambiaba | Ahora solo se calcula la última posición. Medido de nuevo el 17 de octubre con 65 536 piezas: prefijo 0.387 s, pico al escribir 2.53 GB |
| `test_convertir_guardar_cargar` en la A100 | Fallaba: el modelo cargado iba a la GPU y se comparaba en la CPU (un error de la prueba, no de la biblioteca) | La prueba carga en `'cpu'` |
| La proyección del reloj de la capa estable podía pasar por TF32 (primera ejecución en la A100) | La protección era más débil de lo previsto en GPU | `_ProyeccionIEEE` (1.2·10⁻¹⁰ desde la segunda ejecución) |
| El 3.55·10⁻⁶ constante de la capa estable con el reloj corrido no tenía una causa confirmada | Se podía confundir con una deriva del reloj | Las pruebas numéricas corren ahora sin TF32. La segunda ejecución y la del 17 de octubre dieron 5.5·10⁻⁸, constante, lo que confirma que era TF32 en las multiplicaciones que siguen al término de tiempo |
| `test_entrenar_aprende_a_copiar_y_reanuda` en la máquina de la A100 de Colab (PyTorch 2.11) | Falló desde la quinta corrida en GPU (código del núcleo `d9927a1bba79`; con 2.11 pasó en las corridas 2 a 4): pérdida de copia 4.19 tras 600 pasos, con un umbral de 1.5. La prueba entrena en la CPU, y con su tasa de aprendizaje de 3e-3, salir de la meseta inicial de la copia (ln 63 ≈ 4.14) en 600 pasos dependía del orden de las sumas en coma flotante (número de hilos, instrucciones de MKL, versión de PyTorch). Se reprodujo fuera de Colab con PyTorch 2.11 con 6 y 12 hilos de CPU (4.19–4.20); con 2.14.1, la semilla 6 del modelo también se quedó en la meseta (4.22–4.23), incluso con 1 000 pasos (4.28) | La prueba entrena con 6e-3, con los mismos pasos, lote, umbral y comprobación de reanudación. Salió de la meseta antes del paso 250 en todos los casos probados: 22 corridas, semillas del modelo 0–7 con 4 hilos y 0–3 con 12 hilos con PyTorch 2.14.1; semillas 0–3 con 12 hilos, 0–3 con 6 hilos con MKL en AVX-512 y 0–1 con 12 hilos con ATen y MKL en AVX2 con PyTorch 2.11.0; pérdida de copia final 0.118–0.159. La prueba corregida pasa en la máquina de la A100 de Colab (ejecución del 17 de octubre, PyTorch 2.14.1) |

**Ya corregidos durante el ciclo de v89 (forman parte del núcleo en 1.0):**
- NumPy es ahora una dependencia declarada, necesaria para guardar con el modelo repartido. La versión mínima de
  PyTorch es 2.9: con 2.6 FSDP falla, y con 2.7–2.8 la norma del gradiente repartido sale el doble con gloo.
- Reanudar en fp16 un entrenamiento hecho en bf16 o fp32 fallaba al cargar el escalador de gradientes. Está corregido.
- Están corregidos los casos límite de `escribir()`: `top_p=0`, `no_repetir_ngramas` cuando prohíbe todas las piezas y
  piezas fuera del vocabulario.
- Varias configuraciones que fallaban en silencio ahora avisan o dan un error claro, entre ellas `torchrun` sin
  `--fsdp` y los objetivos imposibles de `para_parametros`.
- FSDP arma su malla en la CPU cuando usa gloo.
- Las comprobaciones de `torch.compile` se saltan antes de PyTorch 2.14.
- `_respuesta_racha` convertía un NaN en un índice fuera de rango. Ahora el NaN sigue de largo, así que `entrenar()`
  puede saltar el paso.
- `found_inf` tenía la forma equivocada en `PasoGrafado`. Está corregido.

---

## Cambios incompatibles y migración

**Los módulos pasaron a un paquete, y las carpetas cambiaron de nombre.** Los documentos archivados en `history/` usan
las rutas viejas, y [`history/README.md`](../history/README.md) trae la correspondencia completa.

| Antes (v89 y la organización previa al lanzamiento) | 1.0 |
|---|---|
| `import asrn_v89 as asrn` | `import asrn` (capa estable por omisión) o `import asrn.nucleo as asrn` (el núcleo de v89 tal cual) |
| `import asrn_estable as estable` | `import asrn.estable as estable`, o `from asrn import CapaASRNEstable, Protecciones, RecorteDinamico, convertir` |
| `estable.convertir(asrn_v89.crear_modelo(...))` | `asrn.crear_modelo(...)` (ya convertido) |
| `python asrn_v89.py --datos ...` | `python -m asrn --datos ...` o `asrn-train --datos ...` (capa estable); `python -m asrn.nucleo ...` (capa del núcleo) |
| `torchrun --nproc_per_node=N asrn_v89.py --fsdp ...` | `torchrun --nproc_per_node=N -m asrn --fsdp ...` |
| `asrn_v89.py`, `asrn_estable.py` | `asrn/nucleo.py`, `asrn/estable.py` |
| `pruebas/test_asrn_v89.py`, `test_asrn_estable.py`, `verificar_gpu_v89.py` | `tests/test_nucleo.py`, `tests/test_estable.py`, `tests/verificar_gpu.py` |
| `pruebas/versiones_anteriores/` (`asrn_v85.py`, `asrn_v87.py`) | Quitada: las pruebas de 1.0 ya no comparan con v85/v87 |
| `experimentos/` (programas y cuadernos del banco), `experimentos/fuentes/`, `resultados/banco_a100/`, `resultados/banco_cpu/` | `benchmarks/`, `benchmarks/builders/`, `results/banco_a100/`, `results/banco_cpu/` |
| `experimentos/experimento_cita_v87.py` | Quitado; el experimento pasó a la capa estable de 1.0 en `benchmarks/experimento_citas.py` |
| `experimentos/sondas/`, `medir_velocidad_*` | `history/probes/`, `history/experiments/` |
| `resultados/corrida_*`, `bloques_1`, `velocidad` | `history/results/` |
| `docs/*` (whitepaper, RENDIMIENTO, ESTABILIDAD_Y_BANCO) | `history/docs/` |
| `archive_v89/`, `entrenar_100m/`, `articulo/` | `history/`, `examples/train_100m/`, `paper/` |
| `ARCHITECTURE*.md`, `ASRN_Technical_Report*.md`, `RELEASE_NOTES*.md` en la raíz | `docs/ARCHITECTURE*.md`, `docs/TECHNICAL_REPORT*.md`, `docs/RELEASE_NOTES*.md` |

**Las cuentas por omisión cambiaron un poco.** `import asrn` da ahora la capa estable. Sus salidas difieren de las de
v89 solo por el redondeo (1.3·10⁻⁷ en los logits). Para tener exactamente el comportamiento de v89, usa cualquiera de
estas opciones:
- `import asrn.nucleo as asrn`, o `python -m asrn.nucleo` en la línea de comandos (con esta forma Python muestra un
  `RuntimeWarning` de `runpy`, porque el paquete ya importó `asrn.nucleo`; el entrenamiento corre normalmente).
- `asrn.crear_modelo(..., protecciones=asrn.Protecciones.ninguna())`, o `asrn.convertir(modelo,
  asrn.Protecciones.ninguna())`. Da logits y gradientes iguales bit a bit (prueba en CPU).

**Los archivos guardados cargan sin cambios.** `guardar()` escribe un diccionario simple,
`{'config': dict, 'pesos': state_dict}`, así que no se guarda ninguna ruta de clase ni de módulo. La capa del núcleo y
la estable tienen los mismos nombres de parámetros, así que los archivos cruzan en los dos sentidos (probado: guardado
por el núcleo y cargado por el paquete, y al revés). `cargar()` sigue aceptando archivos de v83 en adelante; las
pruebas de 1.0 no cubren la carga de archivos de versiones anteriores.

**Otros cambios.**
- Escribir paso a paso usa la escritura O(1) (ver [Escritura en O(1)](#escritura-en-o1)): las mismas secuencias de
  piezas que la escritura O(t) anterior y logits a menos de 10⁻⁴, más los anillos fijos en memoria (17 MiB por capa
  ASRN y por secuencia con los tamaños por omisión).
- Al escribir paso a paso, la capa estable guarda el reloj en 64 bits (8 bytes por pieza y capa en vez de 4).
- Las opciones que se quitaron en la limpieza de octubre de 2026 del ciclo de v89 (`partir_valores`, la convolución
  por canal sin memoria, la capa densa con puerta) no están en 1.0.
- `largo_cada=2` (una capa ASRN cada dos bloques) es la forma por omisión desde ese ciclo. `largo_cada=1` da la forma de
  antes, y los modelos guardados cargan con la suya.

---

## Problemas conocidos y limitaciones

**Escritura en la A100: la escritura O(1) todavía no se midió ahí.**
- En la A100 (modelo de ~50 M, lote 1), escribir fue lo más lento de los cuatro modelos: plano en 42.82–43.45 ms por
  pieza con 1 a 64 mil piezas de contexto (pendiente de 3.0 ns por pieza de contexto, IC del 95% de −25 a 31 ns), 2.9
  veces el Transformer++ (14.44–15.23 ms), 1.7 veces Mamba-2 (24.47–25.28 ms) y 1.2 veces RWKV-7 (34.00–39.51 ms).
  Estas cifras se midieron con la escritura O(t) anterior. El tiempo por pieza ya era plano ahí (lo más probable, por
  el costo de lanzar kernels; no se perfiló), así que el término O(t) quedaba oculto.
- Con la escritura O(1) el costo por pieza es constante con el contexto en CPU (ver
  [Escritura en O(1)](#escritura-en-o1)), pero la cifra de la A100 está pendiente: no se afirma ninguna aceleración de
  la escritura O(1) en la A100. La sección de escritura del cuaderno corre ahora la escritura O(1), así que la próxima
  ejecución en Colab la mide.
- La memoria por secuencia sigue creciendo con el texto, más los anillos fijos (17 MiB por capa ASRN y por secuencia
  con los tamaños por omisión). Con la escritura O(t), el pico en la A100 con 65 536 piezas fue de 3.82 GB, contra
  2.53 GB del Transformer++, 1.26 GB de Mamba-2 y 3.69 GB de RWKV-7. Estas cifras incluyen el paso del prefijo.

**Con textos cortos manda el costo de lanzar kernels.**
- Una capa sola tiene un piso de ~12 ms hasta 16 mil piezas.
- En el modelo entero con 1 024 piezas, el Transformer++ es 6% más rápido, y 25% más rápido con los dos compilados.
- CUDA Graphs (`PasoGrafado`) bajaron una capa con 1 024 piezas de 11.4 a 1.6 ms. `PasoGrafado` no está dentro de
  `entrenar()`. Sus límites: una GPU, bf16 o fp32, sin checkpoint, sin FSDP ni `torch.compile`, sin acumulación y con
  una forma de lote fija.

**`torch.compile` necesita PyTorch ≥ 2.14.** Con versiones anteriores, Inductor falla con este modelo: C++ inválido en
CPU y un error interno de Triton en una A100 con 2.11 (la ejecución anterior). `entrenar()` avisa, y con
`compilar='partes'` sigue sin compilar. Con 2.14.1 en la A100 pasa la comprobación de `torch.compile` en GPU (logits a
4.8·10⁻⁴, gradientes a 1.2·10⁻⁷), y la capa estable compilada por partes entrena 1.21–1.23 veces más rápido que sin
compilar. Solo están compiladas las dos filas compiladas de la tabla de entrenamiento; todas las demás cifras de la
A100 en `results/banco_a100/` son sin compilar.

**El aprendizaje de la copia empieza en una meseta, y salir de ella no está garantizado.** Un modelo se queda primero en
la meseta inicial de la copia (ln 511 ≈ 6.24 en la tarea de copia del banco, ln 63 ≈ 4.14 en la prueba del núcleo) y
sale de ella en un paso que depende de la semilla, de la tasa de aprendizaje y hasta del orden de las sumas en coma
flotante. En la A100 con 10⁻³, ASRN salió en 2 de 3 semillas: la semilla 1 se quedó en ella los 2 000 pasos (pérdida
final 6.232; 0.15% con 4×), mientras que las semillas 0 y 2 copiaron con 4× a 99.99% y 99.98% (con 3·10⁻³, la semilla 0
salió en el paso 400). En la ejecución anterior aprendieron las 3 semillas, así que la base salió de la meseta en 5 de 6
corridas entre las dos. En la ablación, 5 de las 24 corridas de las variantes livianas (la base, `estable_AB`,
`sin_puerta`, `puerta_sin_senales`, `sin_reloj`, `sin_hawkes_racha`, `sin_sufijo` y `sin_confirmacion`, 3 semillas cada
una) se quedaron en ella (3 de 24 en la ejecución anterior), y las corridas que aprendieron van de 4.88% a 97.35% con 16
veces el largo entrenado. Es un problema de optimización (salir de la meseta), no de capacidad. La misma sensibilidad
hizo fallar la prueba de copia del núcleo en la máquina de Colab (ver Problemas resueltos).

**Las pruebas de los kernels en CPU con Triton 3.6 necesitan NumPy < 2.4.** En CPU, las pruebas corren los kernels
Triton en el intérprete de Triton. El intérprete de Triton 3.6 (el que va con PyTorch 2.11) convierte arreglos de un
elemento en enteros de Python, y NumPy 2.4 convirtió eso en un error, así que ahí fallan tres pruebas de kernels con
NumPy ≥ 2.4. Triton 3.8 (PyTorch 2.14) y las corridas en GPU no están afectados.

**Notas de método del banco.** Describen qué se midió:
- **Prefijo del Transformer++ al escribir: corregido y medido de nuevo.** En la ejecución anterior el envoltorio
  calculaba los logits de las 65 536 posiciones del prefijo (4.3 GB en bf16), y en los otros modelos solo los de la
  última. Con el envoltorio corregido, el prefijo con 65 536 piezas tarda 0.387 s y la escritura llega a un pico de
  2.53 GB (ASRN 0.140 s y 3.82 GB, Mamba-2 0.139 s y 1.26 GB). Su tiempo por pieza nunca cambió. El prefijo de RWKV-7
  (110.9–130.9 s a todo largo) lo dominan la compilación y el ajuste automático de sus kernels de fla.
- **`mezcla_lse` en MQAR.** En la ejecución anterior la variante experimental divergió (NaN desde el paso 400) por un
  error de implementación en esa variante. Con la implementación corregida ya no diverge, pero tampoco aprende MQAR
  (≤ 3.61% con 4× los pares). No forma parte de 1.0.
- **Tasas de aprendizaje.** La tasa de cada modelo es la mejor de una grilla de dos (10⁻³ y 3·10⁻³) con la semilla 0, y
  los empates van a la tasa más baja. En WikiText-103, el Transformer++ eligió el borde inferior de la grilla (10⁻³) y
  los otros tres modelos el borde superior (3·10⁻³). La tasa de ASRN en MQAR fue 3·10⁻³ en esta ejecución y 10⁻³ en la
  anterior.

**El resultado de RWKV-7 en texto sigue sin resolver.** Aquí RWKV-7 (el `RWKV7Model` original) llegó a una perplejidad
de 75.8 en WikiText-103, contra 67.9 de ASRN (72.4 contra 67.8 en la ejecución anterior). ASRN quedó por delante en
cada semilla (por 6.26, 3.43 y 14.02), pero la diferencia no es significativa (p pareado = 0.13 con 1 024 piezas,
0.084 con 4 096). RWKV-7 varía mucho entre semillas, y su tasa elegida está en el borde de la grilla. En la quinta
corrida de v89, con otro protocolo (30 M de piezas) y una reimplementación de RWKV-7, RWKV-7 fue el mejor, con 47.87
contra 57.02. Ninguno de los dos resultados dice cuál refleja mejor a RWKV-7, así que **no se puede
afirmar que ASRN supere a RWKV-7 en texto en general**.

**Alcance del banco.** Estas condiciones limitan cada afirmación de calidad:
- **Escala chica.** Los modelos tienen 2.5–46 M de parámetros. El entrenamiento en WikiText-103 usó 20 M de
  posiciones, unas 0.44 por parámetro, lejos de converger. Las curvas seguían bajando, y la diferencia de pérdida entre
  ASRN y Mamba-2 se achicó durante el entrenamiento.
- **Búsqueda chica.** Solo dos tasas (10⁻³ y 3·10⁻³) y 3 semillas, y no hay un conjunto aparte para elegir la tasa.
  Con 10⁻³, Mamba-2 queda por delante de ASRN en WikiText (86.1 contra 89.6, una semilla).
- **Un solo corpus de texto,** y premia copiar. Las perplejidades son por pieza sobre el conjunto de validación, así
  que no se comparan con las cifras publicadas por palabra.
- **Todos los modelos atan su salida a los vectores de pieza,** también RWKV-7, cuya receta original no la ata.
- **ASRN trae incorporada la coincidencia exacta de números de pieza** (sufijo, vistos, tablas de palabra). Ese sesgo
  inductivo coincide con copia y MQAR, y este MQAR está en el techo para ASRN.
- **Algunos rivales no aprendieron algunas tareas con este presupuesto.** Mamba-2 no aprendió la copia, RWKV-7 no
  terminó de aprenderla, el Transformer++ no aprendió MQAR y RWKV-7 no convergió en MQAR en 2 de 3 semillas. Son
  efectos del presupuesto o del protocolo, no evidencia contra esas arquitecturas.
- **El MQAR de los rivales varía entre ejecuciones.** Con las mismas semillas, el mismo protocolo y las mismas
  bibliotecas de los rivales, pero otra versión de PyTorch, Mamba-2 con 4× los pares pasó de 54.39 ± 5.89 en la
  ejecución anterior a 26.09 ± 25.55, y RWKV-7 con el largo entrenado de 99.90 ± 0.17 a 65.43 ± 31.76 (lo más probable,
  kernels de GPU no deterministas más entrenamientos sin converger). El MQAR de ASRN fue estable entre ejecuciones.
  Frente a Mamba-2, la ventaja de ASRN en MQAR solo es significativa con 4× los pares (p pareado = 0.039; 0.070 y 0.090
  con 1× y 2×).
- **Dentro del largo entrenado, el Transformer++ aprende más rápido.** Aprende la copia en ≤ 200 pasos, contra 1 000
  de ASRN con la misma tasa cuando ASRN sale de la meseta (1 000–1 200 en la ejecución anterior). Cada corrida de ASRN
  tarda 1.2–1.9 veces más. Nada se comparó a igual cómputo.

**Lo que dice la evidencia de cada componente.** La capa separa dos responsabilidades de diseño. La *recuperación*
decide dónde copiar: la atención dispersa por casilleros (tablas de hash aprendidas, ventanas de candidatos y puntajes
con el sufijo exacto, la confirmación y el término de tiempo) hace la recuperación exacta. La *regulación* decide si
copiar y cuándo parar: la puerta y sus señales, el reloj de tiempo propio y las señales de Hawkes y de racha regulan la
copia al escribir libremente. La ablación corrió en la A100 con 3 semillas y las tasas de la base (10⁻³ en copia,
3·10⁻³ en MQAR; la ejecución anterior usó 10⁻³ en las dos), y su fila base es la propia corrida de calidad.
- **Núcleo de recuperación: validado en copia y MQAR.** Sin la capa ASRN (`solo_local`) las dos tareas quedan en el
  azar en 3 de 3 semillas (por construcción: los bloques locales solo ven las últimas 13 piezas). Con ella, ASRN llega a
  99.99% y 99.98% en copia con 4× el largo entrenado (2 de 3 semillas: la semilla 1 se quedó en la meseta, 0.15%; media
  de las 3 semillas 66.71 ± 57.64; ejecución anterior 99.87 ± 0.18, 3 de 3) y a 99.75% ± 0.43 en MQAR con 4× los pares.
- **Mecanismos de regulación: no interfieren en copia ni en MQAR.** En estas tareas el 100% de las posiciones evaluadas
  pide copiar, así que el núcleo de recuperación basta por sí solo. La presencia de los mecanismos de regulación no
  degrada la recuperación exacta, y quitarlos de a uno no la baja más allá de la variación entre semillas en las
  corridas que aprendieron: entre las corridas de copia que salieron de la meseta, la más baja con 4× es 99.89% en la
  base y las variantes de regulación (`estable_AB`, semilla 0). La semilla 1 es la difícil: se quedó en la meseta los
  2 000 pasos en la base misma, y también sin la puerta (`sin_puerta`), sin el reloj (`sin_reloj`) y con el grupo B
  (`estable_AB`). Como la base misma falló con la semilla 1, una semilla fallida en una variante no se puede atribuir al
  componente quitado. Entre las dos ejecuciones (6 corridas por variante), la copia se aprendió en 5 de 6 corridas de
  la base, 4 de 6 sin la puerta o sin el reloj, y 6 de 6 con la puerta pero sin sus señales y sin Hawkes ni la racha.
  Las medias de MQAR con 4× los pares son 99.75% la base, 99.94% sin la puerta, 99.50% con la puerta pero sin sus
  señales (`puerta_sin_senales`), 99.95% sin el reloj y 99.53% sin Hawkes ni la racha (`sin_hawkes_racha`); todas las
  variantes salvo `solo_local` y `mezcla_lse` dan ≥ 97.92% en cada semilla.
- **Mecanismos de regulación: validados al escribir libremente, en condiciones controladas.** En el experimento
  sintético controlado de citas de largo acotado (`benchmarks/experimento_citas.py`: piezas al azar con citas de 6 a 12
  piezas de vez en cuando; d = 64, 2 bloques locales y 1 capa ASRN, 6 000 pasos, semillas 0–2;
  [resultados](../results/citas/RESUMEN.md)), la capa completa de 1.0 se compara con la misma capa sin las señales de
  Hawkes y de racha (`sin_hawkes_racha`), con una puerta que solo ve el estado oculto (`puerta_sin_senales`), sin puerta
  (`sin_puerta`) y sin el reloj (`sin_reloj`). Solo con la pérdida del modelo de lenguaje, sin supervisión explícita, la
  puerta de la capa completa aprende un "acantilado": un empujón a copiar cuando empieza una cita (+0.6 a +0.8 con una
  racha de una pieza) y un freno fuerte cuando la racha supera las citas de los datos (−7.7 a −9.2 desde 12 piezas). Así
  frena el *aliasing de estado* (sin la racha, una cita de 5 piezas y una de 50 son el mismo estado). Con el muestreo
  por omisión (T = 0.8, top-k 40), la racha escrita más larga es de 10 piezas (texto real: 10) contra 21, 27 y 23 en las
  tres primeras variantes, y las rachas de más de 9 piezas son el 0.3% de todas (texto real: 0.9%) contra 7.2%, 9.3% y
  11.8%. Quitar solo las señales de Hawkes y de racha casi no cambia la pérdida de validación (3.703–3.706 contra
  3.707–3.708); las variantes sin las señales de la puerta o sin puerta pierden 0.03–0.05. El mismo experimento en la
  A100 ([resultados](../results/citas_a100/RESUMEN.md)) lo replica: la puerta aprende el mismo acantilado (+0.6 a +0.7
  con una racha de una pieza, −7.6 a −8.4 desde 12 piezas); con el muestreo por omisión la racha más larga es de 10
  piezas contra 19, 39 y 21, y las rachas de más de 9 son el 0.3% contra 7.9%, 9.2% y 12.0%; en los cuatro modos de
  muestreo la racha más larga por semilla es 9–12 con la capa completa y 9–10 sin el reloj, contra 14–24, 15–39 y 18–31
  en las tres primeras variantes; y la pérdida de validación es 3.703–3.705 con la capa completa contra 3.706–3.709 sin
  las señales de Hawkes y de racha, 3.735–3.748 sin las señales de la puerta, 3.728–3.731 sin puerta y 3.704–3.705 sin
  el reloj. Límites conocidos: sin el reloj, la capa escribe como la completa (racha más larga 10; 0.2% de rachas de más
  de 9 en CPU, 0.4% en la A100; la misma pérdida), así que el papel propio del reloj, que tiene que ver con cambios de
  tema que esta tarea no tiene, no está probado; cuán seguido empieza una cita no está calibrado (fracción citada 20.9%
  con T = 0.8 y 9.3% con T = 1 en CPU, 21.8% y 10.1% en la A100, contra 15.6% del texto real); y es una sola tarea
  sintética con d = 64.
- **Refinamientos de la recuperación: no se pueden separar en estas tareas saturadas.** El sufijo exacto (`sin_sufijo`,
  MQAR 99.69%), la confirmación (`sin_confirmacion`, 99.88%) y el número de tablas refinan la recuperación; su aporte
  individual no se puede separar en copia y MQAR. Sin el sufijo, la copia falló con la semilla 0, y la semilla 2
  aprendió con 1× (87.02%) pero dio 39.77% con 4× y 4.88% con 16×, la corrida aprendida más baja de la ablación.
  `dos_tablas` (1 + 1 tablas) aprendió la copia en 1 de 3 semillas (2 de 6 entre las dos ejecuciones), lo que lo
  sugiere, sin ser concluyente.
- **No adoptada: `mezcla_lse`.** En la ejecución anterior, mezclar las tablas por log-suma-exp salió de la meseta de la
  copia en el paso 400 en 3 de 3 semillas. Eso no se reprodujo aquí: solo aprendió la semilla 0 (pérdida < 1 desde
  el paso 600), así que aprender antes la copia no es un efecto robusto. Con la implementación corregida no aprende
  MQAR (≤ 3.61% con 4× los pares), y cuesta +62% por corrida en copia y +61% en MQAR (solo existe en PyTorch).
- **Grupo B: apagado por omisión.** No tuvo un costo ni un efecto consistentes en la ablación.
- **Lo que sigue.** La ablación a gran escala de los mecanismos de regulación en corpus naturales con distractores y
  cambios de tema está prevista para v1.1 (ver la [hoja de ruta](#hoja-de-ruta--trabajo-futuro-v11)).

**Combinaciones sin probar y otros límites.**
- FSDP, `PasoGrafado` y las comprobaciones en GPU (incluida la de `torch.compile`) están probados solo con la capa del
  núcleo, no con la estable. `compilar_partes` con la capa estable se cronometró en la A100, pero su coincidencia
  numérica solo se comprueba con la capa del núcleo. `RecorteDinamico` no está probado con FSDP.
- FSDP2 está cubierto por una prueba con dos procesos en CPU (gloo) y una comprobación con un proceso NCCL en la A100;
  todavía no se corrió con varias GPUs.
- Guardar con el modelo repartido junta los pesos y el estado de AdamW completos en el proceso 0 (~12 bytes por
  parámetro).
- `evaluar()` repartido repite la misma cuenta en todos los procesos.
- El comportamiento por encima de ~51 M de parámetros todavía no se midió; el kit de 103.5 M (`examples/train_100m/`)
  está listo, y su primera corrida está en la hoja de ruta.
- La capa usa ~40 KiB por pieza y capa en la A100 (el doble que la atención).

---

## Hoja de ruta / Trabajo futuro (v1.1+)

ASRN 1.0 entrena un modelo de ~50 M de parámetros a 163–172 mil piezas/s (199–210 mil compilado por partes), casi
plano de 1 024 a 32 768 piezas por texto, en una sola A100. Para escalar más allá de 100 M de parámetros, el
lanzamiento trae [`examples/train_100m/`](../examples/train_100m/README.md), un kit de Colab listo para correr para un
modelo de 103.5 M de parámetros con FineWeb, y el reparto con FSDP2 (`repartir_fsdp`), cubierto por una prueba de FSDP
con dos procesos en CPU en `tests/test_nucleo.py` y una comprobación con un proceso NCCL en la A100. Las próximas
versiones parten de ahí:

- **Más semillas para mezclar las tablas por su normalizador logarítmico (`mezcla_lse`)**, para saber si sale antes de
  la meseta de la copia (3 de 3 semillas en el paso 400 en la ejecución anterior; aquí solo 1 de 3, desde el paso 600),
  y un kernel Triton para ella solo si es así.
- **Escritura en la A100:** medir ahí la escritura O(1) y recortar los lanzamientos de kernels por pieza al escribir
  (por ejemplo con CUDA Graphs), que son lo que domina en la GPU.
- **La primera corrida de 100 M de parámetros** con FineWeb usando `examples/train_100m/`.
- **Una ablación a gran escala de los mecanismos de regulación** (la puerta y sus señales, el reloj, Hawkes y la racha)
  en corpus con distractores y cambios de tema.
- **Grillas de tasas de aprendizaje más anchas** (incluido el Transformer++ con 3·10⁻⁴ y 6·10⁻⁴). Repetir `mezcla_lse`
  en MQAR y el prefijo del Transformer++ con el envoltorio corregido ya está hecho (ejecución del 17 de octubre).
- **Corridas compiladas más allá de la velocidad de entrenamiento** (capa sola, escritura, calidad), y Mamba-2
  compilado y con `causal-conv1d`.
- **Propuestas de kernel** del reporte técnico (estimadas a partir del código, no medidas): una ventana deslizante
  persistente (−34% de bytes de los kernels), compartir entre capas el orden de los grupos de piezas, un ordenamiento
  radix de 14 bits, un escaneo paralelo en toda la rejilla, registros empaquetados por posición y ajuste automático.

---

## Verificación

| Dónde | Qué | Resultado |
|---|---|---|
| CPU (4 núcleos, Python 3.13, PyTorch 2.14.1, Triton 3.8 en modo intérprete), desde una copia limpia del commit del lanzamiento, 15 de octubre de 2026 | `python pruebas/test_nucleo.py`; `python pruebas/test_estable.py` (ahora en `tests/`; baterías de 62 y 17 pruebas en ese momento) | Pasan **62 / 62** (~4.5 min) y **17 / 17** (~50 s) |
| CPU (4 núcleos, Python 3.13, PyTorch 2.14.1, Triton 3.8, NumPy 2.5.3), 16 de octubre de 2026, después del arreglo de la prueba de copia y de los alias en inglés | `python pruebas/test_nucleo.py`; `python pruebas/test_estable.py` (ahora en `tests/`; batería del núcleo de 62 pruebas en ese momento) | Pasan **62 / 62** (~3 min) y **18 / 18** (~40 s) |
| La misma CPU, PyTorch 2.11.0 (la versión de las corridas anteriores en Colab), Triton 3.6, NumPy 2.3.5, 16 de octubre de 2026 | Las mismas dos baterías (batería del núcleo de 62 pruebas en ese momento) | 60 pasan y 2 se saltan (`torch.compile`, necesita 2.14); pasan **18 / 18**. Con NumPy ≥ 2.4, el intérprete de Triton 3.6 falla en las tres pruebas de kernel contra PyTorch (`TypeError: only 0-dimensional arrays can be converted to Python scalars`), un problema de Triton que Triton 3.8 no tiene |
| La misma CPU, PyTorch 2.11.0 y 2.14.1, 12 hilos (`torch.set_num_threads(12)`), 16 de octubre de 2026 | `test_entrenar_aprende_a_copiar_y_reanuda` | Pasa en las dos. La versión anterior de la prueba (lr 3e-3) falla en 2.11 con pérdida de copia 4.199, la cifra vista en Colab |
| La misma CPU, un hilo (`OMP_NUM_THREADS=1`), después de la reorganización de 1.0 (nombres de carpetas en inglés, sin las comparaciones con v85/v87), 16 de octubre de 2026 | `python tests/test_nucleo.py`; `python tests/test_estable.py` | PyTorch 2.14.1 (Triton 3.8, NumPy 2.5.3): pasan **55 / 55** y **18 / 18**. PyTorch 2.11.0 (Triton 3.6, NumPy 2.3.5): 53 pasan y 2 se saltan (`torch.compile`, necesita 2.14); pasan **18 / 18** |
| Un entorno virtual nuevo (Python 3.13, PyTorch 2.14.1), 15 de octubre de 2026 | `pip install .` (sin aislamiento de compilación y sin resolver dependencias) | Instala `asrn 1.0` y el comando `asrn-train` |
| CPU (4 núcleos, Python 3.13, PyTorch 2.14.1, Triton 3.8), 23 de octubre de 2026, con la escritura O(1) | `python tests/test_nucleo.py`; `python tests/test_estable.py` (las baterías actuales: 57 y 19 pruebas, 76 en total) | Pasan **57 / 57** y **19 / 19** |
| La misma CPU, PyTorch 2.11.0 (Triton 3.6.0, NumPy 2.3.5), 23 de octubre de 2026, con la escritura O(1) | Las mismas dos baterías | 55 pasan y 2 se saltan (`torch.compile`, necesita 2.14); pasan **19 / 19** |
| La misma CPU, PyTorch 2.14.1, 23 de octubre de 2026 | Escritura O(1) contra la escritura O(t) y la ida sobre el texto entero (`test_escritura_o1_igual_que_o_t` en las dos baterías, `test_anillo_y_ngramas_como_su_definicion`): capa de v89 y estable, anillos que se desbordan, lotes de 1 a 3, inicios de 1, 2 y 40 piezas, con y sin la memoria de lo ajeno | Iguales: diferencia máxima 2.2·10⁻⁷ |
| La misma CPU, PyTorch 2.14.1, 23 de octubre de 2026 | `escribir()` contra la implementación O(t) anterior en 192 combinaciones de opciones de muestreo (temperatura, top-k, top-p, memoria de lo ajeno, temperatura que conserva la cita, pena de repetición, prohibición de n-gramas), capa de v89 y estable | Las mismas secuencias de piezas y logits a menos de 10⁻⁴ en **192 / 192** |
| La misma CPU (4 hilos, PyTorch 2.14.1), 23 de octubre de 2026 | `benchmarks/medir_escritura.py`: ms por pieza nueva y tiempo del inicio según el contexto (de 512 a 131 072 piezas), escritura O(1) contra O(t) | [`results/escritura_cpu/`](../results/escritura_cpu/RESUMEN.md): O(1) 9.49–12.47 ms por pieza, pendiente −19.21 ns por pieza de contexto (IC del 95% de −41.54 a 3.12); O(t) 15.80–91.67 ms, pendiente 583.31 ns (537.13 a 629.49) |
| CPU, PyTorch 2.14.1 | Costo exacto (FLOPs, bytes guardados), latencia, escritura (con la versión O(t)), copia a escala chica; sondas del reloj, la fuga, el corrimiento y los gradientes | `results/banco_cpu/`, `history/probes/` |
| NVIDIA A100-SXM4-80GB (Colab; PyTorch 2.14.1+cu130, CUDA 13.0, Python 3.13.15; `mamba_ssm` 2.3.2.post1, `flash-linear-attention` 0.5.2; huella del código `5164e1f32513`), ejecución del 17 de octubre de 2026 con `benchmarks/asrn_colab.ipynb` | `tests/test_nucleo.py`; `tests/test_estable.py` | Pasan **55 / 55** (5.3 min) y **18 / 18** (1.0 min), incluidas la prueba de copia con su nueva tasa de aprendizaje y las tres pruebas del paquete (la API del paquete, su línea de comandos y los alias en inglés) |
| La misma A100, la misma ejecución | Las 7 comprobaciones de `tests/verificar_gpu.py` | Pasan **7 / 7**, `torch.compile` incluido (logits a 4.8·10⁻⁴, gradientes a 1.2·10⁻⁷): entrenamiento en bf16 de un modelo de 30 M (pérdida 9.806 → 0.203, 46 435 piezas/s); checkpoint por niveles 2 009 / 1 708 / 1 099 MB; FSDP con NCCL en un proceso igual que sin repartir (diferencia 0); GPU = CPU (diferencia máxima 6.1·10⁻⁷, gradiente 4.5·10⁻⁷ relativa); causalidad (1.8·10⁻⁷); escritura paso a paso = texto entero en 80 pasos (2.3·10⁻⁷) |
| La misma A100, la misma ejecución | Estabilidad en la GPU, capa sola, modelo de ~50 M (sin compilar y compilado), escritura, copia, MQAR, WikiText-103, ablación; experimento de citas | [`results/banco_a100/`](../results/banco_a100/RESUMEN.md), [`results/citas_a100/`](../results/citas_a100/RESUMEN.md) |
| Ejecución anterior: NVIDIA A100-SXM4-80GB (Colab; PyTorch 2.11.0+cu130, CUDA 13.0, Python 3.13.15; `mamba_ssm` 2.3.2.post1, `flash-linear-attention` 0.5.2; nada compilado), ejecuciones del 9 de octubre de 2026 (desde un ZIP, sin huella del código) y del 11 de octubre de 2026 (huella del código `176cfe250773`) | Estabilidad en la GPU, capa sola, modelo de ~50 M, escritura, copia, MQAR, WikiText-103, ablación | [`history/results/banco_a100_pt211/`](../history/results/banco_a100_pt211/RESUMEN.md) |
| Ejecución anterior (`history/results/banco_a100_pt211/`), con los nombres de antes de 1.0 | `test_asrn_v89.py` (ahora `tests/test_nucleo.py`; 62 pruebas en ese momento) | 59 pasan, 2 se saltan (`torch.compile`, necesita 2.14), 1 falla (`test_entrenar_aprende_a_copiar_y_reanuda`; causa encontrada y corregida después, ver Problemas resueltos) |
| Ejecución anterior (`history/results/banco_a100_pt211/`), con los nombres de antes de 1.0 | `test_asrn_estable.py` (ahora `tests/test_estable.py`; 14 pruebas en ese momento) | 13 pasan y 1 falló por un error de la prueba misma (dispositivo), ya corregido |
| Ejecución anterior (`history/results/banco_a100_pt211/`) | Las 7 comprobaciones de `verificar_gpu_v89.py` (ahora `tests/verificar_gpu.py`) | 6 pasan y `torch.compile` se salta: entrenamiento en bf16 de un modelo de 30 M (pérdida 9.806 → 0.203); checkpoint por niveles 2 009 / 1 708 / 1 099 MB; FSDP con NCCL en un proceso igual que sin repartir; GPU = CPU (diferencia máxima 6.1·10⁻⁷, gradiente 4.4·10⁻⁷ relativa); causalidad (1.8·10⁻⁷); escritura paso a paso = texto entero (2.3·10⁻⁷) |

Las pruebas corren desde el repositorio completo: importan `asrn` desde la raíz, y la batería del núcleo usa además
`benchmarks/rivales_lineales.py`. Para reproducir el banco de la A100, corre `benchmarks/asrn_colab.ipynb` en Colab
(PyTorch 2.14; también corre todas las pruebas y el experimento de citas). Guarda en Drive y retoma donde quedó. Sus
secciones tardaron 284 minutos en total en la ejecución del 17 de octubre. Un flujo de GitHub Actions
([`.github/workflows/tests.yml`](../.github/workflows/tests.yml)) corre las dos baterías en CPU con Python 3.11 y 3.13;
sin Triton, ahí se saltan las pruebas de kernel contra PyTorch. Entornos probados: Python 3.13 con PyTorch 2.11.0 y
2.14.1 (CPU, A100); integración continua con Python 3.11 y 3.13 (CPU).

### Procedencia de los resultados

`results/banco_a100/` y `results/citas_a100/` salen del código con huella `5164e1f32513`: los primeros 12 dígitos
hexadecimales del sha256 de `asrn/*.py`, los `*.py` de la raíz y `benchmarks/*.py`, ordenados por nombre y concatenados
(el cuaderno la muestra). El código de 1.0 difiere de ese código solo en la escritura O(1) (`asrn/nucleo.py`,
`asrn/estable.py`), en que la medición de escritura pasa las piezas nuevas desde la CPU (`benchmarks/banco_gpu.py`), en
docstrings, en el número de versión y en el nuevo `benchmarks/medir_escritura.py`. La ida y los gradientes del modelo
son iguales bit a bit (comprobado), así que cada cifra de la A100 salvo el tiempo de escritura vale sin cambios para
1.0.
