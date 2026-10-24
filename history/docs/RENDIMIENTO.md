# Velocidad de ASRN: dónde se va el tiempo y el plan

Este documento explica por qué la capa ASRN entrena más lento que la atención de un Transformer, qué se arregló ya y
cuál es el plan para el primer kernel propio. Todos los números se pueden reproducir: los de la A100 con
`history/experiments/medir_velocidad_v89.py` (o el cuaderno `medir_velocidad_a100.ipynb`, retirado en 1.0); los conteos de bytes,
contando cada operación de PyTorch de una ida y vuelta. Las secciones van en el orden en que se hizo el trabajo; el
resultado de todo junto está aquí abajo y en la sección 6.

**Palabras que se usan:**
- **kernel**: cada programa que la GPU ejecuta. Cada operación de PyTorch (sumar, ordenar, multiplicar matrices)
  lanza al menos uno.
- **ida y vuelta** (*forward* y *backward*): calcular la salida y después los gradientes para aprender.
- **memoria de la GPU (HBM)** y **SRAM**: la memoria grande y lenta de la GPU (40 u 80 GB) y la memoria chica y
  rapidísima dentro de cada núcleo (unos cientos de KB). Un kernel *fusionado* hace varias operaciones seguidas en la
  SRAM sin escribir los resultados intermedios en la HBM.
- **Triton**: un lenguaje (de OpenAI) para escribir kernels de GPU en Python. Es lo que usa FlashAttention en
  muchas versiones y lo que genera `torch.compile`.

## 1. Lo medido en la A100

De la primera corrida en una A100 de 40 GB (`history/results/corrida_1/`), antes de cualquier arreglo:

| | ASRN | Transformer (FlashAttention) |
|---|---|---|
| Una capa (d = 512), 1 024 piezas | 23.5 ms | 1.5 ms |
| Una capa, 262 144 piezas | 562 ms | 1 449 ms |
| Modelo de 50 M, 4 096 × 8 | 44 033 piezas/s | 184 380 piezas/s |
| Modelo de 50 M, 32 768 × 1 | 44 410 piezas/s | 92 469 piezas/s |

Dos cosas se ven en la capa sola:

1. **Un costo fijo de ~22 ms por capa**, aunque el texto sea corto. Con 1 024 piezas casi todo el tiempo es eso.
2. **Un costo por pieza de ~2.1 µs** (microsegundos) por capa, que es lo que manda en el entrenamiento.

En el modelo de 50 M (8 capas ASRN, d = 576), las capas ASRN se llevan cerca del 80% de cada paso:
8 capas × 32 768 piezas × ~2.3 µs ≈ 0.60 s de los 0.74 s del paso.

## 2. Por qué: la capa está limitada por la memoria, no por las cuentas

Se contó cada operación de una ida y vuelta (d = 576, lote 32 × 1 024, en bf16): cuántos bytes lee y escribe en
la memoria de la GPU y cuántas cuentas hace.

| | Antes | Ahora (este cambio) |
|---|---|---|
| Bytes movidos por pieza | 852 KB | 619 KB |
| Cuentas por pieza | 7.0 MFLOP | 7.0 MFLOP |
| Operaciones (kernels) por ida y vuelta | 927 | 731 |

Una A100 hace unas 150 cuentas en el tiempo en que mueve un byte. La capa hace unas 8 por byte, así que la GPU pasa
casi todo el tiempo **esperando datos**, no calculando. Por eso no ayuda vectorizar más (ya no hay bucles por pieza
ni por tabla) ni usar `torch.gather` o `index_select` en otros lugares: lo que hay que bajar son los **bytes**.

Por partes (bytes por pieza, antes → ahora):

| Parte de la capa | Bytes | Operaciones |
|---|---|---|
| Atención por casilleros (`_acumular`) | 623 → 389 KB | 164 → 163 |
| Hash (`_phi`) | 79 KB | 48 |
| Reloj y confirmación (`_contexto`) | 64 KB | 39 |
| Puerta (`_combinar`) | 62 KB | 100 |
| Repetición: Hawkes y racha (`_repeticion`, `escaneo`) | 4 → 3 KB | 565 → 370 |

- **La atención por casilleros mueve casi todos los bytes.** Cada una de las 8 tablas escribe en memoria su propia
  copia ordenada de todo (los vectores de 576 números incluidos), arma ~25 tablas intermedias de puntajes de
  32 × 64 por bloque (sumar, enmascarar, softmax...) y vuelve todo al orden del texto. Cada paso lee y escribe en la
  memoria de la GPU.
- **El escaneo lanza la mitad de las operaciones** y casi no mueve bytes: es un árbol de ~10 niveles (con 1 024
  piezas) con ~7 operaciones chiquitas por nivel, en la ida y en la vuelta. Eso es el costo fijo: cada operación
  cuesta ~20 µs de Python y lanzamiento, aunque no haga casi nada.

## 3. Lo que ya se arregló (mismas cuentas)

Cuatro cambios en `asrn_v89.py`. La salida es **idéntica bit a bit** a la anterior (lo comprueban las pruebas de
igualdad con v85 y v87, y `medir_velocidad_v89.py` en la GPU); en los gradientes solo cambia el redondeo de la vuelta
del escaneo (≤ 3·10⁻⁷ relativo).

1. **Volver al orden del texto por filas.** Antes se usaba `scatter` con un índice de 64 bits *por cada número* de
   cada vector (8 tablas × 576 números × 8 bytes por pieza, solo de índices). Ahora es `index_select` con un índice
   *por fila*, y la suma de las 8 tablas tiene su propia vuelta (`_DesordenarSumar`), que no pasa por un tensor de
   8 × L × d ni usa sumas atómicas.
2. **El largo del sufijo en 1 byte** (`_largo_seguido`): es un número de 0 a 4 en una tabla del tamaño de los
   puntajes, y se guardaba en 8 bytes.
3. **Los casilleros en 32 bits** (`_ids`): la GPU ordena claves de 32 bits con la mitad de pasadas que de 64.
4. **La vuelta del escaneo** (`_Escaneo`): en vez de derivar el árbol operación por operación, la vuelta es el mismo
   escaneo al revés. ~200 operaciones menos por capa y menos de la mitad de memoria guardada.

Resultado contado: −27% de bytes, −21% de operaciones. En la A100 dio de 1.05 a 1.21 veces más rápido (sección 5).

## 4. Lo que queda: el primer kernel propio

Después de estos arreglos, la atención por casilleros sigue siendo el 63% de los bytes. Lo que sobra ahí no se
puede quitar con operaciones de PyTorch: cada operación escribe su resultado en la memoria de la GPU antes de que la
siguiente lo lea. Hace falta **fusionar**: un solo kernel que haga todo el camino en la SRAM, como FlashAttention
hace con la atención.

### Qué hace el kernel (la parte que reemplaza)

Todo `_acumular` desde que se ordenan las posiciones hasta la salida, para cada tabla `τ` y cada bloque de
`C = 32` preguntas en el orden de esa tabla:

1. Leer de la memoria de la GPU, siguiendo el orden de la tabla (`orden`), las filas de las 32 preguntas y de sus
   64 llaves (bloque anterior + actual): `φ` (112 números), `Qc`/`Kc` (16), el reloj `B`, `log α`, el casillero y
   las 4 piezas anteriores (se leen directo de los números de pieza).
2. Puntaje, en la SRAM:
   `s = φ_q·φ_k / √(T·n_bits) + Qc·Kc / √16 + w_sufijo[largo] + log α_k − (γ_τ / C)(B_q − B_k)`
   (las dos multiplicaciones de matrices con tensor cores, `tl.dot`).
3. Máscara: mismo casillero, llave entre las 32 anteriores en el orden, llave real.
4. Softmax en 32 bits, salida `P · V_k` y las señales de la puerta (hay, máximo, pico).
5. Escribir la salida de cada pregunta **directamente en su posición del texto**, en un espacio por tabla que
   después se suma (sin copia ordenada ni `index_select`).

No se escriben en la memoria: las 8 copias ordenadas de `V` y de `φ`, los ~25 tensores de puntajes, ni la salida en
el orden de la tabla.

### Interfaz (para probarlo contra lo que ya existe)

```
entradas (orden del texto):  φ (B, L, 112) · V (B, L, d) · Qc, Kc (B, L, 16) · piezas (B, L)
                             B_t, log α (B, L) en 32 bits · casilleros (B, L, T) · orden (B·T, L)
parámetros:                  w_sufijo (5) · γ (T) · temperatura
salidas:                     suma de las tablas (B, L, d) · hay, máximo, pico (B, T, L)
```

Es exactamente lo que calcula `_acumular` hoy, así que la versión de PyTorch queda como **referencia** (y como
respaldo en CPU o sin Triton): cada versión del kernel se compara contra ella.

### Cuánto puede ganar (estimación, a confirmar)

Contando lo mínimo que el kernel tiene que leer y escribir, la ida de la atención por casilleros bajaría de ~150 KB
a ~45 KB por pieza, y la vuelta de ~200 KB a ~110 KB. La capa entera pasaría de ~620 KB a ~380 KB por pieza. Además,
un kernel fusionado suele aprovechar mejor la memoria que muchos kernels chicos: hoy la capa saca ~25% del ancho de
banda de la A100 (755 KB por pieza en 2.1 µs). Estimación honesta: **la capa, de 2 a 4 veces más rápida**. Hay que medirlo.

### Lo que el kernel no alcanza a cerrar

Para igualar al Transformer en el modelo de 50 M, suponiendo que el resto del modelo (~0.14 s por paso) no cambia,
las capas ASRN tendrían que ser:

| Largo del texto | Tiempo del paso del Transformer | Las capas ASRN tendrían que ser |
|---|---|---|
| 4 096 | 0.18 s | ~16 veces más rápidas |
| 16 384 | 0.25 s | ~5 veces más rápidas |
| 32 768 | 0.35 s | ~3 veces más rápidas |

Con el kernel, la brecha se puede cerrar con textos largos (≥ 32 mil piezas), y achicar mucho con 16 mil. **Con
textos de 1 a 4 mil, no**: ahí el límite es el diseño. Cada una de las 8 tablas copia el vector completo
(576 números), así que la capa mueve 8 vectores completos por pieza; la atención de un Transformer reparte el vector
entre sus cabezas. Si cada tabla copiara solo su parte del vector (576 / 8 = 72 números, como una cabeza), esa parte
del trabajo bajaría 8 veces. **Eso cambia el modelo**: habría que re-entrenar y comparar la calidad (copia, WikiText).
Es una decisión de diseño, no de código.

## 5. La primera medición en la A100 (y lo que corrigió)

`medir_velocidad_a100.ipynb` en una A100 de 40 GB (PyTorch 2.11), con los 4 arreglos de la sección 3:

| | Antes | Con los 4 arreglos |
|---|---|---|
| Una capa, 1 024 piezas | 21.9 ms | 18.2 ms (1.21×) |
| Una capa, 16 384 piezas | 49.3 ms | 44.9 ms (1.10×) |
| Una capa, 262 144 piezas | 565 ms | 537 ms (1.05×) |
| Modelo de 50 M, 4 096 × 8 | 43 680 piezas/s | 46 177 piezas/s (1.06×) |

La salida fue igual bit a bit en la GPU, en fp32 y en bf16. La ganancia fue menor que el 27% de bytes, y la parte 4
mostró por qué: **la mitad del tiempo de la GPU (18.4 de 36.6 ms con 16 384 piezas) se iba en un solo kernel**, la
vuelta de `w_sufijo[largo]`. Es una suma "atómica" (`index_add_`) de millones de números sobre solo 5 casillas, y la
GPU tiene que hacer esas sumas una por una porque todas caen en las mismas direcciones. La vuelta de la respuesta a
la racha (`u_racha[i]`, 16 casillas) tenía el mismo problema: otros 2.9 ms. El conteo de bytes no lo podía ver: no
es un problema de cuántos datos se mueven, sino de muchos hilos esperando turno para escribir en el mismo lugar.

Arreglo (`_Tomar`): la ida sigue siendo `w[i]`. La vuelta suma el gradiente de cada casilla con una reducción normal
(una pasada por casilla, sin atómicas), que la GPU hace en paralelo. Además el resultado es siempre el mismo: las
sumas atómicas no siguen siempre el mismo orden, así que dos corridas iguales podían dar gradientes un poco distintos
(probablemente la diferencia de 4% en bf16 que vio la medición; el script ahora mide ese ruido aparte).

**Medido en la A100** (segunda corrida, contra la versión anterior a todos estos cambios):

| | Antes | Ahora | Transformer (FlashAttention) |
|---|---|---|---|
| Una capa, 1 024 piezas | 23.0 ms | 19.1 ms (1.20×) | 1.5 ms |
| Una capa, 16 384 piezas | 50.7 ms | 26.6 ms (1.91×) | 7.9 ms |
| Una capa, 65 536 piezas | 145.5 ms | 53.9 ms (2.70×) | 95.6 ms |
| Una capa, 262 144 piezas | 564.7 ms | 201.8 ms (2.80×) | 1 448.7 ms |
| Modelo de 50 M, 1 024 × 32 | 35 811 piezas/s | 57 568 (1.61×) | 207 118 |
| Modelo de 50 M, 4 096 × 8 | 43 755 piezas/s | 87 701 (2.00×) | 184 380 |
| Modelo de 50 M, 16 384 × 2 | 44 386 piezas/s | 89 402 (2.01×) | 128 774 |

(Los números del Transformer son de la corrida de `history/results/`, en la misma clase de GPU.)

- La capa ya es más rápida que FlashAttention desde unas 40 mil piezas (antes, desde unas 100 mil), y 7 veces más
  rápida con 262 144.
- Con 16 384 piezas el modelo entero quedó a 1.4 veces del Transformer (antes, 2.9).
- El ruido de la versión anterior consigo misma en bf16 fue 3.1% (en `afin_contexto.2.bias`) y la diferencia con la
  nueva, 5.6% en el mismo parámetro: del mismo orden. Es el redondeo de sumas en bf16, no un error.

## 5b. El primer kernel Triton: el escaneo

Con las atómicas fuera, la GPU trabaja 15.7 de los 32 ms de la capa con 16 384 piezas (3.3 de 21 ms con 1 024): el
resto del tiempo espera a que Python le mande los kernels, ~1 070 por ida y vuelta. La parte que más lanza es el
escaneo de Hawkes y racha (8.5 ms con 1 024 piezas, 15 ms con 16 384), con ~300 operaciones chicas por capa.

`_escaneo_kernel` (Triton) hace el escaneo entero en un kernel: un programa por fila, en bloques de 1 024 elementos
con `tl.associative_scan` en la SRAM y un acarreo entre bloques. La vuelta es otro kernel del mismo código que corre
de derecha a izquierda y además escribe dL/da_i = l_i S_{i−1}. De ~290 operaciones por capa quedan ~6.

- Se usa en GPU cuando Triton está instalado (viene con PyTorch en Linux). Sin GPU o sin Triton, o dentro de
  `torch.compile`, se usa el escaneo de PyTorch. `ASRN_TRITON=0` lo apaga.
- Suma en otro orden que el árbol de PyTorch: la salida difiere en el redondeo (≤ 10⁻⁶ relativo). Ya no es igual bit
  a bit; `medir_velocidad_v89.py` mide la diferencia y confirma que sin Triton sigue siendo igual bit a bit.
- La prueba `test_escaneo_triton_igual_que_pytorch` corre el mismo kernel en CPU con el intérprete de Triton
  (`TRITON_INTERPRET=1`): ida y vuelta, con varios bloques y acarreo.

**Medido** en la corrida completa del cuaderno en una A100 de 80 GB (`history/results/corrida_2/`), con todo lo de
arriba:

| Modelo de 50 M | Primera corrida | Ahora | Transformer (misma GPU) |
|---|---|---|---|
| 1 024 × 32 | 35 136 piezas/s | 63 107 (1.8×) | 230 907 |
| 4 096 × 8 | 44 033 | 97 627 (2.2×) | 201 759 |
| 16 384 × 2 | 44 503 | 100 918 (2.3×) | 137 567 |
| 32 768 × 1 | 44 410 | **100 658 (2.3×)** | 96 542 |

Con 32 mil piezas, ASRN ya entrena igual de rápido que el Transformer. La capa sola: 15.0 ms con 1 024 piezas (antes
23.5) y empata con FlashAttention desde ~32 mil. Las cuentas siguen dando lo mismo: WikiText-103 da la misma
perplejidad que en la primera corrida (57.2 contra 56.8) y las 56 pruebas que corren en esa versión de PyTorch pasan.
## 5c. El kernel de la atención por casilleros

`_casilleros_ida` y `_casilleros_vuelta` (Triton) hacen todo `_acumular` sin pasar por la memoria de la GPU las 8
copias ordenadas ni los ~25 tensores de puntajes. Hay un programa por (fila de lote y tabla, bloque de 32 preguntas en
el orden de esa tabla):

- **Ida:** lee directo del orden del texto, siguiendo `orden`, las filas de sus 32 preguntas y 64 llaves (φ, Qc/Kc, el
  reloj, log α, el casillero y las 4 piezas anteriores). Arma los puntajes en la SRAM (dos `tl.dot`, el peso del
  sufijo, el núcleo de tiempo y la máscara), hace el softmax, multiplica por V en bloques de 64 columnas y escribe la
  salida de cada pregunta en su posición del texto, junto con las señales de la puerta.
- **Vuelta:** recalcula los puntajes, igual que FlashAttention, así que **no guarda nada de la ida**: solo sus
  entradas. Los gradientes de las filas (φ, Qc, Kc, V, el reloj y log α) se suman con atómicas en 32 bits: cada fila
  recibe pocas, unas 3 por tabla, así que no hay embudo. Los de γ y w_sufijo, que son pocos números, se escriben un
  renglón por programa y se suman después, sin atómicas.
- **Cuentas:** en fp32 coinciden con `_acumular` salvo el redondeo (≤ 10⁻⁷ en la ida, ≤ 4·10⁻⁶ en los gradientes). En
  bf16 los puntajes se suman en 32 bits; comparados con la versión en fp32, los gradientes del kernel quedan igual de
  cerca que los de `_acumular` en bf16, o más. Lo comprueba `test_casilleros_triton_igual_que_pytorch` con el
  intérprete de Triton, que multiplica mal en bf16, así que ahí las multiplicaciones van en 32 bits.
- **Cuándo se usa:** en GPU con Triton, con bloques de al menos 16 (`chunk`, φ y la confirmación), fuera de
  `torch.compile`. `ASRN_TRITON_CASILLEROS=0` lo apaga solo a él; `ASRN_TRITON=0`, a todos.
- **Medido en la A100 de 40 GB** (`medir_velocidad_a100.ipynb`, contra la versión del principio y contra la de
  "solo escaneo", que tiene todo lo anterior menos este kernel):

| | Antes | Solo escaneo | Con este kernel | Transformer |
|---|---|---|---|---|
| Una capa, 1 024 piezas | 23.7 ms | 14.4 ms | 10.5 ms | 1.5 ms |
| Una capa, 16 384 piezas | 54.2 ms | 19.0 ms | 16.7 ms | 7.9 ms |
| Una capa, 262 144 piezas | 564 ms | 201 ms | 169 ms | 1 449 ms |
| Memoria de la capa, 262 144 piezas | 17.9 GB | 15.0 GB | **7.0 GB** | 4.0 GB |
| Modelo de 50 M, 1 024 × 32 | 34 594 piezas/s | 60 676 | **105 454** | 207 118 |
| Modelo de 50 M, 4 096 × 8 | 43 747 | 89 450 | **106 536** | 184 380 |
| Modelo de 50 M, 16 384 × 2 | 44 435 | 92 242 | **106 031** | 128 774 |
| Modelo de 50 M, 32 768 × 1 | 44 256 | 91 796 | **105 611** | 92 469 |

  El modelo quedó 3 veces más rápido que al principio y más rápido que el Transformer con 32 mil piezas. La memoria
  de la capa bajó a la mitad (la vuelta ya no guarda los puntajes).
- **La vuelta tenía atómicas.** En esa medición la ida del kernel tardaba 0.33 ms y la vuelta 6.4 ms (con 16 384
  piezas): la vuelta sumaba con atómicas unas 11 mil veces por pieza (8 tablas × 2 ventanas × los 512 números de V,
  más phi). Ahora cada contribución tiene su lugar: cada posición es pregunta una vez por tabla y cada llave aparece
  en dos ventanas de su tabla, así que se escriben sin atómicas en espacios (tabla, lugar) y se suman después, en
  orden. Además la vuelta da siempre el mismo resultado. **Medido:** la vuelta pasó de 6.4 a 0.99 ms.

| Modelo de 50 M (piezas/s) | Al principio | Con atómicas | Sin atómicas | Transformer |
|---|---|---|---|---|
| 1 024 × 32 | 35 012 | 105 454 | **147 361** | 207 118 |
| 4 096 × 8 | 43 783 | 106 536 | **147 279** | 184 380 |
| 16 384 × 2 | 44 402 | 106 031 | **145 679** | 128 774 |
| 32 768 × 1 | 44 292 | 105 611 | **146 195** | 92 469 |

  Una capa: 10.5 ms con 1 024 piezas (antes 23.4), 11.6 ms con 16 384 (antes 51.5) y 86.6 ms con 262 144 (antes
  564; FlashAttention 1 449). La memoria de la capa con 262 144 piezas es 12.0 GB (antes 17.9): un poco más que con
  atómicas (7.0 GB), por los espacios donde se escribe cada contribución de la vuelta.
- **bf16:** la diferencia de gradientes entre antes y ahora en bf16 (~9.5%, en `afin_contexto.2.bias`, un gradiente
  casi nulo) es del orden del ruido de antes consigo misma (2–5%). Comparados con la cuenta en 32 bits en la GPU,
  antes y ahora quedan exactamente igual de cerca: 1.15·10⁻³ de error relativo en el peor parámetro (`g_conf.weight`).

**Lo que queda.** Con 1 024 piezas, la GPU trabaja 1.6 de los 10.6 ms de la capa, y con 16 384, 7.0 de 13.8 ms. El
resto es Python lanzando ~400 kernels: el reloj y la confirmación, la puerta, las conversiones de bf16. Ese es el
límite con textos cortos, y se ataca juntando esas operaciones (CUDA Graphs o `torch.compile` con PyTorch ≥ 2.14).

## 5d. CUDA Graphs: el paso entero grabado

Con textos cortos la GPU espera a Python: con 1 024 piezas, una capa tiene 1.6 ms de trabajo de GPU en 10.6 ms. Un
**CUDA Graph** graba una vez todos los kernels de un paso de entrenamiento (ida, vuelta, recorte y AdamW) y después
la GPU los repite con una sola llamada. Python ya no lanza los ~3 000 kernels del paso uno por uno, y tampoco espera
a la GPU entre pasos (`entrenar()` lee la pérdida con `.item()` en cada paso, lo que hoy impide que los lanzamientos
de un paso se adelanten mientras la GPU termina el anterior).

- **`asrn.PasoGrafado(modelo, opt, lote, largo)`** hace eso. El optimizador se crea con
  `crear_optimizador(..., capturable=True)` y la tasa de aprendizaje vive en la GPU, así que cambiarla no obliga a
  grabar de nuevo. Si la pérdida o la norma no son finitas, AdamW fusionado no cambia nada en ese paso (el mismo
  mecanismo que usa el escalador de fp16), sin que Python tenga que mirar el número. Las cuentas son las mismas que
  sin grabar: lo comprueba `test_paso_grafado_mismas_cuentas_y_limites` en CPU (sin grabar).
- **Límites:** una GPU, bf16 o fp32, sin checkpoint, sin FSDP ni `torch.compile`, sin acumulación, y siempre la
  misma forma de lote. Por ahora no está dentro de `entrenar()`: primero hay que medir si vale la pena.
- **Arreglo en el camino:** con pesos NaN, `_respuesta_racha` convertía el NaN en un índice basura (en GPU eso mata el
  proceso, y entonces `entrenar()` nunca llegaba a saltar el paso). Ahora el NaN va al nodo 0 y la salida sale NaN.
- **Cómo se mide:** `medir_velocidad_v89.py` agrega la capa con CUDA Graph y, en el modelo de 50 M, ASRN y el
  Transformer **cada uno con y sin CUDA Graph** (también acelera al Transformer), y qué fracción del paso pasa la GPU
  trabajando.
- **Qué esperamos y qué decide:** si con CUDA Graph ASRN iguala o supera al Transformer (también con CUDA Graph) desde
  4 096 piezas y queda a menos de ~10% con 1 024, se integra en `entrenar()`. Si no, el siguiente paso es el cambio de
  diseño (sección 4): que cada tabla copie solo su parte del vector.

**Medido en la A100 de 40 GB:**

| | Sin CUDA Graph | Con CUDA Graph |
|---|---|---|
| Una capa, 1 024 piezas, lote 1 | 11.4 ms | **1.6 ms** (FlashAttention: 1.5 ms) |
| Una capa, 4 096 piezas, lote 1 | 11.0 ms | **2.2 ms** |
| Una capa, 16 384 piezas, lote 1 | 11.8 ms | **6.7 ms** |
| Una capa, 65 536 piezas, lote 1 | 23.1 ms | 22.9 ms |

| Modelo de 50 M, 32 768 piezas por paso | ASRN | GPU ocupada | Transformer | GPU ocupada |
|---|---|---|---|---|
| 1 024 × 32 | 146 122 piezas/s | 97.8% | 207 630 | 99.6% |
| 4 096 × 8 | 141 030 | 94.6% | 184 593 | 99.5% |
| 16 384 × 2 | 143 287 | 97.1% | 128 958 | 99.6% |
| 32 768 × 1 | 143 195 | 97.0% | 92 670 | 99.7% |

- **Una capa sola con un texto corto:** CUDA Graph la hace 7 veces más rápida, casi igual que FlashAttention con
  1 024 piezas. Ahí sí la GPU esperaba a Python. Sirve para escribir texto y para lotes chicos.
- **Entrenar con un lote completo:** la GPU ya trabaja el 95–98% del paso, así que CUDA Graph puede ganar, como
  mucho, esos pocos puntos. Para empatar con el Transformer hacen falta 31% (4 096) y 42% (1 024). **No alcanza: el
  límite en entrenamiento es el trabajo de la GPU, no Python.** Según lo acordado, sigue el cambio de diseño.
- El paso grabado del modelo falló en esa corrida por un error mío: `found_inf` tenía forma `[1]`, y AdamW fusionado
  la necesita escalar (como GradScaler). Corregido, con una prueba en CPU que pasa por el mismo camino (AdamW
  fusionado también existe en CPU).
- **Dónde está ahora el trabajo de la GPU** (una capa, 16 384 piezas): los kernels de casilleros son ~20%; las
  conversiones de tipo (bf16 ↔ fp32, `aten::copy_`), ~21%; el resto se reparte entre las proyecciones, sumas y
  operaciones chicas del reloj, la puerta y Hawkes. Ya no hay un solo cuello de botella.

## 5e. Sin cambiar el modelo: el paso entero y `torch.compile`

Medido en la A100 de 40 GB con PyTorch 2.14 (`medir_velocidad_a100.ipynb`, parte 5), un paso de entrenamiento del
modelo de 50 M con 1 024 × 32 piezas:

| Parte (medida sola) | ms | del paso |
|---|---|---|
| Paso entero | 224 | 100% |
| Vocabulario, salida y pérdida (igual en el Transformer) | 55 | 24% |
| 9 bloques locales | 70 | 31% |
| 8 capas ASRN | 138 | 61% |
| Recorte y AdamW | 4 | 2% |
| Transformer: paso entero | 158 | |

(Medidas por separado suman más que el paso entero: cada una paga sus propias conversiones.) Entre todos los kernels,
**las conversiones de tipo (`aten::copy_`, 1 020 llamadas) son el 21% del trabajo de la GPU**.

- **`torch.compile` en las partes chicas de las capas ASRN** (hash, valores, reloj, Hawkes y racha, puerta): el paso
  bajó de 224 a 191 ms (−15%), de ~147 mil a ~172 mil piezas por segundo. La atención por casilleros sigue con su
  kernel Triton.
- **Siguiente:** `asrn.compilar_partes(modelo)` (o `entrenar(..., compilar='partes')`, `--compilar`) compila además
  los **bloques locales** (31% del paso, que no estaban compilados). Para comparar en igualdad, el Transformer de
  referencia también se mide compilado. **A medir.**
- **La primera corrida con todo compilado no terminó** (se detuvo a los 80 minutos). Causa: al ver un tamaño de lote
  nuevo, `torch.compile` recompilaba en modo de formas variables, que con este modelo tarda unas 5 veces más. En CPU:
  274 s contra 50 s. Ahora `compilar_partes` compila para formas fijas (`dynamic=False`): cada tamaño nuevo tarda unos
  26 s en CPU, y las 9 capas y los modelos nuevos comparten lo compilado (6 grafos en total). El cuaderno muestra el
  avance en vivo.

**Medido con todo compilado** (A100 de 40 GB, PyTorch 2.14; piezas por segundo, los dos modelos en la misma sesión):

| Largo × lote | ASRN | ASRN compilado | Transformer | Transformer compilado | Compilados: ASRN / Transformer |
|---|---|---|---|---|---|
| 1 024 × 32 | 147 837 | 186 811 | 207 624 | 271 710 | 0.69 |
| 4 096 × 8 | 145 991 | 185 896 | 184 824 | 234 088 | 0.79 |
| 16 384 × 2 | 144 846 | 181 590 | 129 318 | 151 833 | **1.20** |
| 32 768 × 1 | 142 397 | 182 677 | 92 866 | 103 939 | **1.76** |

- `compilar_partes` acelera el entrenamiento de ASRN un 26%: es la mejor forma de entrenarlo en GPU.
- **Pero también acelera al Transformer (31%), así que la brecha con textos cortos no se cierra.** No cumple el
  criterio (igualar desde 4 096 y quedar a menos de 10% con 1 024).
- **Por qué, con el paso por partes:** el modelo ASRN tiene en cada nivel un bloque local y una capa ASRN. Sus 9
  bloques locales (70 ms) más sus 8 capas ASRN (139 ms) cuestan casi el doble que los 8 bloques del Transformer
  (108 ms). Con textos cortos la atención es barata, y la diferencia es de cuánto trabajo hace el modelo por pieza, no
  de cómo está programado. Lo que queda son decisiones de diseño: que cada tabla copie solo su parte del vector
  (sección 4), o poner una capa ASRN cada dos bloques (`largo_cada=2`, que ya existe en la configuración). Las dos
  cambian el modelo y hay que comparar su calidad.

## 5f. Una capa ASRN cada dos bloques (`largo_cada=2`)

Cuarta corrida completa de `pruebas_gpu_a100.ipynb` (A100 de 40 GB, PyTorch 2.11 sin compilar, bf16,
`history/results/corrida_4/`). "ASRN cada 2" tiene casi los mismos parámetros que ASRN normal, pero más bloques locales
y la mitad de capas ASRN (modelo de 50 M: 10 bloques y 5 capas ASRN contra 9 y 8). Mismas tasas, semillas y datos.

**Velocidad del modelo de 50 M** (piezas por segundo, misma GPU):

| Largo × lote | ASRN | ASRN cada 2 | Transformer | cada 2 / ASRN | cada 2 / Transformer |
|---|---|---|---|---|---|
| 1 024 × 32 | 148 007 | 172 394 | 207 538 | 1.16 | 0.83 |
| 4 096 × 8 | 145 312 | 171 502 | 184 781 | 1.18 | 0.93 |
| 16 384 × 2 | 145 969 | 170 410 | 129 159 | 1.17 | **1.32** |
| 32 768 × 1 | 146 205 | 169 984 | 92 877 | 1.16 | **1.83** |

Memoria: 22.7 GB (ASRN 23.8, Transformer 22.2).

**Calidad** (3 semillas con la mejor tasa, 0.003 en los dos):

| Prueba | ASRN | ASRN cada 2 | Transformer |
|---|---|---|---|
| WikiText-103, perplejidad con 1 024 | 56.97 ± 0.18 | 57.06 ± 0.47 | 64.06 ± 0.83 |
| WikiText-103, perplejidad con 4 096 | 52.99 ± 0.30 | 53.15 ± 0.69 | 125.3 ± 2.6 |
| WikiText-103, minutos por corrida | 4.34 | **3.86** | 3.06 |
| Copia con 1 024 | 100% | 100% | 97.8% ± 3.7 |
| Copia con 4 096 | 99.9% ± 0.2 | 99.3% ± 1.3 | 0.2% |
| Copia con 16 384 | 90.9% ± 9.4 | 89.6% ± 16.6 | 0.2% |
| Copia, segundos por corrida | 89 | **68** | 36 |

- **Aprende igual:** la diferencia en WikiText (0.09 de perplejidad) es menor que lo que varía entre semillas, y la
  copia a larga distancia se mantiene (las dos fallan con la tasa 0.001 y aprenden con 0.003 en las 3 semillas).
- **Entrena 16–18% más rápido** con cualquier largo, y gasta 1 GB menos. Con textos cortos sigue más lento que el
  Transformer sin compilar (1.20 veces con 1 024, 1.08 con 4 096), pero la brecha baja a la mitad (antes 1.40 y 1.27
  en esta misma GPU).
- **Decisión:** `largo_cada=2` es la mejor forma de ASRN medida hasta ahora: misma calidad y más velocidad. **Desde
  octubre de 2026 es el valor por omisión** (las tallas de ejemplo, `TALLAS`, se ajustaron para quedar del mismo
  tamaño); `largo_cada=1` (`--largo_cada 1`) da la forma de antes y los modelos guardados cargan con la suya. La
  quinta corrida lo confirmó con 3 semillas contra Mamba-2 y RWKV-7 (sección 5i). Falta medirlo en modelos más
  grandes.

**Compilado** (`medir_velocidad_a100.ipynb`, A100 de 40 GB, PyTorch 2.14.1, piezas por segundo):

| Largo × lote | ASRN compilado | cada 2 compilado | Transformer compilado | cada 2 / Transformer |
|---|---|---|---|---|
| 1 024 × 32 | 185 880 | 214 908 | 271 307 | 0.79 |
| 4 096 × 8 | 185 428 | 215 313 | 234 044 | 0.92 |
| 16 384 × 2 | 182 697 | 213 862 | 151 753 | **1.41** |
| 32 768 × 1 | 182 857 | 213 485 | 103 826 | **2.06** |

"Cada 2" compilado gana lo mismo que sin compilar (16–17% sobre ASRN compilado), pero se queda en 214 908 piezas
por segundo con 1 024: justo debajo de la meta de 215 mil (paso 6b). Según lo acordado, se pasa al diseño de las
tablas (sección 5g).

## 5g. Cada tabla copia solo su parte del vector (`partir_valores`)

Hasta ahora, cada una de las 8 tablas copia el vector entero (576 números en el modelo de 50 M) y la salida es el
promedio de las tablas que encontraron algo. Con `partir_valores=True`, la tabla τ copia solo los números
τ·d/8 a (τ+1)·d/8 (72 de 576), como una cabeza de la atención, y la salida son esas 8 partes una al lado de la otra.

- **Mismos parámetros**: no se agrega ni se quita nada; solo cambia qué copia cada tabla. Los puntajes, la máscara,
  las señales de la puerta, el reloj y la memoria de lo repetido son los mismos.
- **Menos bytes**: los kernels leen 8 veces menos de `V` y de su gradiente, y ya no hace falta la salida de 8 × L × d
  por tabla ni sumarla: cada tabla escribe su parte en su lugar. En la vuelta, el gradiente de `V` es de 2 × L × d en
  vez de 2 × 8 × L × d.
- **Dónde estaba** (ya no está): `CapaASRN(partir_valores=True)`, `ConfigASRN(partir_valores=True)` y `--partir_valores`; en
  PyTorch (`_acumular`), en los dos kernels Triton (`PARTIR`) y en la escritura paso a paso (`_atender`). Pide que
  `d` sea múltiplo de `n_tablas`. Estaba apagado por omisión.
- **Comprobado en CPU**: el kernel Triton da lo mismo que PyTorch (ida y vuelta, en fp32 y bf16, también con
  d = 576); escribir paso a paso da lo mismo que el texto entero; el futuro no cambia el pasado.
- **Cambia el modelo**: cada tabla trae menos información. La calidad no se midió (ver abajo por qué).

**Medido** (`medir_velocidad_a100.ipynb`, A100 de 40 GB, PyTorch 2.14.1, piezas por segundo):

| Largo × lote | cada 2 compilado | partido compilado | partido / cada 2 | Transformer compilado |
|---|---|---|---|---|
| 1 024 × 32 | 214 536 | 218 788 | 1.02 | 271 582 |
| 4 096 × 8 | 214 550 | 219 339 | 1.02 | 233 917 |
| 16 384 × 2 | 211 390 | 215 469 | 1.02 | 151 717 |
| 32 768 × 1 | 212 413 | 213 961 | 1.01 | 103 942 |

Una capa sola (d = 512): con 1 024 piezas, 10.6 → 10.5 ms; con 262 144, 84.6 → 67.1 ms (−21%) y 12.6 → 8.6 GB de
memoria (−32%).

- **Casi no acelera el entrenamiento: +1–3%.** La estimación (230–250 mil) estaba mal: suponía que copiar los
  vectores era la mayor parte de los ~8.7 ms de la atención por casilleros, y no lo es. Los kernels Triton de la
  atención son ~2–3 ms por capa; el resto son las partes de alrededor (el hash `_phi`, ordenar, las piezas
  anteriores, conversiones de tipo). Partir los valores solo achica los kernels.
- Con textos muy largos sí ayuda (−21% de tiempo y −32% de memoria por capa con 262 mil piezas), pero ahí ASRN ya
  es mucho más rápido que la atención.
- **Decisión:** no vale la pena cambiar el modelo por 2%. Primero quedó como opción experimental y en la limpieza de
  octubre de 2026 **se quitó del código** (los kernels volvieron a ser los de antes, sin `PARTIR`): no ganaba lo
  suficiente para mantenerla.

## 5h. Menos conversiones de tipo

En el paso de entrenamiento sin compilar, `aten::copy_` (sobre todo pasar de 32 a 16 bits y de vuelta) era el 21% del
trabajo de la GPU. Con autocast, cada multiplicación convierte su entrada a 16 bits por su cuenta: la misma `E` se
convertía tres veces por capa, el estado normalizado otras tres, y cada conversión tiene otra en la vuelta.

Cambios, con las mismas cuentas:

- **`_phi_y_valores`**: la primera capa del hash de palabra y los valores leen `E` en una sola multiplicación (cada
  número sale con el mismo redondeo, solo en otra columna). `v(E_{t+1})` es `v(E)` desplazado una pieza. `E` se
  convierte a 16 bits una vez, y `[palabra, anterior]` para el hash de contexto se arma con `E` ya convertida
  (convertir y juntar da los mismos bits que juntar y convertir).
- **`_contexto`**: `cq`, `ck` y el reloj en una sola multiplicación, con el estado normalizado convertido una vez.
- **La puerta `g(h)`** (un número por pieza) se calcula en 32 bits: así `h` no se copia entera a 16 bits ni su
  gradiente de vuelta. Es más precisa que antes.

Contado en CPU (una capa, d = 576, ida y vuelta en bf16): de 15 954 a **9 032 números convertidos por pieza (−43%)** y
de 81 a 56 conversiones. Las salidas son las de antes salvo el redondeo (en 32 bits, ~1e-7; los modelos de v85 y v87
cargan con esa diferencia).

Un intento descartado: partir el hash de contexto en `W_a E_t + W_b E_{t-1}` (un producto de `E` desplazado) ahorraba
un poco más, pero redondea cada mitad por separado; en bf16, cerca de cero, eso cambiaba casilleros y el error de los
gradientes contra 32 bits subía hasta 20 veces. Se dejó como estaba.

**Falta medirlo en la GPU** (`medir_velocidad_a100.ipynb`). Compilado por partes, Inductor ya juntaba parte de estas
conversiones, así que ahí la ganancia será menor que sin compilar.

## 5i. Mejorar los bloques locales (calidad, no velocidad)

Contra Mamba-2 y RWKV-7 (`history/results/corrida_5/`), RWKV-7 predice mejor WikiText (47.9 contra 56.5). La capa
ASRN copia como ningún otro modelo; lo débil está en los bloques locales que van entre capas ASRN. En el modelo de
44 M, cada bloque tiene 2.1 M de parámetros y **la mitad es una convolución densa (d × d × 4) que solo ve 4 piezas**;
la capa densa tiene un ancho interno de 2d (el Transformer y RWKV usan 4d).

Variantes en `BloqueLocal` (la receta final no cambia), cada una con **los mismos parámetros**
(`ConfigASRN.con_mismos_parametros` ajusta el ancho de la capa densa). Después de esta medición solo quedó `memoria`
(`mezcla='memoria'`); "canal" y la puerta se quitaron del código:

| Variante | Mezcla de piezas | Capa densa (d = 512) |
|---|---|---|
| ASRN (receta) | convolución densa, 4 piezas | GELU, ancho 1 024 |
| canal | convolución por canal, 4 piezas | GELU, ancho 2 048 |
| canal + puerta | convolución por canal | con puerta, ancho 1 360 |
| memoria | por canal + memoria con olvido por canal (escalas de 1 a 1 024 piezas, con el escaneo de ASRN) | GELU, ancho 1 280 |
| memoria + puerta | igual | con puerta, ancho 848 |

Se miden con `pruebas_gpu_a100.ipynb` y `BLOQUES = True` (unos 40 minutos): velocidad, copia y WikiText con una
semilla y la tasa de ASRN (3e-3). La que gane se confirma con 3 semillas.

**Primera medición** (`history/results/bloques_1/`, A100 de 80 GB, PyTorch 2.11, una semilla, tasa 3e-3):

| Variante | WikiText 1 024 | WikiText 4 096 | Copia 1 024 / 4 096 / 16 384 | Piezas/s (50 M, 1 024) |
|---|---|---|---|---|
| ASRN (receta) | 56.24 | 52.32 | 100% / 100% / 99.2% | 175 145 |
| canal | 56.03 | 52.02 | 100% / 99.6% / 76.9% | 172 663 |
| canal + puerta | 58.74 | 54.65 | 99.9% / 86.8% / 23.8% | 165 334 |
| **memoria** | **53.06** | **49.21** | 100% / **35.4%** / **2.7%** | 145 478 |
| memoria + puerta | 53.96 | 50.06 | 100% / 62.6% / 20.4% | 142 371 |

(RWKV-7 en la quinta corrida: 47.87. ASRN da 56.24, igual que la semilla 0 de la quinta corrida: la medición se
repite.)

- **La memoria con olvido mejora la perplejidad 5.7%** (56.2 → 53.1), también con textos 4 veces más largos.
- **Pero rompe la copia con textos más largos que los de entrenamiento** (35% con 4 096 y 3% con 16 384, contra
  100% y 99%); con 1 024 copia perfecto. Causa probable: sus escalas de tiempo llegan a 1 024 piezas, el largo de
  entrenamiento, y con textos más largos el estado de la memoria toma valores que nunca vio. Es una sola semilla.
- Convolución por canal con la capa densa más ancha: igual que la receta. La capa densa con puerta empeora.
- La memoria cuesta 17% de velocidad.

Siguiente prueba (lista, sin correr): memoria con escalas de tiempo más cortas, para que su estado sea el mismo a
cualquier largo y la memoria larga quede a cargo de la capa ASRN. `memoria_max` (por omisión 1 024) fija la escala más
larga; el cuaderno grande con `BLOQUES = True` compara ASRN con memoria de 1 024, 128 y 32 piezas.

## 6. Plan

| Paso | Qué | Cómo se comprueba |
|---|---|---|
| 1 (hecho) | Los 4 arreglos de la sección 3 + `medir_velocidad_v89.py` | Igual bit a bit; 1.05–1.21× en la A100 |
| 1b (hecho) | Vuelta de `w[i]` sin sumas atómicas (`_Tomar`) | Capa 1.2–2.8×, modelo 1.6–2.0× en la A100 |
| 1c (hecho) | Kernel Triton del escaneo (sección 5b): ~290 operaciones → ~6 | Modelo 1.8–2.3× más rápido que al inicio; empata al Transformer con 32 mil piezas |
| 2 y 3 (hecho) | Kernels Triton de la **ida** y la **vuelta** de la atención por casilleros (sección 5c) | Modelo 3× más rápido que al principio; más rápido que el Transformer con 32 mil piezas |
| 3b (hecho) | Vuelta sin atómicas | Modelo ~146 mil piezas/s; más rápido que el Transformer desde 16 mil piezas |
| — | Corrida completa del cuaderno con todo (`history/results/corrida_3/`) | Aprende igual (WikiText 57.0 contra 63.4); ~162 mil piezas/s; WikiText en 4.0 minutos (antes 12.7) |
| 4 (hecho) | CUDA Graphs: el paso entero grabado (`PasoGrafado`, sección 5d) | Capa sola 7× con textos cortos; en entrenamiento no alcanza (la GPU ya está 95–98% ocupada) |
| 5 (hecho) | `torch.compile` por partes (`compilar_partes`, sección 5e) | ASRN +26% (~183 mil piezas/s); el Transformer compilado también gana 31%: con 1 024 piezas sigue 1.45× más rápido |
| 6 (hecho) | Diseño sin código nuevo: una capa ASRN cada dos bloques (`largo_cada=2`), con los mismos parámetros (sección 5f) | Misma calidad (WikiText 57.06 contra 56.97; copia igual) y 16–18% más rápido: ~171 mil piezas/s sin compilar. **Por omisión desde octubre de 2026** |
| 6b (hecho) | "Cada 2" compilado por partes (sección 5f) | 214 908 piezas/s con 1 024: no llega a la meta de 215 mil (Transformer compilado: 271 307) |
| 7 (hecho) | Cada tabla copia solo su parte del vector (`partir_valores`, sección 5g) | +2% (218 788 piezas/s con 1 024): no vale cambiar el modelo; se quitó del código |
| 8 (hecho en CPU) | Menos conversiones de tipo (sección 5h) | −43% de números convertidos por capa (contado en CPU); velocidad en GPU: `medir_velocidad_a100.ipynb` |
| 9 (en curso) | Bloques locales mejores (sección 5i) | Memoria: −5.7% de perplejidad pero rompe la copia larga. Siguiente: `memoria_max` 128 y 32 (`pruebas_gpu_a100.ipynb` con `BLOQUES = True`) |
| 10 | Un modelo de 100 millones con FineWeb (`examples/train_100m/`) | Curvas de pérdida, perplejidad con 1 024 y 4 096 piezas y texto de ejemplo |
