# ASRN v89: análisis, kernel, protecciones y banco de pruebas

Revisión completa hecha desde cero en octubre de 2026: se analizó el código de `asrn_v89.py` pieza por pieza, se
buscaron con mediciones (no con suposiciones) los modos de falla numérica, se implementaron las protecciones en
[`asrn/estable.py`](../../asrn/estable.py) y se volvió a medir todo con un banco de pruebas nuevo
([`benchmarks/banco_asrn.py`](../../benchmarks/banco_asrn.py)). Todo lo medido aquí es nuevo; las pocas veces que
se cita una medición anterior del proyecto en GPU (para dimensionar una propuesta de kernel) se dice explícitamente.

**Dónde se midió.** En dos lugares:

- **CPU** (4 núcleos, sin GPU, PyTorch 2.14.1 y Triton 3.8): lo que no depende del hardware, medido exacto (FLOPs con
  el contador de PyTorch, bytes que se guardan para la vuelta con ganchos de autograd, errores numéricos, causalidad) y
  una primera batería de calidad con modelos de ~0.7 M de parámetros. Los kernels Triton corrieron en el intérprete de
  Triton (`TRITON_INTERPRET=1`), que comprueba las cuentas, no la velocidad.
- **Una NVIDIA A100-SXM4-80GB** (Colab, 9 y 11 de octubre de 2026; PyTorch 2.11 con CUDA 13.0, sin `torch.compile`, que
  este modelo necesita en 2.14) con el cuaderno `banco_a100.ipynb` (retirado en 1.0; hoy [`asrn_colab.ipynb`](../../benchmarks/asrn_colab.ipynb)): la estabilidad
  en la GPU de verdad, velocidad y memoria contra los rivales en su mejor versión disponible (Mamba-2 original de `mamba_ssm`, RWKV-7
  original de `flash-linear-attention`; Transformer++, la receta de Llama con FlashAttention-2) y calidad con 2.5–45 M de parámetros. Resultados en
  [`history/results/banco_a100_pt211/`](../results/banco_a100_pt211/RESUMEN.md).

**Dos ejecuciones.** La primera (9 de octubre) midió todo salvo RWKV-7: nuestro envoltorio armaba un RWKV-7 de una
sola capa y fla divide por (capas − 1). Era el último modelo de cada sección, así que las filas de los otros no se
vieron afectadas. La segunda (11 de octubre, con el código corregido, huella `176cfe250773`) agregó RWKV-7 en las seis
secciones y repitió la estabilidad con TF32 apagado; las demás filas quedaron tal cual. **Siguen pendientes** tres
cosas que la segunda ejecución no hizo (no se pidieron en la configuración): `mezcla_lse` en MQAR con la corrección,
la escritura del Transformer++ sin el artefacto de memoria y las tasas extra del Transformer++ en WikiText
(sección 6).

**Salvedades que valen para toda la parte de GPU** (se repiten donde pesan): nada está compilado ni usa CUDA graphs;
Mamba-2 corrió sin `causal-conv1d` (su camino sin fusionar: velocidad y memoria pesimistas); TF32 estuvo encendido
en toda la primera ejecución salvo en la comparación fp32 del kernel (sección 2.3), y en la segunda, salvo en las
pruebas numéricas de estabilidad; los tiempos son de una corrida por punto
(mediana de 2 a 5 repeticiones en la capa y el modelo; en la escritura, el promedio de 32 pasos de una sola pasada);
el Transformer++ es nuestra implementación de la receta de Llama (con FlashAttention-2, pero RMSNorm, RoPE y SwiGLU
sin fusionar); la calidad es a escala chica (los modelos de WikiText vieron 0.44 piezas por parámetro) y con una
grilla de dos tasas.

## Resumen

| | Resultado |
|---|---|
| Pruebas | En CPU, las 62 pruebas de v89 pasan antes y después de los cambios y las 15 nuevas de la capa estable también. En la A100: v89 59 pasan, 2 se saltan y 1 falla (`test_entrenar_aprende_a_copiar_y_reanuda`, que no aprende la copia: falla desde la quinta corrida del proyecto, sin explicar); la estable (14 pruebas en esa versión) 13 y 1 falla de la prueba misma (dispositivo), ya corregida |
| Fuga causal | **Encontrada y cerrada.** La máscara de v89 es un puntaje de −10⁹, no −∞: si todos los puntajes válidos quedan debajo (núcleo de tiempo enorme), la masa del softmax va a llaves no válidas, que incluyen la propia pregunta (copia la pieza *siguiente*) y piezas posteriores. Medido en PyTorch y en el kernel Triton: el futuro cambia el pasado (0.016 con γ = 10⁹ y 0.075 con γ = 10¹¹, igual por los dos caminos; 0.016 con el kernel compilado en la A100). Con la máscara exacta: 0 exacto, también en la A100 |
| Reloj en 32 bits | **Se degrada desde ~1 millón de piezas.** Con el `cumsum` de CUDA, la diferencia B_t − B_{t−1} entre piezas vecinas pierde 2.2% a 1 M, 8.6% a 4 M y 50% a 16 M, donde el reloj ya se detiene en el 16% de los pasos (1.1% a 4 M, 37% a 32 M): lo mismo que predijo la emulación en CPU. Su efecto en la capa depende de γ: en la A100 (sin TF32), la salida de v89 cambia 9.4·10⁻⁵ si el reloj está corrido 10⁶, 2.8·10⁻³ a 1.6·10⁷ y 8.8·10⁻² a 10⁹ (la misma prueba en CPU: 1.2·10⁻⁴, 2.8·10⁻³ y 7.8·10⁻²). La estable: 5.5·10⁻⁸, idéntico con los tres corrimientos, en CPU y en la A100 sin TF32 (con TF32 encendido daba 3.6·10⁻⁶: era TF32) |
| Reloj con bf16 o TF32 | v89 calcula w·ĥ del reloj en 16 bits: β con 3 cifras, y el error se acumula en B (1.3 unidades tras 4 096 piezas con d = 64 y 3.3 con d = 512; crece con el largo). En fp32 en GPU, TF32 hace lo mismo: B_t de v89 cambia 2.0 unidades en 65 536 piezas al encender TF32; el de la estable, 1.2·10⁻¹⁰ (`_ProyeccionIEEE`) |
| Gradientes y largo | **No crecen con el largo** (de 256 a 65 536 piezas con texto tipo Zipf y a 32 768 con vocabulario uniforme, en un modelo chico sin entrenar): la hipótesis de que el gradiente del reloj (cuya adjunta es una suma acumulada hacia atrás) explota con el largo no se observó |
| Costo por pieza | Constante en ASRN (5.1 MFLOP por capa con d = 512; 5.9 MFLOP el modelo de 0.7 M); el Transformer crece lineal (4.3 → 13.2 MFLOP de 256 a 4 096 piezas en el modelo chico) |
| Velocidad en la A100 | **Capa sola** (d = 512, lote 1, ida y vuelta; cada "capa" es una unidad distinta, sección 5.3): ASRN tiene un piso de ~12 ms hasta 16 mil piezas y pasa a la atención y a Mamba-2 hacia ~19 mil; con 262 144 piezas es 22.5× más rápida que la atención y 2.7× que el bloque de Mamba-2, con 2.0× y 1.3× su memoria. **Modelo de ~50 M** (32 768 piezas por paso): ASRN mantiene 170–173 mil piezas/s de 1 024 a 32 768 piezas; pasa al Transformer++ entre 1 y 4 mil piezas (2.17× con 32 768) y a Mamba-2 sin fusionar por 1.5× a todo largo. **RWKV-7** (fla) es el más lento a largo: con 262 144 piezas su capa tarda 7.8× la de ASRN con 1.9× su memoria, y su modelo entrena a 45–49 mil piezas/s (ASRN 3.5–3.8× más rápido) |
| Memoria guardada | ASRN: 44 KB por pieza y capa en PyTorch, **20 KB con el kernel Triton** (d = 512, fp32), constante con el largo |
| Latencia (CPU) | ASRN crece con exponente 1.14–1.22, el Transformer 1.83; se cruzan entre 2 y 8 mil piezas, y con 32 768 ASRN es 2.6–3.0× más rápido |
| Escritura | **El algoritmo es O(t) por pieza** (recorre toda la memoria): en CPU, 3.8 → 14.6 ms por pieza de 512 a 32 768 piezas de contexto. En la A100 el tiempo es plano (40.1–41.0 ms de 1 a 64 mil piezas), probablemente porque lo fija el lanzamiento de muchos kernels chicos (no se perfiló), y es la escritura más lenta de las medidas: 2.8× la del Transformer++, 1.6× la de Mamba-2 y 1.2× la de RWKV-7 (35 ms). Se puede hacer O(1) (sección 2.4), pero en GPU primero hay que bajar lanzamientos |
| Calidad (A100, 3 semillas) | **Copia** (entrenada con 1 024): ASRN 100% en el largo, 99.9% con 4× y 67.6% ± 27 con 16× (de 43 a 97% según la semilla); el Transformer++ cae al azar fuera de su largo y Mamba-2 no la aprendió. **MQAR** con 2× y 4× los pares: ASRN 99.6% y 98.5% contra 83.4% y 54.4% de Mamba-2. **WikiText-103** (~45 M, 20 M de posiciones de entrenamiento): perplejidad 67.8 ASRN, 72.4 RWKV-7, 74.2 Mamba-2, 93.7 Transformer++; con 4 096 piezas, 63.0, 70.2, 73.0 y 125.6. **RWKV-7** (fla): no terminó de aprender la copia (30% en el largo), MQAR 99.9% en el largo pero 59% y 30% con 2× y 4× los pares, y en texto queda detrás de ASRN en cada semilla, aunque con n = 3 no es significativo (p = 0.15) y en una corrida anterior, con otro protocolo y la reimplementación, fue el mejor. Falta una grilla de tasas más ancha (con 10⁻³, Mamba-2 queda por delante de ASRN) |
| Ablación (A100) | Lo único concluyente: sin la capa ASRN no se resuelve ni copia ni MQAR, y mezclar las tablas por logsumexp sale de la meseta de la copia 600–800 pasos antes (sin mejorar la generalización). Con 3 semillas, quitar la puerta, el reloj o el sufijo no da una diferencia medible en estas tareas, que tampoco las ponen a prueba |

## 1. Análisis arquitectónico y matemático

Notación del código: B lote, L largo, d ancho, T = 8 tablas, b = 14 bits, P = T·b = 112, C = 32 candidatos, cf = 16
(confirmación), K = 4 (sufijo), n = 4 (grupos de 1 a 4 piezas). La capa calcula

```math
h'_t = h_t + g_t \cdot W_o\, c_t, \qquad c_t = \frac{1}{\max(1, n_t)} \sum_{\tau} \sum_{s \in \mathcal{N}_\tau(t)} a_\tau(t,s)\, W_v E_{s+1}
```

### 1.1 El hash aprendido (8 tablas) y la función de puntuar y elegir

**Qué hace.** Dos redes chicas dan φ_t ∈ ℝ¹¹² (4 tablas ven la pieza sola, 4 la pieza y la anterior); el casillero de la
tabla τ son los signos de sus 14 números. Cada pieza compara con a lo sumo C = 32 compañeros de casillero anteriores
por tabla (los más recientes), con el puntaje

```math
s_\tau(t,s) = \frac{\langle \varphi_t, \varphi_s \rangle}{\sqrt{112}} + \frac{\langle W_q \hat h_t, W_k \hat h_s\rangle}{4}
+ w^{\text{suf}}_{\ell(t,s)} + \log\alpha_s - \frac{\gamma_\tau}{C}(B_t - B_s)
```

y un softmax **por tabla**; las salidas de las tablas con candidatos se **promedian**.

**Lo que aporta.**
- Trabajo fijo por pieza: 8 × 32 candidatos = 256 comparaciones como máximo, sin importar el largo. Es la misma idea que
  la atención LSH de Reformer (ordenar por casillero, atender dentro del bloque y el anterior), con tres diferencias: el
  hash se aprende, la búsqueda es causal y por recencia, y el puntaje suma evidencia exacta (sufijo) y tiempo.
- El sufijo exacto (hasta 4 piezas iguales hacia atrás) es lo que permite copiar sin error: es una señal discreta que
  una similitud de vectores no da.
- La recursión a la pieza siguiente (se copia W_v E_{s+1}) convierte la búsqueda en un mecanismo de inducción explícito.

**Cuellos de botella y riesgos.**
1. **El casillero no tiene gradiente.** Los signos de φ no son derivables; el hash aprende solo de forma indirecta: el
   gradiente del puntaje acerca φ_t y φ_s de los pares útiles que *ya* comparten casillero, y eso tiende a alinear sus
   signos. Un par útil que nunca cae junto no recibe señal. Con bf16, un φ cerca de 0 cambia de signo por redondeo (el
   documento de rendimiento ya lo vio en el hash de contexto).
2. **Recencia truncada.** En un casillero frecuente (una pieza común) solo se ven las 32 apariciones más recientes: el
   alcance largo de lo frecuente depende de las tablas de contexto (pares de piezas, casilleros más raros).
3. **Las tablas comparten el puntaje.** El producto ⟨φ_t, φ_s⟩ usa los 112 números, no los 14 de la tabla: las tablas solo
   difieren en *qué candidatos* ven y en su γ_τ.
4. **Promedio uniforme entre tablas.** Una tabla con candidatos malos pesa lo mismo que una con un candidato exacto; la
   puerta corrige solo globalmente (con `mejor` y `pico`). Reformer combina sus rondas pesándolas por su normalizador
   (logsumexp), que equivale a un solo softmax sobre la unión de candidatos. Es una mejora candidata barata (el kernel ya
   calcula el máximo y la suma de cada fila), pero **cambia el modelo** y hay que medirla.
5. **El softmax por tabla ya es estable** (resta el máximo, en 32 bits). Su defecto real era la máscara finita: −10⁹ no
   es −∞ (sección 3).
6. **Cuentas:** por pieza y tabla, 2C·(P + cf) = 8 192 multiplicaciones para los puntajes y 2C·d para la salida; la
   ventana de 2C llaves solo tiene C válidas por pregunta (banda), así que la mitad de esas cuentas se descartan. No
   importa: la capa está limitada por memoria, no por cuentas (sección 2).

### 1.2 El reloj propio y el estado normalizado de h

**Qué hace.** Desde el estado normalizado ĥ_t = LN(h_t):

```math
\beta_t = \mathrm{softplus}(w_\beta \cdot \hat h_t + b_\beta) \ge 0, \qquad
\alpha_t = 2\sigma(w_\alpha \cdot \hat h_t) \in (0,2), \qquad B_t = \sum_{u \le t} \beta_u
```

El núcleo de memoria es α_s e^{−κ(B_t − B_s)}: un decaimiento exponencial medido en **tiempo propio**, que cada pieza
avanza lo que la red decida (β ≈ 0 dentro de una cita: nada se olvida; β grande en un cambio de tema).

**Lo que aporta.**
- Decaimiento que depende del contenido (como el Δ de Mamba), pero aplicado a una búsqueda dispersa y a las
  intensidades de Hawkes.
- Como el núcleo es exponencial, en el puntaje el tiempo entra **lineal** en B: el término se separa en (llave) −
  (pregunta) y la vuelta no necesita guardar un tensor C × 2C. Es una decisión muy buena para la memoria.
- Todo arranca neutro (β = 1, α = 1, γ = 0): con pesos de v85, v89 calcula lo mismo.

**Cuellos de botella y riesgos (medidos).**
1. **B es una suma acumulada en 32 bits.** Un número de 32 bits tiene 24 bits de mantisa: a B = 2²⁰ el paso mínimo es
   0.06, a 2²⁴ es 1. Medido sobre β = softplus(N(0,1)) (media 0.81):

   | Largo | Error de B_t − B_{t−1}, acumulador de 32 bits (CPU) | Salida redondeada (CPU) | **`cumsum` de CUDA (A100)** | Reloj detenido (CPU) | Reloj detenido (A100) |
   |---|---|---|---|---|---|
   | 2¹⁶ | 0.13% | 0.16% | — | 0% | — |
   | 2²⁰ (1 M) | 2.0% | 2.5% | 2.2% | 0.01% | 0.03% |
   | 2²² (4 M) | 8.1% | 10% | 8.6% | 0.9% | 1.1% |
   | 2²⁴ (16 M) | 44% | 64% | 50% | 16% | 16% |
   | 2²⁵ (32 M) | — | — | 100% | — | 37% |
   | 2²⁶ (64 M) | 100% | 100% | — | 61% | — |

   El acumulador de 32 bits es el caso de la escritura paso a paso (`c['B'][:, t] = c['B'][:, t-1] + β`). El `cumsum`
   de la GPU cae entre las dos columnas de CPU, como se había predicho: sumar en paralelo no rescata la precisión,
   porque el límite es la mantisa de B, no el orden de la suma. Con más distancia el error relativo baja (en la A100,
   0.19% entre piezas a 32 de distancia y 0.007% a 1 024, con 2²⁰ piezas), pero el absoluto queda en torno a un ulp de
   B y crece con el largo; ese error, por γ/C, es el que entra al puntaje.
2. **Cancelación en el puntaje.** v89 calcula (log α_k + tasa·B_k) − tasa·B_q: dos números grandes que se restan después
   de sumarles el puntaje. Medido sobre la capa entera, desplazando el reloj (solo importan diferencias, así que la
   salida no debería cambiar), en CPU: 4·10⁻⁵ a 10⁵, 2.8·10⁻⁴ a 10⁶, 4.7·10⁻³ a 1.6·10⁷, 2·10⁻² a 6.4·10⁷ y 0.11 a
   10⁹ (una capa de prueba con γ de 0.5 a 4). La prueba de la A100 usa otra capa (`banco_gpu.py`): con el kernel
   compilado, 1.1·10⁻⁴ a 10⁶, 2.8·10⁻³ a 1.6·10⁷ y 8.8·10⁻² a 10⁹ (809 veces más error para un corrimiento mil veces
   mayor; sin TF32, en la segunda ejecución, 9.4·10⁻⁵, 2.8·10⁻³ y 8.8·10⁻²); esa misma prueba en CPU da 1.2·10⁻⁴,
   2.8·10⁻³ y 7.8·10⁻²: lo mismo. La capa estable: 5.5·10⁻⁸ en CPU (con el intérprete de Triton) y en la A100 con
   TF32 apagado, **idéntico** con los tres corrimientos y también entre dos relojes corridos (10⁶ contra 10⁹). Un
   error que no cambia con el corrimiento no es deriva: es una diferencia de redondeo fija entre la base y cualquier
   reloj corrido. Con TF32 encendido (la primera ejecución), la estable daba 3.55·10⁻⁶, también constante: TF32 en
   las multiplicaciones que siguen al término de tiempo (W_o, `g_conf`) agrandaba ese redondeo, sin deriva.
3. **β no tiene techo.** Con pesos rotos, B pasa de 3.4·10³⁸ en pocas piezas, y el núcleo da ∞ − ∞ = NaN (prueba
   `test_beta_enorme_no_rompe_la_capa`).
4. **Precisión de w·ĥ.** Para ahorrar una lectura de ĥ, v89 calcula el reloj en la misma multiplicación que Qc y Kc, en
   16 bits con autocast. En fp16 eso puede desbordar (d sumandos > 65 504 ⇒ β = ∞); en bf16 β tiene 3 cifras y B
   acumula ese error: con la capa de `test_reloj_en_32_bits_con_autocast`, 0.39, 0.65, 0.93 y 1.27 unidades tras 512,
   1 024, 2 048 y 4 096 piezas con d = 64, y 3.3 tras 4 096 con d = 512. **En GPU hay un tercer caso: TF32.** Con `torch.backends.cuda.matmul.allow_tf32`
   encendido (lo encienden `entrenar()` y el cuaderno), un producto de matrices "de 32 bits" se hace con 10 bits de
   mantisa: la proyección del reloj de la capa estable, aunque se pidiera en 32 bits, pudo ir así en toda la primera
   corrida en A100. Corregido después de esa corrida: en GPU el reloj se proyecta con `_ProyeccionIEEE` (multiplicar y
   sumar elemento a elemento, que no pasa por TF32; son 2 columnas, ~2d cuentas por pieza). Medido en la A100 (65 536
   piezas, d = 512): al encender TF32, B_t de v89 en fp32 cambia hasta 2.0 unidades, y el de la estable 1.2·10⁻¹⁰ (el
   redondeo de 64 bits). Con autocast bf16, v89 no cambia con TF32 (ya calcula en 16 bits, con el error de arriba).
5. **El estado normalizado en sí es seguro:** LayerNorm corre en 32 bits también con autocast y su ε = 10⁻⁵ evita dividir
   por cero; h tendría que pasar de ~10¹⁹ para desbordar el cuadrado. No hizo falta protegerlo.
6. **Hipótesis descartada con datos:** la adjunta de una suma acumulada es otra suma acumulada hacia atrás
   (∂L/∂β_u = Σ_{t≥u} ∂L/∂B_t), así que se esperaba que el gradiente del reloj creciera con el largo. Medido con el mismo
   número de piezas por paso, de 256 a 65 536 piezas con texto tipo Zipf y de 256 a 32 768 con vocabulario uniforme,
   en un modelo chico (d = 64) sin entrenar: la norma del gradiente del reloj queda plana en cada distribución
   (1.7·10⁻⁵ con Zipf, 0.7·10⁻⁵ con vocabulario uniforme) y la de γ entre 2 y 6·10⁻⁷ (con vocabulario uniforme sube
   1.7 veces de 256 a 32 768), ≈ 0.0% de la norma total. Las señales de la puerta se mueven poco: el percentil 1 de
   `mejor` queda entre 0.2 y 0.4, y su mínimo se vuelve menos extremo con el largo (de −1.8 a −0.6). No se agregó
   ninguna protección para esto; en un modelo grande entrenado habría que volver a medirlo.

### 1.3 El escaneo asociativo O(N), los vistos y Hawkes

**Qué hace.** Tres recurrencias lineales S_i = a_i S_{i−1} + b_i con a_i ∈ [0, 1]:

- **Vistos** v_n(t): si el grupo de n piezas que termina en t ya salió antes. Se obtiene ordenando las claves de los
  grupos (orden estable): la aparición anterior queda justo antes.
- **Hawkes** λ_n(t) = Σ α_s e^{−κ_n(B_t − B_s)} sobre las apariciones anteriores del mismo grupo: un escaneo segmentado en
  ese mismo orden, con a_i = e^{−κ ΔB} si sigue el mismo grupo y 0 si empieza otro.
- **Racha** ℓ_t = v_4(t)(e^{−κ_r β_t} ℓ_{t−1} + 1): cuánto lleva la cita en curso. La puerta la lee con una función lineal
  por tramos no monótona, que aprende a cortar las citas demasiado largas.

Las funciones x ↦ a x + b se componen como (a₁, b₁)∘(a₂, b₂) = (a₁a₂, a₂b₁ + b₂), que es asociativa: el escaneo par-impar
hace trabajo O(N) con profundidad O(log N); el kernel Triton hace bloques de 1 024 con `tl.associative_scan` y un
acarreo entre bloques.

**Lo que aporta.** Rompe el aliasing de las señales binarias (una cita de 5 piezas y una de 50 se ven iguales con solo
`vistos` y el sufijo ≤ 4): la puerta sabe *cuánto* y *cuán recientemente* se repitió algo, y cuánto lleva la cita.

**Estabilidad (buena por diseño).** Con a ∈ [0, 1] solo hay productos que se achican y sumas; un a que se va a 0 por
debajo del mínimo de 32 bits solo reinicia la recurrencia, que es lo correcto. λ ≤ 2 × (número de apariciones) y entra a
la puerta como log(1 + λ); la racha entra por una función constante pasado su último nodo. Las claves de los grupos son
sumas en 64 bits con coeficientes al azar: dos grupos distintos con la misma clave tienen probabilidad ~10⁻¹⁶ por par
(un falso "visto" cada ~10¹⁶ pares de posiciones; despreciable).

**Cuellos de botella.**
1. **El orden de los grupos se repite en cada capa.** Depende solo de las piezas (y de `c_vistos`, que sale de la misma
   semilla en todas las capas), pero cada capa ASRN hace sus 4 ordenamientos estables de claves de 64 bits.
2. **El kernel del escaneo usa un programa por fila.** Con lote 1 son 4 filas de Hawkes y 1 de racha: 5 programas, que
   ocupan 5 de los 108 multiprocesadores de una A100 y recorren el largo en serie, de 1 024 en 1 024.
3. **Alrededor del escaneo hay muchas operaciones chicas** (reunir por `orden`, exp, relleno, devolver al orden del
   texto), cada una con su viaje a la memoria.
4. **La escritura no aprovecha nada de esto:** `_repeticion_paso` compara la clave nueva con *todas* las anteriores y
   `_atender` puntúa contra *toda* la memoria antes de quedarse con las 32 más recientes de cada casillero: O(t) por pieza.

## 2. El kernel propio: análisis de hardware y propuestas

### 2.1 Dónde se van los bytes

La capa está limitada por memoria: hace del orden de 10 cuentas por byte, y una A100 necesita ~100 para que las
cuentas sean el límite. Lo que hay que bajar son bytes. Conteo a partir del código de los kernels (por pieza, por capa,
d = 512, bf16; cada posición es pregunta una vez y llave dos veces en cada una de las 8 tablas):

| Parte | Ida | Vuelta | Qué lo causa |
|---|---|---|---|
| Lecturas de llaves (φ, Kc, V, sufijo, reloj) | ~2.6 KB × 8 tablas | lo mismo, más ∂salida | cada llave se lee en **dos** ventanas (bloque actual y siguiente) |
| Salida por tabla `OUT` (B·T, L, d) y su suma | 8 KB escritos + 8 KB leídos + 1 KB | — | cada tabla escribe su vector entero y se suman después |
| Gradientes de llaves en dos lugares `(2, B·T, L, ·)` y su suma | — | ~20 KB escritos + ~23 KB leídos | sin atómicas, cada llave escribe en el lugar de cada ventana |
| **Total de los kernels** | **≈ 41 KB** | **≈ 81 KB** | ≈ 122 KB por pieza y capa |

A 1.55 TB/s (la A100 de 40 GB), 122 KB son ~80 ns por pieza; a ~2.0 TB/s (la de 80 GB, la de esta revisión), ~60 ns.
Es del orden de lo que midió el proyecto antes en una A100 para estos kernels
(2–3 ms por capa con 32 mil piezas, [RENDIMIENTO.md](RENDIMIENTO.md), sección 5g): **los kernels ya están cerca del
ancho de banda**, y lo que queda se gana bajando bytes, no reordenando cuentas. Según esa misma medición, con textos de
1–32 mil piezas pesa más lo que rodea a los kernels (hash, ordenar, piezas anteriores, conversiones) que los kernels.

**Lo que midió la A100 en esta revisión** (sección 5.3) lo confirma: la capa estable entera (ida y vuelta, d = 512,
bf16, lote 1) crece 239 ns por pieza entre 65 y 262 mil piezas, unas 4 veces el piso de ~60 ns de los kernels en esta
GPU, y por debajo de 16 mil piezas su tiempo es un piso de ~12 ms que no depende del largo. Ese piso es, muy
probablemente, el costo de lanzar muchas operaciones chicas sin compilar: en esta corrida no se perfiló, pero el
proyecto midió antes que grabar el paso con CUDA Graph baja una capa con 1 024 piezas de 11.4 a 1.6 ms
([RENDIMIENTO.md](RENDIMIENTO.md), sección 5d). Hay margen en las dos puntas: bajar lanzamientos (compilar, CUDA graphs, fusionar lo que rodea a los
kernels) para los textos cortos, y las propuestas de la tabla siguiente para los largos.

**SRAM y registros.** Un programa de la ida tiene vivos φ de 32 preguntas y 64 llaves (8 + 16 KB en bf16), los puntajes
y sus máscaras (varios tiles de 32 × 64 en 32 bits, ~8 KB cada uno) y un bloque de V: ~60 KB por programa. Con 4 warps
son ~120 registros de 32 bits por hilo: cabe justo en la ida; en la vuelta (con ∂p, ∂s y la atención vivos a la vez) es
muy probable que derrame a memoria local. Hoy `num_warps`, `num_stages` y `BD = 64` están fijos.

### 2.2 Propuestas, en orden de beneficio esperado

| # | Propuesta | Qué ahorra (estimado del código) | ¿Cambia el modelo? |
|---|---|---|---|
| 1 | **Ventana deslizante persistente:** un programa recorre varios bloques seguidos de su tabla y guarda en registros el bloque actual, que es el "anterior" del siguiente. Cada llave se lee una vez, y en la vuelta su gradiente se acumula en registros y se escribe una vez | −34% de bytes de los kernels (~41 de 122 KB por pieza); elimina la mitad de los espacios `(2, B·T, L, ·)`: con 262 mil piezas y d = 512, ~2.7 GB menos (el proyecto midió antes 12 GB sin atómicas contra 7 GB con atómicas) | No: mismas cuentas, determinista |
| 2 | **Compartir el orden de los grupos entre capas:** calcular una vez por paso las claves, `orden` y `vistos` (dependen solo de las piezas) y pasarlos a cada capa | (capas ASRN − 1) × 4 ordenamientos estables de claves de 64 bits (~1 KB por pieza entre los 4) y sus lanzamientos | No: mismos números |
| 3 | **Ordenar por casillero con 14 bits e índices de 32 bits:** `argsort` estable de PyTorch ordena claves de 32 bits con valores de 64 (≈ 4 pasadas × 12 B × 2); un radix de 2 pasadas de 7 bits con índices `int32` (CUB con `end_bit = 14`, o Triton) mueve ~3× menos. Con filas de ≥ 64 mil piezas, un conteo con 2¹⁴ casilleros en SRAM (64 KB) lo hace en una pasada | ~3× menos bytes en los 8 ordenamientos por capa; el tiempo pasa a ser O(N) con constante chica (la "N log N" del README es la de un ordenamiento por comparación; en GPU es radix) | No |
| 4 | **Escaneo con rejilla (filas × bloques) y fusionado:** fase 1 escanea cada bloque en paralelo, fase 2 escanea los acarreos (o look-back desacoplado), fase 3 corrige; y el mismo kernel lee `orden`, B y log α, calcula a y b, y escribe λ ya en el orden del texto | Ocupación con lote 1: de 5 programas a 5·L/1 024; ~10 operaciones y sus viajes a memoria menos por capa | No |
| 5 | **Registro empaquetado por posición:** B alta, B baja, log α, las 4 piezas anteriores y el casillero en 32 bytes contiguos | De ~4 sectores de 32 B a 1 por cada llave reunida (~3 KB por pieza) | No |
| 6 | **Autotune** de `num_warps` ∈ {4, 8}, `num_stages` ∈ {1, 2, 3}, `BD` ∈ {32, 64, 128}, con la vuelta en 8 warps | Evita derrames de registros en la vuelta | No |
| 7 | **Mezcla de tablas por logsumexp** (sección 1.1): el kernel ya tiene máximo y suma por fila; guardar LSE por (tabla, posición) y pesar las tablas por e^{LSE} | +4 B por pieza y tabla | **Sí**: hay que reentrenar y medir |
| 8 | **Escritura O(1) por pieza** (sección 2.4) | De O(t) a O(T·C) por pieza | No |

Descartado por datos propios del proyecto: que cada tabla copie solo d/T números (`partir_valores`) dio +2% al
entrenar con 1–32 mil piezas, porque la salida por tabla no era el cuello de botella con esos largos.

### 2.3 Lo que sí se implementó en el kernel en esta revisión

El modo `ESTABLE` (parámetro `tl.constexpr` de los dos kernels; apagado, el kernel compilado es el de siempre):

- El reloj llega en dos partes de 32 bits (alta y baja de un reloj de 64 bits). El término de tiempo usa
  `dt = (B_k − B_q)_alta + (B_k − B_q)_baja` **antes** de multiplicar por la tasa: entre piezas cercanas la resta de la
  parte alta es exacta (lema de Sterbenz) y la baja corrige el redondeo. Cuesta 4 bytes más leídos por pregunta y llave.
- Máscara exacta: el máximo del softmax se toma solo entre las llaves válidas y las no válidas pesan 0.

Comprobado con el intérprete de Triton: en modo `ESTABLE` el kernel da lo mismo que la versión de PyTorch de la capa
estable (ida y vuelta, fp32 y bf16), no tiene la fuga, y su salida no cambia (≤ 8·10⁻⁸) con el reloj desplazado hasta
10⁹. Sin `ESTABLE`, la prueba de siempre (`test_casilleros_triton_igual_que_pytorch`) sigue pasando.

**Compilado en la A100:** con TF32 apagado, el kernel `ESTABLE` coincide con PyTorch a 4.9·10⁻⁸ en la salida y 4.2·10⁻⁶
en el peor gradiente (fp32; 5.3·10⁻⁶ en la segunda ejecución); en bf16 queda a 0.096 de la respuesta en fp32, contra
0.091–0.107 de PyTorch en bf16 en las dos ejecuciones: el mismo orden de error. Cierra la fuga (v89 0.016, estable 0
exacto). Con el reloj desplazado y TF32 apagado, la salida de la estable cambia 5.5·10⁻⁸ con los tres corrimientos y
la de v89 de 9.4·10⁻⁵ a 8.8·10⁻² (sección 1.2). Una capa con 2²⁰ piezas (solo ida, d = 64, bf16): v89 35.2 ms y
2.185 GB, estable 36.3 ms (32.4 ms en la segunda ejecución, ya con `_ProyeccionIEEE`) y 2.198 GB (+12 bytes por
pieza; una medición por ejecución: las diferencias de tiempo son ruido). Las dos salidas quedan a 8.2·10⁻⁴: a un millón de piezas difieren en
menos de 0.1%, una cifra que mezcla el reloj, la proyección en bf16 de v89, la máscara y el redondeo, y que no aísla
la deriva.

### 2.4 Escritura en O(1) por pieza

Medido (modelo d = 128, CPU): 3.8 ms por pieza con 512 piezas de contexto, 5.1 con 2 048, 8.9 con 8 192 y 14.6 con
32 768. El costo por pieza de contexto baja en este rango (0.81, 0.63 y 0.23 µs entre puntos seguidos: la curva todavía
no es lineal); que el algoritmo sea O(t) sale del código. Si el marginal se quedara en 0.23 µs, con un millón de
piezas detrás serían ~0.23 s por pieza nueva (estimación). La búsqueda solo usa las C más recientes de cada casillero, así que no hace falta mirar toda la memoria:

- **Por tabla, un anillo por casillero** con las posiciones de sus últimas C apariciones: T × 2¹⁴ × C enteros de 32
  bits = 16 MB por capa. Una pieza nueva lee sus 8 anillos (256 posiciones), reúne φ, Kc, V y el reloj de esas
  posiciones y actualiza los anillos: O(T·C·(P + d)), constante.
- **Por grupo de 1 a 4 piezas, una tabla hash** clave → (λ, B y α de su última aparición). Hawkes se actualiza en O(1):
  λ_nuevo = e^{−κ(B_t − B_último)}(λ_último + α_último).

Con eso, la memoria sigue siendo O(t) (como un caché de llaves), pero el tiempo por pieza queda constante como en
Mamba-2 y RWKV-7. Las posiciones que ya no están en ningún anillo no se vuelven a leer, así que se podrían descartar:
la memoria quedaría acotada por T·2¹⁴·C posiciones.

**En la A100 esto todavía no es lo que más pesa.** Con el modelo de ~50 M y lote 1, escribir cuesta 40.1–41.0 ms por
pieza con 1, 4, 16 y 64 mil piezas de contexto: plano. La pendiente medida no se distingue de cero (~7–9 ns por pieza de
contexto entre las puntas o con un ajuste lineal; con el ruido de ±0.4 ms, la cota al 95% es de ~40 ns), contra
0.23 µs en CPU con un modelo mucho más chico: el trabajo O(t) queda escondido bajo un costo fijo por paso que,
muy probablemente, es el lanzamiento de las muchas operaciones chicas de cada paso (el Transformer++, que lee toda su
caché, también queda plano; no se perfiló). El orden de las mejoras en GPU es entonces: primero bajar lanzamientos
(compilar o CUDA graphs, y fusionar el paso), después la escritura O(1), que importa recién con contextos de
cientos de miles de piezas.

## 3. Protecciones de estabilidad

`mascara_exacta`, `reloj_64` y `beta_max` responden a fallas medidas (sección 1); `log_alfa_min`, `mejor_max`,
`tope_puerta` y `kappa_min` son preventivas: la falla que evitan se deduce del código y no se midió. Las del **grupo A** no cambian nada en el régimen normal (la capa estable
da los mismos números que v89 salvo el redondeo: 1.3·10⁻⁷ en los logits) y están encendidas por omisión. Las del
**grupo B** cambian la función: apagadas por omisión (los modelos guardados dan lo mismo) y pensadas para entrenar desde
cero con textos largos.

| Protección | Grupo | Falla que evita | Costo |
|---|---|---|---|
| `mascara_exacta` | A | Fuga causal cuando los puntajes válidos bajan de −10⁹ (PyTorch, kernel y escritura) | Ninguno: en el régimen normal, los mismos bits |
| `reloj_64` | A | Precisión del reloj desde ~1 M de piezas; cancelación en el puntaje; reloj detenido desde 16 M; w·ĥ del reloj en 16 bits o en TF32 (medido: 2.0 unidades de B en 65 536 piezas) | 8 bytes por pieza para B (en vez de 4); +4 B leídos en el kernel; 2 columnas de un producto en 32 bits (en GPU, sin TF32: `_ProyeccionIEEE`). Medido en la A100 para todo el grupo A, con la versión anterior a `_ProyeccionIEEE`: +3% de tiempo con 262 mil piezas y +4–5% de memoria a todo largo (ĥ guardado en 32 bits) |
| `beta_max = 10⁴` | A | B infinito con pesos rotos (NaN en toda la capa) | Nada (tope muy por encima de lo normal) |
| `log_alfa_min = −30` | A | log α → −∞ (otra forma de bajar de −10⁹; preventiva) | Nada |
| `mejor_max = 10⁴` | A | Un −10⁹ entrando a la puerta por la señal `mejor` (preventiva) | Nada |
| `tope_puerta` (10) | B | Puerta saturada (gradiente casi nulo) por señales fuera de lo visto (preventiva) | Cambia la puerta: logit → c·tanh(logit/c) |
| `kappa_min` (1/(4·L_entr)) | B | Con κ → 0, λ crece con el largo del texto: la puerta ve en inferencia valores que nunca vio (preventiva; en la sonda de gradientes, log(1 + λ) quedó plano de 256 a 65 536 piezas) | Cambia Hawkes: memoria máxima acotada |
| `RecorteDinamico` | entrenamiento | Con un solo recorte global, un pico en pocos números (reloj, γ) achica en ese paso el gradiente de todo el modelo | Una norma por grupo y paso; sin sincronizar con la GPU |

**Lo que no hizo falta.** Estabilizar el softmax por tabla (ya resta el máximo en 32 bits), proteger LayerNorm (32 bits
con ε), ni recortar el gradiente del reloj por el largo (se midió que no crece). No se agregó ningún `nan_to_num`
dentro de la capa: esconder un NaN impide que `entrenar()` salte el paso, que es lo correcto cuando algo se rompe.

**Recorte dinámico por grupos.** Para cada grupo (reloj y tasas; puerta; resto) se lleva una media móvil exponencial de la
norma y de su cuadrado; el umbral es min(tope, μ + z·σ) después de un calentamiento, y si la norma lo pasa, solo ese
grupo se escala. La media se actualiza con la norma ya recortada (un pico no sube el umbral siguiente) y no se
actualiza con NaN. Probado: con un pico de 10⁴ en el reloj, el reloj se recorta al umbral y el grupo "resto" queda
intacto bit a bit (el de la puerta no se comprueba).

## 4. Código

La capa está en [`asrn/estable.py`](../../asrn/estable.py): `CapaASRNEstable` hereda de `CapaASRN` (mismos parámetros y
mismos nombres: los archivos guardados cargan tal cual) y reescribe la ida con la forma de cada tensor. Resumen del
recorrido:

```python
def forward(self, E, h, piezas=None):
    # E (B, L, d) vectores de pieza · h (B, L, d) estado · piezas (B, L) enteros
    sig = firmas_de_piezas(piezas)                    # (B, L)      firma exacta de cada pieza
    Qc, Kc, beta, log_alfa = self._contexto(h)        # (B, L, cf) ×2 · (B, L) beta en 64 bits · (B, L) log alfa
    Bt = beta.cumsum(1)                               # (B, L)      reloj acumulado, en 64 bits
    acc = self._acumular(E, Qc, Kc, Bt, log_alfa, sig)
    #   phi (B, L, 112) -> casilleros (B·T, L) -> orden (B·T, L) -> ventanas (W, C, 2C) puntajes
    #   -> softmax con máscara exacta -> (W, C, d) -> de vuelta al texto y sumado: salida (B, L, d)
    #   + señales de la puerta (B, L, 1): tablas con candidatos, mejor, pico, tablas de contexto
    acc.update(self._repeticion(sig, beta, Bt, log_alfa))
    #   vistos (B, L, 4) · hawkes (B, L, 4) · racha (B, L)
    return self._combinar(acc, h)                     # (B, L, d)   h + puerta · W_o c
```

**El reloj propio, con sus protecciones** (`_contexto`):

```python
hh32 = self.ln_ctx(h)                                 # (B, L, d) estado normalizado, en 32 bits
hh = _a_autocast(hh32)                                # en 16 bits si hay autocast
if p.reloj_64 and h.is_cuda:                          # GPU: Qc, Kc como siempre; el reloj en 32 bits sin TF32
    y = F.linear(hh, torch.cat([self.cq.weight, self.ck.weight]))                     # (B, L, 2cf)
    u = _ProyeccionIEEE.apply(hh32.float(), self.w_reloj.float(), self.b_reloj.float())   # (B, L, 2)
elif p.reloj_64 and hh.dtype != hh32.dtype:           # CPU con autocast: Qc, Kc en 16 bits; el reloj en 32
    y = F.linear(hh, torch.cat([self.cq.weight, self.ck.weight]))                     # (B, L, 2cf)
    with torch.autocast(device_type=h.device.type, enabled=False):
        u = F.linear(hh32.float(), self.w_reloj.float(), self.b_reloj.float())        # (B, L, 2)
else:                                                 # sin autocast: una multiplicación, como v89
    y = F.linear(hh, torch.cat([self.cq.weight, self.ck.weight, self.w_reloj]), F.pad(self.b_reloj, (2 * cf, 0)))
    u = y[..., 2 * cf:].float()
beta = F.softplus(u[..., 0]).clamp(max=p.beta_max)                     # (B, L) >= 0 y acotado
log_alfa = (F.logsigmoid(u[..., 1]) + math.log(2)).clamp(min=p.log_alfa_min)
beta = beta.double()                                  # reloj_64: B_t = cumsum(beta) se acumula en 64 bits
```

El término de tiempo resta en 64 bits antes de multiplicar (`_TiempoRelativo`, que para la vuelta guarda solo sus
entradas, igual que v89), y el softmax usa la máscara exacta:

```python
def _softmax_valido(sc, valido, exacta):              # sc, valido (W, C, 2C)
    hay = valido.any(-1, keepdim=True)                # (W, C, 1)
    maximo = sc.detach().masked_fill(~valido, float('-inf')).amax(-1, keepdim=True)
    ref = torch.where(hay, maximo, torch.zeros_like(maximo))
    att = torch.softmax((sc - ref).masked_fill(~valido, prohibido), dim=-1) * hay
    # la mejor válida queda en 0 exacto; una no válida (-1e9) nunca gana. En el régimen normal, los mismos bits:
    # el softmax ya restaba ese mismo máximo por dentro
```

**La puerta, con sus protecciones** (`_combinar`):

```python
salida = acc['salida'] / acc['peso_total'].clamp(min=1)                 # (B, L, d) promedio de tablas
mejor = torch.where(pt > 0, acc['mejor'], 0).clamp(-p.mejor_max, p.mejor_max)
senales = torch.cat([pt / T, mejor, acc['pico'] / pt.clamp(min=1), acc['peso_ctx'] / Tc], -1)   # (B, L, 4)
with torch.autocast(device_type=h.device.type, enabled=False):          # todo el logit en 32 bits
    logit = F.linear(h, self.g.weight.to(h.dtype), self.g.bias.to(h.dtype))                      # (B, L, 1)
logit = logit + self.g_conf(senales) + (acc['vistos'] * self.w_vistos).sum(-1, keepdim=True)
logit = logit + (torch.log1p(acc['hawkes']) * self.u_hawkes).sum(-1, keepdim=True)              # acotado: log1p
logit = logit + self._respuesta_racha(acc['racha']).unsqueeze(-1)       # constante pasado el último nodo
if p.tope_puerta is not None:                                           # grupo B
    logit = p.tope_puerta * torch.tanh(logit / p.tope_puerta)
return h + torch.sigmoid(logit) * self.drop(self.o(salida))
```

**En el kernel** (`asrn_v89._cas_puntajes` y `_cas_softmax`, modo `ESTABLE`):

```python
if ESTABLE:   # reloj en dos partes: la resta antes de multiplicar
    dt = (bk[None, :] - bq[:, None]) + (bk_bajo[None, :] - bq_bajo[:, None])
    sc = sc + peso + lak[None, :] + tasa * dt
...
if ESTABLE:   # máximo entre las válidas; las no válidas pesan 0 exacto
    m = tl.where(hay, tl.max(tl.where(valido, sc, float('-inf')), 1), 0.0)
    e = tl.where(valido, tl.exp(sc - m[:, None]), 0.0)
    att = e / tl.where(hay, tl.sum(e, 1), 1.0)[:, None]
```

**Uso:**

```python
import asrn_v89 as asrn, asrn_estable as estable
modelo = estable.convertir(asrn.crear_modelo(V=32000, d=512, n_capas=12))      # grupo A; mismos pesos
# para entrenar desde cero con textos largos: estable.Protecciones.textos_largos(largo_entrenamiento=4096)
asrn.entrenar(modelo, datos, recorte=estable.RecorteDinamico(modelo, tope=1.0, z=3.0))
```

Pruebas: [`tests/test_estable.py`](../../tests/test_estable.py) (15): bit a bit igual a v89 sin
protecciones; igual salvo el redondeo con el grupo A; la fuga causal (PyTorch, kernel y escritura), el reloj corrido,
el reloj detenido, β enorme y el reloj en 16 bits, cada uno reproducido en v89 y ausente en la estable; el tope de la
puerta y el piso de κ (que actúan, sin reproducir una falla); el kernel en modo `ESTABLE` contra PyTorch con el
intérprete de Triton; la proyección del reloj sin TF32 (`gradcheck` e igual a `F.linear`); escritura paso a paso;
guardar y cargar; y el recorte dinámico (grupos, umbral, NaN, dentro de `entrenar()`). `log_alfa_min` y `mejor_max`
no tienen prueba propia. Las sondas que midieron las fallas de la sección 1.2 están en
[`history/probes/`](../probes/).

**En la A100** corrieron también las pruebas de siempre: `test_asrn_v89.py` dio 59 bien, 2 saltadas y 1 falla,
`test_entrenar_aprende_a_copiar_y_reanuda` (la pérdida de copia queda en 4.19 tras 600 pasos, con umbral 1.5). Falla
desde la quinta corrida del proyecto (código `asrn_v89.py` d9927a1bba79), siempre en Colab con PyTorch 2.11; la misma
prueba pasó con PyTorch 2.11 en las corridas 2 a 4, y pasa en CPU con 2.14. No está aislado si la causa es el cambio
de código, la versión o la máquina ([`history/results/README.md`](../results/README.md)). Puede estar relacionada con la
meseta de la copia de las secciones 5.6 y 5.7 (la pérdida queda en ≈ ln 63 = 4.14), pero no se comprobó; hasta
entenderla, no se puede presentar el aprendizaje de la copia como robusto. `test_asrn_estable.py` (14 pruebas en esa
versión; la de la proyección del reloj sin TF32 se agregó después) dio 13 bien y 1 falla de la prueba misma (cargaba
el modelo en la GPU y comparaba en la CPU), ya corregida. Las 7 verificaciones de GPU de `verificar_gpu_v89.py` pasan
(bf16, checkpoint por niveles, FSDP en un proceso, GPU = CPU, causalidad, escritura = texto entero) salvo
`torch.compile`, que se salta con PyTorch < 2.14.

## 5. Banco de pruebas

### 5.1 Protocolo

Todo con [`benchmarks/banco_asrn.py`](../../benchmarks/banco_asrn.py) (los modelos en
[`banco_modelos.py`](../../benchmarks/banco_modelos.py) y [`modelos_oficiales.py`](../../benchmarks/modelos_oficiales.py))
y, en la GPU, [`banco_gpu.py`](../../benchmarks/banco_gpu.py). Cada número de esta sección está en
[`results/banco_cpu/resultados.json`](../../results/banco_cpu/resultados.json) o en
[`history/results/banco_a100_pt211/resultados.json`](../results/banco_a100_pt211/resultados.json); las tablas salen de ellos con
`python benchmarks/banco_asrn.py resumir` y la última celda del cuaderno.

**En CPU (perfil `cpu`):**

- **Modelos, mismo ancho (d = 128) y casi los mismos parámetros (0.61–0.73 M en copia y MQAR, 0.63–0.74 M en texto):** ASRN (receta: 4 bloques locales y una
  capa ASRN cada dos), Transformer (GPT con RoPE y `scaled_dot_product_attention`), Mamba-2 y RWKV-7 (las
  reimplementaciones de `rivales_lineales.py`; en CPU, Mamba-2 con el algoritmo SSD por bloques y RWKV-7 paso a paso).
  Vocabulario atado a la salida en todos.
- **Mismo entrenamiento:** AdamW (decaimiento 0.1, betas 0.9/0.95), calentamiento 2% y coseno hasta lr/10, recorte en 1,
  lotes de 16 × 256 piezas, **los mismos lotes** para todos los modelos en cada semilla. Para cada modelo y tarea, la
  mejor de dos tasas (10⁻³ y 3·10⁻³) con la semilla 0, y esa tasa con las semillas 1 y 2: se informa media ± desviación.
- **Tareas:**
  - *Copia:* textos de 256 piezas (vocabulario 128) cuya segunda mitad repite la primera; la pérdida cuenta solo la
    mitad copiable. Exactitud con 256, 1 024 y 4 096 piezas (copiar desde 128, 512 y 2 048 atrás). 400 pasos.
  - *MQAR* (recuerdo asociativo con muchas preguntas, de Zoology): L/16 pares llave → valor y después preguntas
    "llave valor" con llaves al azar de esos pares; cuenta solo el valor de cada pregunta. Exactitud con 256, 512 y
    1 024 piezas (16, 32 y 64 pares). 600 pasos.
  - *Texto:* bytes de la documentación de Vim (9.9 MB de prosa técnica en inglés; 95% para entrenar, 5% para validar).
    Bits por byte con 256 y 1 024 piezas, y en *Opticks* de Newton (otro texto, que no se usa para entrenar). 400 pasos.
- **Costo:** FLOPs de ida y vuelta (`FlopCounterMode`) y bytes guardados para la vuelta (`saved_tensors_hooks`, sin contar
  pesos), por pieza, con una capa de d = 512 y lote 1. `FlopCounterMode` no cuenta la atención SDPA de CPU: al
  Transformer se le suman 6·L·d FLOPs por pieza (ida y vuelta de la atención causal).

**En la A100 (perfil `a100`, `banco_a100.ipynb`):**

- **ASRN** es la capa estable con el grupo A (`estable_A`), en la receta de siempre (bloques locales con convolución y
  una capa ASRN cada dos). **Los rivales:** Transformer++ (la receta de Llama: RMSNorm, SwiGLU, RoPE; es **nuestra**
  implementación, en `modelos_oficiales.py`, con la atención de `scaled_dot_product_attention`, que en la A100 usa
  FlashAttention-2, y caché de llaves para escribir; RMSNorm, RoPE y SwiGLU son operaciones sueltas de PyTorch, sin
  fusionar, así que con textos cortos su velocidad puede ser pesimista), Mamba-2 (el bloque `Mamba2` de `mamba_ssm` 2.3.2, con sus kernels Triton del algoritmo SSD,
  estado 128, cabezas de 64; sin `causal-conv1d`, que no tiene rueda para esta versión) y RWKV-7 (`RWKV7Model` de
  `flash-linear-attention` 0.5.2, con su inicialización, cabezas de 64 y la salida atada a los vectores de pieza como
  en todos los modelos; el RWKV-7 oficial no la ata).
- **Parámetros: casi iguales.** Copia: ASRN 2 631 236, Transformer++ 2 541 312 (3 bloques), Mamba-2 2 458 488 (5
  bloques), RWKV-7 2 915 328; ASRN con 4 bloques locales y 2 capas ASRN (d = 256). MQAR: 4 597 316, 4 507 392,
  4 424 568 y 4 881 408 (vocabulario más grande). WikiText: 45 138 568 (8 bloques locales y 4 capas ASRN, d = 512),
  44 416 000 (6 bloques), 44 656 912 (11 bloques) y 46 320 640; sin la tabla de vocabulario, ASRN tiene 3.9% más que
  el Transformer++ y 2.6% más que Mamba-2. RWKV-7 (3, 3 y 6 bloques) tiene más que ASRN: +10.8%, +6.2% y +2.6% en las
  tres tareas; con un bloque menos tendría 25%, 14% y 13% menos (cada bloque tiene 0.93 M con d = 256 y 3.4 M con
  d = 512).
- **Mismo entrenamiento que en CPU** (AdamW 0.9/0.95, decaimiento 0.1 solo en matrices, calentamiento 2%, coseno hasta
  lr/10, recorte en 1), en bf16, con los mismos lotes por semilla y las mismas piezas por paso. Dos tasas {10⁻³,
  3·10⁻³} con la semilla 0; se elige la de mejor medida **en el largo de entrenamiento** y se repite con las semillas
  1 y 2. Un empate (100% con las dos) se resuelve por el orden de la tupla: queda 10⁻³. **No hay un conjunto de
  prueba aparte:** la tasa se elige con la misma evaluación que se informa.
- **Tareas:**
  - *Copia:* vocabulario 512, textos de 1 024 piezas, 2 000 pasos de lotes de 32. Exactitud con 1 024, 4 096 y 16 384
    piezas (azar: 1/511 = 0.196%).
  - *MQAR:* vocabulario 8 192, textos de 512 piezas con 64 pares (L/8), 2 000 pasos de lotes de 64. Exactitud con 512,
    1 024 y 2 048 piezas, es decir con 64, 128 y 256 pares: la evaluación cambia a la vez el largo y la cantidad de
    pares.
  - *WikiText-103:* piezas de GPT-2 (50 257); 1 220 pasos de 16 ventanas de 1 024 piezas con inicio al azar (con
    reemplazo) entre las primeras 19 988 480 piezas del conjunto de entrenamiento: 19 988 480 posiciones, de las que
    cada corrida ve ~63% distintas (~12.6 M).
    Perplejidad por pieza sobre las primeras 245 760 piezas de validación, con 1 024 y 4 096 de contexto (las mismas
    piezas con los dos largos). No es comparable con cifras publicadas (que son por palabra y en el conjunto de prueba).
- **Velocidad:** textos al azar; una corrida por punto (mediana de 3 repeticiones en la capa, 2 desde 65 536 piezas, y
  de 5 en el modelo). *Capa sola:* d = 512, lote 1, bf16, ida y vuelta, con la memoria pico. **Cada "capa" es una unidad
  distinta:** la de ASRN es solo su capa de mezcla (0.65 M de parámetros; en la receta va con dos bloques locales de
  2.1 M), la del Transformer++ es su atención con RMSNorm (1.05 M, sin la SwiGLU) y la de Mamba-2 es su bloque entero
  (1.72 M, con su expansión, que hace de capa densa). *Modelo:* ~50 M de parámetros (vocabulario 32 768, d = 512; ASRN con 14 bloques locales y 7 capas ASRN,
  Transformer++ con 11 bloques, Mamba-2 con 20), 32 768 piezas por paso con cualquier largo. *Escritura:* esos mismos
  modelos, lote 1, el promedio de 32 piezas nuevas (una pasada) después de un prefijo de 1 a 64 mil piezas.

### 5.2 Costo por capa (exacto, independiente del hardware)

| Capa (d = 512) | MFLOP/pieza L=1024 | MFLOP/pieza L=4096 | MFLOP/pieza L=16384 | KB guardados/pieza |
|---|---|---|---|---|
| ASRN (v89) | 5.19 | 5.14 | 5.13 | 43.8 |
| ASRN (estable) | 5.19 | 5.14 | 5.13 | 43.9 |
| Transformer | 9.44 | 18.87 | 56.62 | 12.0 |
| Mamba-2 | 11.48 | 11.48 | 11.48 | 93.8 |
| RWKV-7 | 8.26 | — | — | 183.3 |
| ASRN (v89) + kernel | (igual) | (igual) | (igual) | 20.1 |
| ASRN (estable) + kernel | (igual) | (igual) | (igual) | 20.2 |

- **ASRN cuesta lo mismo por pieza con cualquier largo** (5.1 MFLOP); la atención crece lineal con el largo y con 16 mil
  piezas cuesta 11 veces más. Con d = 512, la capa ASRN cuesta menos FLOPs que una capa de atención incluso a largo
  cero (sus proyecciones son 2d² contra 4d²).
- **Mamba-2:** `FlopCounterMode` cuenta mal la vuelta de una convolución por grupos (la de Mamba-2 es por canales,
  1 152 grupos): le asignaba 10.7 MFLOP por pieza, ~1 000 veces lo real, y la primera versión de esta tabla daba 22.1.
  `banco_asrn.py` usa ahora una fórmula que tiene en cuenta los grupos (con una convolución densa da lo mismo que la de
  PyTorch); las demás filas no cambian.
- **Memoria** (KB de 1 024 bytes): el kernel Triton de ASRN guarda 20 KB por pieza (solo sus entradas), contra 44 KB
  de la versión de PyTorch; las protecciones no agregan nada medible. La capa de atención con SDPA guarda 12 KB: 8 de
  q, k, v, la salida y el logsumexp, y 4 de la entrada y la salida de su LayerNorm.
  Los números de Mamba-2 y RWKV-7 son de sus implementaciones de referencia en PyTorch (RWKV-7 paso a paso guarda el
  estado de cada paso) y **no** representan a sus kernels de GPU, que guardan sus entradas más un estado por bloque de
  64 piezas.

### 5.3 Latencia de una capa

**En CPU (4 núcleos), ms de ida y vuelta:**

| Capa (d = 256) | L=512 | L=2048 | L=8192 | L=32768 | exponente (ajuste de 2 048 a 32 768) |
|---|---|---|---|---|---|
| ASRN (v89) | 22.7 | 63.7 | 317.6 | 1853.8 | 1.22 |
| ASRN (estable) | 26.9 | 67.5 | 297.7 | 1608.8 | 1.14 |
| Transformer | 5.3 | 29.6 | 331.5 | 4796.0 | 1.83 |
| Mamba-2 | 18.2 | 83.6 | 417.7 | 9847.8 | 1.72 |
| RWKV-7 | 310.8 | 1896.0 | — | — | nan |

- ASRN crece casi lineal (exponente 1.14–1.22: el costo por pieza sube algo por los ordenamientos y la caché); la
  atención crece con exponente 1.83. Se cruzan entre 2 y 8 mil piezas; con 32 768 ASRN es 2.6–3.0 veces más rápido.
  Las diferencias entre v89 y la estable (de −13% a +19%, sin una dirección fija) son ruido de la medición en CPU.
- Las latencias de Mamba-2 y RWKV-7 en CPU no sirven para compararlos: RWKV-7 va paso a paso en Python, y la vuelta de
  la versión SSD por bloques de Mamba-2 escala mal en CPU con 32 mil piezas (su exponente 1.72 es de la implementación,
  no del algoritmo, que es lineal).

**En la A100** (d = 512, lote 1, bf16, ida y vuelta; ms y, abajo, GB de memoria pico):

| Capa | L=1 024 | L=4 096 | L=16 384 | L=65 536 | L=262 144 |
|---|---|---|---|---|---|
| ASRN (v89) | 13.6 | 11.5 | 13.0 | 18.2 | 64.2 |
| ASRN (estable) | 12.2 | 11.9 | 12.1 | 19.3 | 66.2 |
| Transformer++ (FlashAttention-2) | 3.2 | 3.3 | 9.9 | 101.5 | 1 488.9 |
| Mamba-2 (`mamba_ssm`, sin `causal-conv1d`) | 7.7 | 7.8 | 11.1 | 35.3 | 180.9 |
| RWKV-7 (fla, `RWKV7Attention`) | 11.2 | 12.0 | 35.3 | 131.4 | 519.7 |
| *Transformer++ / estable* | *0.27* | *0.27* | *0.82* | *5.3* | *22.5* |
| *Mamba-2 / estable* | *0.63* | *0.66* | *0.92* | *1.8* | *2.7* |
| *RWKV-7 / estable* | *0.92* | *1.01* | *2.9* | *6.8* | *7.8* |

| Memoria (GB) | L=1 024 | L=4 096 | L=16 384 | L=65 536 | L=262 144 | KB (de 1 024 B) por pieza |
|---|---|---|---|---|---|---|
| ASRN (v89) | 0.04 | 0.16 | 0.65 | 2.6 | 10.3 | 38 |
| ASRN (estable) | 0.04 | 0.17 | 0.68 | 2.7 | 10.9 | 40 |
| Transformer++ | 0.02 | 0.09 | 0.34 | 1.4 | 5.5 | 20 |
| Mamba-2 | 0.05 | 0.16 | 0.55 | 2.1 | 8.5 | 31 |
| RWKV-7 | 0.08 | 0.33 | 1.32 | 5.3 | 21.1 | 79 |

- **Hasta 16 mil piezas, ASRN es de las más lentas:** empata con RWKV-7 con 1 y 4 mil piezas, y la atención y Mamba-2
  son más rápidas; su tiempo es un piso de ~12 ms que no depende del largo
  (muy probablemente el lanzamiento de muchas operaciones chicas, sin compilar y con lote 1: no se perfiló en esta
  corrida, pero con CUDA Graph el proyecto midió antes 11.4 → 1.6 ms con 1 024 piezas; RENDIMIENTO.md, sección 5d).
  FlashAttention-2 es 3.6–3.8 veces más rápida con 1–4 mil piezas.
- **RWKV-7 es el más lento a partir de 16 mil piezas** y el que más memoria usa (79 KB por pieza, el doble que ASRN):
  con 262 144 piezas tarda 7.8 veces lo de ASRN. Su capa es la primera de un modelo de dos (fla no arma uno de una),
  medida con la implementación de fla, que en la ida y vuelta de entrenamiento usa sus kernels Triton por bloques.
- **Desde ~19 mil piezas ASRN pasa a la atención y a Mamba-2** (interpolando entre 16 y 65 mil; con Mamba-2 el cruce cae en el mismo
  lugar). Con 65 536 piezas es 5.3 veces más rápida que la atención y 1.8 veces que el bloque de Mamba-2; con 262 144,
  22.5 y 2.7 veces. Desde 65 mil crece ~239 ns por pieza. Contra Mamba-2, estas razones son de unidades distintas (su
  bloque tiene 2.6 veces los parámetros de la capa ASRN): con modelos enteros la razón es 1.5 (sección 5.4).
- **A cambio usa más memoria:** ~40 KB por pieza, el doble que la atención (20 KB) y 1.3 veces Mamba-2 (31 KB). Con
  262 mil piezas son 10.9 GB contra 5.5 y 8.5.
- **La capa estable tarda lo mismo que v89** dentro del ruido con textos cortos (+3% con 262 mil piezas) y usa +4–5% de
  memoria a todo largo (guardaba ĥ en 32 bits para el reloj; medido antes de `_ProyeccionIEEE`).
- Mamba-2 corrió por su camino sin fusionar (sin `causal-conv1d`): sus cifras son pesimistas, y la ventaja de ASRN
  sobre él es una cota superior.

### 5.4 Entrenar un modelo de ~50 M (A100)

32 768 piezas por paso con cualquier largo (lote 32 con 1 024 piezas, lote 1 con 32 768), bf16, sin compilar. Piezas
por segundo (GB de memoria pico):

| Modelo | parámetros | L=1 024 | L=4 096 | L=16 384 | L=32 768 |
|---|---|---|---|---|---|
| ASRN | 50.7 M | 173 346 (23.5) | 172 431 (23.5) | 170 474 (23.5) | 170 460 (23.5) |
| Transformer++ | 51.0 M | 182 654 (25.5) | 161 049 (25.5) | 110 493 (25.5) | 78 378 (25.6) |
| Mamba-2 | 51.2 M | 116 745 (26.4) | 116 766 (26.4) | 115 221 (26.4) | 113 211 (26.4) |
| RWKV-7 | 51.1 M | 49 431 (30.5) | 49 127 (30.5) | 47 513 (30.5) | 45 143 (30.5) |
| *ASRN / Transformer++* | | *0.95* | *1.07* | *1.54* | *2.17* |
| *ASRN / Mamba-2* | | *1.48* | *1.48* | *1.48* | *1.51* |
| *ASRN / RWKV-7* | | *3.51* | *3.51* | *3.59* | *3.78* |

- **ASRN no se frena con el largo:** pierde 1.7% de 1 024 a 32 768 piezas. El Transformer++ pierde 57% y Mamba-2 3%.
- **Pasa al Transformer++ entre 1 y 4 mil piezas** (~2.3 mil interpolando), mucho antes que la capa sola: con un lote
  grande, el costo fijo de ASRN se reparte entre todas las piezas del paso. Con 32 768 piezas es 2.17 veces más
  rápido. Con 1 024 el Transformer++ es 5% más rápido.
- **Contra Mamba-2, 1.5 veces más rápido a todo largo**, con la salvedad de que Mamba-2 corrió sin fusionar.
- **Contra RWKV-7, 3.5–3.8 veces más rápido**, con 23% menos memoria. Es el mismo orden que en la quinta corrida con la
  reimplementación de RWKV-7 (también con los kernels de fla): 50–53 mil piezas/s.
- La memoria la dominan los logits del vocabulario (comunes a todos): ASRN usa 8% menos que el Transformer++, 11%
  menos que Mamba-2 y 23% menos que RWKV-7, en una sola medición.

### 5.5 Escritura

**En CPU** (modelo d = 128), ms por pieza nueva:

| Contexto | 512 | 2048 | 8192 | 32768 | exponente |
|---|---|---|---|---|---|
| ASRN | 3.83 | 5.09 | 8.93 | 14.55 | 0.33 |

Cada pieza nueva cuesta más cuanto más texto hay detrás (sección 2.4). El exponente 0.33 (ajuste con los cuatro puntos)
refleja una curva que todavía no es lineal en este rango: el costo por pieza de contexto baja de 0.81 a 0.63 y a
0.23 µs entre puntos seguidos. Que el algoritmo sea O(t) sale del código, no de este ajuste.

**En la A100** (los modelos de ~50 M de la sección 5.4, lote 1, bf16), ms por pieza nueva:

| Modelo | contexto 1 024 | 4 096 | 16 384 | 65 536 | prefijo de 65 536 (s) | memoria con 65 536 (GB) |
|---|---|---|---|---|---|---|
| ASRN | 40.5 | 40.1 | 41.0 | 41.0 | 0.136 | 3.82 |
| Transformer++ | 14.1 | 14.5 | 14.6 | 14.3 | (0.388)* | (6.22)* |
| Mamba-2 | 25.4 | 25.5 | 25.4 | 25.4 | 0.133 | 1.26 |
| RWKV-7 | 34.7 | 35.0 | 35.7 | 35.5 | (122)** | 3.69 |

- **Plano en los cuatro modelos**, también en el Transformer++, que lee toda su caché en cada pieza: con lote 1 lo que
  se mide es, muy probablemente, el lanzamiento de kernels, no el contexto (no se perfiló). El O(t) de ASRN queda
  escondido debajo (sección 2.4).
- **ASRN es la escritura más lenta:** 2.8 veces la del Transformer++, 1.6 veces la de Mamba-2 y 1.2 veces la de RWKV-7, coherente con la
  cantidad de operaciones chicas que hace por pieza (8 tablas, 4 grupos de Hawkes, la racha, la puerta). Su memoria pico
  con 64 mil piezas de contexto es 3 veces la de Mamba-2 (3.82 contra 1.26 GB), pero esa cifra incluye el paso del
  prefijo: la de Mamba-2 también crece con el contexto (de 0.12 a 1.26 GB) aunque su estado es fijo. No separa lo que
  ASRN guarda para escribir (la memoria de cada capa, O(t)) de lo transitorio; hace falta medir la memoria después
  del prefijo.
- \* *El prefijo y la memoria del Transformer++ no son comparables:* nuestro envoltorio calculaba la salida (65 536 ×
  32 768 logits, 4.3 GB en bf16) de todas las posiciones del prefijo, y los otros dos solo de la última. Está corregido
  y falta volver a medirlo (no se pidió en la segunda ejecución). Cada prefijo se midió una sola vez, sin calentar.
  \*\* El prefijo de RWKV-7 tarda 110–122 s con cualquier largo, de 1 a 64 mil piezas: es casi todo la compilación (o el
  ajuste automático) de los kernels de fla para cada forma nueva, no el cálculo.

### 5.6 Calidad

**En CPU** (copia; modelos de ~0.7 M, 400 pasos, 3 semillas con la mejor tasa de cada modelo):

| Modelo | lr | semillas | parámetros | exactitud L=256 | exactitud L=1024 | exactitud L=4096 | s/corrida | piezas/s |
|---|---|---|---|---|---|---|---|---|
| ASRN | 0.003 | 3 | 684,100 | 100.0% ± 0.0 | 99.8% ± 0.1 | 74.3% ± 9.4 | 112 | 14,633 |
| Transformer | 0.001 | 3 | 609,920 | 100.0% ± 0.0 | 0.7% ± 0.0 | 0.8% ± 0.1 | 35 | 46,739 |
| Mamba-2 | 0.001 | 3 | 721,608 | 0.9% ± 0.2 | 0.8% ± 0.0 | 0.8% ± 0.1 | 132 | 12,450 |

RWKV-7 llegó a 5.3% con su única corrida en CPU (la batería se detuvo ahí: en CPU va paso a paso y cada corrida tarda
10 minutos). Un piloto de MQAR en CPU (una semilla, 16 pares; no quedó guardado en el JSON, así que no se puede
comprobar) dio ASRN 100% y 99.9% con 4 veces los pares, y el Transformer 7.8%. La batería grande es la de la A100:

**En la A100** (media ± desviación de 3 semillas con la tasa elegida; las corridas, una por una, en el JSON):

| Copia (entrenada con 1 024) | lr | 1 024 | 4 096 (4×) | 16 384 (16×) | pérdida < 1 desde el paso | s/corrida |
|---|---|---|---|---|---|---|
| ASRN | 10⁻³ | 100.0% ± 0.0 | 99.9% ± 0.2 | 67.6% ± 27.0 (43.5 / 62.5 / 96.8) | 1 000 / 1 000 / 1 200 | 70 |
| Transformer++ | 10⁻³ | 100.0% ± 0.0 | 0.2% ± 0.0 | 0.2% ± 0.0 | 200 / 200 / 200 | 40 |
| Mamba-2 | 3·10⁻³ | 0.2% ± 0.0 | 0.2% ± 0.0 | 0.2% ± 0.0 | nunca | 86 |
| RWKV-7 | 10⁻³ | 30.5% ± 25.7 (24.9 / 8.1 / 58.6) | 0.3% ± 0.0 | 0.2% ± 0.0 | nunca (al final, 3.5 / 4.9 / 1.7) | 251 |

| MQAR (entrenado con 64 pares en 512) | lr | 512 (64 pares) | 1 024 (128) | 2 048 (256) | s/corrida |
|---|---|---|---|---|---|
| ASRN | 10⁻³ | 100.0% ± 0.0 | 99.6% ± 0.6 | 98.5% ± 1.8 | 81 |
| Transformer++ | 3·10⁻³ | 0.8% ± 0.9 | 0.5% ± 0.5 | 0.2% ± 0.2 | 54 |
| Mamba-2 | 3·10⁻³ | 95.9% ± 1.8 | 83.4% ± 3.1 | 54.4% ± 5.9 | 92 |
| RWKV-7 | 10⁻³ | 99.9% ± 0.2 | 59.2% ± 1.3 | 30.0% ± 0.3 | 213 |

| WikiText-103 (perplejidad) | lr | contexto 1 024 | contexto 4 096 | 4 096 / 1 024 | s/corrida |
|---|---|---|---|---|---|
| ASRN | 3·10⁻³ | **67.78 ± 0.88** | **63.00 ± 0.86** | −7.0% | 144 |
| Transformer++ | 10⁻³ | 93.70 ± 1.19 | 125.56 ± 3.83 | +34% | 121 |
| Mamba-2 | 3·10⁻³ | 74.17 ± 1.03 | 72.98 ± 1.03 | −1.6% | 142 |
| RWKV-7 | 3·10⁻³ | 72.35 ± 4.03 | 70.24 ± 4.05 | −2.9% | 305 |

**Copia.** ASRN copia sin error en el largo de entrenamiento y con 4 veces el largo (99.7–99.99%: copia 2 047 piezas
desde 2 048 atrás). Con 16 veces, depende de la semilla: 43.5%, 62.5% y 96.8%; lo sólido es que *a veces* generaliza
a 16×, no cuánto en promedio. La corrida con 3·10⁻³ (que no se eligió: empató en 100% en el largo y quedó 10⁻³ por
orden) dio 99.0% con 16×. El Transformer++ cae **exactamente al azar** fuera de su largo (0.16–0.23% contra 1/511 =
0.196%); hay dos causas posibles que estos datos no separan: RoPE ve distancias que nunca vio, y la distancia de la
copia es siempre la mitad del texto, así que una cabeza que aprendió "mirar 511 atrás" apunta al lugar equivocado en
un texto más largo. Mamba-2 no aprendió ni siquiera en el largo de entrenamiento: con las dos tasas, su pérdida quedó
en ln 511 = 6.236 los 2 000 pasos. ASRN pasó por la misma meseta durante 600 a 1 000 pasos antes de salir, así que esto dice que
Mamba-2 no aprendió con este presupuesto, no que no pueda copiar (su estado de 327 680 números alcanza de sobra).
RWKV-7 estaba aprendiendo: su pérdida bajaba en las tres semillas (al final, 3.5, 4.9 y 1.7) y acertaba 25%, 8% y
59% en el largo de entrenamiento; con 2 000 pasos no llegó, y fuera de su largo cae al azar.

**Lo que va en contra de ASRN en la copia:** dentro del largo entrenado, el Transformer++ la aprende en ≤ 200 pasos
(pérdida 0.0025 en el paso 200), al menos 4 veces antes que ASRN con la misma tasa (la curva tiene un punto
cada 200 pasos, y ASRN sigue por encima de 1 en el paso 800), y cada corrida le lleva el 57%
del tiempo (40 s contra 70). Con 3·10⁻³, ASRN sale de la meseta entre los pasos 200 y 400.

**MQAR.** ASRN mantiene el recuerdo al crecer los pares (99.97%, 99.6% y 98.5%); Mamba-2 baja de 95.9% a 83.4% y a
54.4%, la caída que se espera de una memoria de estado fijo (Zoology, Based), aunque aquí mezclada con la
extrapolación de largo y con que Mamba-2 no había convergido ni en el largo entrenado (su pérdida seguía bajando en
el paso 2 000). Con 128 y 256 pares la separación es completa en las tres semillas (ASRN ≥ 96.4% en cada
una; Mamba-2 ≤ 86.5% y ≤ 61.0%); prueba t pareada p = 0.016 y 0.0098. Con 64 pares, 3 semillas no alcanzan para
afirmarlo (p = 0.059), y Mamba-2 todavía estaba aprendiendo (pérdida final 0.10–0.25). **RWKV-7** (con 10⁻³; con
3·10⁻³ no aprendió) resuelve MQAR en el largo entrenado (99.7–100%) pero cae más que Mamba-2 al crecer los pares:
57.7–60.3% con 128 y 29.7–30.4% con 256, muy parejo entre semillas (separación completa contra ASRN, p pareada =
0.0007 y 0.0003; contra Mamba-2, 0.005 y 0.017). Que acierte casi exactamente 1/2 y 1/4 sugiere que guarda unos 64
pares, los del largo entrenado, y no más; no se comprobó. **El Transformer++ no aprendió
MQAR** con ninguna de las dos tasas (pérdida final 7.4–8.3, cerca de ln 4 096 = 8.32; acierta del orden de 1 entre el
número de pares, como la ablación sin memoria a distancia). Es falta de presupuesto o de tasa en este protocolo, no un
límite de la atención (Zoology la muestra resolviendo MQAR con más entrenamiento): esa fila no es evidencia contra la
atención.

**WikiText-103.** ASRN es el mejor en cada semilla (66.8–68.5) y el Transformer++ el peor (92.9–95.1); contra Mamba-2
(73.3–75.3), la prueba t pareada da p = 0.0077 con 1 024 y 0.0033 con 4 096. **RWKV-7** queda en medio y varía mucho
entre semillas (70.4, 69.7 y 77.0): detrás de ASRN en las tres (por 2.4, 2.9 y 8.5), pero con n = 3 la diferencia no
es significativa (p pareada = 0.15 con 1 024 y 0.067 con 4 096), y contra Mamba-2 gana en dos semillas y pierde en
una. Con 4 096 piezas de contexto (las mismas piezas evaluadas), ASRN mejora 7.0%, RWKV-7 2.9% y Mamba-2 1.6%: usan
el contexto extra; el Transformer++ empeora 34% (RoPE fuera de su largo). Las salvedades pesan mucho aquí:

- **Lejos de converger:** 0.44 posiciones de entrenamiento por parámetro (0.28 si se cuentan las piezas distintas
  que ve cada corrida); las curvas seguían bajando al final (0.2–0.5 nats entre los pasos
  488–732 y 976–1 220), y la diferencia de pérdida entre Mamba-2 y ASRN, sobre los mismos lotes, se achicó
  durante el entrenamiento (de ~0.11–0.15 a ~0.03–0.06 nats).
- **Tasas en el borde de la grilla:** ASRN y Mamba-2 eligieron la más alta (3·10⁻³) y el Transformer++ la más baja
  (10⁻³; con 3·10⁻³ dio 144.9), con un calentamiento corto (24 pasos). Con la misma tasa de 10⁻³ (semilla 0), el orden
  era otro: Mamba-2 86.0, ASRN 88.6, Transformer++ 95.1. Antes de afirmar una ventaja en texto hace falta al menos el
  Transformer++ con 3·10⁻⁴ y 6·10⁻⁴ (pendiente: sección 6) y ASRN y Mamba-2 con 6·10⁻³.
- **Sesgo inductivo y corpus:** ASRN recibe los números de pieza y trae incorporada la coincidencia exacta (sufijo,
  vistos, tablas de palabra). WikiText-103 premia copiar (las entidades se repiten dentro de cada artículo; los modelos
  con caché o kNN mejoran mucho en él). Con un solo corpus, la ventaja puede ser propia de WikiText; falta uno con
  menos repetición.
- **RWKV-7 contradice a la quinta corrida.** Allí (5 de octubre; otro protocolo: 30 M de piezas, la reimplementación
  de RWKV-7 de `rivales_lineales.py`, ASRN v89) RWKV-7 fue el mejor en texto por un margen amplio: 47.87 contra 57.02
  de la misma receta de ASRN con 1 024 piezas (16% menos). Aquí, con el `RWKV7Model` de fla, queda 7% por encima de
  ASRN. Las diferencias de protocolo (datos, evaluación, implementación de los dos modelos) no permiten decir cuál
  refleja mejor a RWKV-7. Aquí, además, su tasa elegida está en el borde de la grilla, varía mucho entre semillas y
  RWKV-7 entrena 2.5 veces más lento por corrida. No se puede afirmar que ASRN supere a RWKV-7 en texto en general.

**Tiempo de entrenamiento** (solo entrenar, a igual cantidad de datos, no a igual cómputo): contra el Transformer++,
ASRN tarda 1.76, 1.51 y 1.19 veces más por corrida en copia, MQAR y WikiText, Mamba-2 2.16, 1.71 y 1.17, y RWKV-7
6.3, 3.9 y 2.5 (con la compilación de la primera corrida incluida). En copia
y MQAR, con modelos de 2.5–4.6 M, esos tiempos miden sobre todo sobrecargas, no la arquitectura; en WikiText, con
~45 M, la diferencia baja a 1.17–1.19 veces.

**Qué es sólido y qué no.** Sólido dentro de este protocolo: cuando ASRN sale de la meseta, copia con 1× y 4× el largo
(≥ 99.06% con 4× en todas las corridas de ASRN y sus variantes que aprendieron, sección 5.7; de las 24 de la base y
las variantes livianas, 3 no salieron de la meseta en 2 000 pasos), y el Transformer++ cae al azar fuera del suyo; ASRN supera a Mamba-2 y a RWKV-7 en MQAR con 128 y 256
pares; en WikiText, ASRN, RWKV-7 y Mamba-2 quedan por debajo del Transformer++ en cada semilla y con las dos tasas
probadas.
Depende del presupuesto chico o del protocolo: la cifra de copia con 16×, que Mamba-2 no copie, que el Transformer++
no haga MQAR, que RWKV-7 no termine de aprender la copia, la diferencia con 64 pares, el orden entre ASRN, RWKV-7 y
Mamba-2 en WikiText (se invierte con 10⁻³, y los dos
eligieron el borde superior de la grilla), el tamaño de las diferencias y las tasas elegidas. Tres semillas miden la variación por inicialización y lotes, no la de la grilla de tasas ni la del
protocolo, que es mayor; y con n = 3 ninguna prueba no paramétrica puede bajar de p = 0.10 bilateral (las pruebas t
de arriba suponen normalidad).

### 5.7 Ablación (A100)

ASRN (capa estable, grupo A) quitando una pieza a la vez, en copia y MQAR, con 3 semillas y la tasa elegida para ASRN
en la sección 5.6 (10⁻³ en las dos tareas). Las corridas de la base son las mismas de la sección 5.6. Las piezas
congeladas (reloj, Hawkes y racha, sufijo, confirmación) arrancan en su valor neutro, así que esas variantes calculan lo
mismo que la base en el paso 0, y en cada semilla todas ven los mismos lotes: el diseño es pareado (en la copia, la
pérdida en el paso 200 coincide en la cuarta cifra decimal entre variantes de una misma semilla, salvo `mezcla_lse`; en
MQAR, en la tercera).

**La copia es bimodal:** o una corrida sale de la meseta inicial (pérdida ≈ ln 511) y llega a 100% en el largo de
entrenamiento, o no sale en los 2 000 pasos y queda en el azar. Una media ± desviación (como 66.7% ± 57.6) mezcla las
dos cosas, así que aquí se separan: cuántas semillas aprendieron, y cómo generalizan las que aprendieron.

| Variante (copia) | aprendieron | 4×, la peor de las que aprendieron | 16× de las que aprendieron (s0 / s1 / s2) | s/corrida |
|---|---|---|---|---|
| estable_A (la base) | 3 de 3 | 99.67% | 43.5 / 62.5 / 96.8% | 70.3 |
| estable_AB (+ grupo B) | 3 de 3 | 99.98% | 65.4 / 91.7 / 68.9% | 70.0 |
| sin_puerta | 2 de 3 | 99.95% | 71.3 / — / 60.6% | 57.3 |
| puerta_sin_senales | 3 de 3 | 99.73% | 49.3 / 85.7 / 89.2% | 59.3 |
| sin_reloj | 2 de 3 | 99.89% | 91.5 / — / 95.7% | 68.2 |
| sin_hawkes_racha | 3 de 3 | 99.42% | 64.0 / 63.1 / 80.7% | 65.5 |
| sin_sufijo | 2 de 3 | 99.06% | — / 50.1 / 87.7% | 68.7 |
| sin_confirmacion | 3 de 3 | 99.95% | 51.8 / 83.3 / 70.8% | 68.4 |
| dos_tablas (1 + 1 en vez de 4 + 4) | 1 de 3 | 99.90% | 40.4 / — / — | 69.1 |
| solo_local (sin capas ASRN) | 0 de 3 | — | — | 25.8 |
| mezcla_lse (tablas por logsumexp) | 3 de 3 | 99.88% | 95.3 / 66.1 / 62.8% | 219.4 |

| Variante (MQAR) | 512, la peor semilla | 1 024, la peor | 2 048 (s0 / s1 / s2) | s/corrida |
|---|---|---|---|---|
| estable_A (la base) | 99.93% | 98.96% | 99.55 / 96.45 / 99.52% | 81.3 |
| estable_AB | 99.67% | 98.67% | 99.24 / 97.84 / 95.47% | 81.9 |
| sin_puerta | 99.93% | 99.56% | 99.09 / 97.94 / 99.61% | 72.5 |
| puerta_sin_senales | 99.71% | 99.32% | 99.37 / 96.76 / 99.32% | 74.2 |
| sin_reloj | 99.90% | 99.66% | 99.19 / 98.67 / 99.89% | 80.4 |
| sin_hawkes_racha | 99.71% | 98.75% | 99.54 / 94.43 / 99.47% | 79.7 |
| sin_sufijo | 99.90% | 99.77% | 99.56 / 99.64 / 98.57% | 80.8 |
| sin_confirmacion | 100.00% | 99.93% | 99.72 / 99.16 / 99.82% | 80.0 |
| dos_tablas | 99.54% | 99.25% | 95.42 / 98.36 / 98.18% | 80.4 |
| solo_local | 1.76% | 0.83% | 0.44 / 0.42 / 0.40% | 41.3 |
| mezcla_lse | NaN desde el paso 400: un error nuestro, corregido; no cuenta | | | 167.9 |

**Lo concluyente:**

1. **La capa ASRN es la que resuelve las dos tareas.** Sin ella (`solo_local`: el modelo ve las últimas 13 piezas, la actual y
   12 hacia atrás, y la copia está a 512) la pérdida queda en el azar los 2 000 pasos, en las 3 semillas de las dos tareas. Es
   concluyente por construcción más que por estadística.
2. **Mezclar las tablas por logsumexp acelera el aprendizaje de la copia:** las 3 semillas salen de la meseta entre
   los pasos 200 y 400 (en el paso 400 su pérdida es ~0.009, la de la base sigue en 6.237), entre 600 y 800 pasos antes
   que la base; ocupa los 3 primeros lugares de 30 corridas, algo que al azar pasaría con probabilidad 1/C(30, 3) =
   0.00025. **Pero** no generaliza mejor con 16× (74.7% contra 67.6%; permutación p = 0.70), por ahora tarda 3 veces más
   por corrida (solo existe en PyTorch; llega a pérdida < 1 en ~43–45 s contra ~34–41 s de la base, estimado por la fracción de pasos), y la base con
   3·10⁻³ también sale entre 200 y 400: puede parecerse a un paso efectivo mayor más que a una mejor solución. Es la
   mejora candidata de la sección 1.1 (punto 4); merece un kernel y una prueba en texto.

**Lo que no se puede afirmar con 3 semillas:**

- **Que quitar la puerta, el reloj o el sufijo dañe el aprendizaje.** Fallaron en 1 de 3 semillas, pero una falla no
  distingue nada (Fisher, 1 de 3 contra 0 de 3: p = 0.50 unilateral, 1.0 bilateral): con las 8 variantes livianas juntas (3 fallas en 24
  corridas) la probabilidad de al menos una falla en 3 semillas es 0.33, y se esperaban ~2.3 de 7 variantes con alguna
  falla (se vieron 3). Sin el reloj y sin el sufijo, además, la variante calcula lo mismo que la base al arrancar: lo
  que cambia es la optimización dentro del presupuesto, no la capacidad.
- **Que menos tablas dificulten aprender** (dos_tablas aprendió en 1 de 3, con el 16× más bajo, 40.4%): los datos lo
  sugieren, sin ser concluyentes (Fisher p = 0.20 unilateral, 0.40 bilateral; entre 9 grupos de 3 con 5 fallas al azar, alguno junta dos con
  probabilidad 0.62).
- **Que alguna variante generalice mejor o peor con 16×:** las 21 corridas que aprendieron (variantes livianas) van de
  43.5% a 96.8%, y la variación entre variantes no supera la de dentro de cada una (ANOVA F = 0.51, p ≈ 0.81). Con 3
  contra 3, una diferencia tendría que superar ~41 puntos para ser significativa.
- **Nada en MQAR:** las 27 corridas válidas aprendieron (≥ 99.5% con 64 pares, ≥ 94.4% con 256); lo que varía en
  2 048 sigue a cuánto convergió cada corrida (Spearman −0.85 entre la pérdida final y la exactitud, p < 0.0001), no a
  la pieza quitada. Este MQAR está en el techo para ASRN: llaves únicas, el valor justo después de la llave, y toda
  posición evaluada pide copiar; basta buscar la pieza exacta y leer la siguiente.

**Por qué estas tareas no prueban la puerta ni sus señales.** Copia y MQAR solo evalúan posiciones donde copiar es
correcto; la puerta (con Hawkes, racha y vistos) existe para decidir cuándo *no* copiar o cortar una cita demasiado
larga, y el reloj para olvidar en un cambio de tema. Sí se mide su costo: sin la puerta una corrida de copia tarda 17%
menos y sin sus señales 14% menos (contra las semillas 1 y 2 de la base, 69.1 s, que corrieron con los kernels ya
compilados); sin Hawkes y racha, 5% menos. El grupo B de protecciones no tiene costo ni efecto medible.

**Para que la ablación diga más:** más semillas en copia (con 0 fallas en n semillas, la cota de la tasa de falla es de
~3/n), repetirla con 3·10⁻³ o con más pasos (para que "no aprendió" no sea un corte del presupuesto), una tarea con
distractores o una ablación corta en WikiText para la puerta y sus señales, un MQAR más exigente, y `mezcla_lse` en
MQAR con la corrección (pendiente: sección 6).

### 5.8 Comparación matemática

Por capa de mezcla (sin la capa densa que la acompaña; el bloque de Mamba-2 no lleva capa densa y por eso trae su
propia expansión), con largo N, ancho d, entrenamiento (ida y vuelta) y FLOPs = 2 × multiplicaciones. T, C, P de ASRN
como arriba; N_s el estado de Mamba-2 (64 en CPU, 128 en la A100), D_h el ancho de cabeza de RWKV-7 (64).

| | Transformer (FlashAttention) | Mamba-2 (SSD) | RWKV-7 | ASRN |
|---|---|---|---|---|
| Cuentas por pieza | 6·(4d²) + **6·N·d** | ~6·(6d²) + O(d·N_s) | ~6·(4d²) + O(d·D_h) | 6·(2d² + hash) + 6·T·2C·(P + cf + d) |
| Crece con N | **sí (lineal)** | no | no | no (ordenar es radix: pasadas fijas) |
| Límite en GPU | cuentas (tensor cores) | cuentas y memoria | cuentas y memoria | **memoria** (~10 cuentas por byte) |
| Memoria de activaciones | O(N·d) | O(N·d) + estados por bloque | O(N·d) + estados por bloque | O(N·(d + P + T)) |
| Escritura: tiempo por pieza | O(N·d) | O(d·N_s), constante | O(d·D_h), constante | hoy O(N·(P + T)); con la sección 2.4, O(T·C·(P + d)) |
| Escritura: memoria | caché 2·N·d | estado fijo 2d·N_s | estado fijo d·D_h | O(N·(d + P)), acotable a T·2¹⁴·C posiciones |
| Recuperar algo exacto de muy atrás | sí (si la posición generaliza) | limitado por el tamaño del estado | limitado por el estado | sí, si cae en el mismo casillero entre las C más recientes |

**Dónde se cruzan en tiempo real.** La estimación de antes de la A100: la atención está limitada por cuentas (~7·N·d
FLOPs por pieza, ida y vuelta, causal, con el recálculo de FlashAttention, a ~200 TFLOP/s efectivos) y la capa ASRN
por memoria (~122 KB por pieza de sus kernels a 1.55 TB/s, el ancho de banda de la A100 de 40 GB: ~80 ns, y del
orden de 2–3 veces más con lo que los rodea); igualando, con d = 512, el cruce caía en ~4 mil piezas contando solo los
kernels y 9–13 mil con todo. **Lo medido:** con una capa sola y lote 1, el cruce está en ~19 mil piezas, más tarde que
lo estimado, porque por debajo de 16 mil manda un piso de ~12 ms (muy probablemente lanzamientos, sin compilar) que la
estimación no tenía; en el régimen largo el costo marginal, 239 ns por pieza, es 3 veces los 80 ns estimados y ~4
veces el piso de ~60 ns de la A100 de 80 GB usada aquí (~2.0 TB/s): lo que rodea a los kernels pesa algo más de lo
estimado. **Con un modelo entero y lote
grande, el cruce baja a entre 1 y 4 mil piezas** (~2.3 mil), porque el piso se reparte entre todo el lote. Bajar
lanzamientos (compilar, CUDA graphs) acercaría el cruce de la capa sola al del modelo; las propuestas 1–4 de la
sección 2.2 bajarían la pendiente.

### 5.9 Conclusión: dónde domina ASRN y dónde no

Con lo medido en la A100 contra Mamba-2 y RWKV-7 originales y nuestro Transformer++, y con las salvedades de arriba
(sin compilar, Mamba-2 sin fusionar, escala chica, dos tasas):

**Domina:**

1. **Entrenar con textos largos.** Un modelo de ~50 M entrena a la misma velocidad con 1 o con 32 mil piezas
   (170–173 mil piezas/s), mientras el Transformer++ pierde 57%: ASRN es 1.07 veces más rápido con 4 096 piezas, 1.54
   con 16 384 y 2.17 con 32 768, con algo menos de memoria; contra Mamba-2 sin fusionar, 1.5 veces a todo largo, y contra
   RWKV-7, 3.5–3.8 veces (también en la quinta corrida RWKV-7 fue el más lento). El cruce con el Transformer++ está entre 1 y 4 mil piezas; en la quinta corrida, contra otro Transformer con
   FlashAttention (26% más rápido que este con 1 024 piezas), la misma receta de ASRN lo alcanzaba recién hacia 4 mil.
2. **Una capa con contextos muy largos** (lote 1): desde ~19 mil piezas pasa a la atención y al bloque de Mamba-2; con
   262 144 es 22.5 veces más rápida que la atención y 2.7 veces que el bloque de Mamba-2 (una unidad con 2.6 veces sus
   parámetros: con modelos enteros la razón es 1.5), a cambio de 2.0 y 1.3 veces su memoria; contra RWKV-7, 7.8 veces
   más rápida y con la mitad de memoria.
3. **Recuperar algo exacto más allá del largo entrenado:** copia con 4 veces el largo casi sin error (≥ 99.67% en las
   3 semillas), donde el Transformer++, que la aprendió en su largo, cae al azar (Mamba-2 no la aprendió y RWKV-7 no terminó de
   aprenderla en el largo de entrenamiento, así que con ellos no hay comparación de extrapolación); y MQAR con 2 y 4
   veces los pares, en textos 2 y 4 veces más largos (99.6% y 98.5% contra 83.4% y 54.4% de Mamba-2 y 59.2% y 30.0% de
   RWKV-7, que en el largo entrenado acierta 99.9%; separación completa por semilla), sin separar
   el efecto del largo del de los pares, con Mamba-2 sin converger y en un MQAR que está en el techo para ASRN
   (sección 5.7). Es la tarea que coincide con su mecanismo incorporado. La caída del Transformer++ puede venir de
   RoPE fuera de su largo o de una cabeza que aprendió un desplazamiento fijo: estos datos no lo separan, ni dicen
   cómo le iría a un Transformer con ALiBi, NoPE o YaRN. Casos de uso posibles, no medidos: código, registros,
   documentos con citas textuales; en ellos además hay que decidir cuándo *no* copiar, que estas tareas no ponen a
   prueba.
4. **Perplejidad en WikiText-103 a este presupuesto** (20 M de posiciones de entrenamiento, ~45 M de parámetros),
   frente a RWKV-7, Mamba-2 y Transformer++: 67.8 contra 72.4, 74.2 y 93.7 con 1 024 piezas, y 63.0 contra 70.2, 73.0
   y 125.6 con 4 096, mejor en cada semilla, con la mejor de las dos tasas de cada modelo (contra RWKV-7, que varía
   mucho entre semillas, no es significativo con n = 3: p = 0.15). Vale solo para estas dos tasas (con 10⁻³ Mamba-2 queda
   por delante), este corpus y este presupuesto, lejos de converger: la diferencia con Mamba-2 se achicó durante el
   entrenamiento (de ~0.14 a ~0.06 nats), y en la quinta corrida, con 30 M de piezas, los márgenes eran menores (ASRN
   57.0, Mamba-2 59.5, Transformer 64.3) y RWKV-7 (la reimplementación) fue el mejor, con 47.9: esa contradicción
   queda sin resolver.
5. **Estabilidad numérica (la capa estable contra v89):** sin la fuga causal (0 contra 0.016 con γ = 10⁹), con el
   reloj sin deriva (5.5·10⁻⁸ constante contra 8.8·10⁻² con el reloj corrido 10⁹), el reloj inmune a TF32 y sin NaN con β enorme, a un costo
   chico (+3% de tiempo con 262 mil piezas y +4–5% de memoria). **Para textos largos conviene desde ~1 millón de
   piezas y hace falta desde ~16 millones:** el reloj de 32 bits pierde precisión desde 1 M (2.2% en la diferencia
   entre vecinas; 8.6% a 4 M) y se detiene en el 1.1% de los pasos a 4 M y en el 16% a 16 M, también en la GPU.
   Cuánto pesa en la salida depende de γ: con un millón de piezas, la capa de prueba cambió menos de 0.1%.

**No domina:**

1. **Textos cortos con lote chico:** hasta 16 mil piezas, una capa ASRN tarda ~12 ms, contra 3.2–9.9 de la atención
   y 7.7–11.1 de Mamba-2 (RWKV-7 tarda lo mismo que ASRN con 1 y 4 mil y más desde 16 mil); en el modelo con 1 024 piezas, el Transformer++ es 5% más rápido. Probablemente es el costo
   de lanzar kernels (no se perfiló en esta corrida; con CUDA Graph el proyecto midió antes 11.4 → 1.6 ms) y es lo
   primero a atacar (compilar, CUDA graphs).
2. **Memoria por capa:** ~40 KB por pieza, contra 20 de la atención y 31 de Mamba-2 (RWKV-7 usa 79).
3. **Escribir:** es la más lenta con lote 1 (40–41 ms por pieza, 2.8 veces el Transformer++, 1.6 veces Mamba-2 y 1.2 veces RWKV-7) y
   guarda una memoria que crece con el contexto (su pico con 64 mil es 3 veces el de Mamba-2, aunque esa cifra incluye
   el prefijo). El algoritmo es O(t) por pieza; la sección 2.4 da el camino a O(1) en tiempo; la memoria sigue
   creciendo con el contexto, aunque se puede acotar a T·2¹⁴·C ≈ 4.2 millones de posiciones.
4. **Aprender rápido dentro del largo entrenado:** el Transformer++ aprende la copia en ≤ 200 pasos, ASRN con la misma
   tasa en 1 000–1 200, y cada corrida de ASRN tarda 1.2–1.8 veces la del Transformer++. Nada se comparó a igual
   cómputo.
5. **Robustez del aprendizaje de la copia:** 3 de 24 corridas livianas de la ablación no salieron de la meseta en 2 000
   pasos, la copia con 16× va de 43 a 97% según la semilla, y la prueba de copia de v89 falla desde la quinta corrida.
6. **MQAR en el largo entrenado contra Mamba-2** (100% contra 95.9%, p = 0.059, con Mamba-2 aún aprendiendo) y **texto
   con la misma tasa de 10⁻³** (Mamba-2 86.0 contra ASRN 88.6, una semilla): no hay ventaja demostrada.
7. **Texto contra RWKV-7, sin conclusión:** aquí ASRN gana en cada semilla, pero sin significancia (p = 0.15) y con la
   tasa de RWKV-7 en el borde de la grilla; en la quinta corrida, con otro protocolo y la reimplementación de RWKV-7,
   RWKV-7 ganó por 16%. Hace falta repetir con el mismo protocolo, más tasas y más datos.
8. **Generalidad de la ventaja en texto:** un solo corpus, que premia copiar, y la tasa del Transformer++ en el borde
   de la grilla.

### 5.10 Lo que falta

De la A100 quedan tres cosas cortas (sección 6): `mezcla_lse` en MQAR con la corrección, la escritura del
Transformer++ sin el artefacto de memoria y las tasas extra del Transformer++ en WikiText. Después, en orden de valor:
más tasas y más datos en texto para ASRN, RWKV-7 y Mamba-2 (la contradicción con la quinta corrida); compilar
(PyTorch ≥ 2.14) y medir con `causal-conv1d` para Mamba-2; un segundo corpus de texto y una tarea con distractores para
juzgar la puerta; la ablación con 3·10⁻³ y más semillas; entender la falla de `test_entrenar_aprende_a_copiar_y_reanuda`
(desde la quinta corrida); pruebas propias para `log_alfa_min` y `mejor_max`; y las propuestas de kernel de la
sección 2.2, empezando por la ventana persistente y la escritura O(1).

## 6. Cómo reproducir

```bash
pip install -r requirements.txt
python pruebas/test_asrn_v89.py                 # 62 pruebas (~4 min en CPU, con los kernels en el intérprete)
python pruebas/test_asrn_estable.py             # 15 pruebas de la capa estable (~1 min)
python benchmarks/banco_asrn.py costo latencia escritura      # ~5 min
TRITON_INTERPRET=1 python benchmarks/banco_asrn.py costo_kernel
python benchmarks/banco_asrn.py calidad ablacion              # horas en CPU; minutos en una GPU
python benchmarks/banco_asrn.py resumir                       # tablas en results/banco_cpu/RESUMEN.md
python history/probes/reloj.py                  # las sondas de la sección 1.2: reloj, fuga, corrimiento, gradientes
```

La tarea de texto en CPU lee la documentación de Vim 9.1 de `/usr/share/vim/vim91/doc` (paquete `vim-runtime`;
*Opticks*, opcional, se busca en `/usr/local`); sin ella, use `--tareas copia,mqar`.

**En una A100 (Colab):** `benchmarks/banco_a100.ipynb` (retirado en 1.0; hoy
[`benchmarks/asrn_colab.ipynb`](../../benchmarks/asrn_colab.ipynb)) corre el banco a escala de GPU (`--perfil a100` de
`banco_asrn.py`, más `banco_gpu.py`; el costo exacto de la sección 5.2 queda para la CPU) contra los rivales de
[`modelos_oficiales.py`](../../benchmarks/modelos_oficiales.py): Transformer++ (nuestra receta de Llama con
FlashAttention-2), Mamba-2 de `mamba_ssm` y RWKV-7 de `flash-linear-attention`. Primero comprueba que cada original
entrena y escribe en esa GPU: si uno no entrena, usa en su lugar la reimplementación de `rivales_lineales.py` (y lo
anota en la columna `implementacion` de cada fila); si solo falla al escribir, no mide su escritura. Guarda cada
resultado en Drive apenas termina, y si Colab se desconecta retoma solo lo que falta. La corrida completa se estimó en
~3 h 20 min (~43 unidades de Colab); la primera tardó 2 h 40 min sin RWKV-7, y la segunda, solo con RWKV-7 y la
estabilidad, 1 h 30 min.

**Para completar lo que queda** (sección 5.10), con la misma carpeta de Drive, la rama actualizada y, en la celda de
configuración, **todo** esto (en la segunda ejecución no se pusieron `REHACER` ni `TASAS_EXTRA`, y sin ellos no hay
nada que borrar ni que agregar):

```python
SECCIONES = ['escritura', 'wikitext', 'ablacion']
MAMBA_COMPILADO = False                  # causal-conv1d no tiene rueda para esta versión: ahorra 8 min
REHACER = ['ablacion/ASRN:mezcla_lse/mqar', 'escritura_gpu/Transformer++']
TASAS_EXTRA = {'wikitext/Transformer++': (3e-4, 6e-4)}
SEMILLAS = None
```

Unos 25–35 minutos (≈ 6–8 unidades). Solo corre eso; lo demás se conserva tal cual. El cuaderno guarda en `entorno` la
fecha y una huella (sha256) del código de cada ejecución, aunque venga de un ZIP, y conserva los de las ejecuciones
anteriores en `entornos_anteriores` (la primera, del 9 de octubre, corrió desde un ZIP sin huella; la del 11, con la
huella `176cfe250773`).
