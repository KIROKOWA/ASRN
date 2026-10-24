# Experimento de citas acotadas (ASRN 1.0)

Semillas 0, 1, 2; 6000 pasos; CPU, 1 hilo(s), PyTorch 2.14.1+cu130; commit 6d94c21; 2026-10-16 23:16 UTC. Generado por `benchmarks/experimento_citas.py --resumir` desde `semilla_*.json` (los datos crudos).

Rachas: posiciones seguidas cuyo grupo de 4 piezas ya salio antes. En el texto real duran a lo sumo ~9 (citas de 6 a 12 piezas). Las rachas se juntan de todas las semillas; "max por semilla" es el maximo de cada una.

## Perdida de validacion

| Modelo | semilla 0 | semilla 1 | semilla 2 |
|---|---|---|---|
| ASRN 1.0 | 3.7056 | 3.7027 | 3.7036 |
| sin Hawkes ni racha | 3.7079 | 3.7076 | 3.7070 |
| puerta sin senales | 3.7586 | 3.7290 | 3.7465 |
| sin puerta | 3.7396 | 3.7372 | 3.7320 |
| sin reloj | 3.7053 | 3.7050 | 3.7042 |

## T=1.0, top_k=0, memoria_de_lo_ajeno=True

| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |
|---|---|---|---|---|---|---|
| texto real | 15.6% | 228 | 5.60 | 10 |  | 0.9% |
| ASRN 1.0 | 9.3% | 439 | 5.21 | 9 | 9 / 9 / 9 | 0.0% |
| sin Hawkes ni racha | 8.4% | 398 | 5.18 | 19 | 16 / 16 / 19 | 7.3% |
| puerta sin senales | 6.8% | 338 | 4.92 | 20 | 19 / 20 / 19 | 11.8% |
| sin puerta | 7.0% | 353 | 4.89 | 29 | 17 / 15 / 29 | 9.1% |
| sin reloj | 9.3% | 442 | 5.17 | 10 | 10 / 10 / 9 | 0.5% |

## T=1.0, top_k=0, memoria_de_lo_ajeno=True (conservar_cita=False)

| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |
|---|---|---|---|---|---|---|
| texto real | 15.6% | 228 | 5.60 | 10 |  | 0.9% |
| ASRN 1.0 | 9.3% | 439 | 5.21 | 9 | 9 / 9 / 9 | 0.0% |

## T=1.0, top_k=0, memoria_de_lo_ajeno=False

| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |
|---|---|---|---|---|---|---|
| texto real | 15.6% | 228 | 5.60 | 10 |  | 0.9% |
| ASRN 1.0 | 11.2% | 509 | 5.38 | 10 | 9 / 10 / 10 | 0.4% |
| sin Hawkes ni racha | 11.0% | 502 | 5.39 | 18 | 18 / 18 / 17 | 8.4% |
| puerta sin senales | 6.2% | 313 | 4.86 | 24 | 24 / 19 / 18 | 10.9% |
| sin puerta | 6.7% | 334 | 4.92 | 36 | 19 / 20 / 36 | 9.9% |
| sin reloj | 10.9% | 503 | 5.30 | 10 | 9 / 10 / 10 | 0.4% |

## T=1.0, top_k=0, memoria_de_lo_ajeno=False (conservar_cita=False)

| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |
|---|---|---|---|---|---|---|
| texto real | 15.6% | 228 | 5.60 | 10 |  | 0.9% |
| ASRN 1.0 | 11.2% | 509 | 5.38 | 10 | 9 / 10 / 10 | 0.4% |

## T=0.8, top_k=40, memoria_de_lo_ajeno=True

| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |
|---|---|---|---|---|---|---|
| texto real | 15.6% | 228 | 5.60 | 10 |  | 0.9% |
| ASRN 1.0 | 20.9% | 980 | 5.24 | 10 | 9 / 10 / 10 | 0.3% |
| sin Hawkes ni racha | 19.0% | 895 | 5.22 | 21 | 21 / 20 / 20 | 7.2% |
| puerta sin senales | 13.9% | 728 | 4.69 | 27 | 17 / 23 / 27 | 9.3% |
| sin puerta | 14.7% | 735 | 4.91 | 23 | 19 / 21 / 23 | 11.8% |
| sin reloj | 20.2% | 947 | 5.25 | 10 | 10 / 10 / 9 | 0.2% |

## T=0.8, top_k=40, memoria_de_lo_ajeno=True (conservar_cita=False)

| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |
|---|---|---|---|---|---|---|
| texto real | 15.6% | 228 | 5.60 | 10 |  | 0.9% |
| ASRN 1.0 | 27.6% | 925 | 7.34 | 10 | 10 / 10 / 10 | 2.8% |

## T=0.8, top_k=40, memoria_de_lo_ajeno=False

| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |
|---|---|---|---|---|---|---|
| texto real | 15.6% | 228 | 5.60 | 10 |  | 0.9% |
| ASRN 1.0 | 22.2% | 1017 | 5.36 | 10 | 9 / 10 / 10 | 0.4% |
| sin Hawkes ni racha | 23.1% | 1064 | 5.33 | 21 | 21 / 20 / 19 | 8.5% |
| puerta sin senales | 13.1% | 700 | 4.59 | 32 | 26 / 20 / 32 | 10.0% |
| sin puerta | 14.1% | 737 | 4.69 | 35 | 21 / 26 / 35 | 9.5% |
| sin reloj | 23.5% | 1085 | 5.32 | 10 | 10 / 10 / 10 | 0.9% |

## T=0.8, top_k=40, memoria_de_lo_ajeno=False (conservar_cita=False)

| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |
|---|---|---|---|---|---|---|
| texto real | 15.6% | 228 | 5.60 | 10 |  | 0.9% |
| ASRN 1.0 | 30.5% | 1004 | 7.47 | 11 | 10 / 10 / 11 | 4.3% |

## Respuesta de la puerta a la racha (ASRN 1.0, primera capa ASRN)

| Semilla | 0 | 0.4 | 1.1 | 2 | 3.4 | 5.4 | 8.2 | 12.3 | 18.3 | 26.9 | 39.4 | 57.5 | 83.7 | 121.6 | 176.5 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -0.23 | +0.29 | +0.76 | +0.75 | +0.13 | -0.66 | -2.37 | -9.19 | -9.17 | -9.17 | -9.17 | -9.17 | -9.17 | -9.17 | -9.17 | -9.17 |
| 1 | -0.11 | +0.29 | +0.62 | +0.51 | -0.13 | -0.65 | -1.98 | -7.68 | -7.68 | -7.68 | -7.68 | -7.68 | -7.68 | -7.68 | -7.68 | -7.68 |
| 2 | -0.14 | +0.27 | +0.64 | +0.58 | +0.06 | -0.50 | -2.00 | -7.82 | -7.80 | -7.80 | -7.80 | -7.80 | -7.80 | -7.80 | -7.80 | -7.80 |
