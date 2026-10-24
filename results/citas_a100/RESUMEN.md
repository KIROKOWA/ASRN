# Experimento de citas acotadas (ASRN 1.0)

Semillas 0, 1, 2; 6000 pasos; GPU NVIDIA A100-SXM4-80GB, PyTorch 2.14.1+cu130; commit ; 2026-10-17 06:47 UTC. Generado por `benchmarks/experimento_citas.py --resumir` desde `semilla_*.json` (los datos crudos).

Rachas: posiciones seguidas cuyo grupo de 4 piezas ya salio antes. En el texto real duran a lo sumo ~9 (citas de 6 a 12 piezas). Las rachas se juntan de todas las semillas; "max por semilla" es el maximo de cada una.

## Perdida de validacion

| Modelo | semilla 0 | semilla 1 | semilla 2 |
|---|---|---|---|
| ASRN 1.0 | 3.7054 | 3.7032 | 3.7049 |
| sin Hawkes ni racha | 3.7094 | 3.7069 | 3.7058 |
| puerta sin senales | 3.7409 | 3.7477 | 3.7348 |
| sin puerta | 3.7312 | 3.7282 | 3.7311 |
| sin reloj | 3.7049 | 3.7038 | 3.7047 |

## T=1.0, top_k=0, memoria_de_lo_ajeno=True

| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |
|---|---|---|---|---|---|---|
| texto real | 15.6% | 228 | 5.60 | 10 |  | 0.9% |
| ASRN 1.0 | 10.1% | 476 | 5.23 | 9 | 9 / 9 / 9 | 0.0% |
| sin Hawkes ni racha | 9.0% | 419 | 5.30 | 20 | 14 / 15 / 20 | 6.9% |
| puerta sin senales | 6.7% | 339 | 4.87 | 27 | 15 / 22 / 27 | 11.8% |
| sin puerta | 7.1% | 363 | 4.83 | 20 | 19 / 20 / 18 | 10.2% |
| sin reloj | 8.8% | 422 | 5.13 | 9 | 9 / 9 / 9 | 0.0% |

## T=1.0, top_k=0, memoria_de_lo_ajeno=True (conservar_cita=False)

| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |
|---|---|---|---|---|---|---|
| texto real | 15.6% | 228 | 5.60 | 10 |  | 0.9% |
| ASRN 1.0 | 10.1% | 476 | 5.23 | 9 | 9 / 9 / 9 | 0.0% |

## T=1.0, top_k=0, memoria_de_lo_ajeno=False

| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |
|---|---|---|---|---|---|---|
| texto real | 15.6% | 228 | 5.60 | 10 |  | 0.9% |
| ASRN 1.0 | 11.8% | 539 | 5.40 | 10 | 9 / 10 / 10 | 0.4% |
| sin Hawkes ni racha | 11.2% | 522 | 5.29 | 20 | 16 / 18 / 20 | 7.3% |
| puerta sin senales | 6.0% | 297 | 5.01 | 28 | 24 / 19 / 28 | 10.8% |
| sin puerta | 7.5% | 357 | 5.14 | 27 | 27 / 20 / 20 | 11.8% |
| sin reloj | 10.9% | 504 | 5.33 | 10 | 9 / 10 / 9 | 0.2% |

## T=1.0, top_k=0, memoria_de_lo_ajeno=False (conservar_cita=False)

| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |
|---|---|---|---|---|---|---|
| texto real | 15.6% | 228 | 5.60 | 10 |  | 0.9% |
| ASRN 1.0 | 11.8% | 539 | 5.40 | 10 | 9 / 10 / 10 | 0.4% |

## T=0.8, top_k=40, memoria_de_lo_ajeno=True

| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |
|---|---|---|---|---|---|---|
| texto real | 15.6% | 228 | 5.60 | 10 |  | 0.9% |
| ASRN 1.0 | 21.8% | 992 | 5.39 | 10 | 9 / 10 / 10 | 0.3% |
| sin Hawkes ni racha | 19.6% | 910 | 5.30 | 19 | 17 / 15 / 19 | 7.9% |
| puerta sin senales | 14.9% | 759 | 4.83 | 39 | 24 / 21 / 39 | 9.2% |
| sin puerta | 14.6% | 723 | 4.97 | 21 | 21 / 20 / 20 | 12.0% |
| sin reloj | 19.4% | 907 | 5.26 | 10 | 9 / 10 / 9 | 0.4% |

## T=0.8, top_k=40, memoria_de_lo_ajeno=True (conservar_cita=False)

| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |
|---|---|---|---|---|---|---|
| texto real | 15.6% | 228 | 5.60 | 10 |  | 0.9% |
| ASRN 1.0 | 28.7% | 960 | 7.34 | 11 | 10 / 11 / 10 | 2.5% |

## T=0.8, top_k=40, memoria_de_lo_ajeno=False

| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |
|---|---|---|---|---|---|---|
| texto real | 15.6% | 228 | 5.60 | 10 |  | 0.9% |
| ASRN 1.0 | 24.2% | 1083 | 5.49 | 12 | 10 / 10 / 12 | 1.0% |
| sin Hawkes ni racha | 22.4% | 1005 | 5.48 | 24 | 19 / 24 / 18 | 9.2% |
| puerta sin senales | 13.2% | 676 | 4.80 | 22 | 18 / 21 / 22 | 11.1% |
| sin puerta | 15.4% | 778 | 4.87 | 31 | 31 / 21 / 18 | 11.1% |
| sin reloj | 22.6% | 1045 | 5.32 | 10 | 10 / 9 / 10 | 0.3% |

## T=0.8, top_k=40, memoria_de_lo_ajeno=False (conservar_cita=False)

| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |
|---|---|---|---|---|---|---|
| texto real | 15.6% | 228 | 5.60 | 10 |  | 0.9% |
| ASRN 1.0 | 31.0% | 1007 | 7.57 | 12 | 12 / 11 / 10 | 6.0% |

## Respuesta de la puerta a la racha (ASRN 1.0, primera capa ASRN)

| Semilla | 0 | 0.4 | 1.1 | 2 | 3.4 | 5.4 | 8.2 | 12.3 | 18.3 | 26.9 | 39.4 | 57.5 | 83.7 | 121.6 | 176.5 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -0.16 | +0.32 | +0.75 | +0.68 | +0.11 | -0.59 | -2.10 | -8.41 | -8.39 | -8.39 | -8.39 | -8.39 | -8.39 | -8.39 | -8.39 | -8.39 |
| 1 | -0.11 | +0.29 | +0.64 | +0.56 | +0.00 | -0.55 | -1.90 | -7.73 | -7.73 | -7.73 | -7.73 | -7.73 | -7.73 | -7.73 | -7.73 | -7.73 |
| 2 | -0.14 | +0.28 | +0.65 | +0.62 | +0.13 | -0.46 | -1.95 | -7.67 | -7.65 | -7.65 | -7.65 | -7.65 | -7.65 | -7.65 | -7.65 | -7.65 |
