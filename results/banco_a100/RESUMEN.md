## ASRN 1.0 en NVIDIA A100-SXM4-80GB con PyTorch 2.14.1+cu130 (2026-10-17 02:12 UTC, modo completo)

- originales: listo (13.1 min)
- pruebas: listo (10.2 min)
- estabilidad: listo (0.5 min)
- capa: listo (9.5 min)
- modelo: listo (10.4 min)
- escritura: listo (9.4 min)
- copia: listo (31.6 min)
- mqar: listo (31.2 min)
- wikitext: listo (50.8 min)
- ablacion: listo (80.8 min)
- citas: listo (36.9 min)

### Reloj en 32 bits (cumsum de la GPU)

| Largo | error B_t − B_{t−1} | error a 32 | error a 1 024 | reloj parado |
|---|---|---|---|---|
| 2^20 | 2.16e-02 | 1.92e-03 | 6.92e-05 | 0.03% |
| 2^22 | 8.60e-02 | 7.68e-03 | 2.95e-04 | 1.11% |
| 2^24 | 5.02e-01 | 3.27e-02 | 1.31e-03 | 16.08% |
| 2^25 | 1.00e+00 | 6.70e-02 | 2.90e-03 | 37.21% |

- Kernel ESTABLE contra PyTorch: fp32 salida 4.9e-08, gradientes 4.5e-06; bf16 9.6e-02 de fp32 (PyTorch en bf16: 9.4e-02)
- Fuga causal con γ = 1e9: v89 1.6e-02, estable 0.0e+00
- Reloj corrido, v89: 1e+06: 9.4e-05, 1.6e+07: 2.8e-03, 1e+09: 8.8e-02, 1e6 contra 1e9: 8.8e-02
- Reloj corrido, estable: 1e+06: 5.5e-08, 1.6e+07: 5.5e-08, 1e+09: 5.5e-08, 1e6 contra 1e9: 5.5e-08
- Reloj con TF32 encendido contra apagado (diferencia máxima de B_t, 65 536 piezas): v89: 1.98e+00, v89 bf16: 1.16e-10, estable: 1.16e-10, estable bf16: 5.82e-11
- Una capa con 2^20 piezas: {'v89': {'segundos': 0.041694986000038625, 'memoria_gb': 2.185232896}, 'estable': {'segundos': 0.033261810000112746, 'memoria_gb': 2.197815808}, 'diferencia_final': 0.0008180793956853449}

### Una capa (ms de ida y vuelta)

| Capa | L=1024 | L=4096 | L=16384 | L=65536 | L=262144 |
|---|---|---|---|---|---|
| ASRN (v89) | 12.0 | 11.7 | 11.7 | 18.0 | 64.0 |
| ASRN (estable) | 12.1 | 12.2 | 12.5 | 19.3 | 68.7 |
| Transformer++ | 3.3 | 3.4 | 10.0 | 102.4 | 1486.0 |
| Mamba-2 | 7.7 | 9.9 | 11.1 | 34.8 | 174.3 |
| RWKV-7 | 11.6 | 12.0 | 35.0 | 131.1 | 519.4 |

### Una capa (GB de memoria pico)

| Capa | L=1024 | L=4096 | L=16384 | L=65536 | L=262144 |
|---|---|---|---|---|---|
| ASRN (v89) | 0.0 | 0.2 | 0.6 | 2.6 | 10.3 |
| ASRN (estable) | 0.0 | 0.2 | 0.7 | 2.7 | 10.9 |
| Transformer++ | 0.0 | 0.1 | 0.3 | 1.4 | 5.5 |
| Mamba-2 | 0.1 | 0.2 | 0.5 | 2.1 | 8.5 |
| RWKV-7 | 0.1 | 0.3 | 1.3 | 5.3 | 21.1 |

### Entrenar modelos de ~50 M (piezas por segundo; GB)

| Modelo | parámetros | L=1024 | L=4096 | L=16384 | L=32768 |
|---|---|---|---|---|---|
| ASRN | 50.7 M | 172,065 (23.5 GB) | 171,275 (23.5 GB) | 163,503 (23.5 GB) | 163,023 (23.5 GB) |
| Transformer++ | 51.0 M | 182,116 (25.5 GB) | 160,772 (25.5 GB) | 110,342 (25.5 GB) | 78,400 (25.6 GB) |
| Mamba-2 | 51.2 M | 118,002 (26.4 GB) | 118,314 (26.4 GB) | 116,935 (26.4 GB) | 114,544 (26.4 GB) |
| RWKV-7 | 51.1 M | 50,439 (30.5 GB) | 50,082 (30.5 GB) | 48,397 (30.5 GB) | 45,895 (30.5 GB) |
| ASRN compilado | 50.7 M | 210,413 (22.3 GB) | 207,945 (22.3 GB) | 200,800 (22.3 GB) | 198,826 (22.3 GB) |
| Transformer++ compilado | 51.0 M | 262,909 (22.9 GB) | 220,468 (22.9 GB) | 135,614 (22.9 GB) | 90,540 (22.9 GB) |

### Escribir (ms por pieza nueva)

| Modelo | contexto 1024 | contexto 4096 | contexto 16384 | contexto 65536 |
|---|---|---|---|---|
| ASRN | 42.84 | 42.82 | 43.45 | 43.09 |
| Transformer++ | 14.60 | 14.44 | 14.72 | 15.23 |
| Mamba-2 | 25.28 | 24.93 | 24.69 | 24.47 |
| RWKV-7 | 34.32 | 38.21 | 34.00 | 39.51 |

### Calidad: copia

| Modelo | lr | semillas | parámetros | exactitud L=1024 | exactitud L=4096 | exactitud L=16384 | s/corrida | piezas/s |
|---|---|---|---|---|---|---|---|---|
| ASRN | 0.001 | 3 | 2,631,236 | 66.7% ± 57.7 | 66.7% ± 57.6 | 44.5% ± 38.4 | 78 | 840,856 |
| Transformer++ | 0.001 | 3 | 2,541,312 | 100.0% ± 0.0 | 0.2% ± 0.0 | 0.2% ± 0.0 | 40 | 1,623,770 |
| Mamba-2 | 0.003 | 3 | 2,458,488 | 0.2% ± 0.1 | 0.2% ± 0.0 | 0.2% ± 0.0 | 88 | 744,354 |
| RWKV-7 | 0.001 | 3 | 2,915,328 | 25.8% ± 20.0 | 0.2% ± 0.0 | 0.2% ± 0.0 | 245 | 266,968 |

### Calidad: mqar

| Modelo | lr | semillas | parámetros | exactitud L=512 | exactitud L=1024 | exactitud L=2048 | s/corrida | piezas/s |
|---|---|---|---|---|---|---|---|---|
| ASRN | 0.003 | 3 | 4,597,316 | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.7% ± 0.4 | 87 | 756,695 |
| Transformer++ | 0.003 | 3 | 4,507,392 | 1.5% ± 0.2 | 0.6% ± 0.2 | 0.3% ± 0.1 | 54 | 1,206,957 |
| Mamba-2 | 0.003 | 3 | 4,424,568 | 93.2% ± 3.3 | 57.2% ± 23.8 | 26.1% ± 25.6 | 94 | 700,821 |
| RWKV-7 | 0.001 | 3 | 4,881,408 | 65.4% ± 31.8 | 34.2% ± 18.5 | 16.5% ± 9.3 | 208 | 314,339 |

### Calidad: wikitext

| Modelo | lr | semillas | parámetros | perplejidad L=1024 | perplejidad L=4096 | s/corrida | piezas/s |
|---|---|---|---|---|---|---|---|
| ASRN | 0.003 | 3 | 45,138,568 | 67.91 ± 1.00 | 63.08 ± 0.97 | 147 | 136,300 |
| Transformer++ | 0.001 | 3 | 44,416,000 | 93.27 ± 1.07 | 128.87 ± 6.56 | 121 | 164,951 |
| Mamba-2 | 0.003 | 3 | 44,656,912 | 74.25 ± 0.92 | 73.09 ± 0.88 | 141 | 141,675 |
| RWKV-7 | 0.003 | 3 | 46,320,640 | 75.81 ± 6.39 | 73.96 ± 6.71 | 300 | 66,646 |

### Ablación: copia (exactitud)

| Variante | semillas | L=1024 | L=4096 | L=16384 |
|---|---|---|---|---|
| estable_A | 3 | 66.7% ± 57.7 | 66.7% ± 57.6 | 44.5% ± 38.4 |
| estable_AB | 3 | 66.7% ± 57.6 | 66.7% ± 57.6 | 46.1% ± 42.8 |
| sin_puerta | 3 | 66.7% ± 57.7 | 66.7% ± 57.6 | 49.6% ± 42.8 |
| puerta_sin_senales | 3 | 100.0% ± 0.0 | 100.0% ± 0.0 | 87.6% ± 11.6 |
| sin_reloj | 3 | 66.8% ± 57.6 | 66.7% ± 57.6 | 59.0% ± 51.7 |
| sin_hawkes_racha | 3 | 100.0% ± 0.0 | 100.0% ± 0.0 | 78.9% ± 8.7 |
| sin_sufijo | 3 | 62.4% ± 54.3 | 46.6% ± 50.1 | 21.2% ± 32.3 |
| sin_confirmacion | 3 | 100.0% ± 0.0 | 100.0% ± 0.0 | 65.4% ± 22.4 |
| dos_tablas | 3 | 33.5% ± 57.6 | 33.3% ± 57.4 | 12.3% ± 20.9 |
| solo_local | 3 | 0.2% ± 0.0 | 0.2% ± 0.0 | 0.2% ± 0.0 |
| mezcla_lse | 3 | 33.3% ± 57.3 | 31.0% ± 53.4 | 17.3% ± 29.6 |

### Ablación: mqar (exactitud)

| Variante | semillas | L=512 | L=1024 | L=2048 |
|---|---|---|---|---|
| estable_A | 3 | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.7% ± 0.4 |
| estable_AB | 3 | 100.0% ± 0.0 | 99.9% ± 0.2 | 99.3% ± 1.2 |
| sin_puerta | 3 | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.9% ± 0.0 |
| puerta_sin_senales | 3 | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.5% ± 0.5 |
| sin_reloj | 3 | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.9% ± 0.1 |
| sin_hawkes_racha | 3 | 100.0% ± 0.1 | 99.9% ± 0.1 | 99.5% ± 0.7 |
| sin_sufijo | 3 | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.7% ± 0.5 |
| sin_confirmacion | 3 | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.9% ± 0.2 |
| dos_tablas | 3 | 99.9% ± 0.1 | 99.8% ± 0.2 | 99.0% ± 0.6 |
| solo_local | 3 | 1.3% ± 0.3 | 0.7% ± 0.1 | 0.3% ± 0.0 |
| mezcla_lse | 3 | 6.1% ± 2.1 | 3.9% ± 1.1 | 2.6% ± 1.3 |

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

