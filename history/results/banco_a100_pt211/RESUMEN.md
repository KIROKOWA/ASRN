## Banco ASRN en NVIDIA A100-SXM4-80GB (2026-10-11 03:05 UTC, modo completo)

- originales: listo (12.3 min)
- pruebas: listo (5.0 min)
- estabilidad: listo (0.5 min)
- capa: listo (7.1 min)
- modelo: listo (5.4 min)
- escritura: listo (8.5 min)
- copia: listo (17.5 min)
- mqar: listo (15.6 min)
- wikitext: listo (23.6 min)
- ablacion: listo (0.0 min)

### Reloj en 32 bits (cumsum de la GPU)

| Largo | error B_t − B_{t−1} | error a 32 | error a 1 024 | reloj parado |
|---|---|---|---|---|
| 2^20 | 2.16e-02 | 1.92e-03 | 6.82e-05 | 0.03% |
| 2^22 | 8.60e-02 | 7.70e-03 | 3.17e-04 | 1.11% |
| 2^24 | 5.02e-01 | 3.27e-02 | 1.31e-03 | 16.08% |
| 2^25 | 1.00e+00 | 6.69e-02 | 2.76e-03 | 37.21% |

- Kernel ESTABLE contra PyTorch: fp32 salida 4.9e-08, gradientes 5.3e-06; bf16 9.6e-02 de fp32 (PyTorch en bf16: 9.1e-02)
- Fuga causal con γ = 1e9: v89 1.6e-02, estable 0.0e+00
- Reloj corrido, v89: 1e+06: 9.4e-05, 1.6e+07: 2.8e-03, 1e+09: 8.8e-02, 1e6 contra 1e9: 8.8e-02
- Reloj corrido, estable: 1e+06: 5.5e-08, 1.6e+07: 5.5e-08, 1e+09: 5.5e-08, 1e6 contra 1e9: 5.5e-08
- Reloj con TF32 encendido contra apagado (diferencia máxima de B_t, 65 536 piezas): v89: 1.98e+00, v89 bf16: 1.16e-10, estable: 1.16e-10, estable bf16: 5.82e-11
- Una capa con 2^20 piezas: {'v89': {'segundos': 0.03515708400004769, 'memoria_gb': 2.185232896}, 'estable': {'segundos': 0.03240366599993649, 'memoria_gb': 2.197815808}, 'diferencia_final': 0.0008180793956853449}

### Una capa (ms de ida y vuelta)

| Capa | L=1024 | L=4096 | L=16384 | L=65536 | L=262144 |
|---|---|---|---|---|---|
| ASRN (v89) | 13.6 | 11.5 | 13.0 | 18.2 | 64.2 |
| ASRN (estable) | 12.2 | 11.9 | 12.1 | 19.3 | 66.2 |
| Transformer++ | 3.2 | 3.3 | 9.9 | 101.5 | 1488.9 |
| Mamba-2 | 7.7 | 7.8 | 11.1 | 35.3 | 180.9 |
| RWKV-7 | 11.2 | 12.0 | 35.3 | 131.4 | 519.7 |

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
| ASRN | 50.7 M | 173,346 (23.5 GB) | 172,431 (23.5 GB) | 170,474 (23.5 GB) | 170,460 (23.5 GB) |
| Transformer++ | 51.0 M | 182,654 (25.5 GB) | 161,049 (25.5 GB) | 110,493 (25.5 GB) | 78,378 (25.6 GB) |
| Mamba-2 | 51.2 M | 116,745 (26.4 GB) | 116,766 (26.4 GB) | 115,221 (26.4 GB) | 113,211 (26.4 GB) |
| RWKV-7 | 51.1 M | 49,431 (30.5 GB) | 49,127 (30.5 GB) | 47,513 (30.5 GB) | 45,143 (30.5 GB) |

### Escribir (ms por pieza nueva)

| Modelo | contexto 1024 | contexto 4096 | contexto 16384 | contexto 65536 |
|---|---|---|---|---|
| ASRN | 40.53 | 40.15 | 40.98 | 40.98 |
| Transformer++ | 14.11 | 14.53 | 14.59 | 14.31 |
| Mamba-2 | 25.39 | 25.45 | 25.36 | 25.39 |
| RWKV-7 | 34.68 | 35.00 | 35.68 | 35.48 |

### Calidad: copia

| Modelo | lr | semillas | parámetros | exactitud L=1024 | exactitud L=4096 | exactitud L=16384 | s/corrida | piezas/s |
|---|---|---|---|---|---|---|---|---|
| ASRN | 0.001 | 3 | 2,631,236 | 100.0% ± 0.0 | 99.9% ± 0.2 | 67.6% ± 27.0 | 70 | 932,296 |
| Transformer++ | 0.001 | 3 | 2,541,312 | 100.0% ± 0.0 | 0.2% ± 0.0 | 0.2% ± 0.0 | 40 | 1,637,044 |
| Mamba-2 | 0.003 | 3 | 2,458,488 | 0.2% ± 0.0 | 0.2% ± 0.0 | 0.2% ± 0.0 | 86 | 758,393 |
| RWKV-7 | 0.001 | 3 | 2,915,328 | 30.5% ± 25.7 | 0.3% ± 0.0 | 0.2% ± 0.0 | 251 | 260,671 |

### Calidad: mqar

| Modelo | lr | semillas | parámetros | exactitud L=512 | exactitud L=1024 | exactitud L=2048 | s/corrida | piezas/s |
|---|---|---|---|---|---|---|---|---|
| ASRN | 0.001 | 3 | 4,597,316 | 100.0% ± 0.0 | 99.6% ± 0.6 | 98.5% ± 1.8 | 81 | 806,591 |
| Transformer++ | 0.003 | 3 | 4,507,392 | 0.8% ± 0.9 | 0.5% ± 0.5 | 0.2% ± 0.2 | 54 | 1,216,165 |
| Mamba-2 | 0.003 | 3 | 4,424,568 | 95.9% ± 1.8 | 83.4% ± 3.1 | 54.4% ± 5.9 | 92 | 710,264 |
| RWKV-7 | 0.001 | 3 | 4,881,408 | 99.9% ± 0.2 | 59.2% ± 1.3 | 30.0% ± 0.3 | 213 | 308,073 |

### Calidad: wikitext

| Modelo | lr | semillas | parámetros | perplejidad L=1024 | perplejidad L=4096 | s/corrida | piezas/s |
|---|---|---|---|---|---|---|---|
| ASRN | 0.003 | 3 | 45,138,568 | 67.78 ± 0.88 | 63.00 ± 0.86 | 144 | 138,579 |
| Transformer++ | 0.001 | 3 | 44,416,000 | 93.70 ± 1.19 | 125.56 ± 3.83 | 121 | 165,414 |
| Mamba-2 | 0.003 | 3 | 44,656,912 | 74.17 ± 1.03 | 72.98 ± 1.03 | 142 | 141,131 |
| RWKV-7 | 0.003 | 3 | 46,320,640 | 72.35 ± 4.03 | 70.24 ± 4.05 | 305 | 65,594 |

### Ablación: copia (exactitud)

| Variante | semillas | L=1024 | L=4096 | L=16384 |
|---|---|---|---|---|
| estable_A | 3 | 100.0% ± 0.0 | 99.9% ± 0.2 | 67.6% ± 27.0 |
| estable_AB | 3 | 100.0% ± 0.0 | 100.0% ± 0.0 | 75.4% ± 14.3 |
| sin_puerta | 3 | 66.7% ± 57.6 | 66.7% ± 57.6 | 44.1% ± 38.4 |
| puerta_sin_senales | 3 | 100.0% ± 0.0 | 99.9% ± 0.1 | 74.7% ± 22.1 |
| sin_reloj | 3 | 66.7% ± 57.7 | 66.7% ± 57.6 | 62.5% ± 54.0 |
| sin_hawkes_racha | 3 | 100.0% ± 0.0 | 99.8% ± 0.3 | 69.3% ± 9.9 |
| sin_sufijo | 3 | 66.8% ± 57.6 | 66.4% ± 57.3 | 46.0% ± 43.9 |
| sin_confirmacion | 3 | 100.0% ± 0.0 | 100.0% ± 0.0 | 68.6% ± 15.8 |
| dos_tablas | 3 | 33.5% ± 57.6 | 33.4% ± 57.6 | 13.6% ± 23.2 |
| solo_local | 3 | 0.2% ± 0.1 | 0.2% ± 0.0 | 0.2% ± 0.0 |
| mezcla_lse | 3 | 100.0% ± 0.0 | 99.9% ± 0.1 | 74.7% ± 17.9 |

### Ablación: mqar (exactitud)

| Variante | semillas | L=512 | L=1024 | L=2048 |
|---|---|---|---|---|
| estable_A | 3 | 100.0% ± 0.0 | 99.6% ± 0.6 | 98.5% ± 1.8 |
| estable_AB | 3 | 99.9% ± 0.2 | 99.5% ± 0.7 | 97.5% ± 1.9 |
| sin_puerta | 3 | 100.0% ± 0.0 | 99.8% ± 0.2 | 98.9% ± 0.9 |
| puerta_sin_senales | 3 | 99.9% ± 0.2 | 99.7% ± 0.4 | 98.5% ± 1.5 |
| sin_reloj | 3 | 99.9% ± 0.1 | 99.8% ± 0.2 | 99.3% ± 0.6 |
| sin_hawkes_racha | 3 | 99.9% ± 0.2 | 99.6% ± 0.7 | 97.8% ± 2.9 |
| sin_sufijo | 3 | 100.0% ± 0.1 | 99.9% ± 0.1 | 99.3% ± 0.6 |
| sin_confirmacion | 3 | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.6% ± 0.4 |
| dos_tablas | 3 | 99.8% ± 0.2 | 99.5% ± 0.3 | 97.3% ± 1.6 |
| solo_local | 3 | 1.8% ± 0.0 | 0.9% ± 0.1 | 0.4% ± 0.0 |
| mezcla_lse | 3 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 |
