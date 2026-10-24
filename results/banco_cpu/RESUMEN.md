### Costo por capa

| Capa (d = 512) | MFLOP/pieza L=1024 | MFLOP/pieza L=4096 | MFLOP/pieza L=16384 | KB guardados/pieza |
|---|---|---|---|---|
| ASRN (v89) | 5.19 | 5.14 | 5.13 | 43.8 |
| ASRN (estable) | 5.19 | 5.14 | 5.13 | 43.9 |
| Transformer | 9.44 | 18.87 | 56.62 | 12.0 |
| Mamba-2 | 11.48 | 11.48 | 11.48 | 93.8 |
| RWKV-7 | 8.26 | — | — | 183.3 |
| ASRN (v89) + kernel | (igual) | (igual) | (igual) | 20.1 |
| ASRN (estable) + kernel | (igual) | (igual) | (igual) | 20.2 |

### Latencia de ida y vuelta de una capa (ms, CPU)

| Capa (d = 256) | L=512 | L=2048 | L=8192 | L=32768 | exponente |
|---|---|---|---|---|---|
| ASRN (v89) | 22.7 | 63.7 | 317.6 | 1853.8 | 1.22 |
| ASRN (estable) | 26.9 | 67.5 | 297.7 | 1608.8 | 1.14 |
| Transformer | 5.3 | 29.6 | 331.5 | 4796.0 | 1.83 |
| Mamba-2 | 18.2 | 83.6 | 417.7 | 9847.8 | 1.72 |
| RWKV-7 | 310.8 | 1896.0 | — | — | nan |

### Escritura (ms por pieza nueva)

| Contexto | 512 | 2048 | 8192 | 32768 | exponente |
|---|---|---|---|---|---|
| ASRN | 3.83 | 5.09 | 8.93 | 14.55 | 0.33 |

### Calidad: copia

| Modelo | lr | semillas | parámetros | exactitud L=256 | exactitud L=1024 | exactitud L=4096 | s/corrida | piezas/s |
|---|---|---|---|---|---|---|---|---|
| ASRN | 0.003 | 3 | 684,100 | 100.0% ± 0.0 | 99.8% ± 0.1 | 74.3% ± 9.4 | 112 | 14,633 |
| Transformer | 0.001 | 3 | 609,920 | 100.0% ± 0.0 | 0.7% ± 0.0 | 0.8% ± 0.1 | 35 | 46,739 |
| Mamba-2 | 0.001 | 3 | 721,608 | 0.9% ± 0.2 | 0.8% ± 0.0 | 0.8% ± 0.1 | 132 | 12,450 |
