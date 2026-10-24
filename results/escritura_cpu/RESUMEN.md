# Escritura O(1) contra O(t) (2026-10-23 13:37 UTC, cpu, 4 hilos, PyTorch 2.14.1+cu130)

Modelo ASRN de 700,484 parametros (V = 256, d = 128, 4 bloques, 2 capas ASRN), lote 1, sin muestrear. Mediana de 9 tandas de 32 piezas.

| contexto | O(1): ms por pieza | O(t): ms por pieza | O(1): prefijo (s) | O(t): prefijo (s) |
|---|---|---|---|---|
| 512 | 12.45 | 15.80 | 0.10 | 0.06 |
| 2,048 | 12.47 | 16.73 | 0.26 | 0.20 |
| 8,192 | 10.68 | 19.02 | 0.77 | 0.70 |
| 32,768 | 11.26 | 37.12 | 3.60 | 2.73 |
| 131,072 | 9.49 | 91.67 | 29.04 | 20.44 |

- O(1): pendiente -19.21 ns por pieza de contexto (IC 95% -41.54 a 3.12)
- O(t): pendiente 583.31 ns por pieza de contexto (IC 95% 537.13 a 629.49)
