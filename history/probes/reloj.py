# Copyright 2026 Alexis Calvario
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sonda 2: precision del reloj B_t = cumsum(beta) en 32 bits.
  - 'GPU' (acumulador de 32 bits, como cumsum en CUDA y el reloj de _Escritor.paso): np.cumsum en float32
  - 'CPU' (acumulador de 64 bits redondeado a 32 en cada salida, como torch.cumsum en CPU)
  - referencia: float64
Errores relativos de la diferencia B_t - B_s (lo unico que usa el modelo) para |t - s| = 1, 32, 1024."""
import numpy as np

rng = np.random.default_rng(0)
print(f"{'largo':>10} {'beta media':>10} | {'acum. 32 bits: err. rel. dif. 1 / 32 / 1024':>44} | "
      f"{'reloj parado':>12} | {'redondeo a 32: err. rel. dif. 1 / 32':>36}")
for log2L in (16, 20, 22, 24, 25, 26):
    L = 2 ** log2L
    u = rng.standard_normal(L).astype(np.float32)
    beta = np.log1p(np.exp(u)).astype(np.float32)            # softplus(N(0,1)): media ~0.81
    B64 = np.cumsum(beta.astype(np.float64))
    B32a = np.cumsum(beta, dtype=np.float32)                  # acumulador de 32 bits
    B32r = B64.astype(np.float32)                             # exacto y redondeado (CPU)
    fila = []
    for k in (1, 32, 1024):
        ref = B64[k:] - B64[:-k]
        err = np.abs((B32a[k:].astype(np.float64) - B32a[:-k]) - ref) / ref
        fila.append(np.median(err[-L // 4:]))                # ultimo cuarto del texto (B grande)
    parado = np.mean(B32a[1:] == B32a[:-1])
    filar = []
    for k in (1, 32):
        ref = B64[k:] - B64[:-k]
        err = np.abs((B32r[k:].astype(np.float64) - B32r[:-k]) - ref) / ref
        filar.append(np.median(err[-L // 4:]))
    print(f"{'2^%d' % log2L:>10} {beta.mean():>10.3f} | {fila[0]:>12.2e} {fila[1]:>12.2e} {fila[2]:>12.2e}       | "
          f"{parado:>12.2%} | {filar[0]:>12.2e} {filar[1]:>12.2e}")
