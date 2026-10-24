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
"""Sonda 1: la mascara de v89 es -1e9 (no -inf). Si todos los puntajes validos de una pregunta quedan por debajo de
-1e9 (nucleo de tiempo enorme), el softmax manda la masa a llaves NO validas: la propia pregunta (que copia E_{t+1})
y posiciones posteriores. Se mide si cambiar el futuro cambia el pasado."""
import os
import sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import asrn.estable as est

torch.manual_seed(0)
d, L, corte = 32, 96, 60
capa = est.CapaASRNEstable(d, chunk=16, n_bits=3)        # pocos bits: casilleros llenos, muchas llaves por pregunta
E = torch.randn(1, L, d)
h = torch.randn(1, L, d)
x = torch.randint(0, 7, (1, L))
E2, h2, x2 = E.clone(), h.clone(), x.clone()
E2[:, corte:] = torch.randn(1, L - corte, d)
h2[:, corte:] = torch.randn(1, L - corte, d)
x2[:, corte:] = torch.randint(0, 7, (1, L - corte))

for gamma in (0.0, 1.0, 1e3, 1e9, 1e11):
    with torch.no_grad():
        capa.gamma.fill_(gamma)
        a, b = capa(E, h, x), capa(E2, h2, x2)
    dif = (a[:, :corte] - b[:, :corte]).abs().max().item()
    print(f"gamma={gamma:>8.0e}  max |salida(pasado) con futuro A - con futuro B| = {dif:.3e}"
          + ("   <-- FUGA CAUSAL" if dif > 1e-6 else ""))
