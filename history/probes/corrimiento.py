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
"""La capa depende solo de diferencias del reloj: sumarle una constante (como estar a millones de piezas del inicio)
no deberia cambiar la salida. Se mide cuanto cambia segun la constante, en v89 y en la capa estable."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import torch, asrn.nucleo as v89, asrn.estable as est
torch.manual_seed(0)
d, L = 64, 256
x = torch.randint(0, 20, (2, L)); x[:, 128:192] = x[:, :64]
E, h = torch.randn(20, d)[x], torch.randn(2, L, d)
for nombre, capa in [('v89', est.CapaASRNEstable(d, est.Protecciones.ninguna(), chunk=16)),
                     ('estable', est.CapaASRNEstable(d, chunk=16))]:
    torch.manual_seed(1)
    with torch.no_grad():
        for n, p in capa.named_parameters():
            if n in v89._NUEVOS_V87 or n in ('cq.weight', 'w_sufijo', 'w_vistos', 'g_conf.weight'):
                p.add_(torch.randn_like(p) * 0.5)
        capa.gamma.copy_(torch.linspace(0.5, 4, capa.n_tablas))
        sig = v89.firmas_de_piezas(x)
        Qc, Kc, beta, la = capa._contexto(h)
        Bt = beta.cumsum(1)
        def salida(off):
            Bo = Bt + off
            acc = capa._acumular(E, Qc, Kc, Bo, la, sig)
            acc.update(capa._repeticion(sig, beta, Bo, la))
            return capa._combinar(acc, h)
        base = salida(0.0)
        fila = []
        for off in (1e5, 1e6, 4e6, 1.6e7, 6.4e7, 1e9):
            fila.append(f'{(salida(off) - base).abs().max().item() / base.abs().max().item():.1e}')
        print(f'{nombre:8} corrimiento 1e5, 1e6, 4e6, 1.6e7, 6.4e7, 1e9 -> error relativo de la salida: ' + '  '.join(fila))
