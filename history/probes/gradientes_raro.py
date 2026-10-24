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
"""Sonda 3: como crecen con el largo (a igual numero de piezas por paso) los gradientes de cada grupo de parametros
y las senales que entran a la puerta ('mejor', log1p(hawkes), racha). Modelo chico con los parametros nuevos fuera de
su valor neutro (como despues de entrenar): gamma > 0, reloj con pesos."""
import os
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import asrn.nucleo as v89

V, d, TOKENS = 8192, 64, 2 ** 15
GRUPOS = {
    'gamma (nucleo de tiempo)': lambda n: n.endswith('.gamma'),
    'reloj (w_reloj, b_reloj)': lambda n: '_reloj' in n,
    'kappas (Hawkes, racha)': lambda n: 'log_kappa' in n,
    'puerta (g, g_conf, w_vistos, u_hawkes, d_racha)': lambda n: any(k in n for k in ('.g.', 'g_conf', 'w_vistos',
                                                                                         'u_hawkes', 'd_racha')),
}


def texto_zipf(B, L, g):
    p = 1.0 / torch.arange(1, V + 1).double() ** 0.0
    x = torch.multinomial(p.expand(B, -1), L, replacement=True, generator=g)
    n = max(4, L // 16)                                # una cita larga en cada fila: hay casilleros con llaves lejanas
    x[:, L - n:] = x[:, :n]
    return x


def modelo(semilla):
    m = v89.crear_modelo(V=V, d=d, n_capas=4, semilla=semilla)
    g = torch.Generator().manual_seed(100 + semilla)
    with torch.no_grad():
        for c in m.largos:
            c.gamma.copy_(torch.rand(c.gamma.shape, generator=g) * 2.0)        # olvido entre 0 y 2
            c.w_reloj.normal_(std=0.05, generator=g)
            c.g_conf.weight.normal_(std=0.3, generator=g)
            c.u_hawkes.normal_(std=0.3, generator=g)
            c.cq.weight.normal_(std=0.1, generator=g)
    return m


senales = []


def espiar(capa, acc, h):
    pt = acc['peso_total']
    senales.append({'mejor': acc['mejor'][pt > 0].detach(), 'hawkes': torch.log1p(acc['hawkes']).detach(),
                    'racha': acc['racha'].detach()})
    return v89.CapaASRN._combinar(capa, acc, h)


print(f"{'largo':>7} | " + " | ".join(f"{k[:24]:>24}" for k in GRUPOS) + f" | {'resto':>8} | {'gamma/total':>11}"
      + f" | {'mejor p1':>9} {'mejor min':>10} | {'log1p(lam) max':>14} | {'racha max':>9}")
for L in (256, 1024, 4096, 16384, 32768):
    B = TOKENS // L
    filas = []
    for semilla in range(3):
        m = modelo(semilla)
        for c in m.largos:
            c._combinar = (lambda cc: (lambda acc, h: espiar(cc, acc, h)))(c)
        g = torch.Generator().manual_seed(semilla)
        x = texto_zipf(B, L + 1, g)
        senales.clear()
        perdida = F.cross_entropy(m(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))
        perdida.backward()
        normas = {k: 0.0 for k in GRUPOS}
        resto = 0.0
        for n, p in m.named_parameters():
            if p.grad is None:
                continue
            q = p.grad.double().square().sum().item()
            for k, f in GRUPOS.items():
                if f(n):
                    normas[k] += q
                    break
            else:
                resto += q
        mej = torch.cat([s['mejor'] for s in senales])
        filas.append(([normas[k] ** 0.5 for k in GRUPOS] + [resto ** 0.5],
                      torch.quantile(mej[:200000], 0.01).item(), mej.min().item(),
                      max(s['hawkes'].max().item() for s in senales), max(s['racha'].max().item() for s in senales)))
    med = [sum(f[0][i] for f in filas) / len(filas) for i in range(len(GRUPOS) + 1)]
    total = sum(v ** 2 for v in med) ** 0.5
    p1 = sum(f[1] for f in filas) / 3
    mn = min(f[2] for f in filas)
    hk = max(f[3] for f in filas)
    rc = max(f[4] for f in filas)
    print(f"{L:>7} | " + " | ".join(f"{v:>24.3e}" for v in med[:-1]) + f" | {med[-1]:>8.3f} | {med[0] / total:>11.1%}"
          + f" | {p1:>9.1f} {mn:>10.1f} | {hk:>14.2f} | {rc:>9.0f}")
