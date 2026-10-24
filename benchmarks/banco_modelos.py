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
"""Los modelos que compara banco_asrn.py, todos con el mismo esqueleto (vocabulario atado a la salida, pre-norma):

  - ASRN (asrn.nucleo, receta: una capa ASRN cada dos bloques) y sus variantes de ablacion
  - Transformer: GPT con RoPE y F.scaled_dot_product_attention (el mismo de las corridas de desarrollo de v89)
  - Mamba-2 y RWKV-7: los de rivales_lineales.py. Sin GPU, Mamba-2 usa aqui el algoritmo SSD por bloques
    (nucleo_ssd_bloques: la misma recurrencia que el paso a paso, en bloques de 64 con productos de matrices) en vez
    del paso a paso; RWKV-7 sigue paso a paso (su forma por bloques, con la regla delta, no esta escrita en PyTorch).
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [p for p in (os.path.dirname(_AQUI), _AQUI) if p not in sys.path]
import asrn.nucleo as asrn
import asrn.estable as estable
import rivales_lineales as rivales


# ----------------------------------------------------------------------------------------------------------------
# Mamba-2 por bloques (CPU)
# ----------------------------------------------------------------------------------------------------------------

def nucleo_ssd_bloques(q, k, v, g, Q=64):
    """h_t = exp(g_t) h_{t-1} + k_t v_t^T, o_t = q_t . h_t, en bloques de Q piezas (el algoritmo SSD de Mamba-2): dentro
    del bloque, una atencion causal con decaimiento exp(G_i - G_j); entre bloques, el estado (H, N, P) pasa de uno al
    siguiente. q, k (B, L, H, N), v (B, L, H, P), g (B, L, H) <= 0. En 32 bits. Mismo resultado que nucleo_ssd_lento."""
    B, L, H, N = q.shape
    P = v.shape[-1]
    pad = (-L) % Q
    q, k, v = (F.pad(t.float(), (0, 0, 0, 0, 0, pad)) for t in (q, k, v))
    g = F.pad(g.float(), (0, 0, 0, pad))                       # relleno: decaimiento 1 y k = 0, no aporta
    n = (L + pad) // Q
    q, k = q.view(B, n, Q, H, N), k.view(B, n, Q, H, N)
    v, g = v.view(B, n, Q, H, P), g.view(B, n, Q, H)
    G = g.cumsum(2)                                            # (B, n, Q, H) decaimiento acumulado en el bloque
    causal = torch.ones(Q, Q, dtype=torch.bool, device=q.device).tril()
    dec = (G.unsqueeze(3) - G.unsqueeze(2)).masked_fill(~causal[:, :, None], float('-inf')).exp()   # (B,n,Q,Q,H)
    o = torch.einsum('bnijh,bnjhp->bnihp', torch.einsum('bnihd,bnjhd->bnijh', q, k) * dec, v)
    hasta_fin = (G[:, :, -1:] - G).exp()                       # (B, n, Q, H) de j al final del bloque
    dH = torch.einsum('bnjh,bnjhd,bnjhp->bnhdp', hasta_fin, k, v)   # (B, n, H, N, P)
    A = G[:, :, -1].exp()                                      # (B, n, H)
    Hc, previos = q.new_zeros(B, H, N, P), []
    for c in range(n):
        previos.append(Hc)
        Hc = A[:, c, :, None, None] * Hc + dH[:, c]
    Hprev = torch.stack(previos, 1)                            # (B, n, H, N, P) estado al empezar cada bloque
    o = o + torch.einsum('bnihd,bnhdp->bnihp', q * G.exp().unsqueeze(-1), Hprev)
    return o.reshape(B, n * Q, H, P)[:, :L]


def _nucleo_ssd(q, k, v, g):
    if rivales._usar_fla(q):
        return rivales.chunk_simple_gla(q, k, v, g=g.float(), scale=1.0)[0]
    return nucleo_ssd_bloques(q, k, v, g).to(v.dtype)


rivales.nucleo_ssd = _nucleo_ssd                               # Mamba2.forward lo busca por nombre en su modulo


# ----------------------------------------------------------------------------------------------------------------
# Transformer de referencia (el de las corridas de desarrollo de v89)
# ----------------------------------------------------------------------------------------------------------------

def _rope_tablas(L, dh, disp, base=10000.0):
    inv = 1.0 / base ** (torch.arange(0, dh, 2, device=disp, dtype=torch.float32) / dh)
    ang = torch.outer(torch.arange(L, device=disp, dtype=torch.float32), inv)
    return ang.cos()[None, None], ang.sin()[None, None]


def _rope(x, cos, sin):
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).type_as(x)


class Atencion(nn.Module):
    def __init__(self, d, cabezas, rope=True):
        super().__init__()
        self.cabezas, self.rope = cabezas, rope
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proy = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.cabezas, d // self.cabezas).permute(2, 0, 3, 1, 4)
        if self.rope:
            cos, sin = _rope_tablas(L, d // self.cabezas, x.device)
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proy(y.transpose(1, 2).reshape(B, L, d))


class BloqueTransformer(nn.Module):
    def __init__(self, d, cabezas):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.at = Atencion(d, cabezas)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h):
        h = h + self.at(self.ln1(h))
        return h + self.ff(self.ln2(h))


class Transformer(nn.Module):
    def __init__(self, V, d, n_capas, cabezas):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.bloques = nn.ModuleList([BloqueTransformer(d, cabezas) for _ in range(n_capas)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight
        for n, p in self.named_parameters():
            if p.dim() >= 2:
                residual = n.endswith(('proy.weight', 'ff.2.weight'))
                nn.init.normal_(p, std=0.02 / (2 * n_capas) ** 0.5 if residual else 0.02)
            elif n.endswith('bias'):
                nn.init.zeros_(p)

    def forward(self, x):
        h = self.emb(x)
        for b in self.bloques:
            h = b(h)
        return self.head(self.ln_f(h))

    def num_parametros(self):
        return sum(p.numel() for p in self.parameters())


def transformer_parecido(parametros, V, d, cabezas, semilla=0):
    with torch.device('meta'):
        por_capa = sum(p.numel() for p in BloqueTransformer(d, cabezas).parameters())
    n = max(1, round((parametros - V * d - 2 * d) / por_capa))
    torch.manual_seed(semilla)
    return Transformer(V, d, n, cabezas)


# ----------------------------------------------------------------------------------------------------------------
# Capas solas (para medir costo por capa) y modelos completos
# ----------------------------------------------------------------------------------------------------------------

class CapaSola(nn.Module):
    """Una capa de mezcla sola con norma y residual: h -> h + mezcla(LN(h)). tipo: 'transformer' (atencion causal,
    sin RoPE, SDPA), 'mamba2', 'rwkv7'. ASRN se mide con asrn.CapaASRN directamente (h = capa(E, h, piezas))."""

    def __init__(self, tipo, d, cabezas=None):
        super().__init__()
        self.tipo, self.ln = tipo, nn.LayerNorm(d)
        if tipo == 'transformer':
            self.mezcla = Atencion(d, cabezas or max(1, d // 64), rope=False)
        elif tipo == 'mamba2':
            self.mezcla = rivales.Mamba2(d)
        else:
            self.mezcla = rivales.MezclaTiempoRWKV7(d, 0, 1)

    def forward(self, h):
        x = self.ln(h)
        return h + (self.mezcla(x, None)[0] if self.tipo == 'rwkv7' else self.mezcla(x))


class SinPuerta(estable.CapaASRNEstable):
    """Ablacion: la puerta fija en 1 (siempre suma lo copiado): h' = h + W_o c."""

    def _combinar(self, acc, h):
        salida = acc['salida'] / acc['peso_total'].clamp(min=1)
        return h + self.drop(self.o(salida))


class PuertaSinSenales(estable.CapaASRNEstable):
    """Ablacion: la puerta solo ve el estado (sigma(w_g . h + b)), sin las senales de la busqueda ni de lo repetido."""

    def _combinar(self, acc, h):
        salida = acc['salida'] / acc['peso_total'].clamp(min=1)
        with torch.autocast(device_type=h.device.type, enabled=False):
            logit = F.linear(h, self.g.weight.to(h.dtype), self.g.bias.to(h.dtype))
        return h + torch.sigmoid(logit).to(salida.dtype) * self.drop(self.o(salida))


class MezclaLSE(estable.CapaASRNEstable):
    """Propuesta (docs/TECHNICAL_REPORT.md; history/docs/ESTABILIDAD_Y_BANCO.md, 1.1 y 2.2 #7): en vez del promedio
    uniforme de las tablas con candidatos, pesar cada tabla por su normalizador Z_tau = sum_s exp(puntaje): es un
    solo softmax sobre la union de candidatos de todas las tablas (como las rondas de Reformer). Solo PyTorch (para
    medir la calidad en CPU)."""

    def _acumular(self, E, Qc, Kc, Bt, log_alfa, sig):
        B, L, _ = E.shape
        T, C, cf = self.n_tablas, self.chunk, self.confirmar
        n_pad = (-L) % C
        R = (L + n_pad) // C + 1
        BT = B * T
        dv = E.device
        phi, V = self._phi_y_valores(E)
        ids = self._ids(phi.detach()).transpose(1, 2).reshape(BT, L)
        orden = ids.argsort(dim=-1, stable=True)
        nulo = B * L
        fuente = F.pad(orden + torch.arange(0, nulo, L, device=dv).repeat_interleave(T).unsqueeze(1),
                       (C, n_pad), value=nulo)
        real = fuente != nulo
        ids_o = F.pad(torch.gather(ids, 1, orden), (C, n_pad), value=-1)

        def ordenar(x):
            k = x.shape[-1]
            return torch.cat([x.reshape(nulo, k), x.new_zeros(1, k)]).index_select(0, fuente.view(-1)).view(BT, -1, k)

        def preguntas(x):
            return x.reshape(BT * R, C, *x.shape[2:])[1:]

        def llaves(x):
            return x.reshape(BT * R * C, *x.shape[2:]).unfold(0, 2 * C, C).movedim(-1, 1)

        en_orden = torch.empty_like(orden).scatter_(1, orden, torch.arange(L, device=dv).expand(BT, L))
        fila = (en_orden + torch.arange(0, BT * R * C, R * C, device=dv).unsqueeze(1)).reshape(-1)

        def desordenar(x):
            k = x.shape[-1]
            return x.reshape(-1, k).index_select(0, fila).view(B, T, L, k)

        phi_o, V_o = ordenar(phi), ordenar(V)
        Qc_o, Kc_o, Pf_o = ordenar(Qc), ordenar(Kc), ordenar(asrn._anteriores(sig, self.sufijo))
        sc = torch.bmm(preguntas(phi_o), llaves(phi_o).transpose(1, 2)) / self.temperatura
        sc = sc + torch.bmm(preguntas(Qc_o), llaves(Kc_o).transpose(1, 2)) * cf ** -0.5
        sc = sc + self._peso_sufijo(asrn._largo_seguido(preguntas(Pf_o).unsqueeze(2), llaves(Pf_o).unsqueeze(1)))
        tasa = (self.gamma / C).repeat(B).unsqueeze(1)
        B_o = ordenar(Bt.unsqueeze(-1)).squeeze(-1)
        la_o = ordenar(log_alfa.unsqueeze(-1)).squeeze(-1)
        tasa_w = preguntas(tasa.expand(BT, R * C))[:, :1].unsqueeze(-1)
        sc = sc + llaves(la_o).unsqueeze(1) + estable._TiempoRelativo.apply(preguntas(B_o), llaves(B_o), tasa_w)
        i = torch.arange(C, device=dv).unsqueeze(1)
        j = torch.arange(2 * C, device=dv)
        anteriores = (j >= i) & (j < C + i)
        valido = (preguntas(ids_o).unsqueeze(2) == llaves(ids_o).unsqueeze(1)) & llaves(real).unsqueeze(1) & anteriores
        att, hay, sc_max = estable._softmax_valido(sc, valido, True)                    # (W, C, 2C)
        # log Z de cada (pregunta, tabla) y su peso entre las tablas de esa pieza: softmax sobre las tablas.
        # Todo enmascarado ANTES de exp: un puntaje no valido grande daria exp = inf, y su gradiente (0 * inf) NaN
        # aunque despues se descarte (asi fallo MQAR en la A100). Las filas sin llaves validas no aportan.
        entrada = torch.where(hay, sc.masked_fill(~valido, float('-inf')), torch.zeros_like(sc))
        logz = torch.where(hay, torch.logsumexp(entrada, -1, keepdim=True), torch.zeros_like(sc_max))   # (W, C, 1)
        logz_t, hay_t = desordenar(logz), desordenar(hay.float())                     # (B, T, L, 1) cada uno
        m = torch.where(hay_t > 0, logz_t, torch.full_like(logz_t, float('-inf'))).amax(1, keepdim=True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m)).detach()
        e = torch.where(hay_t > 0, (logz_t - m).clamp(max=0).exp(), torch.zeros_like(logz_t))
        peso_t = e / e.sum(1, keepdim=True).clamp(min=1e-30)                          # (B, T, L, 1), suma 1 o 0
        peso_w = torch.zeros(att.shape[0] * C, device=dv, dtype=att.dtype).index_copy(
            0, fila, peso_t.reshape(-1)).view(att.shape[0], C, 1)                       # (W, C, 1)
        out = torch.bmm(att * peso_w, llaves(V_o))
        salida = asrn._DesordenarSumar.apply(out.reshape(-1, out.shape[-1]), fila, fuente.view(-1)[C:], B, T, L)
        senal = desordenar(torch.cat([hay.float(), sc_max.float(), att.detach().amax(-1, keepdim=True).float()], -1))
        return self._senales(salida, senal)

    def _combinar(self, acc, h):
        # la salida ya es un promedio pesado (los pesos suman 1): no se divide por el numero de tablas
        acc = dict(acc, salida=acc['salida'] * acc['peso_total'].clamp(min=1))
        return super()._combinar(acc, h)


# congelados en su valor neutro (todo lo nuevo de v86/v87 arranca sin efecto: congelarlo = quitarlo)
_CONGELAR = {
    'sin_reloj': ('w_reloj', 'b_reloj', 'gamma'),                    # beta = 1, alfa = 1, gamma = 0
    'sin_hawkes_racha': ('u_hawkes', 'd_racha', 'log_kappa', 'log_kappa_racha'),
    'sin_sufijo': ('w_sufijo',),
    'sin_confirmacion': ('cq',),
}

VARIANTES = ('receta', 'estable_A', 'estable_AB', 'sin_puerta', 'puerta_sin_senales', 'sin_reloj',
             'sin_hawkes_racha', 'sin_sufijo', 'sin_confirmacion', 'dos_tablas', 'solo_local', 'mezcla_lse')


def asrn_variante(variante, V, d, n_capas, semilla, largo=256):
    """Un ModeloASRN de la receta (largo_cada=2) con una pieza quitada (ver VARIANTES)."""
    kw = dict(n_tablas=2, tablas_contexto=1) if variante == 'dos_tablas' else {}
    m = asrn.crear_modelo(V, d=d, n_capas=n_capas, semilla=semilla, **kw)
    if variante == 'receta':
        return m
    if variante == 'solo_local':
        m.largos, m.pos_largo = nn.ModuleList(), []
        return m
    prot = estable.Protecciones.textos_largos(largo) if variante == 'estable_AB' else estable.Protecciones()
    estable.convertir(m, prot)
    clase = {'sin_puerta': SinPuerta, 'puerta_sin_senales': PuertaSinSenales, 'mezcla_lse': MezclaLSE}.get(variante)
    for c in m.largos:
        if clase is not None:
            c.__class__ = clase
        for nombre, p in c.named_parameters():
            if nombre.split('.')[0] in _CONGELAR.get(variante, ()):
                p.requires_grad_(False)
    return m


def construir(nombre, V, d, n_capas, semilla, largo=256):
    """'ASRN' (y 'ASRN:<variante>'), 'Transformer', 'Mamba-2', 'RWKV-7': los rivales con el mismo ancho y el numero de
    bloques que deja sus parametros mas cerca de los de ASRN."""
    if nombre.startswith('ASRN'):
        return asrn_variante(nombre.split(':')[1] if ':' in nombre else 'receta', V, d, n_capas, semilla, largo)
    objetivo = asrn.crear_modelo(V, d=d, n_capas=n_capas).num_parametros()
    if nombre == 'Transformer':
        return transformer_parecido(objetivo, V, d, max(1, d // 64), semilla)
    tipo = {'Mamba-2': 'mamba2', 'RWKV-7': 'rwkv7'}[nombre]
    return rivales.rival_parecido(tipo, objetivo, V, d, semilla)


def num_parametros(m):
    return sum(p.numel() for p in m.parameters())
