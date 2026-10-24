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
"""Mamba-2 y RWKV-7: dos modelos de lenguaje que, como ASRN, cuestan lo mismo por pieza sin importar el largo del texto.

Sirven de rivales en las pruebas (benchmarks/asrn_colab.ipynb cuando falta el original, las pruebas del nucleo y
history/experiments/medir_velocidad_v89.py), con el mismo esqueleto que el
Transformer de referencia: vocabulario atado a la salida, pre-norma, residual. Lo caro de cada uno (la recurrencia
lineal) corre con los kernels Triton de flash-linear-attention (`pip install flash-linear-attention`) cuando hay GPU;
todo lo demas es PyTorch normal. Sin GPU (o sin la libreria) la recurrencia se hace paso a paso en PyTorch: mismas
cuentas, muy lento, solo para revisar que todo corre. `comparar_nucleos()` compara las dos versiones en la GPU.

Mamba-2 (Dao y Gu, 2024): bloque = RMSNorm -> Mamba2 -> residual, sin capa densa aparte (como en el articulo; por eso
lleva unas dos veces mas bloques que un Transformer del mismo tamano). Mamba2: proyeccion a (z, x, B, C, dt),
convolucion causal corta, nucleo SSD con decaimiento escalar por cabeza (h_t = exp(A dt_t) h_{t-1} + dt_t B_t x_t^T,
y_t = C_t h_t + D x_t), norma con puerta silu(z) y proyeccion de salida. Escrito siguiendo mamba_ssm (Mamba2Simple)
con ngroups = 1; el nucleo es chunk_simple_gla de flash-linear-attention, que calcula exactamente esa recurrencia.

RWKV-7 "Goose" (Peng y otros, 2025): bloque = LN -> mezcla en el tiempo -> residual, LN -> mezcla de canales ->
residual. La mezcla en el tiempo sigue RWKV7Attention de flash-linear-attention (y el repositorio oficial
BlinkDL/RWKV-LM, x070): desplazamiento de una pieza, r, k, v, decaimiento por canal w, regla delta con a y kk, puerta
g, 'v first' entre capas, GroupNorm por cabeza y la correccion r.k. El nucleo es chunk_rwkv7.

Son reimplementaciones para comparar en igualdad de condiciones (mismo optimizador, mismos datos, mismo tamano), no
los modelos oficiales entrenados.

Proyectos de origen (este archivo sigue su diseno; los nucleos se importan de flash-linear-attention, no se copian):
  - mamba_ssm (state-spaces/mamba, Tri Dao y Albert Gu), Mamba2Simple: licencia Apache-2.0
  - flash-linear-attention (fla-org/flash-linear-attention, Songlin Yang, Yu Zhang y Zhiyuan Li), RWKV7Attention,
    chunk_simple_gla y chunk_rwkv7: licencia MIT
  - BlinkDL/RWKV-LM (Bo Peng), RWKV-7 x070: licencia Apache-2.0
Los derechos de esos proyectos son de sus autores; sus licencias se aplican a su codigo.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.ops.simple_gla import chunk_simple_gla
    from fla.ops.rwkv7 import chunk_rwkv7
except Exception:                     # sin la libreria (o sin GPU donde importarla): nucleos en PyTorch
    chunk_simple_gla = chunk_rwkv7 = None


def _usar_fla(x):
    return chunk_simple_gla is not None and x.is_cuda


# ----------------------------------------------------------------------------------------------------------------
# Nucleos: la recurrencia lineal (Triton en GPU, paso a paso en PyTorch sin ella)
# ----------------------------------------------------------------------------------------------------------------

def nucleo_ssd_lento(q, k, v, g):
    """h_t = exp(g_t) h_{t-1} + k_t v_t^T, o_t = q_t . h_t. q, k (B, L, H, N), v (B, L, H, P), g (B, L, H). En 32 bits."""
    B, L, H, N = q.shape
    q, k, v, g = q.float(), k.float(), v.float(), g.float()
    h = q.new_zeros(B, H, N, v.shape[-1])
    salida = []
    for t in range(L):
        h = h * g[:, t].exp()[..., None, None] + k[:, t, :, :, None] * v[:, t, :, None, :]
        salida.append(torch.einsum('bhn,bhnp->bhp', q[:, t], h))
    return torch.stack(salida, 1)


def nucleo_ssd(q, k, v, g):
    if _usar_fla(q):
        return chunk_simple_gla(q, k, v, g=g.float(), scale=1.0)[0]
    return nucleo_ssd_lento(q, k, v, g).to(v.dtype)


def nucleo_rwkv7_lento(r, w, k, v, a, b):
    """S_t = S_{t-1} diag(exp(w_t)) + (S_{t-1} a_t) b_t^T + v_t k_t^T, o_t = S_t r_t (S de V x K, por cabeza): la regla
    delta generalizada de RWKV-7. Todo (B, L, H, D). En 32 bits."""
    Bn, L, H, D = r.shape
    r, w, k, v, a, b = (x.float() for x in (r, w, k, v, a, b))
    S = r.new_zeros(Bn, H, v.shape[-1], D)
    salida = []
    for t in range(L):
        Sa = torch.einsum('bhvk,bhk->bhv', S, a[:, t])
        S = S * w[:, t].exp()[:, :, None, :] + Sa[..., None] * b[:, t, :, None, :] + v[:, t, ..., None] * k[:, t, :, None, :]
        salida.append(torch.einsum('bhvk,bhk->bhv', S, r[:, t]))
    return torch.stack(salida, 1)


def nucleo_rwkv7(r, w, k, v, a, b):
    if _usar_fla(r):
        return chunk_rwkv7(r=r, w=w, k=k, v=v, a=a, b=b, scale=1.0, safe_gate=True, chunk_size=64)[0]
    return nucleo_rwkv7_lento(r, w, k, v, a, b).to(v.dtype)


@torch.no_grad()
def comparar_nucleos(dispositivo='cuda', B=2, L=200, H=3, D=32, semilla=0):
    """Diferencia relativa maxima entre los kernels de flash-linear-attention y la version paso a paso (en 32 bits),
    para los dos nucleos. None si no hay kernels."""
    if chunk_simple_gla is None or torch.device(dispositivo).type != 'cuda' or not torch.cuda.is_available():
        return None
    g = torch.Generator(device=dispositivo).manual_seed(semilla)

    def al_azar(*forma):
        return torch.randn(*forma, generator=g, device=dispositivo)
    q, k, v = al_azar(B, L, H, D), al_azar(B, L, H, D) * D ** -0.5, al_azar(B, L, H, D)
    gg = -torch.rand(B, L, H, generator=g, device=dispositivo)
    rel = {}
    rapido, lento = nucleo_ssd(q, k, v, gg).float(), nucleo_ssd_lento(q, k, v, gg)
    rel['mamba2'] = ((rapido - lento).abs().max() / lento.abs().max()).item()
    r, kk = al_azar(B, L, H, D), F.normalize(al_azar(B, L, H, D), dim=-1)
    w = -0.6065306597126334 * torch.rand(B, L, H, D, generator=g, device=dispositivo)
    a = torch.rand(B, L, H, D, generator=g, device=dispositivo)
    rapido, lento = nucleo_rwkv7(r, w, k, v, -kk, kk * a).float(), nucleo_rwkv7_lento(r, w, k, v, -kk, kk * a)
    rel['rwkv7'] = ((rapido - lento).abs().max() / lento.abs().max()).item()
    return rel


# ----------------------------------------------------------------------------------------------------------------
# Mamba-2
# ----------------------------------------------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class Mamba2(nn.Module):
    """La mezcla de Mamba-2 (ngroups = 1). d_estado N, cabezas de P = 64 numeros, expansion 2, convolucion de 4."""

    def __init__(self, d, d_estado=64, expandir=2, ancho_cabeza=64, conv=4, dt_min=1e-3, dt_max=0.1):
        super().__init__()
        self.d_in = expandir * d
        ancho_cabeza = min(ancho_cabeza, self.d_in)
        assert self.d_in % ancho_cabeza == 0, f'Mamba2: {self.d_in} no es multiplo de {ancho_cabeza}'
        self.H, self.P, self.N = self.d_in // ancho_cabeza, ancho_cabeza, d_estado
        canales = self.d_in + 2 * d_estado                        # x, B, C pasan por la convolucion
        self.in_proj = nn.Linear(d, 2 * self.d_in + 2 * d_estado + self.H, bias=False)
        self.conv = nn.Conv1d(canales, canales, conv, groups=canales, padding=conv - 1)
        dt = torch.exp(torch.rand(self.H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))      # softplus^-1(dt)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.H).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.H))
        self.norm = RMSNorm(self.d_in)
        self.out_proj = nn.Linear(self.d_in, d, bias=False)

    def forward(self, u):
        Bn, L, _ = u.shape
        z, xBC, dt = self.in_proj(u).split([self.d_in, self.d_in + 2 * self.N, self.H], dim=-1)
        xBC = F.silu(self.conv(xBC.transpose(1, 2))[..., :L].transpose(1, 2))
        x, Bm, Cm = xBC.split([self.d_in, self.N, self.N], dim=-1)
        dt = F.softplus(dt.float() + self.dt_bias)                          # (B, L, H)
        g = -torch.exp(self.A_log.float()) * dt                              # log del decaimiento, <= 0
        x = x.reshape(Bn, L, self.H, self.P)
        q = Cm.unsqueeze(2).expand(Bn, L, self.H, self.N).contiguous()
        k = Bm.unsqueeze(2).expand(Bn, L, self.H, self.N).contiguous()
        y = nucleo_ssd(q, k, (x * dt.unsqueeze(-1).to(x.dtype)).contiguous(), g)
        y = y + x * self.D.view(1, 1, -1, 1).to(x.dtype)
        y = self.norm(y.reshape(Bn, L, self.d_in) * F.silu(z))
        return self.out_proj(y)


class BloqueMamba2(nn.Module):
    def __init__(self, d, **kw):
        super().__init__()
        self.ln, self.mezcla = RMSNorm(d), Mamba2(d, **kw)

    def forward(self, h, _v_first=None):
        return h + self.mezcla(self.ln(h)), None


# ----------------------------------------------------------------------------------------------------------------
# RWKV-7
# ----------------------------------------------------------------------------------------------------------------

def _desplazar(x):
    """La pieza anterior (ceros en la primera)."""
    return F.pad(x, (0, 0, 1, -1))


class _LoRA(nn.Module):
    def __init__(self, d_in, d_out, rango, activacion=None, sesgo=True):
        super().__init__()
        self.a = nn.Linear(d_in, rango, bias=False)
        self.b = nn.Linear(rango, d_out, bias=sesgo)
        self.act = activacion or (lambda t: t)
        nn.init.zeros_(self.a.weight)
        nn.init.orthogonal_(self.b.weight, gain=0.1)

    def forward(self, x):
        return self.b(self.act(self.a(x)))


def _rango(d, f):
    return max(32, int(round(f * d ** 0.5 / 32) * 32))


class MezclaTiempoRWKV7(nn.Module):
    def __init__(self, d, capa, n_capas, ancho_cabeza=64):
        super().__init__()
        ancho_cabeza = min(ancho_cabeza, d)             # modelos chiquitos (pruebas en CPU): una sola cabeza
        assert d % ancho_cabeza == 0, f'RWKV7: d={d} no es multiplo de {ancho_cabeza}'
        self.H, self.D, self.capa = d // ancho_cabeza, ancho_cabeza, capa
        r01 = capa / max(1, n_capas - 1)
        r10 = 1.0 - capa / n_capas
        ddd = torch.arange(d) / d
        lineal = torch.arange(d) / (d - 1) - 0.5
        zig = ((torch.arange(d) % ancho_cabeza) - (ancho_cabeza - 1) / 2) / ((ancho_cabeza - 1) / 2)
        zig = zig * zig.abs()
        www = -6 + 6 * (torch.arange(d) / (d - 1)) ** (1 + r01 ** 0.3)
        self.x_r, self.x_w = nn.Parameter(1 - ddd ** (0.2 * r10)), nn.Parameter(1 - ddd ** (0.9 * r10))
        self.x_k, self.x_v = nn.Parameter(1 - ddd ** (0.7 * r10)), nn.Parameter(1 - ddd ** (0.7 * r10))
        self.x_a, self.x_g = nn.Parameter(1 - ddd ** (0.9 * r10)), nn.Parameter(1 - ddd ** (0.2 * r10))
        self.k_k = nn.Parameter(0.71 - lineal * 0.1)
        self.k_a = nn.Parameter(torch.full((d,), 1.02))
        self.r_k = nn.Parameter(torch.full((self.H, ancho_cabeza), -0.04))
        self.r_proj, self.k_proj = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.v_proj, self.o_proj = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        nn.init.orthogonal_(self.r_proj.weight)
        nn.init.orthogonal_(self.k_proj.weight, gain=0.1)
        nn.init.orthogonal_(self.v_proj.weight)
        nn.init.zeros_(self.o_proj.weight)
        self.w_lora = _LoRA(d, d, _rango(d, 2.5), torch.tanh)
        self.a_lora = _LoRA(d, d, _rango(d, 2.5))
        self.g_lora = _LoRA(d, d, _rango(d, 5), torch.sigmoid, sesgo=False)
        with torch.no_grad():
            self.w_lora.b.bias.copy_(www + 0.5 + zig * 2.5)
            self.a_lora.b.bias.copy_(-0.19 + zig * 0.3 + lineal * 0.4)
        if capa > 0:
            self.v_lora = _LoRA(d, d, _rango(d, 1.7))
            with torch.no_grad():
                self.v_lora.b.bias.copy_(0.73 - lineal * 0.4)
        self.g_norm = nn.GroupNorm(self.H, d, eps=ancho_cabeza * 1e-5)
        with torch.no_grad():
            self.g_norm.weight.fill_(((capa + 1) / n_capas) ** 0.7)

    def forward(self, x, v_first):
        Bn, L, d = x.shape
        delta = _desplazar(x) - x
        xr, xw, xk, xv, xa, xg = (x + delta * m for m in (self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g))
        r = self.r_proj(xr)
        w = -0.6065306597126334 * torch.sigmoid(self.w_lora(xw).float())        # log del decaimiento por canal
        k, v = self.k_proj(xk), self.v_proj(xv)
        if self.capa == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, torch.sigmoid(self.v_lora(xv)))
        a = torch.sigmoid(self.a_lora(xa))
        g = self.g_lora(xg)
        cab = lambda t: t.reshape(Bn, L, self.H, self.D)            # noqa: E731
        kk = F.normalize(cab(k * self.k_k), dim=-1)
        k = k * (1 + (a - 1) * self.k_a)
        r_, k_, v_, a_ = cab(r), cab(k), cab(v), cab(a)
        o = nucleo_rwkv7(r_, cab(w).to(r_.dtype), k_, v_, (-kk).to(r_.dtype), (kk * a_).to(r_.dtype))
        o = self.g_norm(o.reshape(Bn * L, d)).view(Bn, L, d)
        o = o + ((r_ * k_ * self.r_k).sum(-1, keepdim=True) * v_).reshape(Bn, L, d)
        return self.o_proj(o * g), v_first


class MezclaCanalesRWKV7(nn.Module):
    def __init__(self, d, capa, n_capas):
        super().__init__()
        r10 = 1.0 - capa / n_capas
        self.x_k = nn.Parameter(1 - (torch.arange(d) / d) ** (r10 ** 4))
        self.key, self.value = nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)
        nn.init.orthogonal_(self.key.weight)
        nn.init.zeros_(self.value.weight)

    def forward(self, x):
        k = torch.relu(self.key(x + (_desplazar(x) - x) * self.x_k)).square()
        return self.value(k)


class BloqueRWKV7(nn.Module):
    def __init__(self, d, capa, n_capas, **kw):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.tiempo, self.canales = MezclaTiempoRWKV7(d, capa, n_capas, **kw), MezclaCanalesRWKV7(d, capa, n_capas)

    def forward(self, h, v_first=None):
        o, v_first = self.tiempo(self.ln1(h), v_first)
        h = h + o
        return h + self.canales(self.ln2(h)), v_first


# ----------------------------------------------------------------------------------------------------------------
# El modelo de lenguaje
# ----------------------------------------------------------------------------------------------------------------

class ModeloLineal(nn.Module):
    """Vocabulario atado a la salida, n_capas bloques de `tipo` ('mamba2' o 'rwkv7') y norma final."""

    def __init__(self, tipo, V, d, n_capas):
        super().__init__()
        self.tipo = tipo
        self.emb = nn.Embedding(V, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        if tipo == 'mamba2':
            self.bloques = nn.ModuleList([BloqueMamba2(d) for _ in range(n_capas)])
            self.ln_f = RMSNorm(d)
            for b in self.bloques:                      # salida de cada bloque achicada, como en mamba_ssm
                nn.init.normal_(b.mezcla.in_proj.weight, std=0.02)
                nn.init.normal_(b.mezcla.out_proj.weight, std=0.02 / math.sqrt(n_capas))
        elif tipo == 'rwkv7':
            self.bloques = nn.ModuleList([BloqueRWKV7(d, i, n_capas) for i in range(n_capas)])
            self.ln_f = nn.LayerNorm(d)
            self.ln0 = nn.LayerNorm(d)                  # RWKV normaliza tambien la entrada
        else:
            raise ValueError(f"tipo={tipo!r}: 'mamba2' o 'rwkv7'")
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, x):
        h = self.emb(x)
        if self.tipo == 'rwkv7':
            h = self.ln0(h)
        v_first = None
        for b in self.bloques:
            h, v_first = b(h, v_first)
        return self.head(self.ln_f(h))

    def num_parametros(self):
        return sum(p.numel() for p in self.parameters())


NOMBRES = {'mamba2': 'Mamba-2', 'rwkv7': 'RWKV-7'}


def rival_parecido(tipo, parametros, V, d, semilla=0):
    """El modelo `tipo` de ancho d con el numero de bloques que mas se acerca a `parametros` (al menos 1)."""
    with torch.device('meta'):
        uno = ModeloLineal(tipo, V, d, 1).num_parametros()
        dos = ModeloLineal(tipo, V, d, 2).num_parametros()
    por_bloque = dos - uno
    n = max(1, round((parametros - (uno - por_bloque)) / por_bloque))
    torch.manual_seed(semilla)
    return ModeloLineal(tipo, V, d, n)
