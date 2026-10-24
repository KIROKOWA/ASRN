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
"""Los rivales en su version original y mas rapida, para la GPU (asrn_colab.ipynb):

  - Transformer++ (la receta de Llama, la referencia de los articulos de Mamba y RWKV): RMSNorm, SwiGLU, RoPE y
    atencion causal con F.scaled_dot_product_attention, que en una A100 con bf16 es FlashAttention-2. Escribe con
    cache de llaves y valores reservada de antemano.
  - Mamba-2 oficial: mamba_ssm (Dao y Gu), MambaLMHeadModel con capas Mamba2 y sus valores por omision (estado 128,
    cabezas de 64, expansion 2): los kernels SSD de Triton de los autores, y con causal-conv1d el camino fusionado.
    Escribe con su propio estado (InferenceParams, selective_state_update).
  - RWKV-7: flash-linear-attention (fla), la implementacion de referencia en Hugging Face, RWKV7Model con sus
    kernels Triton y su salida atada a los vectores de pieza. Escribe con su cache (modo recurrente).

Todos con la misma interfaz: modelo(x) -> logits (B, L, V), num_parametros(), y escritura paso a paso:
    estado, logits = modelo.prefijo(x, capacidad)       # x (B, L0)
    logits = modelo.paso(pieza, estado)                  # pieza (B,)
`disponibles()` dice cuales se pueden usar aqui (y por que no). Sin GPU solo funciona el Transformer++."""
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mamba_ssm():
    """Importa mamba_ssm. Instalado sin compilar (MAMBA_SKIP_CUDA_BUILD=TRUE) le falta selective_scan_cuda, la
    extension de Mamba-1, que Mamba-2 no usa (sus kernels son Triton): se reemplaza por un modulo vacio."""
    try:
        import mamba_ssm
    except ImportError as e:
        if 'selective_scan_cuda' not in str(e):
            raise
        sys.modules['selective_scan_cuda'] = types.ModuleType('selective_scan_cuda')
        for k in [k for k in sys.modules if k.startswith('mamba_ssm')]:
            del sys.modules[k]
        import mamba_ssm
    return mamba_ssm


def conv1d_fusionada():
    """True si esta causal-conv1d compilado: Mamba-2 usa entonces su camino mas rapido (use_mem_eff_path)."""
    try:
        import causal_conv1d_cuda  # noqa: F401
    except ImportError:
        return False
    return True


DESACTIVADOS = set()       # los que fallaron su prueba en esta GPU (asrn_colab.ipynb): se usa la reimplementacion
SIN_ESCRITURA = set()      # los que entrenan bien pero cuya escritura paso a paso fallo: no se mide su escritura


def disponibles():
    """{'Transformer++': 'si', 'Mamba-2': 'si (...)' o 'no: motivo', 'RWKV-7': ...}"""
    return {k: ('no: fallo su prueba en esta GPU' if k in DESACTIVADOS else v) for k, v in _disponibles().items()}


def _disponibles():
    out = {'Transformer++': 'si'}
    if not torch.cuda.is_available():
        sin = 'no: sus kernels Triton necesitan GPU'
        return {**out, 'Mamba-2': sin, 'RWKV-7': sin}
    try:
        m = _mamba_ssm()
        out['Mamba-2'] = (f"si (mamba_ssm {getattr(m, '__version__', '?')}, "
                          + ('causal-conv1d fusionado)' if conv1d_fusionada() else 'conv1d de PyTorch, sin compilar)'))
    except Exception as e:                                  # noqa: BLE001
        out['Mamba-2'] = f'no: {type(e).__name__}: {e}'
    try:
        import fla
        from fla.models.rwkv7.modeling_rwkv7 import RWKV7Model  # noqa: F401
        out['RWKV-7'] = f"si (fla {getattr(fla, '__version__', '?')})"
    except Exception as e:                                  # noqa: BLE001
        out['RWKV-7'] = f'no: {type(e).__name__}: {e}'
    return out


# ----------------------------------------------------------------------------------------------------------------
# Transformer++ (Llama)
# ----------------------------------------------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def _rope(desde, L, dh, disp, base=10000.0):
    inv = 1.0 / base ** (torch.arange(0, dh, 2, device=disp, dtype=torch.float32) / dh)
    ang = torch.outer(torch.arange(desde, desde + L, device=disp, dtype=torch.float32), inv)
    return ang.cos(), ang.sin()                                       # (L, dh/2)


def _rotar(x, cos, sin):
    """RoPE de Llama (mitades): x (B, H, L, dh)."""
    x1, x2 = x[..., : x.shape[-1] // 2].float(), x[..., x.shape[-1] // 2:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).type_as(x)


class AtencionPP(nn.Module):
    def __init__(self, d, cabezas, rope=True):
        super().__init__()
        self.h, self.dh, self.rope = cabezas, d // cabezas, rope
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cache=None):
        """x (B, L, d). cache: {'k', 'v': (B, H, capacidad, dh), 't': piezas ya guardadas} o None."""
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)    # (B, H, L, dh) cada uno
        t0 = 0 if cache is None else cache['t']
        if self.rope:
            cos, sin = _rope(t0, L, self.dh, x.device)
            q, k = _rotar(q, cos, sin), _rotar(k, cos, sin)
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            cache['k'][:, :, t0:t0 + L] = k
            cache['v'][:, :, t0:t0 + L] = v
            kk, vv = cache['k'][:, :, :t0 + L], cache['v'][:, :, :t0 + L]
            y = F.scaled_dot_product_attention(q, kk, vv, is_causal=t0 == 0 and L > 1)
        return self.o(y.transpose(1, 2).reshape(B, L, d))


class BloquePP(nn.Module):
    def __init__(self, d, cabezas, ff):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(d), RMSNorm(d)
        self.at = AtencionPP(d, cabezas)
        self.w13 = nn.Linear(d, 2 * ff, bias=False)                    # SwiGLU: puerta y valor en un producto
        self.w2 = nn.Linear(ff, d, bias=False)

    def forward(self, h, cache=None):
        h = h + self.at(self.ln1(h), cache)
        a, b = self.w13(self.ln2(h)).chunk(2, -1)
        return h + self.w2(F.silu(a) * b)


def ancho_swiglu(d):
    """El ancho de la capa densa de Llama: 2/3 de 4d, redondeado a multiplo de 64."""
    return max(64, int(round(8 * d / 3 / 64)) * 64)


class TransformerPP(nn.Module):
    def __init__(self, V, d, n_capas, cabezas=None, ff=None):
        super().__init__()
        cabezas = cabezas or max(1, d // 64)
        self.V, self.cabezas = V, cabezas
        self.emb = nn.Embedding(V, d)
        self.bloques = nn.ModuleList([BloquePP(d, cabezas, ff or ancho_swiglu(d)) for _ in range(n_capas)])
        self.ln_f = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight
        for n, p in self.named_parameters():                          # inicializacion de GPT-2 / Llama
            if p.dim() >= 2:
                residual = n.endswith(('o.weight', 'w2.weight'))
                nn.init.normal_(p, std=0.02 / (2 * n_capas) ** 0.5 if residual else 0.02)

    def forward(self, x, caches=None, solo_ultima=False):
        h = self.emb(x)
        for i, b in enumerate(self.bloques):
            h = b(h, None if caches is None else caches[i])
        if caches is not None:
            for c in caches:
                c['t'] += x.shape[1]
        if solo_ultima:                     # escribir: solo hace falta la ultima (65 536 x V logits serian 4 GB)
            h = h[:, -1:]
        return self.head(self.ln_f(h))

    def num_parametros(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def prefijo(self, x, capacidad):
        B = x.shape[0]
        dh = self.emb.weight.shape[1] // self.cabezas
        p = next(self.parameters())
        tipo = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else p.dtype
        caches = [{'k': torch.empty(B, self.cabezas, capacidad, dh, device=p.device, dtype=tipo),
                   'v': torch.empty(B, self.cabezas, capacidad, dh, device=p.device, dtype=tipo), 't': 0}
                  for _ in self.bloques]
        return caches, self(x, caches, solo_ultima=True)[:, -1]

    @torch.no_grad()
    def paso(self, pieza, caches):
        return self(pieza.view(-1, 1), caches)[:, -1]


# ----------------------------------------------------------------------------------------------------------------
# Mamba-2 oficial (mamba_ssm)
# ----------------------------------------------------------------------------------------------------------------

class Mamba2Oficial(nn.Module):
    def __init__(self, V, d, n_capas, d_state=128, headdim=64):
        super().__init__()
        _mamba_ssm()
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        self.V = V
        cfg = MambaConfig(d_model=d, n_layer=n_capas, d_intermediate=0, vocab_size=V,
                          ssm_cfg=dict(layer='Mamba2', d_state=d_state, headdim=headdim,
                                       use_mem_eff_path=conv1d_fusionada()),
                          rms_norm=True, residual_in_fp32=True, fused_add_norm=True, pad_vocab_size_multiple=8,
                          tie_embeddings=True)
        self.m = MambaLMHeadModel(cfg)

    def forward(self, x):
        return self.m(x).logits[..., :self.V]

    def num_parametros(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def prefijo(self, x, capacidad):
        from mamba_ssm.utils.generation import InferenceParams
        ip = InferenceParams(max_seqlen=capacidad, max_batch_size=x.shape[0])
        lg = self.m(x, inference_params=ip, num_last_tokens=1).logits[:, -1, :self.V]
        ip.seqlen_offset += x.shape[1]
        return ip, lg

    @torch.no_grad()
    def paso(self, pieza, ip):
        lg = self.m(pieza.view(-1, 1), inference_params=ip).logits[:, -1, :self.V]
        ip.seqlen_offset += 1
        return lg


# ----------------------------------------------------------------------------------------------------------------
# RWKV-7 (fla)
# ----------------------------------------------------------------------------------------------------------------

class RWKV7Oficial(nn.Module):
    """RWKV7Model de fla con la salida atada a sus vectores de pieza. (RWKV7ForCausalLM de fla con la salida atada no
    se construye con transformers 5.x; su diferencia con esto es solo la regularizacion l2warp de los logits.)"""

    def __init__(self, V, d, n_capas, head_dim=64):
        super().__init__()
        from fla.models import RWKV7Config
        from fla.models.rwkv7.modeling_rwkv7 import RWKV7Model
        self.m = RWKV7Model(RWKV7Config(hidden_size=d, num_hidden_layers=n_capas, vocab_size=V, head_dim=head_dim,
                                       tie_word_embeddings=False, fuse_cross_entropy=False))
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.m.embeddings.weight

    def forward(self, x):
        return self.head(self.m(input_ids=x).last_hidden_state)

    def num_parametros(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def prefijo(self, x, capacidad):
        from fla.models.utils import Cache
        out = self.m(input_ids=x, past_key_values=Cache(), use_cache=True)
        return out.past_key_values, self.head(out.last_hidden_state[:, -1])

    @torch.no_grad()
    def paso(self, pieza, cache):
        out = self.m(input_ids=pieza.view(-1, 1), past_key_values=cache, use_cache=True)
        return self.head(out.last_hidden_state[:, -1])


# ----------------------------------------------------------------------------------------------------------------
# Construccion con un numero de parametros y capas solas
# ----------------------------------------------------------------------------------------------------------------

CLASES = {'Transformer++': TransformerPP, 'Mamba-2': Mamba2Oficial, 'RWKV-7': RWKV7Oficial}


def _contar(clase, V, d, n):
    try:
        with torch.device('meta'):
            return clase(V, d, n).num_parametros()
    except Exception:                                       # noqa: BLE001  (alguna inicializacion no acepta 'meta')
        return clase(V, d, n).num_parametros()


# fla no arma un RWKV-7 de 1 capa: su inicializacion divide por num_hidden_layers - 1 (fla/layers/rwkv7.py)
MINIMO_CAPAS = {'RWKV-7': 2}


def parecido(nombre, objetivo, V, d, semilla=0, maximo=96):
    """El modelo `nombre` de ancho d con el numero de capas (al menos MINIMO_CAPAS, 1 si no dice) que deja sus
    parametros mas cerca de `objetivo`."""
    clase = CLASES[nombre]
    minimo = MINIMO_CAPAS.get(nombre, 1)
    base, mas_una = _contar(clase, V, d, minimo), _contar(clase, V, d, minimo + 1)
    n = min(maximo, max(minimo, minimo + round((objetivo - base) / (mas_una - base))))
    torch.manual_seed(semilla)
    return clase(V, d, n)


class CapaOficial(nn.Module):
    """Una capa de mezcla sola (norma + mezcla + residual) en su version original, para medir tiempo y memoria por
    capa: 'Transformer++' (atencion con RoPE, SDPA), 'Mamba-2' (mamba_ssm Mamba2) o 'RWKV-7' (fla RWKV7Attention)."""

    def __init__(self, nombre, d):
        super().__init__()
        self.nombre, self.ln = nombre, RMSNorm(d)
        if nombre == 'Transformer++':
            self.mezcla = AtencionPP(d, max(1, d // 64))
        elif nombre == 'Mamba-2':
            _mamba_ssm()
            from mamba_ssm.modules.mamba2 import Mamba2
            self.mezcla = Mamba2(d, d_state=128, headdim=64, use_mem_eff_path=conv1d_fusionada(), layer_idx=0)
        else:
            from fla.layers import RWKV7Attention
            # la primera capa de un modelo de 2 (fla no arma uno de 1: ver MINIMO_CAPAS)
            self.mezcla = RWKV7Attention(hidden_size=d, head_dim=64, layer_idx=0, num_hidden_layers=2)

    def forward(self, h):
        x = self.ln(h)
        if self.nombre == 'RWKV-7':
            return h + self.mezcla(x)[0]
        return h + self.mezcla(x)


if __name__ == '__main__':
    print(disponibles())
    m = TransformerPP(100, 64, 2)
    x = torch.randint(0, 100, (2, 33))
    with torch.no_grad():
        entero = m(x)
        c, lg = m.prefijo(x[:, :20], 40)
        pasos = [lg] + [m.paso(x[:, t], c) for t in range(20, 33)]
    print('Transformer++ escritura = entero:', (torch.stack(pasos, 1) - entero[:, 19:]).abs().max().item())
    if os.environ.get('VERBOSO'):
        print(m)
