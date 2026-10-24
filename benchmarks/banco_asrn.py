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
"""Banco de pruebas de ASRN contra Transformer, Mamba-2 y RWKV-7 (mismo ancho, casi los mismos parametros, mismos
datos y mismo bucle de entrenamiento), mas la ablacion de ASRN. Correr desde la raiz del repositorio:

    python benchmarks/banco_asrn.py [secciones ...] [--salida results/banco_cpu] [--pasos 400] [--semillas 3]

Secciones (por omision, todas):
  costo      por capa (d = 512, lote 1): FLOPs de ida y vuelta (FlopCounterMode, exactos) y bytes que se guardan para
             la vuelta (saved_tensors_hooks, exactos, sin contar pesos) segun el largo. Independiente del hardware.
  latencia   tiempo de ida y vuelta de una capa (d = 256) segun el largo, y su exponente (1 = lineal, 2 = cuadratico)
  escritura  tiempo por pieza nueva de asrn.escribir segun cuanto texto hay detras
  calidad    tres tareas, 4 modelos, la mejor de 2 tasas con la semilla 0 y esa tasa con las demas semillas:
               copia  copiar desde L/2 piezas atras (la perdida cuenta solo la mitad copiable); exactitud con 1, 4 y
                      16 veces el largo de entrenamiento
               mqar   recuerdo asociativo con muchas preguntas (Zoology): L/16 pares llave-valor y despues preguntas;
                      exactitud con 1, 2 y 4 veces el largo (y el numero de pares)
               texto  bytes de la documentacion de Vim (prosa tecnica en ingles); bits por byte con 1 y 4 veces el
                      largo, y en Opticks de Newton (otro texto, sin entrenar)
  ablacion   ASRN quitando una pieza a la vez (banco_modelos.VARIANTES) en copia y mqar, con la tasa de ASRN
  resumir    las tablas en Markdown (RESUMEN.md) desde resultados.json

Sin GPU todo corre en CPU a escala chica (d = 128, ~0.7 M de parametros, 400-600 pasos de 16 x 256 piezas): los numeros
de calidad comparan arquitecturas con el mismo presupuesto chico, no son los de un modelo grande. Con GPU, el mismo
script usa los kernels (Triton de ASRN, flash-linear-attention de los rivales si esta instalada)."""
import argparse
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [p for p in (os.path.dirname(_AQUI), _AQUI) if p not in sys.path]
import asrn.nucleo as asrn
import asrn.estable as estable
import banco_modelos as bm

GPU = torch.cuda.is_available()
DISP = torch.device('cuda' if GPU else 'cpu')
MODELOS = ('ASRN', 'Transformer', 'Mamba-2', 'RWKV-7')


def sincronizar():
    if GPU:
        torch.cuda.synchronize()


def pendiente(xs, ys):
    """Exponente de crecimiento por minimos cuadrados en escala log-log (1 = lineal, 2 = cuadratico)."""
    p = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if y and y > 0]
    if len(p) < 2:
        return float('nan')
    mx, my = statistics.mean(a for a, _ in p), statistics.mean(b for _, b in p)
    return sum((a - mx) * (b - my) for a, b in p) / sum((a - mx) ** 2 for a, _ in p)


def media_desvio(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)


# ----------------------------------------------------------------------------------------------------------------
# 1. Costo por capa: FLOPs y memoria guardada (exactos)
# ----------------------------------------------------------------------------------------------------------------

def bytes_guardados(fn, excluir=()):
    """Bytes de los tensores que autograd guarda para la vuelta durante fn() (cada almacenamiento una vez), sin los de
    `excluir` (los pesos)."""
    fuera = {t.untyped_storage().data_ptr() for t in excluir}
    vistos = {}

    def empacar(t):
        s = t.untyped_storage()
        if s.data_ptr() not in fuera and s.device.type != 'meta':
            vistos[s.data_ptr()] = s.nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(empacar, lambda t: t):
        salida = fn()
    return sum(vistos.values()), salida


def _capa(nombre, d):
    if nombre.startswith('ASRN'):
        prot = estable.Protecciones.ninguna() if nombre == 'ASRN (v89)' else estable.Protecciones()
        return estable.CapaASRNEstable(d, prot).to(DISP)
    return bm.CapaSola({'Transformer': 'transformer', 'Mamba-2': 'mamba2', 'RWKV-7': 'rwkv7'}[nombre], d).to(DISP)


def _ida(capa, E, h, x):
    return capa(E, h, x) if isinstance(capa, asrn.CapaASRN) else capa(h)


def _vuelta_conv(grad_out_shape, x_shape, w_shape, _bias, _stride, _padding, _dilation, transposed, _output_padding,
                 _groups, output_mask, out_shape):
    """FLOPs de la vuelta de una convolucion, tambien por grupos. La formula de FlopCounterMode cuenta el gradiente del
    peso como si la convolucion fuera densa: con la de Mamba-2 (por canales, 1 152 grupos) da ~1 000 veces lo real.
    Cada gradiente (entrada y peso) cuesta lo mismo que la ida."""
    from torch.utils.flop_counter import conv_flop_count
    ida = conv_flop_count(x_shape, w_shape, grad_out_shape, transposed=transposed)
    return ida * (int(bool(output_mask[0])) + int(bool(output_mask[1])))


def seccion_costo(res, d=512, largos=(1024, 4096, 16384)):
    from torch.utils.flop_counter import FlopCounterMode
    out = res.setdefault('costo', {'d': d, 'filas': []})
    print(f"\n== Costo por capa (d = {d}, lote 1): FLOPs de ida+vuelta por pieza y bytes guardados por pieza ==")
    for nombre in ('ASRN (v89)', 'ASRN (estable)', 'Transformer', 'Mamba-2', 'RWKV-7'):
        torch.manual_seed(0)
        capa = _capa(nombre, d)
        for L in largos:
            if nombre == 'RWKV-7' and L > 1024:     # paso a paso en CPU: el costo por pieza ya es constante en 1 024
                continue
            x = torch.randint(0, 50, (1, L), device=DISP)
            x[:, L // 2:L // 2 + L // 4] = x[:, :L // 4]
            E = torch.randn(50, d, device=DISP)[x]
            h = torch.randn(1, L, d, device=DISP, requires_grad=True)
            with FlopCounterMode(display=False,
                                 custom_mapping={torch.ops.aten.convolution_backward: _vuelta_conv}) as fc:
                guardado, y = bytes_guardados(lambda: _ida(capa, E, h, x), excluir=list(capa.parameters()))
                y.square().mean().backward()
            fila = {'modelo': nombre, 'largo': L, 'flops_por_pieza': fc.get_total_flops() / L,
                    'bytes_guardados_por_pieza': guardado / L}
            out['filas'].append(fila)
            print(f"  {nombre:15} L={L:>6}: {fila['flops_por_pieza'] / 1e6:9.2f} MFLOP/pieza   "
                  f"{fila['bytes_guardados_por_pieza'] / 1024:8.1f} KB/pieza guardados")
    # el kernel Triton de ASRN guarda solo sus entradas: se mide con el interprete (CPU) en un largo chico, aparte
    return out


def seccion_costo_kernel(res, d=512, largos=(512, 1024)):
    """Bytes guardados por la capa ASRN con el kernel Triton (en CPU con TRITON_INTERPRET=1: el mismo autograd que en
    la GPU). Correr en un proceso aparte con TRITON_INTERPRET=1."""
    out = res.setdefault('costo_kernel', {'d': d, 'filas': []})
    for nombre in ('ASRN (v89)', 'ASRN (estable)'):
        torch.manual_seed(0)
        capa = _capa(nombre, d)
        assert asrn._usar_triton_casilleros(torch.zeros(1), capa.chunk, capa.n_tablas * capa.n_bits, capa.confirmar)
        for L in largos:
            x = torch.randint(0, 50, (1, L))
            E = torch.randn(50, d)[x]
            h = torch.randn(1, L, d, requires_grad=True)
            guardado, _ = bytes_guardados(lambda: capa(E, h, x), excluir=list(capa.parameters()))
            out['filas'].append({'modelo': nombre + ' + kernel', 'largo': L, 'bytes_guardados_por_pieza': guardado / L})
            print(f"  {nombre + ' + kernel':24} L={L:>6}: {guardado / L / 1024:8.1f} KB/pieza guardados")
    return out


# ----------------------------------------------------------------------------------------------------------------
# 2. Latencia por capa y 3. escritura
# ----------------------------------------------------------------------------------------------------------------

def _medir(fn, rep=3):
    fn()
    ts = []
    for _ in range(rep):
        sincronizar()
        t = time.perf_counter()
        fn()
        sincronizar()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def seccion_latencia(res, d=256, largos=(512, 2048, 8192, 32768), max_rwkv=2048):
    out = res.setdefault('latencia', {'d': d, 'dispositivo': str(DISP), 'filas': [], 'exponente': {}})
    print(f"\n== Latencia de ida+vuelta de una capa (d = {d}, lote 1, {DISP}) ==")
    for nombre in ('ASRN (v89)', 'ASRN (estable)', 'Transformer', 'Mamba-2', 'RWKV-7'):
        torch.manual_seed(0)
        capa = _capa(nombre, d)
        xs, ys = [], []
        for L in largos:
            if nombre == 'RWKV-7' and L > max_rwkv:
                continue
            x = torch.randint(0, 50, (1, L), device=DISP)
            E = torch.randn(50, d, device=DISP)[x]
            h = torch.randn(1, L, d, device=DISP, requires_grad=True)

            def paso():
                _ida(capa, E, h, x).square().mean().backward()
            t = _medir(paso, rep=2 if L >= 8192 else 3)
            xs.append(L)
            ys.append(t)
            out['filas'].append({'modelo': nombre, 'largo': L, 'segundos': t})
            print(f"  {nombre:15} L={L:>6}: {1e3 * t:9.1f} ms   ({1e6 * t / L:7.2f} us/pieza)")
        out['exponente'][nombre] = pendiente(xs[1:], ys[1:])
        print(f"  {nombre:15} exponente (desde {xs[1]}): {out['exponente'][nombre]:.2f}")
    return out


@torch.no_grad()
def seccion_escritura(res, contextos=(512, 2048, 8192, 32768), nuevos=16):
    """Tiempo por pieza de escribir() segun el texto que ya hay: ASRN con su memoria (_Escritor)."""
    out = res.setdefault('escritura', {'filas': []})
    print("\n== Escritura: tiempo por pieza nueva segun el contexto (ASRN, d = 128, 4 bloques) ==")
    m = asrn.crear_modelo(256, d=128, n_capas=4, semilla=0).to(DISP).eval()
    xs, ys = [], []
    for L0 in contextos:
        x = torch.randint(0, 256, (1, L0), device=DISP)
        esc = asrn._Escritor(m)
        esc.inicio(x, capacidad=L0 + nuevos + 1)
        pieza = torch.zeros(1, dtype=torch.long, device=DISP)
        esc.paso(pieza)
        sincronizar()
        t = time.perf_counter()
        for _ in range(nuevos):
            esc.paso(pieza)
        sincronizar()
        t = (time.perf_counter() - t) / nuevos
        xs.append(L0)
        ys.append(t)
        out['filas'].append({'contexto': L0, 'segundos_por_pieza': t})
        print(f"  contexto {L0:>6}: {1e3 * t:8.2f} ms por pieza")
    out['exponente'] = pendiente(xs, ys)
    print(f"  exponente: {out['exponente']:.2f}  (0 = constante como Mamba-2/RWKV-7; 1 = lineal como la atencion "
          "con cache de llaves)")
    return out


# ----------------------------------------------------------------------------------------------------------------
# 4. Calidad: tareas, entrenamiento y evaluacion
# ----------------------------------------------------------------------------------------------------------------

def tarea_copia(n, L, V, semilla):
    """n textos de L piezas: la primera mitad al azar (1..V-1) y la segunda, la misma otra vez (como en las corridas de v89)."""
    g = torch.Generator().manual_seed(semilla)
    mitad = torch.randint(1, V, (n, L // 2), generator=g)
    x = torch.cat([mitad, mitad], 1)
    y = x.clone()
    return x, y


def tarea_mqar(n, L, V, semilla, div_pares=16):
    """Recuerdo asociativo (MQAR, Zoology), largo L con p = L/div_pares pares: [k1 v1 ... kp vp] y despues (L - 2p)/2
    preguntas [k v] con llaves al azar de las p. Llaves en [0, V/2), valores en [V/2, V). Solo cuentan los valores de
    las preguntas (el resto del objetivo es -100)."""
    g = torch.Generator().manual_seed(semilla)
    p = L // div_pares
    nq = (L - 2 * p) // 2
    assert p <= V // 2, (L, V)
    llaves = torch.argsort(torch.rand(n, V // 2, generator=g), -1)[:, :p]
    valores = torch.randint(V // 2, V, (n, p), generator=g)
    cual = torch.randint(0, p, (n, nq), generator=g)
    kq, vq = torch.gather(llaves, 1, cual), torch.gather(valores, 1, cual)
    x = torch.cat([torch.stack([llaves, valores], -1).flatten(1), torch.stack([kq, vq], -1).flatten(1)], 1)
    y = torch.full_like(x, -100)
    y[:, 2 * p + 1::2] = x[:, 2 * p + 1::2]
    return x, y


_TEXTO = {}


def corpus_texto():
    """Bytes de la documentacion de Vim (entrenamiento: 95%, validacion: 5%) y de Opticks (prueba fuera de dominio)."""
    if not _TEXTO:
        carpeta = '/usr/share/vim/vim91/doc'
        archivos = sorted(f for f in os.listdir(carpeta) if f.endswith('.txt'))
        datos = b''.join(open(os.path.join(carpeta, f), 'rb').read() for f in archivos)
        t = torch.frombuffer(bytearray(datos), dtype=torch.uint8).long()
        corte = int(0.95 * t.numel())
        _TEXTO['entrenar'], _TEXTO['validar'] = t[:corte], t[corte:]
        opticks = [os.path.join(r, f) for r, _, fs in os.walk('/usr/local') for f in fs
                   if f == 'Isaac.Newton-Opticks.txt']
        _TEXTO['opticks'] = (torch.frombuffer(bytearray(open(opticks[0], 'rb').read()), dtype=torch.uint8).long()
                             if opticks else None)
    return _TEXTO


def lote_texto(n, L, semilla):
    t = corpus_texto()['entrenar']
    g = torch.Generator().manual_seed(semilla)
    ini = torch.randint(0, t.numel() - L - 1, (n,), generator=g)
    w = t[ini.unsqueeze(1) + torch.arange(L + 1)]
    return w[:, :-1], w[:, 1:]


_WIKI = {}


def corpus_wikitext(piezas_entrenar):
    """WikiText-103 con el tokenizador de GPT-2 (como en las corridas de desarrollo de v89): las primeras `piezas_entrenar` piezas
    de entrenamiento y la validacion entera. Necesita internet y los paquetes datasets y tiktoken."""
    if _WIKI.get('n', 0) < piezas_entrenar:
        import datasets
        import numpy as np
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
        ds = datasets.load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1')

        def tokenizar(parte, tope=None):
            partes, total = [], 0
            for bloque in parte.iter(batch_size=20000):
                for ids in enc.encode_ordinary_batch(bloque['text'], num_threads=8):
                    partes.append(np.asarray(ids, dtype=np.int32))
                    total += len(ids)
                if tope and total >= tope:
                    break
            todo = np.concatenate(partes)
            return torch.from_numpy(todo[:tope] if tope else todo).long()
        _WIKI.update(entrenar=tokenizar(ds['train'], piezas_entrenar), validar=tokenizar(ds['validation']),
                     n=piezas_entrenar)
    return _WIKI


# Tareas de cada perfil. 'cpu': lo que corre en esta maquina sin GPU (d = 128). 'a100': asrn_colab.ipynb, las escalas
# de las corridas en A100 del proyecto (copia de 1 024, MQAR con 64 pares como el mas dificil de Zoology, WikiText-103
# con modelos de ~45 M), con los rivales en su version original (modelos_oficiales.py).
PERFILES = {
    'cpu': dict(d=128, n_capas=4, semillas=3, tasas=(1e-3, 3e-3), modelos=MODELOS, base_ablacion='receta',
                tareas={'copia': dict(V=128, largo=256, evaluar=(256, 1024, 4096), pasos=400, lote=16),
                        'mqar': dict(V=128, largo=256, evaluar=(256, 512, 1024), pasos=600, lote=16),
                        'texto': dict(V=256, largo=256, evaluar=(256, 1024), pasos=400, lote=16)}),
    'a100': dict(d=256, n_capas=4, semillas=3, tasas=(1e-3, 3e-3), base_ablacion='estable_A',
                 modelos=('ASRN', 'Transformer++', 'Mamba-2', 'RWKV-7'),
                 tareas={'copia': dict(V=512, largo=1024, evaluar=(1024, 4096, 16384), pasos=2000, lote=32),
                         'mqar': dict(V=8192, largo=512, evaluar=(512, 1024, 2048), pasos=2000, lote=64,
                                      div_pares=8),
                         'wikitext': dict(V=50257, largo=1024, evaluar=(1024, 4096), pasos=1220, lote=16, d=512,
                                          parametros='45m')}),
}
PERFIL = dict(PERFILES['cpu'], nombre='cpu')
TAREAS = dict(PERFIL['tareas'])
# en la ablacion no va 'estable_A' en CPU: calcula lo mismo que la receta salvo el redondeo (tests/test_estable.py)
ABLACION = tuple(v for v in bm.VARIANTES if v != 'estable_A')


def usar_perfil(nombre, **cambios):
    """Cambia el perfil ('cpu' o 'a100'). cambios: lo que se quiera ajustar (semillas=2, o tareas con otros pasos)."""
    global ABLACION
    PERFIL.clear()
    PERFIL.update(PERFILES[nombre], nombre=nombre, **{k: v for k, v in cambios.items() if k != 'tareas'})
    TAREAS.clear()
    TAREAS.update({k: dict(v) for k, v in PERFILES[nombre]['tareas'].items()})
    for t, c in cambios.get('tareas', {}).items():
        TAREAS.setdefault(t, {}).update(c)
    base = PERFIL['base_ablacion']
    ABLACION = (base,) + tuple(v for v in bm.VARIANTES if v not in (base, 'receta', 'estable_A'))
    return PERFIL


def forma_de(tarea):
    """(d, n_capas) del modelo ASRN de esa tarea en el perfil."""
    t = TAREAS[tarea]
    d = t.get('d', PERFIL['d'])
    if 'parametros' in t:
        return d, asrn.ConfigASRN.para_parametros(t['parametros'], V=t['V'], d=d).n_capas
    return d, t.get('n_capas', PERFIL['n_capas'])


def construir_modelo(nombre, tarea, semilla):
    """El modelo `nombre` para la tarea: ASRN (y sus variantes 'ASRN:...') o un rival con casi los mismos parametros.
    En el perfil 'a100', 'ASRN' es la capa estable (grupo A) y los rivales son los originales (modelos_oficiales),
    con la reimplementacion de rivales_lineales si el original no esta disponible. Devuelve (modelo, implementacion)."""
    t = TAREAS[tarea]
    d, n = forma_de(tarea)
    if nombre == 'ASRN' and PERFIL['nombre'] != 'cpu':
        nombre = 'ASRN:estable_A'
    if nombre.startswith('ASRN'):
        return bm.construir(nombre, t['V'], d, n, semilla, t['largo']), 'asrn'
    objetivo = asrn.crear_modelo(t['V'], d=d, n_capas=n).num_parametros()
    if nombre in ('Transformer++', 'Mamba-2', 'RWKV-7') and PERFIL['nombre'] != 'cpu':
        import modelos_oficiales as mo
        if mo.disponibles().get(nombre, 'no').startswith('si'):
            return mo.parecido(nombre, objetivo, t['V'], d, semilla), 'original'
    reimpl = {'Transformer++': 'Transformer'}.get(nombre, nombre)
    return bm.construir(reimpl, t['V'], d, n, semilla, t['largo']), 'reimplementacion'


def lote_de(tarea, n, L, semilla):
    V = TAREAS[tarea]['V']
    if tarea == 'wikitext':
        t = corpus_wikitext(TAREAS[tarea]['pasos'] * TAREAS[tarea]['lote'] * TAREAS[tarea]['largo'])['entrenar']
        g = torch.Generator().manual_seed(semilla)
        ini = torch.randint(0, t.numel() - L - 1, (n,), generator=g)
        w = t[ini.unsqueeze(1) + torch.arange(L + 1)]
        return w[:, :-1], w[:, 1:]
    if tarea == 'copia':                               # como en las corridas de v89: textos de L, entrada de L - 1
        x, y = tarea_copia(n, L, V, semilla)
        y = y[:, 1:].clone()
        y[:, :L // 2] = -100                           # la primera mitad es azar: solo cuenta la copia (como Zoology)
        return x[:, :-1], y
    if tarea == 'mqar':
        x, y = tarea_mqar(n, L + 1 if (L + 1) % 2 == 0 else L + 2, V, semilla, TAREAS[tarea].get('div_pares', 16))
        return x[:, :L], y[:, 1:L + 1]
    return lote_texto(n, L, semilla)


@torch.no_grad()
def evaluar(modelo, tarea, L, n=16, lote=4):
    """copia/mqar: exactitud (en la segunda mitad / en los valores de las preguntas). texto: bits por byte. wikitext:
    perplejidad en la validacion (ventanas seguidas sin solaparse, hasta ~245 mil piezas)."""
    modelo.eval()
    if tarea == 'wikitext':
        t = _WIKI['validar']
        n_vent = min((t.numel() - 1) // L, 245_760 // L)
        total, cuenta = 0.0, 0
        for i in range(0, n_vent, lote):
            idx = torch.arange(i, min(n_vent, i + lote)).unsqueeze(1) * L + torch.arange(L + 1)
            w = t[idx].to(DISP)
            with torch.autocast(DISP.type, dtype=torch.bfloat16, enabled=GPU):
                lg = modelo(w[:, :-1])
            total += F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), w[:, 1:].reshape(-1),
                                     reduction='sum').item()
            cuenta += w[:, 1:].numel()
        return {'perplejidad': math.exp(total / cuenta)}
    if tarea == 'texto':
        res = {}
        for parte, maximo in (('validar', 192 if L <= 256 else 48), ('opticks', 192 if L <= 256 else 48)):
            t = corpus_texto()[parte]
            if t is None:
                continue
            n_vent = min(maximo, (t.numel() - 1) // L)
            total, cuenta = 0.0, 0
            for i in range(0, n_vent, lote):
                idx = torch.arange(i, min(n_vent, i + lote)).unsqueeze(1) * L + torch.arange(L + 1)
                w = t[idx].to(DISP)
                lg = modelo(w[:, :-1])
                total += F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), w[:, 1:].reshape(-1),
                                         reduction='sum').item()
                cuenta += w[:, 1:].numel()
            res[parte] = total / cuenta / math.log(2)
        return res
    aciertos = total = 0
    for i in range(0, n, lote):
        x, y = lote_de(tarea, min(lote, n - i), L, semilla=999_000 + i)
        x, y = x.to(DISP), y.to(DISP)
        with torch.autocast(DISP.type, dtype=torch.bfloat16, enabled=GPU):
            lg = modelo(x)
        if tarea == 'copia':
            lg, y = lg[:, L // 2:], y[:, L // 2:]
        cuenta = y != -100
        aciertos += ((lg.argmax(-1) == y) & cuenta).sum().item()
        total += cuenta.sum().item()
    return {'exactitud': aciertos / max(total, 1)}


def entrenar(modelo, tarea, semilla, lr, pasos, lote=None):
    """El mismo bucle para todos: AdamW (decaimiento 0.1 en matrices, betas 0.9/0.95), calentamiento 2% y coseno hasta
    lr/10, recorte del gradiente en 1, fp32 (bf16 con autocast en GPU). Los mismos lotes para todos los modelos."""
    L = TAREAS[tarea]['largo']
    lote = lote or TAREAS[tarea].get('lote', 16)
    opt = asrn.crear_optimizador(modelo, lr, 0.1, (0.9, 0.95))
    cal = max(1, pasos // 50)
    modelo.train()
    t0, curva = time.perf_counter(), []
    for paso in range(pasos):
        for g in opt.param_groups:
            g['lr'] = asrn.tasa_aprendizaje(paso, lr, pasos, cal, lr / 10)
        x, y = lote_de(tarea, lote, L, semilla=10_000 + 1_000_003 * semilla + paso)
        x, y = x.to(DISP), y.to(DISP)
        with torch.autocast(DISP.type, dtype=torch.bfloat16, enabled=GPU):
            lg = modelo(x)
        perdida = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1), ignore_index=-100)
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (paso + 1) % max(1, pasos // 10) == 0:
            curva.append((paso + 1, perdida.item()))
    sincronizar()
    return curva, time.perf_counter() - t0


def correr(nombre, tarea, semilla, lr, pasos=None, d=None, n_capas=None):
    """Entrena y evalua un modelo. (d y n_capas: los del perfil; se aceptan por compatibilidad.)"""
    pasos = pasos or TAREAS[tarea]['pasos']
    torch.manual_seed(semilla)
    m, impl = construir_modelo(nombre, tarea, semilla)
    m = m.to(DISP)
    curva, seg = entrenar(m, tarea, semilla, lr, pasos)
    ev = {}
    for L in TAREAS[tarea]['evaluar']:
        largo_entr = TAREAS[tarea]['largo']
        r = evaluar(m, tarea, L, n=16 if L <= 4 * largo_entr else 8, lote=4 if L <= 4 * largo_entr else 1)
        ev[str(L)] = r
    fila = {'modelo': nombre, 'tarea': tarea, 'semilla': semilla, 'lr': lr, 'pasos': pasos,
            'parametros': bm.num_parametros(m), 'segundos': seg, 'curva': curva, 'evaluacion': ev,
            'implementacion': impl, 'perfil': PERFIL['nombre']}
    del m
    if GPU:
        torch.cuda.empty_cache()
    medida = ' | '.join(f"{L}: " + ', '.join(f"{k} {v:.3f}" for k, v in r.items()) for L, r in ev.items())
    print(f"  {nombre:26} {tarea:6} s{semilla} lr {lr:g}: {medida}  ({seg:.0f} s)", flush=True)
    return fila


def _clave(fila):
    """Que tan buena es una corrida (para elegir la tasa): exactitud en el largo de entrenamiento, o -bits por byte."""
    r = fila['evaluacion'][str(TAREAS[fila['tarea']]['largo'])]
    if 'perplejidad' in r:
        return -r['perplejidad']
    return r['exactitud'] if 'exactitud' in r else -r['validar']


def seccion_calidad(res, guardar, tareas=('copia', 'mqar', 'texto'), modelos=None, semillas=None, pasos=None,
                    d=None, n_capas=None, tasas=None):
    modelos, semillas = modelos or PERFIL['modelos'], semillas or PERFIL['semillas']
    tasas, d, n_capas = tasas or PERFIL['tasas'], d or PERFIL['d'], n_capas or PERFIL['n_capas']
    out = res.setdefault('calidad', {'d': d, 'n_capas': n_capas, 'pasos': pasos, 'tareas': TAREAS, 'corridas': [],
                                     'mejor_lr': {}})
    out['tareas'] = {**out.get('tareas', {}), **TAREAS}
    hechas = {(c['modelo'], c['tarea'], c['semilla'], c['lr']) for c in out['corridas']}

    def una(nombre, tarea, s, lr):
        if (nombre, tarea, s, lr) not in hechas:
            out['corridas'].append(correr(nombre, tarea, s, lr, pasos, d, n_capas))
            hechas.add((nombre, tarea, s, lr))
            guardar()
        return next(c for c in out['corridas']
                    if (c['modelo'], c['tarea'], c['semilla'], c['lr']) == (nombre, tarea, s, lr))

    for tarea in tareas:
        print(f"\n== Calidad: {tarea} (d = {forma_de(tarea)[0]}, {pasos or TAREAS[tarea]['pasos']} pasos de "
              f"{TAREAS[tarea].get('lote', 16)} x {TAREAS[tarea]['largo']}) ==", flush=True)
        for nombre in modelos:
            # tasas extra de un modelo en una tarea (PERFIL['tasas_extra']['tarea/modelo']), p. ej. si su mejor tasa
            # quedo en el borde de la grilla; se elige entre todas las probadas
            extra = tuple(PERFIL.get('tasas_extra', {}).get(f'{tarea}/{nombre}', ()))
            previas = {c['lr'] for c in out['corridas']
                       if (c['modelo'], c['tarea'], c['semilla']) == (nombre, tarea, 0)}
            probadas = sorted(set(tasas) | set(extra) | previas)
            mejor = max((una(nombre, tarea, 0, lr) for lr in probadas), key=_clave)
            out['mejor_lr'][f'{tarea}/{nombre}'] = mejor['lr']
            for s in range(1, semillas):
                una(nombre, tarea, s, mejor['lr'])
    return out


def seccion_ablacion(res, guardar, tareas=('copia', 'mqar'), semillas=None, pasos=None, d=None, n_capas=None):
    semillas = semillas or PERFIL['semillas']
    d, n_capas = d or PERFIL['d'], n_capas or PERFIL['n_capas']
    out = res.setdefault('ablacion', {'corridas': []})
    mejor_lr = res.get('calidad', {}).get('mejor_lr', {})
    hechas = {(c['modelo'], c['tarea'], c['semilla']) for c in out['corridas']}
    for tarea in tareas:
        lr = mejor_lr.get(f'{tarea}/ASRN', 3e-3)
        print(f"\n== Ablacion: {tarea} (tasa de ASRN: {lr:g}) ==", flush=True)
        for variante in ABLACION:
            nombre = f'ASRN:{variante}'
            for s in range(semillas):
                if (nombre, tarea, s) in hechas:
                    continue
                previa = next((c for c in res.get('calidad', {}).get('corridas', [])
                               if variante == PERFIL['base_ablacion'] and c['modelo'] == 'ASRN' and c['tarea'] == tarea
                               and c['semilla'] == s and c['lr'] == lr), None)
                fila = dict(previa, modelo=nombre) if previa else correr(nombre, tarea, s, lr, pasos, d, n_capas)
                out['corridas'].append(fila)
                hechas.add((nombre, tarea, s))
                guardar()
    return out


# ----------------------------------------------------------------------------------------------------------------
# Resumen en tablas (Markdown) desde resultados.json
# ----------------------------------------------------------------------------------------------------------------

def _fmt(media, desvio, pct=False, dec=3):
    if media is None:
        return '—'
    if pct:
        return f'{100 * media:.1f}% ± {100 * desvio:.1f}'
    return f'{media:.{dec}f} ± {desvio:.{dec}f}'


def _tabla(encabezados, filas):
    lineas = ['| ' + ' | '.join(encabezados) + ' |', '|' + '---|' * len(encabezados)]
    return '\n'.join(lineas + ['| ' + ' | '.join(str(c) for c in f) + ' |' for f in filas])


def _grupo(corridas, modelo, tarea, lr=None):
    return [c for c in corridas if c['modelo'] == modelo and c['tarea'] == tarea and (lr is None or c['lr'] == lr)]


def _medida(c, L):
    r = c['evaluacion'][str(L)]
    return r.get('exactitud', r.get('validar', r.get('perplejidad')))


def resumir(res):
    partes = []
    if 'costo' in res:
        filas = {}
        for f in res['costo']['filas']:
            filas.setdefault(f['modelo'], {})[f['largo']] = f
        largos = sorted({f['largo'] for f in res['costo']['filas']})
        d = res['costo']['d']
        enc = ['Capa (d = %d)' % d] + [f'MFLOP/pieza L={L}' for L in largos] + ['KB guardados/pieza']
        tab = []
        for m, fl in filas.items():
            extra = (lambda L: 6 * L * d) if m == 'Transformer' else (lambda L: 0)     # atencion causal no contada
            mflops = [f"{(fl[L]['flops_por_pieza'] + extra(L)) / 1e6:.2f}" if L in fl else '—' for L in largos]
            tab.append([m] + mflops + [f"{max(x['bytes_guardados_por_pieza'] for x in fl.values()) / 1024:.1f}"])
        for f in res.get('costo_kernel', {}).get('filas', []):
            if f['largo'] == max(x['largo'] for x in res['costo_kernel']['filas']):
                tab.append([f['modelo']] + ['(igual)'] * len(largos) + [f"{f['bytes_guardados_por_pieza'] / 1024:.1f}"])
        partes.append('### Costo por capa\n\n' + _tabla(enc, tab))
    if 'latencia' in res:
        filas = {}
        for f in res['latencia']['filas']:
            filas.setdefault(f['modelo'], {})[f['largo']] = f['segundos']
        largos = sorted({f['largo'] for f in res['latencia']['filas']})
        tab = [[m] + [f'{1e3 * fl[L]:.1f}' if L in fl else '—' for L in largos]
               + [f"{res['latencia']['exponente'].get(m, float('nan')):.2f}"] for m, fl in filas.items()]
        enc = ['Capa (d = %d)' % res['latencia']['d']] + [f'L={L}' for L in largos] + ['exponente']
        partes.append('### Latencia de ida y vuelta de una capa (ms, CPU)\n\n' + _tabla(enc, tab))
    if 'escritura' in res:
        e = res['escritura']
        partes.append('### Escritura (ms por pieza nueva)\n\n' + _tabla(
            ['Contexto'] + [str(f['contexto']) for f in e['filas']] + ['exponente'],
            [['ASRN'] + [f"{1e3 * f['segundos_por_pieza']:.2f}" for f in e['filas']] + [f"{e['exponente']:.2f}"]]))
    if 'calidad' in res:
        cal = res['calidad']
        presentes = list(dict.fromkeys(c['modelo'] for c in cal['corridas']))
        for tarea, info in cal['tareas'].items():
            filas = []
            for m in presentes:
                lr = cal['mejor_lr'].get(f'{tarea}/{m}')
                g = _grupo(cal['corridas'], m, tarea, lr) if lr is not None else []
                if not g:                                   # todavia sin elegir la tasa (corrida en curso)
                    continue
                celdas = []
                for L in info['evaluar']:
                    v = [_medida(c, L) for c in g]
                    celdas.append(_fmt(*media_desvio(v), pct=tarea in ('copia', 'mqar'),
                                       dec=2 if tarea == 'wikitext' else 3))
                if tarea == 'texto':
                    celdas.append(_fmt(*media_desvio([c['evaluacion'][str(info['largo'])].get('opticks')
                                                      for c in g])))
                n_tokens = info['largo'] * info.get('lote', 16) * g[0]['pasos']
                celdas += [f"{statistics.mean(c['segundos'] for c in g):.0f}",
                           f"{n_tokens / statistics.mean(c['segundos'] for c in g):,.0f}"]
                filas.append([m, f'{lr:g}', len(g), f"{g[0]['parametros']:,}"] + celdas)
            if not filas:
                continue
            nombre = {'copia': 'exactitud', 'mqar': 'exactitud', 'texto': 'bits por byte (validación)',
                      'wikitext': 'perplejidad'}[tarea]
            enc = (['Modelo', 'lr', 'semillas', 'parámetros'] + [f'{nombre} L={L}' for L in info['evaluar']]
                   + (['Opticks (bits/byte)'] if tarea == 'texto' else []) + ['s/corrida', 'piezas/s'])
            partes.append(f'### Calidad: {tarea}\n\n' + _tabla(enc, filas))
    if 'ablacion' in res:
        ab = res['ablacion']['corridas']
        for tarea in ('copia', 'mqar'):
            info = res.get('calidad', {}).get('tareas', {}).get(tarea, TAREAS.get(tarea))
            if info is None:
                continue
            filas = []
            for v in dict.fromkeys(c['modelo'].split(':', 1)[1] for c in ab):
                g = _grupo(ab, f'ASRN:{v}', tarea)
                if g:
                    filas.append([v, len(g)] + [_fmt(*media_desvio([_medida(c, L) for c in g]), pct=True)
                                                for L in info['evaluar']])
            if filas:
                partes.append(f'### Ablación: {tarea} (exactitud)\n\n' + _tabla(
                    ['Variante', 'semillas'] + [f'L={L}' for L in info['evaluar']], filas))
    return '\n\n'.join(partes)


# ----------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('secciones', nargs='*', default=['costo', 'latencia', 'escritura', 'calidad', 'ablacion'])
    ap.add_argument('--salida', default=os.path.join(os.path.dirname(_AQUI), 'results', 'banco_cpu'))
    ap.add_argument('--pasos', type=int, default=None, help='por omision, los de cada tarea (TAREAS)')
    ap.add_argument('--semillas', type=int, default=None, help='por omision, las del perfil')
    ap.add_argument('--tareas', default=None, help='por omision: copia,mqar,texto (cpu) o copia,mqar,wikitext (a100)')
    ap.add_argument('--perfil', default='cpu', choices=sorted(PERFILES))
    a = ap.parse_args()
    usar_perfil(a.perfil)
    a.tareas = a.tareas or ','.join(TAREAS)
    os.makedirs(a.salida, exist_ok=True)
    archivo = os.path.join(a.salida, 'resultados.json')
    res = json.load(open(archivo)) if os.path.exists(archivo) else {}
    res['entorno'] = {'torch': torch.__version__, 'dispositivo': str(DISP), 'hilos': torch.get_num_threads(),
                      'gpu': torch.cuda.get_device_name(0) if GPU else None}

    def guardar():
        tmp = archivo + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(res, f, indent=1)
        os.replace(tmp, archivo)

    tareas = tuple(a.tareas.split(','))
    for s in a.secciones:
        t = time.time()
        if s == 'costo':
            res.pop('costo', None)
            seccion_costo(res)
        elif s == 'costo_kernel':
            res.pop('costo_kernel', None)
            seccion_costo_kernel(res)
        elif s == 'latencia':
            res.pop('latencia', None)
            seccion_latencia(res)
        elif s == 'escritura':
            res.pop('escritura', None)
            seccion_escritura(res)
        elif s == 'calidad':
            seccion_calidad(res, guardar, tareas=tareas, semillas=a.semillas, pasos=a.pasos)
        elif s == 'ablacion':
            seccion_ablacion(res, guardar, tareas=tuple(t for t in tareas if t in ('copia', 'mqar')),
                             semillas=a.semillas, pasos=a.pasos)
        elif s == 'resumir':
            texto = resumir(res)
            print(texto)
            with open(os.path.join(a.salida, 'RESUMEN.md'), 'w') as f:
                f.write(texto + '\n')
            continue
        else:
            raise SystemExit(f'seccion desconocida: {s}')
        guardar()
        print(f"[{s}: {time.time() - t:.0f} s]", flush=True)


if __name__ == '__main__':
    main()
