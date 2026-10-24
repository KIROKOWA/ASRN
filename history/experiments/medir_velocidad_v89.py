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
"""Velocidad de ASRN en GPU, en la misma maquina y la misma sesion. Correr con:
    python history/experiments/medir_velocidad_v89.py [--rapido] [--json velocidad.json]

  1. Mismas cuentas: con los kernels Triton la capa da lo mismo que en PyTorch puro (salvo el redondeo)
  2. Una capa (d = 512, lote 1): tiempo de ida y vuelta segun el largo, en PyTorch, con Triton y grabada
  3. Modelo de ~50 millones (32 768 piezas por paso, bf16): piezas por segundo de ASRN, el Transformer, Mamba-2 y
     RWKV-7, sin y con torch.compile
  4. Donde se va el tiempo: cada parte de la capa por separado, cuantos kernels (programas de GPU) lanza una ida y
     vuelta, si la CPU tiene que esperar a la GPU (sincronizaciones) y los kernels que mas tardan
  5. El paso de entrenamiento entero por partes, con el lote completo (vocabulario y perdida, bloques locales, capas
     ASRN y cada parte de ellas, AdamW) y el del Transformer; con PyTorch >= 2.14, tambien con torch.compile en las
     partes chicas de la capa
Con CUDA Graph (asrn.PasoGrafado): el paso entero grabado, sin que Python lance kernel por kernel. Se mide tambien
para el Transformer de referencia (el mismo del cuaderno de verificacion de v89, pruebas_gpu_a100.ipynb), para comparar en igualdad.
Sin GPU corre en CPU con tamanos chicos (solo para revisar el script; sus numeros no sirven)."""
import contextlib
import json
import os
import statistics
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(os.path.dirname(_AQUI))             # history/experiments -> la raiz del repositorio
sys.path[:0] = [_RAIZ, os.path.join(_RAIZ, 'benchmarks')]    # el paquete asrn y rivales_lineales
import asrn.nucleo as asrn
import rivales_lineales as rivales

RAPIDO = '--rapido' in sys.argv
GPU = torch.cuda.is_available()
disp = torch.device('cuda' if GPU else 'cpu')
resultado = {}


@contextlib.contextmanager
def _variable(nombre, valor):
    previo = os.environ.get(nombre)
    os.environ[nombre] = valor
    try:
        yield
    finally:
        if previo is None:
            del os.environ[nombre]
        else:
            os.environ[nombre] = previo


def sin_triton():
    """La capa sin los kernels Triton (PyTorch puro): la referencia para comparar."""
    return _variable('ASRN_TRITON', '0')


# ----------------------------------------------------------------------------------------------------------------
# Herramientas
# ----------------------------------------------------------------------------------------------------------------

def sincronizar():
    if GPU:
        torch.cuda.synchronize()


def autocast():
    return torch.autocast(disp.type, dtype=torch.bfloat16) if GPU else contextlib.nullcontext()


def medir(paso, repeticiones=5, calentamiento=2):
    """Mediana de los segundos por llamada (esperando a que la GPU termine)."""
    for _ in range(calentamiento):
        paso()
    sincronizar()
    tiempos = []
    for _ in range(repeticiones):
        t = time.perf_counter()
        paso()
        sincronizar()
        tiempos.append(time.perf_counter() - t)
    return statistics.median(tiempos)


def grabar(f, tensores_con_gradiente=(), calentar=3):
    """f() grabada como CUDA Graph: la llama `calentar` veces sin grabar (en otra cola), la graba y devuelve la
    funcion que la repite. Antes de grabar pone en None los gradientes de `tensores_con_gradiente`: dentro del grafo
    la vuelta los asigna en vez de sumarlos."""
    cola = torch.cuda.Stream()
    cola.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cola):
        for _ in range(calentar):
            f()
    torch.cuda.current_stream().wait_stream(cola)
    for t in tensores_con_gradiente:
        t.grad = None
    grafo = torch.cuda.CUDAGraph()
    with torch.cuda.graph(grafo):
        f()
    return grafo.replay


def ocupacion(paso, segundos):
    """Fraccion del paso en que la GPU esta trabajando: tiempo de sus kernels (una llamada, con el perfilador) entre
    el tiempo del paso. Cerca de 1: el limite es la GPU. Bajo: la GPU espera a Python."""
    if not GPU:
        return float('nan')
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        paso()
        sincronizar()
    kernels = sum((getattr(e, 'device_time', None) or getattr(e, 'cuda_time', 0) or 0) for e in prof.events()
                  if e.device_type == torch.autograd.DeviceType.CUDA)
    return kernels / 1e6 / segundos


def _rope_tablas(L, dh, disp_, base=10000.0):
    inv = 1.0 / base ** (torch.arange(0, dh, 2, device=disp_, dtype=torch.float32) / dh)
    ang = torch.outer(torch.arange(L, device=disp_, dtype=torch.float32), inv)
    return ang.cos()[None, None], ang.sin()[None, None]


def _rope(x, cos, sin):
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).type_as(x)


class BloqueTransformer(nn.Module):
    """El mismo bloque que el Transformer de pruebas_gpu_a100.ipynb: pre-norma, atencion causal con RoPE
    (F.scaled_dot_product_attention: FlashAttention en A100) y capa densa de 4d."""

    def __init__(self, d, cabezas):
        super().__init__()
        self.cabezas = cabezas
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proy = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h):
        B, L, d = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(B, L, 3, self.cabezas, d // self.cabezas).permute(2, 0, 3, 1, 4)
        cos, sin = _rope_tablas(L, d // self.cabezas, h.device)
        y = F.scaled_dot_product_attention(_rope(q, cos, sin), _rope(k, cos, sin), v, is_causal=True)
        h = h + self.proy(y.transpose(1, 2).reshape(B, L, d))
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


def transformer_parecido(parametros, V, d, cabezas):
    """El Transformer de ancho d con el numero de capas que mas se acerca a `parametros` (como el del cuaderno)."""
    with torch.device('meta'):
        por_capa = sum(p.numel() for p in BloqueTransformer(d, cabezas).parameters())
    torch.manual_seed(0)
    return Transformer(V, d, max(1, round((parametros - V * d - 2 * d) / por_capa)), cabezas)


def texto_con_repeticiones(B, L, V, semilla=0):
    g = torch.Generator().manual_seed(semilla)
    x = torch.randint(0, V, (B, L), generator=g)
    for _ in range(max(1, L // 128)):
        n = int(torch.randint(8, 64, (1,), generator=g))
        if L > 2 * n:
            i, j = (int(v) for v in torch.randint(0, L - n, (2,), generator=g))
            x[:, j:j + n] = x[:, i:i + n].clone()
    return x


def capa_movida(d, semilla=0, **kw):
    """Una capa con todos sus parametros fuera del valor inicial (como despues de entrenar)."""
    torch.manual_seed(semilla)
    capa = asrn.CapaASRN(d, **kw)
    with torch.no_grad():
        for n, p in capa.named_parameters():
            p.add_(torch.randn_like(p) * (0.5 if n in asrn._NUEVOS_V87 else 0.02))
    return capa.to(disp)


def entradas_capa(d, L, V=32000, semilla=0):
    x = texto_con_repeticiones(1, L, V, semilla).to(disp)
    g = torch.Generator().manual_seed(semilla)
    E = (torch.randn(V, d, generator=g) * 0.02).to(disp)[x]
    h = torch.randn(1, L, d, generator=g).to(disp)
    return x, E, h


def ida_y_vuelta_capa(capa, x, E, h):
    E, h = E.detach().requires_grad_(), h.detach().requires_grad_()
    with autocast():
        y = capa(E, h, x)
    y.float().square().mean().backward()
    for t in [E, h, *capa.parameters()]:
        t.grad = None
    return y.detach()


# ----------------------------------------------------------------------------------------------------------------
# 1. Mismas cuentas
# ----------------------------------------------------------------------------------------------------------------

def mismas_cuentas():
    """Con los kernels Triton la capa da lo mismo que en PyTorch puro, salvo el redondeo (suman en otro orden). En
    bf16, las dos se comparan con la cuenta en 32 bits (la "verdad"): Triton no debe quedar mas lejos que PyTorch."""
    d, L = (512, 4096) if GPU else (64, 300)
    filas, verdad = [], None
    for bf16 in ([False, True] if GPU else [False]):
        x, E, h = entradas_capa(d, L)
        salidas, grads = [], []
        for triton in (False, True):
            capa = capa_movida(d)
            Ei, hi = E.clone().requires_grad_(), h.clone().requires_grad_()
            with (autocast() if bf16 else contextlib.nullcontext()), (contextlib.nullcontext() if triton
                                                                       else sin_triton()):
                y = capa(Ei, hi, x)
            y.float().square().mean().backward()
            salidas.append(y.detach().float())
            grads.append({'E': Ei.grad, 'h': hi.grad, **{n: p.grad for n, p in capa.named_parameters()}})

        def lejos(g, ref, piso=1e-30):
            difs = {n: ((g[n] - ref[n]).abs().max() / (ref[n].abs().max() + piso)).item()
                    for n in ref if ref[n] is not None and g.get(n) is not None}
            n = max(difs, key=difs.get)
            return difs[n], n
        dif_ida = ((salidas[0] - salidas[1]).abs().max() / salidas[0].abs().max()).item()
        dif, n_dif = lejos(grads[1], grads[0])
        fila = {'precision': 'bf16' if bf16 else 'fp32', 'salida_dif_relativa_max': dif_ida,
                'gradiente_dif_relativa_max': dif, 'gradiente_peor_parametro': n_dif}
        print(f"  {fila['precision']}: Triton contra PyTorch: salida {dif_ida:.1e}, gradientes {dif:.1e} ({n_dif})")
        if not bf16:
            verdad = grads[0]
        elif verdad is not None:                # contra fp32, con un piso para gradientes casi nulos
            e_py, n_py = lejos(grads[0], verdad, 1e-3)
            e_tr, n_tr = lejos(grads[1], verdad, 1e-3)
            fila.update({'pytorch_contra_fp32': e_py, 'triton_contra_fp32': e_tr})
            print(f"        contra fp32 (la verdad): PyTorch {e_py:.1e} ({n_py}), Triton {e_tr:.1e} ({n_tr})")
        filas.append(fila)
    resultado['mismas_cuentas'] = filas


# ----------------------------------------------------------------------------------------------------------------
# 2. Una capa
# ----------------------------------------------------------------------------------------------------------------

def una_capa():
    d = 512 if GPU else 64
    largos = ([1024, 4096, 16384] if RAPIDO else [1024, 4096, 16384, 65536, 262144]) if GPU else [256, 1024]
    filas = []
    print(f"  {'largo':>8} {'PyTorch':>10} {'Triton':>10} {'con grafo':>10}   memoria PyTorch / Triton")
    for L in largos:
        x, E, h = entradas_capa(d, L)
        fila = {'largo': L}
        for nombre, modo in (('sin_triton', sin_triton()), ('ahora', contextlib.nullcontext())):
            capa = capa_movida(d)
            if GPU:
                torch.cuda.reset_peak_memory_stats()
            with modo:
                fila[nombre + '_ms'] = 1000 * medir(lambda: ida_y_vuelta_capa(capa, x, E, h))
            fila[nombre + '_gb'] = torch.cuda.max_memory_allocated() / 1e9 if GPU else float('nan')
            del capa
        fila['grafo_ms'] = float('nan')
        if GPU:                         # la misma ida y vuelta con Triton, grabada como CUDA Graph
            capa = capa_movida(d)
            Eg, hg = E.detach().clone().requires_grad_(), h.detach().clone().requires_grad_()

            def f():
                with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                    y = capa(Eg, hg, x)
                y.float().square().mean().backward()
            try:
                fila['grafo_ms'] = 1000 * medir(grabar(f, [Eg, hg, *capa.parameters()]))
            except Exception as e:      # si no se puede grabar, se anota por que y se sigue
                fila['grafo_error'] = f'{type(e).__name__}: {e}'[:400]
                print(f"  (CUDA Graph de la capa fallo: {fila['grafo_error']})")
            del capa, f
            torch.cuda.empty_cache()
        filas.append(fila)
        print(f"  {L:>8} {fila['sin_triton_ms']:8.1f}ms {fila['ahora_ms']:8.1f}ms {fila['grafo_ms']:8.1f}ms   "
              f"{fila['sin_triton_gb']:.2f} / {fila['ahora_gb']:.2f} GB")
    resultado['capa'] = {'d': d, 'lote': 1, 'filas': filas}


# ----------------------------------------------------------------------------------------------------------------
# 3. Modelo de ~50 millones
# ----------------------------------------------------------------------------------------------------------------

def modelo_50m():
    """Piezas por segundo de un paso de entrenamiento entero (ida, vuelta, recorte y AdamW), con 32 768 piezas por paso:
    ASRN (la receta: una capa ASRN cada dos bloques), el Transformer de referencia y Mamba-2 y RWKV-7
    (rivales_lineales.py), todos con el mismo ancho y casi los mismos parametros, sin y con torch.compile (compilar
    necesita PyTorch >= 2.14; con una version anterior esas columnas no aparecen). ASRN se compila por partes
    (compilar_partes); los demas, bloque por bloque.
    'GPU ocupada': la fraccion del paso en que la GPU trabaja (baja = espera a Python)."""
    V = 32768 if GPU else 500
    cfg = asrn.ConfigASRN.para_parametros('50m', V=V) if GPU else asrn.ConfigASRN(V=V, d=32, n_capas=3)
    parametros = asrn.crear_desde_config(cfg).num_parametros()
    piezas = 32768 if GPU else 512
    largos = ([1024, 4096] if RAPIDO else [1024, 4096, 16384, 32768]) if GPU else [128]
    cabezas = 9 if GPU else 4
    variantes = []
    for base in ('asrn', 'transformer', 'mamba2', 'rwkv7'):
        variantes += [base] + ([base + '_compilado'] if PUEDE_COMPILAR else [])
    filas = []
    print(f"  {'largo x lote':>14} " + ' '.join(f'{v:>18}' for v in variantes) + "   (piezas/s; GPU ocupada)")
    for L in largos:
        lote = piezas // L
        x = texto_con_repeticiones(lote, L + 1, V).to(disp)
        entrada, salida = x[:, :-1].contiguous(), x[:, 1:].contiguous()
        fila = {'largo': L, 'lote': lote}
        for nombre in variantes:
            base, compilar = nombre.split('_')[0], nombre.endswith('_compilado')
            m = opt = paso = None
            try:
                if base == 'asrn':
                    m = asrn.crear_desde_config(cfg, semilla=0).to(disp)
                    if compilar:
                        asrn.compilar_partes(m)
                else:
                    m = (transformer_parecido(parametros, V, cfg.d, cabezas) if base == 'transformer'
                         else rivales.rival_parecido(base, parametros, V, cfg.d)).to(disp)
                    if compilar:
                        compilar_transformer(m)         # torch.compile en cada bloque (los kernels Triton quedan)
                m.train()
                opt = asrn.crear_optimizador(m, 3e-4)

                def paso():
                    with autocast():
                        lg = m(entrada)
                    F.cross_entropy(lg.float().reshape(-1, V), salida.reshape(-1)).backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                if compilar:
                    print(f"  (compilando {nombre} para {L} x {lote}: la primera vez tarda unos minutos)", flush=True)
                seg = medir(paso, repeticiones=3, calentamiento=5 if compilar else 2)
                fila[nombre + '_piezas_s'] = piezas / seg
                fila[nombre + '_gpu_ocupada'] = ocupacion(paso, seg)
            except Exception as e:      # un modelo que falla se anota y se sigue con los demas
                fila[nombre + '_piezas_s'] = fila[nombre + '_gpu_ocupada'] = float('nan')
                fila[nombre + '_error'] = f'{type(e).__name__}: {e}'[:400]
                print(f"  ({nombre} fallo: {fila[nombre + '_error']})")
            m = opt = paso = None
            if GPU:
                torch.cuda.empty_cache()
        filas.append(fila)
        print(f"  {f'{L} x {lote}':>14} " + ' '.join(
            f"{fila[v + '_piezas_s']:9,.0f} ({100 * fila[v + '_gpu_ocupada']:3.0f}%)" for v in variantes))
    resultado['modelo'] = {
        'parametros': parametros, 'piezas_por_paso': piezas, 'asrn': {'d': cfg.d, 'n_capas': cfg.n_capas,
                                                                     'largo_cada': cfg.largo_cada},
        'rivales': {t: rivales.rival_parecido(t, parametros, V, cfg.d).num_parametros() for t in ('mamba2', 'rwkv7')},
        'filas': filas}
    if not PUEDE_COMPILAR:
        print('  (sin compilar: usa PyTorch >= 2.14 para la medicion de verdad)')


# ----------------------------------------------------------------------------------------------------------------
# 4. Donde se va el tiempo
# ----------------------------------------------------------------------------------------------------------------

def partes_de_la_capa(capa, x, E, h):
    """Cada parte de la capa con sus propias entradas, para medir su ida y vuelta por separado."""
    sig = asrn.firmas_de_piezas(x)

    def hoja(t):
        return t.detach().requires_grad_()

    def ida_vuelta(f):
        def paso():
            with autocast():
                salida = f()
            vals = salida.values() if isinstance(salida, dict) else (
                salida if isinstance(salida, (tuple, list)) else [salida])
            perdida = sum(v.float().square().mean() for v in vals if v.requires_grad)
            perdida.backward()
        return paso

    with torch.no_grad(), autocast():
        Qc, Kc, beta, log_alfa = capa._contexto(h)
        Bt = beta.cumsum(1)
        acc = capa._acumular(E, Qc, Kc, Bt, log_alfa, sig)
        acc.update(capa._repeticion(sig, beta, Bt, log_alfa))
    Eh, hh = hoja(E), hoja(h)
    Qch, Kch, Bth, lah, betah = hoja(Qc), hoja(Kc), hoja(Bt), hoja(log_alfa), hoja(beta)
    acch = {k: hoja(v) if v.is_floating_point() else v for k, v in acc.items()}
    return {
        'contexto (reloj y confirmacion)': ida_vuelta(lambda: capa._contexto(hh)),
        'acumular (atencion por casilleros)': ida_vuelta(lambda: capa._acumular(Eh, Qch, Kch, Bth, lah, sig)),
        'repeticion (Hawkes y racha)': ida_vuelta(lambda: capa._repeticion(sig, betah, Bth, lah)),
        'combinar (puerta)': ida_vuelta(lambda: capa._combinar(acch, hh)),
        'capa entera': ida_vuelta(lambda: capa(Eh, hh, x)),
    }


def donde_se_va_el_tiempo():
    d = 512 if GPU else 64
    largos = [1024, 16384] if GPU else [256]
    capa = capa_movida(d)
    salida = []
    for L in largos:
        x, E, h = entradas_capa(d, L)
        partes = partes_de_la_capa(capa, x, E, h)
        tiempos = {n: 1000 * medir(p) for n, p in partes.items()}
        # kernels lanzados en una ida y vuelta de la capa entera
        from torch.profiler import ProfilerActivity, profile
        actividades = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if GPU else [])
        with profile(activities=actividades) as prof:
            partes['capa entera']()
            sincronizar()
        if GPU:
            kernels = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
            tabla = prof.key_averages().table(sort_by='cuda_time_total', row_limit=15)
        else:
            kernels = sum(1 for e in prof.events() if e.name.startswith('aten::'))
            tabla = prof.key_averages().table(sort_by='cpu_time_total', row_limit=10)
        # sincronizaciones: la CPU se detiene a esperar a la GPU
        esperas = 0
        if GPU:
            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter('always')
                torch.cuda.set_sync_debug_mode('warn')
                try:
                    partes['capa entera']()
                    sincronizar()
                finally:
                    torch.cuda.set_sync_debug_mode('default')
            esperas = sum('synchroniz' in str(a.message).lower() for a in avisos)
        entera = tiempos['capa entera']
        print(f"\n  largo {L} (d = {d}, lote 1): {kernels} {'kernels' if GPU else 'operaciones'} por ida y vuelta, "
              f"{esperas} sincronizaciones")
        for n, t in tiempos.items():
            print(f"    {n:38} {t:8.2f} ms" + ('' if n == 'capa entera' else f"  ({100 * t / entera:4.0f}%)"))
        print('\n  Los kernels que mas tardan:\n' + '\n'.join('    ' + r for r in tabla.splitlines()))
        salida.append({'largo': L, 'd': d, 'kernels': kernels, 'sincronizaciones': esperas,
                       'ms_por_parte': tiempos, 'kernels_que_mas_tardan': tabla})
    resultado['donde_se_va_el_tiempo'] = salida


# ----------------------------------------------------------------------------------------------------------------
# 5. El paso de entrenamiento entero, por partes (con el lote completo: el caso de entrenar de verdad)
# ----------------------------------------------------------------------------------------------------------------

def compilar_transformer(modelo):
    """Lo mismo que asrn.compilar_partes para el Transformer de referencia: torch.compile en cada bloque, para
    comparar en igualdad (compilar tambien lo acelera a el)."""
    for b in modelo.bloques:
        b.forward = torch.compile(b.forward, dynamic=False)
    return modelo


PUEDE_COMPILAR = torch.__version__ >= asrn._TORCH_COMPILE_MINIMO


def _tiempo_ida_vuelta(f, entradas):
    """Mediana de una ida y vuelta de f(*entradas) (las entradas con gradiente son hojas)."""
    def paso():
        with autocast():
            y = f(*entradas)
        y.float().square().mean().backward()
        for t in entradas:
            if isinstance(t, torch.Tensor) and t.requires_grad:
                t.grad = None
    return medir(paso, repeticiones=3, calentamiento=2)


def paso_por_partes():
    """Donde se va el tiempo de un paso de entrenamiento del modelo de 50 M con 32 768 piezas por paso (lo que mide
    la parte 3), y lo mismo para el Transformer. Cada parte se mide sola, con las entradas del tamano real; la suma
    debe dar cerca del paso entero. Con PyTorch >= 2.14, tambien el modelo ASRN con sus partes chicas compiladas."""
    V = 32768 if GPU else 500
    cfg = asrn.ConfigASRN.para_parametros('50m', V=V) if GPU else asrn.ConfigASRN(V=V, d=32, n_capas=3)
    piezas = 32768 if GPU else 512
    largos = [1024, 4096] if GPU else [128]
    salida = []
    for L in largos:
        lote = piezas // L
        x = texto_con_repeticiones(lote, L, V).to(disp)
        fila = {'largo': L, 'lote': lote, 'ms': {}}
        m = asrn.crear_desde_config(cfg, semilla=0).to(disp).train()
        opt = asrn.crear_optimizador(m, 3e-4)

        def paso():
            with autocast():
                lg = m(x)
            F.cross_entropy(lg.float().reshape(-1, V), x.reshape(-1)).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        ms = fila['ms']
        ms['paso entero'] = 1000 * medir(paso, repeticiones=3, calentamiento=2)

        def vocabulario(xx):
            return m.head(m.ln_f(m.emb(xx)))

        def perdida_vocabulario():
            with autocast():
                lg = vocabulario(x)
            F.cross_entropy(lg.float().reshape(-1, V), x.reshape(-1)).backward()
            m.zero_grad(set_to_none=True)
        ms['vocabulario, salida y perdida'] = 1000 * medir(perdida_vocabulario, repeticiones=3, calentamiento=2)
        h = (m.emb(x).detach() * 1.0).requires_grad_()
        E = m.emb(x).detach().requires_grad_()
        ms[f'{len(m.bloques)} bloques locales'] = len(m.bloques) * 1000 * _tiempo_ida_vuelta(m.bloques[0], [h])
        capa = m.largos[0]
        ms[f'{len(m.largos)} capas ASRN'] = len(m.largos) * 1000 * _tiempo_ida_vuelta(capa, [E, h, x])
        for nombre, t in partes_de_la_capa(capa, x, E.detach(), h.detach()).items():
            if nombre != 'capa entera':
                ms[f'   {nombre} (x{len(m.largos)})'] = len(m.largos) * 1000 * medir(t, repeticiones=3, calentamiento=2)
        params = [p for p in m.parameters()]
        with autocast():
            lg = m(x)
        F.cross_entropy(lg.float().reshape(-1, V), x.reshape(-1)).backward()
        del lg

        def optimizador():
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        ms['recorte y AdamW'] = 1000 * medir(optimizador, repeticiones=3, calentamiento=2)
        opt.zero_grad(set_to_none=True)
        if GPU:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                paso()
                sincronizar()
            fila['kernels_que_mas_tardan'] = prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=20)
        # el Transformer con el mismo lote
        tr = transformer_parecido(m.num_parametros(), V, cfg.d, 9 if GPU else 4).to(disp).train()
        opt_t = asrn.crear_optimizador(tr, 3e-4)

        def paso_t():
            with autocast():
                lg = tr(x)
            F.cross_entropy(lg.float().reshape(-1, V), x.reshape(-1)).backward()
            torch.nn.utils.clip_grad_norm_(tr.parameters(), 1.0)
            opt_t.step()
            opt_t.zero_grad(set_to_none=True)
        ms['Transformer: paso entero'] = 1000 * medir(paso_t, repeticiones=3, calentamiento=2)
        ht = h.detach().requires_grad_()
        ms[f'Transformer: {len(tr.bloques)} bloques (atencion + densa)'] = (
            len(tr.bloques) * 1000 * _tiempo_ida_vuelta(tr.bloques[0], [ht]))
        if PUEDE_COMPILAR:
            try:
                compilar_transformer(tr)
                ms['Transformer compilado: paso entero'] = 1000 * medir(paso_t, repeticiones=3, calentamiento=3)
            except Exception as e:
                fila['compilar_transformer_error'] = f'{type(e).__name__}: {e}'[:400]
        del tr, opt_t
        # ASRN con las partes chicas compiladas
        if PUEDE_COMPILAR:
            try:
                asrn.compilar_partes(m)
                ms['ASRN compilado por partes: paso entero'] = 1000 * medir(paso, repeticiones=3, calentamiento=3)
            except Exception as e:
                fila['compilar_error'] = f'{type(e).__name__}: {e}'[:400]
        else:
            fila['compilar_error'] = (f'PyTorch {torch.__version__}: compilar necesita {asrn._TORCH_COMPILE_MINIMO} o '
                                      'mas nuevo (en el cuaderno: TORCH_NUEVO = True)')
        del m, opt
        if GPU:
            torch.cuda.empty_cache()
        total = ms['paso entero']
        print(f"\n  {L} x {lote} piezas por paso:")
        for nombre, t in ms.items():
            print(f"    {nombre:52} {t:8.1f} ms" + (f"  ({100 * t / total:4.0f}%)" if 'Transformer' not in nombre
                                                    and 'compilado' not in nombre and nombre != 'paso entero' else ''))
        if 'compilar_error' in fila:
            print(f"    (sin partes compiladas: {fila['compilar_error']})")
        if 'kernels_que_mas_tardan' in fila:
            print('\n  Los kernels que mas tardan en el paso entero:\n' +
                  '\n'.join('    ' + r for r in fila['kernels_que_mas_tardan'].splitlines()))
        salida.append(fila)
    resultado['paso_por_partes'] = salida


if __name__ == '__main__':
    entorno = {'torch': torch.__version__,
               'gpu': torch.cuda.get_device_name(0) if GPU else 'SIN GPU (solo revisa el script)'}
    resultado['entorno'] = entorno
    print(f"torch {entorno['torch']} | {entorno['gpu']}")
    for titulo, f in (('1. Mismas cuentas (Triton = PyTorch)', mismas_cuentas),
                      ('2. Una capa: ida y vuelta', una_capa),
                      ('3. Modelo de ~50 millones: ASRN, Transformer, Mamba-2 y RWKV-7', modelo_50m),
                      ('4. Donde se va el tiempo (ahora)', donde_se_va_el_tiempo),
                      ('5. El paso de entrenamiento entero, por partes', paso_por_partes)):
        print(f"\n=== {titulo} ===")
        t = time.time()
        f()
        print(f"  [{time.time() - t:.0f} s]")
    if '--json' in sys.argv:
        with open(sys.argv[sys.argv.index('--json') + 1], 'w') as f:
            json.dump(resultado, f, indent=1, ensure_ascii=False)
