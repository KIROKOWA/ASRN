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
"""Las secciones del banco que solo tienen sentido en una GPU (asrn_colab.ipynb). Cada una agrega su parte a `res`
(el diccionario que se guarda en resultados.json):

  estabilidad_gpu  lo del informe de estabilidad (docs/TECHNICAL_REPORT.md, secciones 3 y 5), ahora en la GPU de
                   verdad: el reloj con cumsum de CUDA en 32 bits, el kernel ESTABLE contra PyTorch (fp32 y bf16),
                   la fuga causal y el reloj corrido con el kernel compilado, y una capa con 1 M de piezas
  capa_gpu         una capa de mezcla sola (d = 512, lote 1): tiempo de ida y vuelta y memoria pico segun el largo,
                   ASRN (v89 y estable) contra la atencion (SDPA, FlashAttention-2), Mamba-2 y RWKV-7 originales
  modelo_gpu       modelos de ~50 M, 32 768 piezas por paso: piezas por segundo y memoria segun el largo, cada uno en su
                   mejor version (ASRN y el Transformer tambien compilados si la version de PyTorch lo permite)
  escritura_gpu    los mismos modelos escribiendo: tiempo del prefijo y milisegundos por pieza nueva segun el contexto
Sin GPU corren en tamano minimo (solo para revisar que el codigo funciona)."""
import contextlib
import math
import os
import statistics
import time

import torch
import torch.nn.functional as F

import asrn.nucleo as asrn
import asrn.estable as estable
import banco_asrn as banco
import banco_modelos as bm
import modelos_oficiales as mo

GPU = torch.cuda.is_available()
DISP = torch.device('cuda' if GPU else 'cpu')
COMPILA = torch.__version__ >= asrn._TORCH_COMPILE_MINIMO


def sincronizar():
    if GPU:
        torch.cuda.synchronize()


def autocast():
    return torch.autocast(DISP.type, dtype=torch.bfloat16, enabled=GPU)


def _medir(fn, rep=3, calentar=1):
    """(segundos medianos, GB pico sobre lo que ya habia) de fn()."""
    for _ in range(calentar):
        fn()
    sincronizar()
    if GPU:
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    ts = []
    for _ in range(rep):
        sincronizar()
        t = time.perf_counter()
        fn()
        sincronizar()
        ts.append(time.perf_counter() - t)
    pico = (torch.cuda.max_memory_allocated() - base) / 1e9 if GPU else float('nan')
    return statistics.median(ts), pico


def _sin_memoria(e):
    return isinstance(e, torch.OutOfMemoryError) or 'out of memory' in str(e).lower()


def _limpiar():
    import gc
    gc.collect()
    if GPU:
        torch.cuda.empty_cache()


@contextlib.contextmanager
def _variable(nombre, valor):
    previo = os.environ.get(nombre)
    os.environ[nombre] = valor
    try:
        yield
    finally:
        if previo is None:
            os.environ.pop(nombre, None)
        else:
            os.environ[nombre] = previo


# ----------------------------------------------------------------------------------------------------------------
# Estabilidad en la GPU
# ----------------------------------------------------------------------------------------------------------------

def _capa_movida(prot, d, semilla=0, **kw):
    torch.manual_seed(semilla)
    capa = estable.CapaASRNEstable(d, prot, **kw)
    g = torch.Generator().manual_seed(semilla + 1)
    with torch.no_grad():
        for n, p in capa.named_parameters():
            p.add_(torch.randn(p.shape, generator=g) * (0.5 if n in asrn._NUEVOS_V87 else 0.05))
        capa.gamma.abs_()
    return capa.to(DISP)


def _grads(capa, x, E0, h0, kernel, bf16):
    with _variable('ASRN_TRITON_CASILLEROS', '1' if kernel else '0'):
        capa.zero_grad()
        E, h = E0.clone().requires_grad_(), h0.clone().requires_grad_()
        with torch.autocast(DISP.type, dtype=torch.bfloat16, enabled=bf16):
            y = capa(E, h, x)
        (y.float() * torch.randn(y.shape, generator=torch.Generator().manual_seed(1)).to(DISP)).sum().backward()
    return y.detach().float(), {'E': E.grad, 'h': h.grad, **{n: p.grad.clone() for n, p in capa.named_parameters()
                                                             if p.grad is not None}}


def seccion_estabilidad_gpu(res):
    out = res.setdefault('estabilidad_gpu', {})
    print('\n== Estabilidad en la GPU ==')
    # 1. el reloj: cumsum de la GPU en 32 bits contra 64
    filas = []
    for log2L in ((20, 22, 24, 25) if GPU else (16,)):
        L = 2 ** log2L
        g = torch.Generator(device=DISP).manual_seed(0)
        beta = F.softplus(torch.randn(L, device=DISP, generator=g))
        B32, B64 = beta.cumsum(0), beta.double().cumsum(0)
        fila = {'largo': L}
        for k in (1, 32, 1024):
            ref = B64[k:] - B64[:-k]
            err = ((B32[k:].double() - B32[:-k].double()) - ref).abs() / ref
            fila[f'err_dif_{k}'] = err[-L // 4:].median().item()
        fila['parado'] = (B32[1:] == B32[:-1]).float().mean().item()
        filas.append(fila)
        print(f"  reloj 32 bits en {DISP}, L=2^{log2L}: error de B_t-B_s vecinas {fila['err_dif_1']:.2e}, a 32 "
              f"{fila['err_dif_32']:.2e}, a 1024 {fila['err_dif_1024']:.2e}; reloj parado {fila['parado']:.2%}")
        del beta, B32, B64
    out['reloj_cumsum'] = filas

    kernel = asrn._usar_triton_casilleros(torch.zeros(1, device=DISP), 16, 32, 16)
    out['kernel_triton'] = kernel
    # 2. kernel ESTABLE = PyTorch (fp32 casi exacto; bf16 no mas lejos de fp32 que PyTorch en bf16)
    if kernel:
        capa = _capa_movida(estable.Protecciones(), 96, chunk=32)
        x = torch.randint(0, 12, (2, 600))
        x[:, 300:450] = x[:, :150]
        E0, h0 = torch.randn(12, 96)[x].to(DISP), torch.randn(2, 600, 96).to(DISP)
        x = x.to(DISP)
        torch.backends.cuda.matmul.allow_tf32 = False
        (y0, g0), (y1, g1) = _grads(capa, x, E0, h0, False, False), _grads(capa, x, E0, h0, True, False)
        torch.backends.cuda.matmul.allow_tf32 = True
        err_y = ((y0 - y1).abs().max() / y0.abs().max()).item()
        err_g = max(((g0[k] - g1[k]).abs().max() / (g0[k].abs().max() + 1e-3)).item() for k in g0)
        _, ref16 = _grads(capa, x, E0, h0, False, True)
        _, tri16 = _grads(capa, x, E0, h0, True, True)
        peor16 = max(((tri16[k] - g0[k]).abs().max() / (g0[k].abs().max() + 1e-3)).item() for k in g0)
        ref = max(((ref16[k] - g0[k]).abs().max() / (g0[k].abs().max() + 1e-3)).item() for k in g0)
        out['kernel_estable'] = {'fp32_salida': err_y, 'fp32_gradientes': err_g, 'bf16_kernel': peor16,
                                 'bf16_pytorch': ref}
        print(f"  kernel ESTABLE contra PyTorch: fp32 salida {err_y:.1e}, gradientes {err_g:.1e}; bf16: kernel "
              f"{peor16:.1e} de fp32 (PyTorch en bf16: {ref:.1e})")
    # 3. la fuga causal y 4. el reloj corrido, con el camino que use la GPU (kernel si hay). Sin TF32: lo que se mide
    # es la aritmetica del reloj, no el redondeo de TF32 en las demas multiplicaciones (en la primera corrida en A100,
    # con TF32 encendido, la estable daba un 3.6e-6 constante con cualquier corrimiento)
    tf32_antes = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    fugas, corr = {}, {}
    for nombre, prot in (('v89', estable.Protecciones.ninguna()), ('estable', estable.Protecciones())):
        torch.manual_seed(0)
        capa = estable.CapaASRNEstable(32, prot, chunk=16, n_bits=3).to(DISP)
        with torch.no_grad():
            capa.gamma.fill_(1e9)
            L, corte = 96, 60
            E, h, x = torch.randn(1, L, 32), torch.randn(1, L, 32), torch.randint(0, 7, (1, L))
            E2, h2, x2 = E.clone(), h.clone(), x.clone()
            E2[:, corte:], h2[:, corte:] = torch.randn(1, L - corte, 32), torch.randn(1, L - corte, 32)
            x2[:, corte:] = torch.randint(0, 7, (1, L - corte))
            a = capa(E.to(DISP), h.to(DISP), x.to(DISP))[:, :corte]
            b = capa(E2.to(DISP), h2.to(DISP), x2.to(DISP))[:, :corte]
        fugas[nombre] = (a - b).abs().max().item()
        capa = _capa_movida(prot, 64, chunk=16)
        with torch.no_grad():
            torch.manual_seed(3)
            x = torch.randint(0, 20, (2, 256))
            x[:, 128:192] = x[:, :64]
            E, h, x = torch.randn(20, 64)[x].to(DISP), torch.randn(2, 256, 64).to(DISP), x.to(DISP)
            sig = asrn.firmas_de_piezas(x)
            Qc, Kc, beta, la = capa._contexto(h)
            Bt = beta.cumsum(1)

            def salida(c):
                acc = capa._acumular(E, Qc, Kc, Bt + c, la, sig)
                acc.update(capa._repeticion(sig, beta, Bt + c, la))
                return capa._combinar(acc, h)
            base = salida(0.0)
            corr[nombre] = {f'{c:g}': ((salida(c) - base).abs().max() / base.abs().max()).item()
                            for c in (1e6, 1.6e7, 1e9)}
            # dos relojes corridos entre si (sin la base, que sigue otro camino aritmetico: con B chico la resta de
            # la parte alta no es exacta, con B grande si)
            a6, a9 = salida(1e6), salida(1e9)
            corr[nombre]['1e6 contra 1e9'] = ((a9 - a6).abs().max() / a6.abs().max()).item()
    torch.backends.cuda.matmul.allow_tf32 = tf32_antes
    out['fuga_gamma_1e9'] = fugas
    out['reloj_corrido'] = corr
    out['reloj_corrido_tf32'] = False
    print(f"  fuga causal con gamma = 1e9 (cambiar el futuro cambia el pasado en): v89 {fugas['v89']:.1e}, "
          f"estable {fugas['estable']:.1e}")
    for nombre, v in corr.items():
        print(f"  reloj corrido {nombre}: " + ', '.join(f'{k}: {e:.1e}' for k, e in v.items()))
    # 6. el reloj y TF32: beta y B_t con TF32 encendido y apagado (con autocast bf16 y sin el). En la estable no deberia
    # cambiar nada (_ProyeccionIEEE); en v89 el reloj sale de la misma multiplicacion que Qc y Kc
    if GPU:
        tf32 = {}
        for nombre, prot in (('v89', estable.Protecciones.ninguna()), ('estable', estable.Protecciones())):
            capa = _capa_movida(prot, 512)
            h = torch.randn(1, 65536, 512, device=DISP) * 2
            for bf16 in (False, True):
                relojes = []
                for permitir in (False, True):
                    torch.backends.cuda.matmul.allow_tf32 = permitir
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                        relojes.append(capa._contexto(h)[2].double().cumsum(1))
                torch.backends.cuda.matmul.allow_tf32 = tf32_antes
                tf32[f"{nombre}{' bf16' if bf16 else ''}"] = (relojes[1] - relojes[0]).abs().max().item()
        out['reloj_tf32'] = tf32
        print('  reloj con TF32 encendido contra apagado (|B_t| max de diferencia en 65 536 piezas): '
              + ', '.join(f'{k}: {v:.2e}' for k, v in tf32.items()))
    # 5. una capa con 1 M de piezas (solo ida, d = 64): tiempo, memoria y cuanto se separan v89 y la estable
    if GPU:
        L = 2 ** 20
        torch.manual_seed(0)
        x = torch.randint(0, 2000, (1, L), device=DISP)
        x[:, L // 2:] = x[:, :L // 2]
        E = torch.randn(2000, 64, device=DISP)[x]
        h = torch.randn(1, L, 64, device=DISP)
        larga = {}
        salidas = {}
        for nombre, prot in (('v89', estable.Protecciones.ninguna()), ('estable', estable.Protecciones())):
            capa = _capa_movida(prot, 64)
            try:
                with torch.no_grad(), autocast():
                    t, mem = _medir(lambda: capa(E, h, x), rep=1)
                    salidas[nombre] = capa(E, h, x).float()[:, -4096:]
                larga[nombre] = {'segundos': t, 'memoria_gb': mem}
            except Exception as e:                          # noqa: BLE001
                if not _sin_memoria(e):
                    raise
                larga[nombre] = {'segundos': None, 'memoria_gb': None}
            _limpiar()
        if len(salidas) == 2:
            larga['diferencia_final'] = ((salidas['v89'] - salidas['estable']).abs().max()
                                         / salidas['estable'].abs().max()).item()
        out['capa_1M'] = larga
        print(f"  una capa con 2^20 piezas (ida, d = 64): {larga}")
    return out


# ----------------------------------------------------------------------------------------------------------------
# Una capa sola
# ----------------------------------------------------------------------------------------------------------------

CAPAS = ('ASRN (v89)', 'ASRN (estable)', 'Transformer++', 'Mamba-2', 'RWKV-7')


def _capa(nombre, d):
    if nombre.startswith('ASRN'):
        prot = estable.Protecciones.ninguna() if nombre == 'ASRN (v89)' else estable.Protecciones()
        return estable.CapaASRNEstable(d, prot).to(DISP), 'asrn'
    if mo.disponibles().get(nombre, 'no').startswith('si'):
        return mo.CapaOficial(nombre, d).to(DISP), 'original'
    tipo = {'Transformer++': 'transformer', 'Mamba-2': 'mamba2', 'RWKV-7': 'rwkv7'}[nombre]
    return bm.CapaSola(tipo, d).to(DISP), 'reimplementacion'


def seccion_capa_gpu(res, d=512, largos=None):
    largos = largos or ((1024, 4096, 16384, 65536, 262144) if GPU else (256, 1024))
    out = res.setdefault('capa_gpu', {'d': d, 'filas': []})
    hechas = {(f['modelo'], f['largo']) for f in out['filas']}
    print(f'\n== Una capa (d = {d}, lote 1, bf16): ida y vuelta ==')
    for nombre in CAPAS:
        capa, impl = _capa(nombre, d)
        for L in largos:
            if (nombre, L) in hechas:
                continue
            fila = {'modelo': nombre, 'largo': L, 'implementacion': impl}
            try:
                x = torch.randint(0, 50, (1, L), device=DISP)
                x[:, L // 2:L // 2 + L // 4] = x[:, :L // 4]
                E = torch.randn(50, d, device=DISP)[x]
                h = torch.randn(1, L, d, device=DISP, requires_grad=True)

                def paso():
                    with autocast():
                        y = capa(E, h, x) if nombre.startswith('ASRN') else capa(h)
                    y.float().square().mean().backward()
                fila['segundos'], fila['memoria_gb'] = _medir(paso, rep=2 if L >= 65536 else 3)
            except Exception as e:                          # noqa: BLE001
                if not _sin_memoria(e):
                    raise
                fila['segundos'] = fila['memoria_gb'] = None
            out['filas'].append(fila)
            s = 'sin memoria' if fila['segundos'] is None else (f"{1e3 * fila['segundos']:8.1f} ms  "
                                                                 f"{fila['memoria_gb']:6.2f} GB")
            print(f'  {nombre:15} L={L:>7}: {s}', flush=True)
            _limpiar()
        capa = None
        _limpiar()
    return out


# ----------------------------------------------------------------------------------------------------------------
# Modelos de ~50 M: entrenar y escribir
# ----------------------------------------------------------------------------------------------------------------

def modelos_50m(V=32768, d=512, objetivo='50m'):
    """{nombre: (constructor, implementacion)}: ASRN (estable) de ~50 M con d = 512 y los rivales del mismo ancho y
    casi los mismos parametros."""
    cfg = asrn.ConfigASRN.para_parametros(objetivo, V=V, d=d)
    n_asrn = asrn.crear_desde_config(cfg, dispositivo='meta').num_parametros()
    hay = mo.disponibles()

    def rival(nombre):
        if hay.get(nombre, 'no').startswith('si'):
            return (lambda: mo.parecido(nombre, n_asrn, V, d)), 'original'
        tipo = {'Transformer++': 'Transformer', 'Mamba-2': 'Mamba-2', 'RWKV-7': 'RWKV-7'}[nombre]
        return (lambda: bm.construir(tipo, V, d, cfg.n_capas, 0)), 'reimplementacion'
    out = {'ASRN': ((lambda: estable.convertir(asrn.crear_desde_config(cfg, semilla=0))), 'asrn')}
    for nombre in ('Transformer++', 'Mamba-2', 'RWKV-7'):
        out[nombre] = rival(nombre)
    return out, cfg


def seccion_modelo_gpu(res, largos=None, piezas=None, compilar=True):
    largos = largos or ((1024, 4096, 16384, 32768) if GPU else (128, 256))
    piezas = piezas or (32768 if GPU else 512)
    constructores, cfg = modelos_50m() if GPU else modelos_50m(V=512, d=64, objetivo='1m')
    out = res.setdefault('modelo_gpu', {'piezas_por_paso': piezas, 'filas': [], 'compila': COMPILA,
                                        'config_asrn': cfg.a_dict()})
    hechas = {(f['modelo'], f['largo']) for f in out['filas']}
    variantes = [(n, False) for n in constructores]
    if compilar and COMPILA:
        variantes += [('ASRN', True), ('Transformer++', True)]
    print(f'\n== Modelos de ~{cfg.contar_parametros() / 1e6:.0f} M: entrenar con {piezas} piezas por paso (bf16) ==')
    for nombre, comp in variantes:
        etiqueta = nombre + (' compilado' if comp else '')
        if all((etiqueta, L) in hechas for L in largos):
            continue
        fabrica, impl = constructores[nombre]
        torch.manual_seed(0)
        m = fabrica().to(DISP)
        n_par = sum(p.numel() for p in m.parameters())
        adelante = m
        t_comp = 0.0
        if comp:
            t0 = time.perf_counter()
            adelante = asrn.compilar_partes(m) if nombre == 'ASRN' else torch.compile(m, dynamic=False)
            t_comp = time.perf_counter() - t0
        opt = asrn.crear_optimizador(m, 1e-3)
        V = cfg.V
        for L in largos:
            if (etiqueta, L) in hechas:
                continue
            B = max(1, piezas // L)
            x = torch.randint(0, V, (B, L + 1), device=DISP)
            fila = {'modelo': etiqueta, 'largo': L, 'lote': B, 'parametros': n_par, 'implementacion': impl}

            def paso():
                with autocast():
                    lg = adelante(x[:, :-1])
                F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1)).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            try:
                t1 = time.perf_counter()
                paso()                                      # primera vez (con compilar: aqui compila)
                sincronizar()
                fila['primer_paso_s'] = time.perf_counter() - t1 + t_comp
                t, mem = _medir(paso, rep=5 if GPU else 2, calentar=1)
                fila.update(segundos_paso=t, piezas_por_s=B * L / t, memoria_gb=mem)
            except Exception as e:                          # noqa: BLE001
                if not _sin_memoria(e):
                    raise
                fila.update(segundos_paso=None, piezas_por_s=None, memoria_gb=None)
            out['filas'].append(fila)
            s = 'sin memoria' if fila['segundos_paso'] is None else (
                f"{fila['piezas_por_s']:>10,.0f} piezas/s  {fila['memoria_gb']:6.2f} GB")
            print(f'  {etiqueta:24} ({n_par / 1e6:5.1f} M) L={L:>6} x {B:>3}: {s}', flush=True)
            _limpiar()
        m = adelante = opt = None
        _limpiar()
    return out


@torch.no_grad()
def seccion_escritura_gpu(res, contextos=None, nuevos=32):
    contextos = contextos or ((1024, 4096, 16384, 65536) if GPU else (64, 256))
    constructores, cfg = modelos_50m() if GPU else modelos_50m(V=512, d=64, objetivo='1m')
    out = res.setdefault('escritura_gpu', {'nuevos': nuevos, 'filas': []})
    hechas = {(f['modelo'], f['contexto']) for f in out['filas']}
    print('\n== Escribir: ms por pieza nueva segun el contexto (lote 1, bf16) ==')
    for nombre, (fabrica, impl) in constructores.items():
        if all((nombre, c) in hechas for c in contextos):
            continue
        torch.manual_seed(0)
        m = fabrica().to(DISP).eval()
        if impl == 'reimplementacion' or nombre in mo.SIN_ESCRITURA:
            print(f'  {nombre}: sin version original que escriba paso a paso, no se mide la escritura')
            continue
        for L0 in contextos:
            if (nombre, L0) in hechas:
                continue
            fila = {'modelo': nombre, 'contexto': L0, 'implementacion': impl}
            try:
                x = torch.randint(0, cfg.V, (1, L0), device=DISP)
                pieza = torch.zeros(1, dtype=torch.long, device=DISP)
                if GPU:
                    torch.cuda.reset_peak_memory_stats()
                    base = torch.cuda.memory_allocated()
                with autocast():
                    sincronizar()
                    t0 = time.perf_counter()
                    if nombre == 'ASRN':
                        esc = asrn._Escritor(m)
                        esc.inicio(x, capacidad=L0 + nuevos + 2)
                        paso = esc.paso
                        pieza = pieza.cpu()          # como en escribir(): la pieza nueva sale del muestreo en la CPU
                    else:
                        estado, _ = m.prefijo(x, L0 + nuevos + 2)
                        paso = (lambda p, e=estado: m.paso(p, e))
                    sincronizar()
                    fila['prefijo_s'] = time.perf_counter() - t0
                    paso(pieza)
                    sincronizar()
                    t0 = time.perf_counter()
                    for _ in range(nuevos):
                        paso(pieza)
                    sincronizar()
                fila['ms_por_pieza'] = 1e3 * (time.perf_counter() - t0) / nuevos
                fila['memoria_gb'] = (torch.cuda.max_memory_allocated() - base) / 1e9 if GPU else None
            except Exception as e:                          # noqa: BLE001
                if not _sin_memoria(e):
                    raise
                fila.update(prefijo_s=None, ms_por_pieza=None, memoria_gb=None)
            out['filas'].append(fila)
            s = 'sin memoria' if fila['ms_por_pieza'] is None else (
                f"prefijo {fila['prefijo_s']:7.3f} s   {fila['ms_por_pieza']:7.2f} ms por pieza")
            print(f'  {nombre:15} contexto {L0:>6}: {s}', flush=True)
            _limpiar()
        m = None
        _limpiar()
    return out


# ----------------------------------------------------------------------------------------------------------------
# Tablas y graficas
# ----------------------------------------------------------------------------------------------------------------

def resumir_gpu(res):
    partes = []
    e = res.get('estabilidad_gpu')
    if e:
        filas = [[f"2^{int(math.log2(f['largo']))}", f"{f['err_dif_1']:.2e}", f"{f['err_dif_32']:.2e}",
                  f"{f['err_dif_1024']:.2e}", f"{f['parado']:.2%}"] for f in e.get('reloj_cumsum', [])]
        partes.append('### Reloj en 32 bits (cumsum de la GPU)\n\n' + banco._tabla(
            ['Largo', 'error B_t − B_{t−1}', 'error a 32', 'error a 1 024', 'reloj parado'], filas))
        lineas = []
        if 'kernel_estable' in e:
            k = e['kernel_estable']
            lineas.append(f"- Kernel ESTABLE contra PyTorch: fp32 salida {k['fp32_salida']:.1e}, gradientes "
                          f"{k['fp32_gradientes']:.1e}; bf16 {k['bf16_kernel']:.1e} de fp32 (PyTorch en bf16: "
                          f"{k['bf16_pytorch']:.1e})")
        if 'fuga_gamma_1e9' in e:
            f = e['fuga_gamma_1e9']
            lineas.append(f"- Fuga causal con γ = 1e9: v89 {f['v89']:.1e}, estable {f['estable']:.1e}")
        for n, v in e.get('reloj_corrido', {}).items():
            lineas.append(f"- Reloj corrido, {n}: " + ', '.join(f'{k}: {x:.1e}' for k, x in v.items())
                          + ('' if e.get('reloj_corrido_tf32') is False else ' (con TF32 encendido)'))
        if 'reloj_tf32' in e:
            lineas.append('- Reloj con TF32 encendido contra apagado (diferencia máxima de B_t, 65 536 piezas): '
                          + ', '.join(f'{k}: {x:.2e}' for k, x in e['reloj_tf32'].items()))
        if 'capa_1M' in e:
            lineas.append(f"- Una capa con 2^20 piezas: {e['capa_1M']}")
        partes.append('\n'.join(lineas))
    c = res.get('capa_gpu')
    if c:
        largos = sorted({f['largo'] for f in c['filas']})
        por = {}
        for f in c['filas']:
            por.setdefault(f['modelo'], {})[f['largo']] = f
        fmt = (lambda f, k, m: '—' if f is None or f.get(k) is None else f'{m * f[k]:.1f}')
        partes.append('### Una capa (ms de ida y vuelta)\n\n' + banco._tabla(
            ['Capa'] + [f'L={L}' for L in largos],
            [[m] + [fmt(v.get(L), 'segundos', 1e3) for L in largos] for m, v in por.items()]))
        partes.append('### Una capa (GB de memoria pico)\n\n' + banco._tabla(
            ['Capa'] + [f'L={L}' for L in largos],
            [[m] + [fmt(v.get(L), 'memoria_gb', 1) for L in largos] for m, v in por.items()]))
    mdl = res.get('modelo_gpu')
    if mdl:
        largos = sorted({f['largo'] for f in mdl['filas']})
        por = {}
        for f in mdl['filas']:
            por.setdefault(f['modelo'], {})[f['largo']] = f
        partes.append('### Entrenar modelos de ~50 M (piezas por segundo; GB)\n\n' + banco._tabla(
            ['Modelo', 'parámetros'] + [f'L={L}' for L in largos],
            [[m, f"{next(iter(v.values()))['parametros'] / 1e6:.1f} M"]
             + ['—' if v.get(L, {}).get('piezas_por_s') is None else
                f"{v[L]['piezas_por_s']:,.0f} ({v[L]['memoria_gb']:.1f} GB)" for L in largos]
             for m, v in por.items()]))
    esc = res.get('escritura_gpu')
    if esc:
        ctx = sorted({f['contexto'] for f in esc['filas']})
        por = {}
        for f in esc['filas']:
            por.setdefault(f['modelo'], {})[f['contexto']] = f
        partes.append('### Escribir (ms por pieza nueva)\n\n' + banco._tabla(
            ['Modelo'] + [f'contexto {L}' for L in ctx],
            [[m] + ['—' if v.get(L, {}).get('ms_por_pieza') is None else f"{v[L]['ms_por_pieza']:.2f}" for L in ctx]
             for m, v in por.items()]))
    return '\n\n'.join(partes)


def graficas(res, carpeta):
    """Graficas PNG (claras) de todo lo que haya en res. Devuelve la lista de archivos."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hechas = []

    def guardar(fig, nombre):
        ruta = os.path.join(carpeta, nombre)
        fig.tight_layout()
        fig.savefig(ruta, dpi=140)
        plt.close(fig)
        hechas.append(ruta)

    def lineas(filas, clave_x, clave_y, titulo, ylabel, nombre, escala=1.0, logy=True):
        por = {}
        for f in filas:
            v = f.get(clave_y)
            if v is not None and math.isfinite(v) and v > 0:
                por.setdefault(f['modelo'], []).append((f[clave_x], escala * v))
        if not por:
            return
        fig, eje = plt.subplots(figsize=(7, 4.3))
        for m, pts in por.items():
            pts.sort()
            eje.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', label=m)
        eje.set_xscale('log', base=2)
        if logy:
            eje.set_yscale('log')
        eje.set_xlabel(clave_x)
        eje.set_ylabel(ylabel)
        eje.set_title(titulo)
        eje.grid(True, alpha=0.3)
        eje.legend(fontsize=8)
        guardar(fig, nombre)

    def seguro(fn, *args, **kw):                         # una grafica que falla no detiene las demas
        try:
            fn(*args, **kw)
        except Exception as e:                          # noqa: BLE001
            print(f'(una grafica de {fn.__name__} no se pudo hacer: {type(e).__name__}: {e})')
            plt.close('all')

    if 'capa_gpu' in res:
        seguro(lineas, res['capa_gpu']['filas'], 'largo', 'segundos', 'Una capa: ida y vuelta', 'ms',
               'capa_tiempo.png', 1e3)
        seguro(lineas, res['capa_gpu']['filas'], 'largo', 'memoria_gb', 'Una capa: memoria pico', 'GB',
               'capa_memoria.png')
    if 'modelo_gpu' in res:
        seguro(lineas, res['modelo_gpu']['filas'], 'largo', 'piezas_por_s', 'Entrenar ~50 M (32 768 piezas por paso)',
               'piezas por segundo', 'modelo_velocidad.png', logy=False)
    if 'escritura_gpu' in res:
        seguro(lineas, res['escritura_gpu']['filas'], 'contexto', 'ms_por_pieza', 'Escribir: costo de cada pieza nueva',
               'ms por pieza', 'escritura.png')
    seguro(_graficas_calidad, res, plt, guardar)
    return hechas


def _graficas_calidad(res, plt, guardar):
    cal = res.get('calidad')
    if cal:
        for tarea, info in cal['tareas'].items():
            filas = []
            for m in dict.fromkeys(c['modelo'] for c in cal['corridas']):
                lr = cal['mejor_lr'].get(f'{tarea}/{m}')
                for c in banco._grupo(cal['corridas'], m, tarea, lr) if lr is not None else []:
                    for L in info['evaluar']:
                        filas.append({'modelo': m, 'largo': L, 'v': banco._medida(c, L)})
            if not filas:
                continue
            medias = {}
            for f in filas:
                medias.setdefault((f['modelo'], f['largo']), []).append(f['v'])
            fig, eje = plt.subplots(figsize=(7, 4.3))
            for m in dict.fromkeys(f['modelo'] for f in filas):
                xs = sorted({L for (mm, L) in medias if mm == m})
                ys = [statistics.mean(medias[(m, L)]) for L in xs]
                lo = [min(medias[(m, L)]) for L in xs]
                hi = [max(medias[(m, L)]) for L in xs]
                pct = tarea in ('copia', 'mqar')
                k = 100 if pct else 1
                eje.plot(xs, [k * y for y in ys], 'o-', label=m)
                eje.fill_between(xs, [k * y for y in lo], [k * y for y in hi], alpha=0.15)
            eje.set_xscale('log', base=2)
            eje.set_xlabel('largo al evaluar (piezas)')
            eje.set_ylabel('exactitud (%)' if tarea in ('copia', 'mqar') else 'perplejidad')
            eje.set_title(f'{tarea}: media y rango entre semillas (entrenado con {info["largo"]})')
            eje.grid(True, alpha=0.3)
            eje.legend(fontsize=8)
            guardar(fig, f'calidad_{tarea}.png')
    ab = res.get('ablacion', {}).get('corridas')
    if ab:
        for tarea in ('copia', 'mqar'):
            g = [c for c in ab if c['tarea'] == tarea]
            if not g:
                continue
            info = cal['tareas'][tarea] if cal else banco.TAREAS[tarea]
            L = info['evaluar'][min(1, len(info['evaluar']) - 1)]   # el primer largo mas alla del de entrenar
            vs = {}
            for c in g:
                vs.setdefault(c['modelo'].split(':', 1)[1], []).append(100 * banco._medida(c, L))
            fig, eje = plt.subplots(figsize=(8, 4.3))
            nombres = list(vs)
            eje.bar(range(len(nombres)), [statistics.mean(vs[n]) for n in nombres],
                    yerr=[statistics.stdev(vs[n]) if len(vs[n]) > 1 else 0 for n in nombres], capsize=3)
            eje.set_xticks(range(len(nombres)), nombres, rotation=35, ha='right', fontsize=8)
            eje.set_ylabel(f'exactitud con {L} piezas (%)')
            eje.set_title(f'Ablación de ASRN: {tarea}')
            eje.grid(True, axis='y', alpha=0.3)
            guardar(fig, f'ablacion_{tarea}.png')
