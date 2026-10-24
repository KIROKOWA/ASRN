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
"""Verificacion de ASRN 1.0 (v89) en GPU (por ejemplo, Google Colab). Correr con:
    python tests/verificar_gpu.py [--rapido] [--json resultados.json]

Lo que las pruebas en CPU (tests/test_nucleo.py y tests/test_estable.py) no pueden comprobar:
  1. bf16 en GPU: entrena una talla real ('30m') unos pasos sin NaN y la perdida baja
  2. checkpoint 'no' / 'asrn' / 'todo': misma perdida y memoria pico de GPU medida de verdad (largo 2048)
  3. FSDP con NCCL (el backend de GPU) en 1 proceso: repartido = sin repartir, entrena, guarda y carga
     (con 1 GPU; repartir entre VARIAS GPUs necesita una maquina con varias y todavia no se ha probado)
  4. torch.compile en GPU: mismas cuentas que sin compilar (se salta con PyTorch < 2.14, donde el compilador falla)
  5. GPU = CPU: el mismo modelo da la misma salida y el mismo gradiente en los dos (32 bits)
  6. causalidad en GPU: cambiar las piezas futuras no cambia ninguna salida anterior
  7. escribir paso a paso en GPU da lo mismo que procesar el texto entero
Al final imprime un resumen PASA / SALTADA / FALLA (y lo guarda en --json, si se pide). Sin GPU corre en CPU con
tamanos chicos (solo para revisar el script)."""
import contextlib
import copy
import json
import os
import sys
import tempfile
import time
import traceback

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # la raiz del repositorio
import asrn.nucleo as asrn

RAPIDO = '--rapido' in sys.argv
GPU = torch.cuda.is_available()
disp = torch.device('cuda' if GPU else 'cpu')
resultados = []                 # (nombre, 'PASA' / 'SALTADA' / 'FALLA', detalle)


class Saltada(Exception):
    """La prueba no aplica en este entorno (no es una falla de ASRN); el mensaje dice por que."""


def prueba(nombre):
    def deco(f):
        def correr():
            t = time.time()
            try:
                detalle = f()
                resultados.append((nombre, 'PASA', detalle))
                print(f"PASA   {nombre} ({time.time() - t:.0f}s) {detalle or ''}")
            except Saltada as e:
                resultados.append((nombre, 'SALTADA', str(e)))
                print(f"SALTADA {nombre}: {e}")
            except Exception as e:
                resultados.append((nombre, 'FALLA', f"{type(e).__name__}: {e}"))
                print(f"FALLA  {nombre}: {type(e).__name__}: {e}")
                traceback.print_exc()
        return correr
    return deco


def _texto_rep(B, L, V, semilla=0):
    """Piezas al azar con un trozo repetido: casilleros llenos, sufijos exactos y 'vistos'."""
    g = torch.Generator().manual_seed(semilla)
    x = torch.randint(0, V, (B, L), generator=g)
    x[:, L // 2:L // 2 + L // 4] = x[:, :L // 4]
    return x


def _activar(modelo, semilla=0):
    """Saca de cero los parametros que arrancan sin efecto (confirmacion, sufijo, vistos, reloj, Hawkes, racha), para
    que todas las partes de la capa cuenten en la comparacion."""
    g = torch.Generator().manual_seed(semilla)
    with torch.no_grad():
        for c in modelo.largos:
            for n, p in c.named_parameters():
                if n in ('cq.weight', 'w_sufijo', 'w_vistos', 'g_conf.weight') or n in asrn._NUEVOS_V87:
                    p.add_(torch.randn(p.shape, generator=g).to(p.device) * 0.3)
    return modelo


@contextlib.contextmanager
def _sin_tf32():
    """Cuentas de 32 bits completas: la GPU (A100) usa TF32 por defecto en las convoluciones, que redondea mas."""
    antes = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = antes


def texto(n, V, semilla=0):
    """Piezas al azar con trozos que se repiten (se aprende copiando)."""
    g = torch.Generator().manual_seed(semilla)
    trozos = torch.randint(0, V, (200, 32), generator=g)
    return trozos[torch.randint(0, 200, (n // 32,), generator=g)].reshape(-1).to(torch.int32)


@prueba("1. bf16 en GPU, talla 30m")
def p1():
    cfg = asrn.TALLAS['30m'] if GPU else asrn.ConfigASRN(V=500, d=64, n_capas=4)
    if not GPU:
        cfg = cfg.con(V=500)
    m = asrn.crear_desde_config(cfg, semilla=0).to(disp)
    datos = texto(400_000, cfg.V)
    h = asrn.entrenar(m, datos, largo=512 if GPU else 128, lote=8, pasos=40 if RAPIDO else 150, lr=1e-3,
                      registrar_cada=10, registro=lambda *a: None)
    p = [f['perdida'] for f in h if 'perdida' in f]
    assert all(map(lambda v: v == v, p)), "hubo NaN"
    assert p[-1] < p[0], p
    return f"{m.num_parametros():,} parametros | perdida {p[0]:.3f} -> {p[-1]:.3f} | {h[-1]['piezas_s']:,.0f} piezas/s"


@prueba("2. checkpoint por niveles: misma perdida, menos memoria")
def p2():
    cfg = asrn.ConfigASRN(V=8000, d=512, n_capas=8) if GPU else asrn.ConfigASRN(V=200, d=32, n_capas=4)
    largo = 2048 if GPU else 256
    m = asrn.crear_desde_config(cfg, semilla=0).to(disp).train()
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, cfg.V, (4, largo), generator=g).to(disp)
    y = torch.randint(0, cfg.V, (4, largo), generator=g).to(disp)
    precision = 'bf16' if GPU and torch.cuda.is_bf16_supported() else ('fp16' if GPU else 'fp32')
    out = {}
    for modo in ('no', 'asrn', 'todo'):
        m.checkpoint = modo
        m.zero_grad(set_to_none=True)
        if GPU:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base_mem = torch.cuda.memory_allocated()
        with asrn._autocast(precision, disp):
            logits = m(x)
        perdida = F.cross_entropy(logits.float().reshape(-1, cfg.V), y.reshape(-1))
        perdida.backward()
        pico = (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 20 if GPU else float('nan')
        out[modo] = (perdida.item(), pico)
        del logits, perdida
    for modo in ('asrn', 'todo'):
        assert abs(out[modo][0] - out['no'][0]) < 1e-3, out
    if GPU:
        assert out['todo'][1] < out['asrn'][1] < out['no'][1], out
    return " | ".join(f"{k}: {v[1]:,.0f} MB" for k, v in out.items()) + f" (largo {largo}, lote 4, {precision})"


def _fsdp_un_proceso():
    import torch.distributed as dist
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29533')
    dist.init_process_group('nccl' if GPU else 'gloo', rank=0, world_size=1)
    try:
        cfg = asrn.ConfigASRN(V=2000, d=256, n_capas=6, checkpoint='todo')
        ref = asrn.crear_desde_config(cfg, semilla=4).to(disp).eval()
        m = asrn.repartir_fsdp(asrn.crear_desde_config(cfg, dispositivo='meta'))
        asrn.materializar(m, disp, 4)
        x = torch.randint(0, cfg.V, (2, 256), device=disp)
        with torch.no_grad():
            dif = (m.eval()(x) - ref(x)).abs().max().item()
        assert dif < 1e-4, dif
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, 'm.pt')
            asrn.entrenar(m, texto(200_000, cfg.V), largo=256, lote=8, pasos=30, lr=2e-3, guardar_en=f,
                          precision='bf16' if GPU else 'fp32', registrar_cada=10, registro=lambda *a: None)
            uno = asrn.cargar(f, dispositivo=disp)
            with torch.no_grad():
                dif2 = (uno(x) - m.eval()(x)).abs().max().item()
            assert dif2 < 1e-3, dif2
        return f"repartido = sin repartir (dif {dif:.1e}); entrena, guarda y carga (dif {dif2:.1e})"
    finally:
        dist.destroy_process_group()


@prueba("3. FSDP con NCCL en 1 proceso")
def p3():
    return _fsdp_un_proceso()


@prueba("4. torch.compile en GPU: mismas cuentas")
def p4():
    if torch.__version__ < asrn._TORCH_COMPILE_MINIMO:
        raise Saltada(f"PyTorch {torch.__version__}: el compilador de PyTorch (Inductor/Triton) falla con este modelo "
                      f"antes de {asrn._TORCH_COMPILE_MINIMO} (en CPU, y en una A100 con 2.11 y Triton). Es un error "
                      "del compilador, no de ASRN: sin compilar las cuentas son las mismas (pruebas 5 a 7)")
    cfg = asrn.ConfigASRN(V=1000, d=128, n_capas=4)
    m = asrn.crear_desde_config(cfg, semilla=0).to(disp).train()
    x = torch.randint(0, cfg.V, (2, 512), device=disp)
    m.zero_grad(set_to_none=True)
    lg = m(x)
    lg.square().mean().backward()
    g_e = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    lg_c = torch.compile(m)(x)
    lg_c.square().mean().backward()
    dif = (lg_c.detach() - lg.detach()).abs().max().item()
    assert dif < 1e-3, dif
    peor = max((p.grad - g_e[n]).abs().max().item() for n, p in m.named_parameters() if n in g_e)
    assert peor < 1e-3, peor
    return f"logits dif {dif:.1e}, gradientes dif {peor:.1e}"


@prueba("5. GPU = CPU: misma salida y mismo gradiente (32 bits)")
def p5():
    cfg = asrn.ConfigASRN(V=300, d=64, n_capas=4, chunk=16, n_bits=6)
    m_cpu = _activar(asrn.crear_desde_config(cfg, semilla=1))
    m_gpu = copy.deepcopy(m_cpu).to(disp)
    x = _texto_rep(2, 400, cfg.V)
    with _sin_tf32():
        lc, lg = m_cpu(x), m_gpu(x.to(disp))
        lc.square().mean().backward()
        lg.square().mean().backward()
    # un casillero es el signo de un numero: si uno cae justo en 0, CPU y GPU pueden redondearlo distinto y esa
    # posicion copia de otro lado. Por eso se pide que coincida el 99% de las posiciones, no todas
    dif = (lg.detach().cpu() - lc.detach()).abs().amax(-1)
    iguales = (dif < 1e-3).float().mean().item()
    gc = torch.cat([p.grad.flatten() for p in m_cpu.parameters() if p.grad is not None])
    gg = torch.cat([p.grad.flatten().cpu() for p in m_gpu.parameters() if p.grad is not None])
    rel = ((gg - gc).norm() / gc.norm()).item()
    assert iguales >= 0.99, f"solo {100 * iguales:.1f}% de las posiciones coinciden"
    assert rel < 1e-3, f"gradiente distinto: {rel:.1e}"
    return (f"{100 * iguales:.1f}% de las posiciones iguales (dif max {dif.max():.1e}); gradiente: dif relativa "
            f"{rel:.1e}" + ("" if GPU else " (sin GPU: compara CPU con CPU)"))


@prueba("6. Causalidad en GPU: el futuro no cambia el pasado")
def p6():
    cfg = asrn.ConfigASRN(V=50, d=64, n_capas=4, chunk=8, n_bits=4)
    m = _activar(asrn.crear_desde_config(cfg, semilla=2), 1).to(disp).eval()
    x = _texto_rep(2, 513, cfg.V).to(disp)
    g = torch.Generator().manual_seed(0)
    peor = 0.0
    with torch.no_grad(), _sin_tf32():
        base = m(x)
        for t in (0, 100, 255, 400, 511):
            y = x.clone()
            y[:, t + 1:] = torch.randint(0, cfg.V, (2, x.shape[1] - t - 1), generator=g).to(disp)
            peor = max(peor, (m(y)[:, :t + 1] - base[:, :t + 1]).abs().max().item())
    assert peor < 1e-4, peor
    return f"dif max en el pasado {peor:.1e} (5 cortes, largo 513)"


@prueba("7. Escribir paso a paso = texto entero en GPU")
def p7():
    cfg = asrn.ConfigASRN(V=60, d=64, n_capas=3, chunk=8, n_bits=4)
    m = _activar(asrn.crear_desde_config(cfg, semilla=3), 2).to(disp).eval()
    x0 = _texto_rep(2, 60, cfg.V)
    with _sin_tf32():
        sec, pasos = asrn.escribir(m, x0, n_nuevos=80, memoria_de_lo_ajeno=False, top_k=0, conservar_cita=False)
        with torch.no_grad():
            entero = m(sec[:, :-1].to(disp))[:, x0.shape[1] - 1:].cpu()
    dif = (entero - pasos).abs().amax(-1)
    iguales = (dif < 1e-3).float().mean().item()
    assert iguales >= 0.99, f"solo {100 * iguales:.1f}% de los pasos coinciden (dif max {dif.max():.1e})"
    return f"{100 * iguales:.1f}% de los 80 pasos iguales (dif max {dif.max():.1e})"


if __name__ == '__main__':
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0) if GPU else 'SIN GPU (solo revisa el script)'}")
    for p in (p1, p2, p3, p4, p5, p6, p7):
        p()
    print("\n=== RESUMEN ===")
    for nombre, estado, detalle in resultados:
        print(f"{estado:7} {nombre}: {detalle}")
    if '--json' in sys.argv:
        with open(sys.argv[sys.argv.index('--json') + 1], 'w') as f:
            json.dump([{'prueba': n, 'resultado': e, 'detalle': str(d or '')}
                       for n, e, d in resultados], f, indent=1, ensure_ascii=False)
    sys.exit(0 if all(e != 'FALLA' for _, e, _ in resultados) else 1)
