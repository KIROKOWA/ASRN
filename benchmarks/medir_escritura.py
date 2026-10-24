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
"""Tiempo por pieza nueva al escribir, segun cuanto texto ya hay: la escritura O(1) de ASRN 1.0 (anillos por
casillero y Hawkes por clave) contra la de antes, O(t) (cada pieza contra toda la memoria). Las dos dan las mismas
salidas (tests/test_nucleo.py: test_escritura_o1_igual_que_o_t); aqui solo se mide el tiempo.

    python benchmarks/medir_escritura.py                      # CPU: results/escritura_cpu/
    python benchmarks/medir_escritura.py --dispositivo cuda   # GPU: results/escritura_gpu/

Por cada contexto: el tiempo de leer la entrada (prefijo) y la mediana de ms por pieza en `repeticiones` tandas de
`nuevos` piezas (lote 1, sin muestrear: solo el modelo); las dos escrituras se miden alternadas en cada contexto,
despues de calentar las dos, y antes de cada largo se lee una entrada de ese largo sin medirla (la primera vez
que aparece un tamano, el sistema reserva la memoria y eso no es de ninguna de las dos). La pendiente es la de una recta por minimos cuadrados de ms por pieza contra el
contexto, con su intervalo de 95%."""
import argparse
import json
import os
import platform
import statistics as st
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))

import torch                                         # noqa: E402

import asrn                                          # noqa: E402
from asrn import nucleo                              # noqa: E402


def sincronizar(dv):
    if dv.type == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def medir(m, L0, o1, nuevos, repeticiones, dv, V):
    g = torch.Generator().manual_seed(L0)
    x = torch.randint(0, V, (1, L0), generator=g).to(dv)
    piezas = torch.randint(0, V, (repeticiones, nuevos + 1), generator=g)    # en la CPU, como las pasa escribir()
    sincronizar(dv)
    t0 = time.perf_counter()
    esc = nucleo._Escritor(m, o1=o1)
    esc.inicio(x, capacidad=L0 + repeticiones * (nuevos + 1) + 1)
    sincronizar(dv)
    prefijo = time.perf_counter() - t0
    tandas = []
    for r in range(repeticiones):
        esc.paso(piezas[r, 0])                       # calentar
        sincronizar(dv)
        t0 = time.perf_counter()
        for k in range(1, nuevos + 1):
            esc.paso(piezas[r, k])
        sincronizar(dv)
        tandas.append(1e3 * (time.perf_counter() - t0) / nuevos)
    return prefijo, st.median(tandas), tandas


def recta(xs, ys):
    """Pendiente (ms por pieza de contexto) y su intervalo de 95% (t de Student, n - 2 grados de libertad)."""
    from scipy import stats
    r = stats.linregress(xs, ys)
    t = stats.t.ppf(0.975, len(xs) - 2)
    return r.slope, r.slope - t * r.stderr, r.slope + t * r.stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dispositivo', default='cpu')
    ap.add_argument('--contextos', default='512,2048,8192,32768,131072')
    ap.add_argument('--nuevos', type=int, default=32)
    ap.add_argument('--repeticiones', type=int, default=9)
    ap.add_argument('--salida', default=None)
    a = ap.parse_args()
    dv = torch.device(a.dispositivo)
    contextos = [int(c) for c in a.contextos.split(',')]
    salida = a.salida or os.path.join(os.path.dirname(AQUI), 'results',
                                      'escritura_gpu' if dv.type == 'cuda' else 'escritura_cpu')
    os.makedirs(salida, exist_ok=True)
    V = 256
    m = asrn.crear_modelo(V, d=128, n_capas=4, semilla=0).to(dv).eval()     # el modelo del banco de CPU
    res = {'entorno': {'fecha': time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()), 'torch': torch.__version__,
                       'python': platform.python_version(), 'dispositivo': dv.type,
                       'gpu': torch.cuda.get_device_name(dv) if dv.type == 'cuda' else None,
                       'hilos': torch.get_num_threads(), 'cpu': platform.processor() or platform.machine()},
           'modelo': {'V': V, 'd': 128, 'n_capas': 4, 'parametros': sum(p.numel() for p in m.parameters()),
                      'capas_asrn': len(m.largos), 'n_tablas': m.largos[0].n_tablas, 'n_bits': m.largos[0].n_bits,
                      'chunk': m.largos[0].chunk},
           'nuevos': a.nuevos, 'repeticiones': a.repeticiones, 'filas': []}
    print(f"Escritura, lote 1, {dv.type}: ms por pieza nueva segun el contexto")
    for o1 in (True, False):                         # calentar las dos rutas antes de medir
        medir(m, contextos[0], o1, a.nuevos, 1, dv, V)
    for L0 in contextos:
        medir(m, L0, False, 1, 1, dv, V)             # la primera lectura de cada largo paga la memoria nueva: sin medir
        for o1 in (True, False):                     # alternadas: el mismo estado de la maquina para las dos
            prefijo, ms, tandas = medir(m, L0, o1, a.nuevos, a.repeticiones, dv, V)
            res['filas'].append({'escritura': 'O(1)' if o1 else 'O(t)', 'contexto': L0, 'prefijo_s': prefijo,
                                 'ms_por_pieza': ms, 'tandas_ms': tandas})
            print(f"  {'O(1)' if o1 else 'O(t)'}  contexto {L0:>7}: {ms:8.2f} ms por pieza  (prefijo {prefijo:.2f} s)")
    res['pendiente_ns_por_pieza_de_contexto'] = {}
    for modo in ('O(1)', 'O(t)'):
        f = [r for r in res['filas'] if r['escritura'] == modo]
        p, lo, hi = recta([r['contexto'] for r in f], [r['ms_por_pieza'] for r in f])
        res['pendiente_ns_por_pieza_de_contexto'][modo] = {'pendiente': 1e6 * p, 'ic95': [1e6 * lo, 1e6 * hi]}
        print(f"  {modo}: pendiente {1e6 * p:.2f} ns por pieza de contexto (IC 95% {1e6 * lo:.2f} a {1e6 * hi:.2f})")
    with open(os.path.join(salida, 'resultados.json'), 'w') as f:
        json.dump(res, f, indent=1)
    lineas = [f"# Escritura O(1) contra O(t) ({res['entorno']['fecha']}, {dv.type}"
              + (f", {res['entorno']['gpu']}" if res['entorno']['gpu'] else f", {res['entorno']['hilos']} hilos")
              + f", PyTorch {torch.__version__})", '',
              f"Modelo ASRN de {res['modelo']['parametros']:,} parametros (V = {V}, d = 128, 4 bloques, "
              f"{res['modelo']['capas_asrn']} capas ASRN), lote 1, sin muestrear. Mediana de {a.repeticiones} tandas "
              f"de {a.nuevos} piezas.", '',
              '| contexto | O(1): ms por pieza | O(t): ms por pieza | O(1): prefijo (s) | O(t): prefijo (s) |',
              '|---|---|---|---|---|']
    for L0 in contextos:
        u = next(r for r in res['filas'] if r['escritura'] == 'O(1)' and r['contexto'] == L0)
        t = next(r for r in res['filas'] if r['escritura'] == 'O(t)' and r['contexto'] == L0)
        lineas.append(f"| {L0:,} | {u['ms_por_pieza']:.2f} | {t['ms_por_pieza']:.2f} | {u['prefijo_s']:.2f} | "
                      f"{t['prefijo_s']:.2f} |")
    lineas.append('')
    for modo, v in res['pendiente_ns_por_pieza_de_contexto'].items():
        lineas.append(f"- {modo}: pendiente {v['pendiente']:.2f} ns por pieza de contexto (IC 95% "
                      f"{v['ic95'][0]:.2f} a {v['ic95'][1]:.2f})")
    with open(os.path.join(salida, 'RESUMEN.md'), 'w') as f:
        f.write('\n'.join(lineas) + '\n')
    print('->', salida)


if __name__ == '__main__':
    main()
