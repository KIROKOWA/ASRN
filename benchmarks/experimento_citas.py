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
"""Experimento sintetico de citas acotadas con ASRN 1.0: ¿los mecanismos de regulacion (la puerta, la intensidad de
Hawkes y la racha) le ensenan al modelo cuando dejar de copiar?

Texto: piezas al azar (V = 64, imposibles de predecir) con citas: con probabilidad 0.04 por posicion se copia un trozo
de las ultimas 200 piezas, de largo uniforme entre 6 y 12. Asi las citas terminan siempre: en el texto real una racha
de grupos de 4 piezas ya vistos dura a lo sumo ~9. Un modelo que no sabe cuando termina una cita sigue copiando.

Modelos (todos con la capa estable de 1.0, grupo A de protecciones, 2 bloques locales y 1 capa ASRN,
d = 64; ver benchmarks/banco_modelos.py):
  ASRN 1.0             la capa completa
  sin Hawkes ni racha  u_hawkes, d_racha y sus dos kappas congelados en su valor inicial neutro (la puerta no ve lo
                       repetido)
  puerta sin senales   la puerta solo ve el estado: sigma(w_g . h + b)
  sin puerta           la puerta fija en 1: siempre suma lo copiado
  sin reloj            el reloj de tiempo propio congelado en su valor inicial (beta = 1, alfa = 1, gamma = 0): el
                       tiempo cuenta piezas
Los cinco se entrenan con la misma semilla, los mismos datos y la misma receta (lr 3e-3, lote 16, largo 256,
--pasos pasos con texto fresco). Despues se les da el inicio (128 piezas) de 64 textos de validacion y se compara lo que
escriben (128 piezas nuevas) con la continuacion real: fraccion de piezas citadas (su grupo de 4 ya salio antes) y largo
de las rachas de cita. Se escribe con temperatura 1 (sin top_k) y 0.8 (top_k 40), con y sin memoria de lo ajeno (si el
inicio cuenta como ya visto); ASRN 1.0 tambien con conservar_cita=False.

Uso:
    python benchmarks/experimento_citas.py --semillas 0 1 2              # una corrida por semilla -> semilla_<s>.json
    python benchmarks/experimento_citas.py --resumir                     # junta las semillas -> resultados.json y
                                                                         # RESUMEN.md
Cada modelo tardo 39-87 min en CPU con un hilo (6 000 pasos, 4 nucleos compartidos); las semillas se pueden correr en
procesos separados. Con --dispositivo cuda corre en la GPU (lo usa benchmarks/asrn_colab.ipynb). Si una corrida se corta,
volver a lanzarla retoma desde el ultimo modelo terminado de cada semilla."""
import argparse
import datetime
import glob
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter

import torch

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_AQUI)
sys.path[:0] = [p for p in (_RAIZ, _AQUI) if p not in sys.path]
import asrn.nucleo as nucleo
import banco_modelos

V, LARGO, D, CAPAS, LOTE, LR = 64, 256, 64, 2, 16, 3e-3
VARIANTES = (('ASRN 1.0', 'estable_A'), ('sin Hawkes ni racha', 'sin_hawkes_racha'),
             ('puerta sin senales', 'puerta_sin_senales'), ('sin puerta', 'sin_puerta'), ('sin reloj', 'sin_reloj'))
MODOS = [(t, k, ajeno) for t, k in ((1.0, 0), (0.8, 40)) for ajeno in (True, False)]
SALIDA = os.path.join(_RAIZ, 'results', 'citas')


def texto_con_citas(n_piezas, semilla, largo_min=6, largo_max=12, p_cita=0.04):
    """Piezas al azar en [1, V) y, con probabilidad p_cita por posicion, una cita de un trozo de las ultimas 200."""
    g = torch.Generator().manual_seed(semilla)
    x = torch.randint(1, V, (n_piezas,), generator=g)
    tiradas = torch.rand(n_piezas, generator=g)
    t = 64
    while t < n_piezas:
        if tiradas[t] < p_cita:
            largo = int(torch.randint(largo_min, largo_max + 1, (1,), generator=g))
            desde = int(torch.randint(max(0, t - 200), t - largo, (1,), generator=g))
            fin = min(n_piezas, t + largo)
            x[t:fin] = x[desde:desde + fin - t].clone()
            t = fin + 1                                     # despues de una cita, al menos una pieza nueva
        else:
            t += 1
    return x


def rachas_de_cita(sec, desde, n=4):
    """Para cada fila: las posiciones >= desde cuyo grupo de n piezas ya salio antes, agrupadas en rachas seguidas.
    Devuelve (fraccion citada, lista de largos de racha)."""
    citadas, rachas = 0, []
    for fila in sec.tolist():
        vistos, racha = set(), 0
        for t in range(len(fila)):
            clave = tuple(fila[t - n + 1:t + 1]) if t >= n - 1 else None
            ya = clave is not None and clave in vistos
            if clave is not None:
                vistos.add(clave)
            if t < desde:
                continue
            if ya:
                citadas += 1
                racha += 1
            elif racha:
                rachas.append(racha)
                racha = 0
        if racha:
            rachas.append(racha)
    return citadas / (sec.shape[0] * (sec.shape[1] - desde)), rachas


def estadisticas(fracciones, rachas):
    """Lo que se reporta de un conjunto de escrituras: % citado, cuantas rachas, media, maximo y % de rachas > 9."""
    return {'citado_pct': round(100 * sum(fracciones) / len(fracciones), 2), 'rachas': len(rachas),
            'media': round(sum(rachas) / len(rachas), 2) if rachas else 0.0, 'max': max(rachas, default=0),
            'mayores_de_9_pct': round(100 * sum(1 for r in rachas if r > 9) / max(1, len(rachas)), 2)}


def respuesta_racha(modelo):
    """Por capa ASRN: (racha, valor) de la respuesta aprendida de la puerta a la racha en cada nodo."""
    salida = []
    for c in modelo.largos:
        nodos = torch.expm1(torch.arange(c.d_racha.shape[0]) * c.racha_paso)
        salida.append([[round(r, 1), round(u, 3)] for r, u in zip(nodos.tolist(), c.u_racha.detach().tolist())])
    return salida


def modo_clave(t, k, ajeno):
    return f"T={t}, top_k={k}, memoria_de_lo_ajeno={ajeno}"


def entorno(dispositivo='cpu'):
    try:
        commit = subprocess.run(['git', '-C', _RAIZ, 'rev-parse', '--short', 'HEAD'], capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = ''
    return {'fecha': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
            'torch': torch.__version__, 'python': platform.python_version(), 'hilos': torch.get_num_threads(),
            'dispositivo': str(dispositivo),
            'gpu': torch.cuda.get_device_name(0) if str(dispositivo).startswith('cuda') else None,
            'cpu': platform.processor() or platform.machine(), 'commit': commit}


def correr_semilla(semilla, pasos, salida, dispositivo='cpu'):
    datos = texto_con_citas(pasos * LOTE * (LARGO + 1) + LARGO, semilla=1)     # texto fresco: no se repite
    val = texto_con_citas(64 * LARGO, semilla=2)
    docs, L0 = val.view(64, LARGO), LARGO // 2
    frac_real, rachas_real = rachas_de_cita(docs, L0)
    res = {'semilla': semilla, 'pasos': pasos, 'entorno': entorno(dispositivo),
           'receta': {'V': V, 'largo': LARGO, 'd': D, 'capas': CAPAS, 'lote': LOTE, 'lr': LR, 'inicio': L0,
                      'n_nuevos': LARGO - L0, 'textos': 64, 'citas': '6-12 piezas, p = 0.04, de las ultimas 200'},
           'texto_real': {'fraccion': frac_real, 'rachas': dict(Counter(rachas_real))}, 'modelos': {}}
    archivo = os.path.join(salida, f'semilla_{semilla}.json')
    if os.path.exists(archivo):                             # retomar: los modelos ya terminados no se repiten
        previo = json.load(open(archivo))
        if previo.get('pasos') == pasos:
            res['modelos'] = previo['modelos']
            res['entorno_previo'] = previo.get('entorno_previo', []) + [previo['entorno']]
    for nombre, variante in VARIANTES:
        if nombre in res['modelos']:
            print(f"semilla {semilla} | {nombre}: ya estaba", flush=True)
            continue
        m = banco_modelos.asrn_variante(variante, V, D, CAPAS, semilla, largo=LARGO).to(dispositivo)
        t0 = time.time()
        nucleo.entrenar(m, datos, largo=LARGO, lote=LOTE, pasos=pasos, lr=LR, semilla=semilla,
                        registrar_cada=pasos, registro=lambda *_: None)
        seg = time.time() - t0
        ev = nucleo.evaluar(m, val, largo=LARGO, lote=16)
        m.eval()
        entrada = {'variante': variante, 'segundos_entrenando': round(seg, 1), 'perdida_validacion': ev['perdida'],
                   'respuesta_racha': respuesta_racha(m), 'escritura': {}}
        opciones = [('', {})] + ([(' (conservar_cita=False)', {'conservar_cita': False})] if variante == 'estable_A'
                                 else [])
        for t, k, ajeno in MODOS:
            for sufijo, kw in opciones:
                with torch.no_grad():
                    sec, _ = nucleo.escribir(m, docs[:, :L0], n_nuevos=LARGO - L0, temperatura=t, top_k=k,
                                             semilla=semilla, memoria_de_lo_ajeno=ajeno, devolver_logits=False, **kw)
                frac, rachas = rachas_de_cita(sec, L0)
                entrada['escritura'][modo_clave(t, k, ajeno) + sufijo] = {'fraccion': frac,
                                                                          'rachas': dict(Counter(rachas))}
        res['modelos'][nombre] = entrada
        print(f"semilla {semilla} | {nombre}: {seg:.0f} s, perdida de validacion {ev['perdida']:.4f}", flush=True)
        os.makedirs(salida, exist_ok=True)
        with open(archivo + '.tmp', 'w') as f:              # se guarda despues de cada modelo
            json.dump(res, f, indent=1, ensure_ascii=False)
        os.replace(archivo + '.tmp', archivo)
    return res


def _rachas(contador):
    return [int(largo) for largo, n in contador.items() for _ in range(n)]


def resumir(salida):
    archivos = sorted(glob.glob(os.path.join(salida, 'semilla_*.json')))
    corridas = [json.load(open(a)) for a in archivos]
    corridas = [c for c in corridas if len(c['modelos']) == len(VARIANTES)]
    if not corridas:
        sys.exit(f'no hay semillas completas en {salida}')
    semillas = [c['semilla'] for c in corridas]
    real = corridas[0]['texto_real']
    tabla = {'texto real': estadisticas([real['fraccion']], _rachas(real['rachas']))}
    resumen = {'semillas': semillas, 'pasos': corridas[0]['pasos'], 'receta': corridas[0]['receta'],
               'entornos': [c['entorno'] for c in corridas], 'perdida_validacion': {}, 'respuesta_racha': {},
               'escritura': {}, 'max_por_semilla': {}}
    for nombre, _ in VARIANTES:
        resumen['perdida_validacion'][nombre] = {str(c['semilla']): c['modelos'][nombre]['perdida_validacion']
                                                 for c in corridas}
        resumen['respuesta_racha'][nombre] = {str(c['semilla']): c['modelos'][nombre]['respuesta_racha']
                                              for c in corridas}
    for clave in corridas[0]['modelos']['ASRN 1.0']['escritura']:
        filas = {'texto real': tabla['texto real']}
        maximos = {}
        for nombre, _ in VARIANTES:
            if clave not in corridas[0]['modelos'][nombre]['escritura']:
                continue
            esc = [c['modelos'][nombre]['escritura'][clave] for c in corridas]
            todas = [r for e in esc for r in _rachas(e['rachas'])]
            filas[nombre] = estadisticas([e['fraccion'] for e in esc], todas)
            maximos[nombre] = [max(_rachas(e['rachas']), default=0) for e in esc]
        resumen['escritura'][clave] = filas
        resumen['max_por_semilla'][clave] = maximos
    with open(os.path.join(salida, 'resultados.json'), 'w') as f:
        json.dump(resumen, f, indent=1, ensure_ascii=False)
    with open(os.path.join(salida, 'RESUMEN.md'), 'w') as f:
        f.write(_resumen_md(resumen))
    print(_resumen_md(resumen))


def _resumen_md(r):
    e = r['entornos'][0]
    lineas = [f"# Experimento de citas acotadas (ASRN 1.0)", '',
              f"Semillas {', '.join(map(str, r['semillas']))}; {r['pasos']} pasos; "
              f"{('GPU ' + e['gpu']) if e.get('gpu') else 'CPU, ' + str(e['hilos']) + ' hilo(s)'}, PyTorch {e['torch']}; "
              f"commit {e['commit']}; {r['entornos'][-1]['fecha']}. Generado por "
              "`benchmarks/experimento_citas.py --resumir` desde `semilla_*.json` (los datos crudos).", '',
              'Rachas: posiciones seguidas cuyo grupo de 4 piezas ya salio antes. En el texto real duran a lo sumo ~9 '
              '(citas de 6 a 12 piezas). Las rachas se juntan de todas las semillas; "max por semilla" es el maximo de '
              'cada una.', '', '## Perdida de validacion', '', '| Modelo | ' +
              ' | '.join(f'semilla {s}' for s in r['semillas']) + ' |', '|---|' + '---|' * len(r['semillas'])]
    for nombre, por in r['perdida_validacion'].items():
        lineas.append(f'| {nombre} | ' + ' | '.join(f'{por[str(s)]:.4f}' for s in r['semillas']) + ' |')
    for clave, filas in r['escritura'].items():
        lineas += ['', f'## {clave}', '', '| | Citado | Rachas | Media | Max | Max por semilla | Rachas > 9 |',
                   '|---|---|---|---|---|---|---|']
        for nombre, s in filas.items():
            maxs = r['max_por_semilla'][clave].get(nombre)
            lineas.append(f"| {nombre} | {s['citado_pct']:.1f}% | {s['rachas']} | {s['media']:.2f} | {s['max']} | "
                          f"{' / '.join(map(str, maxs)) if maxs else ''} | {s['mayores_de_9_pct']:.1f}% |")
    lineas += ['', '## Respuesta de la puerta a la racha (ASRN 1.0, primera capa ASRN)', '',
               '| Semilla | ' + ' | '.join(f'{x[0]:g}' for x in next(iter(r['respuesta_racha']['ASRN 1.0'].values()))[0])
               + ' |', '|---|' + '---|' * len(next(iter(r['respuesta_racha']['ASRN 1.0'].values()))[0])]
    for s, capas in r['respuesta_racha']['ASRN 1.0'].items():
        lineas.append(f'| {s} | ' + ' | '.join(f'{x[1]:+.2f}' for x in capas[0]) + ' |')
    return '\n'.join(lineas) + '\n'


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--pasos', type=int, default=6000)
    a.add_argument('--semillas', type=int, nargs='+', default=[0, 1, 2])
    a.add_argument('--hilos', type=int, default=0, help='hilos de CPU (0: los de PyTorch por omision)')
    a.add_argument('--dispositivo', default='cpu', help="'cpu' o 'cuda' (la GPU)")
    a.add_argument('--salida', default=SALIDA)
    a.add_argument('--resumir', action='store_true', help='solo juntar las semillas ya corridas')
    o = a.parse_args()
    if o.hilos:
        torch.set_num_threads(o.hilos)
    if o.resumir:
        return resumir(o.salida)
    for s in o.semillas:
        correr_semilla(s, o.pasos, o.salida, o.dispositivo)


if __name__ == '__main__':
    main()
