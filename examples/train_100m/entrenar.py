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
"""Entrena un ASRN de ~100 millones de parametros con los datos de preparar_fineweb.py.

    python examples/train_100m/entrenar.py --datos DIR_DATOS --salida DIR_MODELO [opciones]

En DIR_MODELO quedan:
  modelo.pt          el punto de control (pesos, AdamW, azar, historial): se guarda cada --guardar_cada pasos
  modelo.mejor.pt    el modelo con la mejor perdida de validacion (solo pesos)
  registro.txt       todo lo que se imprime
  asrn_100m.pt       el modelo final (solo pesos), al terminar
Si DIR_MODELO/modelo.pt ya existe, sigue desde ahi (Colab se desconecta: volver a correr el mismo comando continua
donde quedo, con los mismos lotes y la misma curva de la tasa de aprendizaje). La forma del modelo sale del archivo.

La receta por omision: 12 capas de ancho 768 (la forma de GPT-2 small) con una capa ASRN cada dos bloques: 103.5
millones de parametros con el vocabulario de GPT-2. Lotes de 16 textos de 1 024 piezas, 32 acumulados: ~524 mil piezas
por paso. Tasa 2e-3 con calentamiento y coseno hasta 2e-4, AdamW (decaimiento 0.1), bf16."""
import argparse
import math
import os
import re
import signal
import sys
import time

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(os.path.dirname(_AQUI))         # examples/train_100m -> la raiz del repositorio
sys.path[:0] = [_RAIZ, _AQUI]                           # el paquete asrn (en la raiz) y preparar_fineweb

import torch                                            # noqa: E402

import asrn                                               # noqa: E402  (capa estable por omision)
from preparar_fineweb import V_GPT2, leer_datos         # noqa: E402


def configuracion(ancho, capas, mezcla='conv'):
    return asrn.ConfigASRN(V=V_GPT2, d=ancho, n_capas=capas, mezcla=mezcla)


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--datos', required=True, help='carpeta de preparar_fineweb.py (con info.json)')
    a.add_argument('--salida', required=True, help='carpeta del modelo (por ejemplo, en Google Drive)')
    a.add_argument('--ancho', type=int, default=768)
    a.add_argument('--capas', type=int, default=12)
    a.add_argument('--minimo', type=float, default=100e6, help='parametros minimos (si no llega, se detiene)')
    a.add_argument('--largo', type=int, default=1024, help='piezas por texto')
    a.add_argument('--lote', type=int, default=16, help='textos por lote (bajalo si falta memoria)')
    a.add_argument('--acumular', type=int, default=32, help='lotes por paso (piezas por paso = lote x acumular x largo)')
    a.add_argument('--piezas', type=float, default=None,
                   help='piezas a entrenar en total (por omision, todas las de los datos: una pasada)')
    a.add_argument('--lr', type=float, default=2e-3)
    a.add_argument('--calentamiento', type=int, default=200, help='pasos de calentamiento de la tasa')
    a.add_argument('--compilar', action='store_true', help='torch.compile por partes (necesita PyTorch >= 2.14)')
    a.add_argument('--evaluar_cada', type=int, default=250)
    a.add_argument('--lotes_eval', type=int, default=50)
    a.add_argument('--guardar_cada', type=int, default=100)
    a.add_argument('--semilla', type=int, default=0)
    o = a.parse_args(argv)

    # Ctrl+C (o detener la celda en Colab) y SIGTERM guardan y terminan: entrenar() atrapa KeyboardInterrupt. Se
    # activa a mano porque un proceso lanzado en segundo plano puede heredar Ctrl+C ignorado.
    def _parar(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _parar)
    signal.signal(signal.SIGTERM, _parar)

    os.makedirs(o.salida, exist_ok=True)
    punto = os.path.join(o.salida, 'modelo.pt')
    archivo_registro = open(os.path.join(o.salida, 'registro.txt'), 'a', buffering=1)

    def registro(texto):
        print(texto, flush=True)
        archivo_registro.write(texto + '\n')

    if not torch.cuda.is_available():
        registro('aviso: no hay GPU; entrenar 100 millones de parametros en CPU tardaria semanas')
    disp = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    t = time.time()
    datos, val, info = leer_datos(o.datos)
    registro(f'datos: {info["fuente"]}, {datos.numel():,} piezas de entrenamiento y {val.numel():,} de validacion '
             f'({time.time() - t:.0f} s)')

    reanudar = punto if os.path.exists(punto) else None
    if reanudar:
        cfg = asrn.ConfigASRN.desde_config_guardada(asrn.nucleo._leer(punto, mmap=True)['config'])
        registro(f'reanudando desde {punto}')
    else:
        cfg = configuracion(o.ancho, o.capas)
    n = cfg.contar_parametros()
    des = cfg.contar_parametros(desglose=True)
    registro(f'modelo: {cfg.n_capas} capas de ancho {cfg.d}, una capa ASRN cada {cfg.largo_cada} bloques: '
             f'{n:,} parametros ({n - des["vocabulario"]:,} sin el vocabulario)')
    if n < o.minimo:
        raise SystemExit(f'el modelo tiene {n:,} parametros, menos que --minimo {o.minimo:,.0f}: sube --ancho o --capas')

    por_paso = o.lote * o.acumular * o.largo
    piezas = int(o.piezas) if o.piezas else datos.numel()
    pasos = max(1, math.ceil(piezas / por_paso))
    registro(f'entrenamiento: {pasos:,} pasos de {por_paso:,} piezas ({pasos * por_paso / 1e9:.2f} mil millones; '
             f'{pasos * por_paso / n:.0f} piezas por parametro)')

    modelo = asrn.crear_desde_config(cfg, semilla=o.semilla).to(disp)
    inicio = {'t': time.time(), 'paso': None}

    def registro_con_tiempo(texto):         # agrega cuanto falta a las lineas de cada paso
        m = re.match(r'paso\s+(\d+)', texto)
        if m:
            paso = int(m.group(1))
            if inicio['paso'] is None:
                inicio['paso'], inicio['t'] = paso, time.time()
            elif paso > inicio['paso']:
                seg = (time.time() - inicio['t']) / (paso - inicio['paso']) * (pasos - paso)
                texto += f' | faltan ~{seg / 3600:.1f} h'
        registro(texto)

    asrn.entrenar(modelo, datos, validacion=val, largo=o.largo, lote=o.lote, acumular=o.acumular, pasos=pasos,
                  lr=o.lr, lr_min=o.lr / 10, calentamiento=o.calentamiento, decaimiento=0.1, betas=(0.9, 0.95),
                  recorte=1.0, precision='auto', compilar='partes' if o.compilar else False,
                  evaluar_cada=o.evaluar_cada, lotes_eval=o.lotes_eval, guardar_en=punto,
                  guardar_cada=o.guardar_cada, reanudar=reanudar, semilla=o.semilla, registrar_cada=10,
                  registro=registro_con_tiempo)
    final = os.path.join(o.salida, 'asrn_100m.pt')
    estado = asrn.nucleo._leer(punto, mmap=True)
    if estado['paso'] >= pasos:
        asrn.guardar(modelo, final)
        registro(f'listo: modelo final en {final}; el mejor en validacion, en {punto[:-3]}.mejor.pt')
    else:
        registro(f'detenido en el paso {estado["paso"]} de {pasos}: vuelve a correr el mismo comando para seguir')


if __name__ == '__main__':
    main()
