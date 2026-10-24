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
"""Arma examples/train_100m/entrenar_asrn_100m.ipynb (sin salidas). Correr desde la raiz del repositorio:
    python benchmarks/builders/construir_entrenar_100m.py examples/train_100m/entrenar_asrn_100m.ipynb
(sin argumento escribe ese mismo archivo)."""
import os
import sys

import nbformat

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # benchmarks/builders -> raiz
SALIDA_POR_OMISION = os.path.join(RAIZ, 'examples', 'train_100m', 'entrenar_asrn_100m.ipynb')

CELDAS = []


def md(texto):
    CELDAS.append(nbformat.v4.new_markdown_cell(texto.strip('\n')))


def code(texto):
    CELDAS.append(nbformat.v4.new_code_cell(texto.strip('\n')))


md(r'''
# Entrenar ASRN de 100 millones de parámetros con FineWeb

Este cuaderno entrena desde cero un modelo de lenguaje **ASRN 1.0** de **103.5 millones de parámetros** con
**FineWeb** (texto web en inglés, filtrado; `HuggingFaceFW/fineweb`, muestra de 10 mil millones de piezas) en Google
Colab. Todo queda en tu **Google Drive**, así que puedes cortar y seguir otro día.

| | |
|---|---|
| Modelo | 12 capas de ancho 768 (la forma de GPT-2 small), una capa ASRN cada dos bloques (6 capas ASRN) |
| Parámetros | 103.5 millones (64.9 millones sin contar el vocabulario) |
| Tokenizador | GPT-2 (`tiktoken`, 50 257 piezas) |
| Datos | 2 mil millones de piezas de FineWeb (~20 por parámetro) + 10 millones para validar |
| Paso | 16 textos de 1 024 piezas × 32 acumulados = ~524 mil piezas |
| Optimizador | AdamW (decaimiento 0.1), tasa 2e-3 con calentamiento y coseno hasta 2e-4, bf16 |

**Tiempo aproximado en una A100:** preparar los datos, 20–40 minutos (una sola vez); entrenar con 2 mil millones de
piezas, entre 5 y 7 horas. Con `PIEZAS = 1_000_000_000` es la mitad. El cuaderno muestra cuánto falta mientras entrena.

**Cómo correrlo**

1. **Entorno de ejecución → Cambiar tipo de entorno de ejecución → A100 GPU** (y, si puedes, *RAM alta*).
2. Revisa la celda de configuración (abajo). Lo normal es no cambiar nada.
3. **Entorno de ejecución → Ejecutar todas**. Te pedirá permiso para usar tu Google Drive y, si el repositorio es
   privado, que subas su ZIP (en GitHub: botón verde **Code → Download ZIP**).
4. **Si Colab se desconecta** (pasa en sesiones largas): vuelve a abrir el cuaderno y dale otra vez a **Ejecutar
   todas**. Los datos ya preparados no se repiten y el entrenamiento sigue desde el último punto guardado (cada 100
   pasos, unos 10 minutos).
5. **Para detenerlo a propósito**: el botón ■ de la celda de entrenamiento. Guarda antes de salir.

**Espacio en Drive:** unos 6 GB (4 GB de datos, 1.2 GB del punto de control y dos copias del modelo de 0.4 GB).
''')

code(r'''
# Configuración. Lo normal es no cambiar nada.
CARPETA_DRIVE = '/content/drive/MyDrive/asrn_100m'  # datos, modelo y registros (sobreviven entre sesiones)
FUENTE = 'fineweb'             # 'fineweb' o 'fineweb-edu' (solo páginas educativas: suele dar mejor modelo)
PIEZAS = 2_000_000_000         # piezas de entrenamiento: ~20 por parámetro. 1_000_000_000 tarda la mitad
VAL = 10_000_000               # piezas de validación (de un archivo que el entrenamiento no usa)
ANCHO, CAPAS = 768, 12         # 103.5 millones de parámetros. Con CAPAS = 14: 114 millones
LARGO, LOTE, ACUMULAR = 1024, 16, 32   # si falta memoria: LOTE = 8 y ACUMULAR = 64 (mismo paso, algo más lento)
LR = 2e-3                      # tasa de aprendizaje máxima
COMPILAR = True                # instala PyTorch 2.14 y compila por partes (~25% más rápido; la primera vez tarda unos minutos)
REPO = 'https://github.com/KIROKOWA/ASRN.git'
RAMA = 'main'
''')

md(r'''
## 1. GPU y Google Drive
''')

code(r'''
import os, subprocess, sys
try:
    print(subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True).stdout)
except (OSError, subprocess.CalledProcessError):
    print('No hay GPU. Entorno de ejecución → Cambiar tipo de entorno de ejecución → A100 GPU, y vuelve a empezar.')
EN_COLAB = 'google.colab' in sys.modules
if EN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
os.makedirs(CARPETA_DRIVE, exist_ok=True)
DATOS = os.path.join(CARPETA_DRIVE, 'datos')
MODELO = os.path.join(CARPETA_DRIVE, 'modelo')
print('Todo queda en', CARPETA_DRIVE)
''')

md(r'''
## 2. El código de ASRN

Se usa una copia guardada en Drive (`asrn_100m/codigo`) para que todas las sesiones entrenen con el mismo código. Si no
existe, se baja de GitHub (o del ZIP que subas) y se guarda ahí. Para usar una versión nueva del código, borra esa
carpeta de Drive. Una copia con la organización anterior del repositorio (sin `examples/train_100m/`) se reemplaza
sola por la nueva.
''')

code(r'''
import glob, shutil, zipfile
CODIGO = os.path.join(CARPETA_DRIVE, 'codigo')

def _tiene(carpeta):
    return (os.path.exists(os.path.join(carpeta, 'asrn', 'nucleo.py'))
            and os.path.isdir(os.path.join(carpeta, 'examples', 'train_100m')))

if not _tiene(CODIGO):
    tmp = '/content/asrn_repo'
    shutil.rmtree(tmp, ignore_errors=True)
    if os.environ.get('ASRN_DIR'):                       # para revisar el cuaderno fuera de Colab
        shutil.copytree(os.environ['ASRN_DIR'], tmp, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    else:
        r = subprocess.run(['git', 'clone', '-q', '--depth', '1', '-b', RAMA, REPO, tmp], capture_output=True,
                           text=True, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
        if r.returncode != 0:
            if not EN_COLAB:
                raise RuntimeError(f'No se pudo descargar {REPO}: {r.stderr.strip()}')
            print('El repositorio es privado. Sube su ZIP (en GitHub: botón verde Code → Download ZIP).')
            from google.colab import files
            subido = files.upload()
            if not subido:
                raise RuntimeError('No se subió ningún archivo')
            shutil.rmtree('/content/zip_asrn', ignore_errors=True)
            with zipfile.ZipFile(next(iter(subido))) as z:
                z.extractall('/content/zip_asrn')
            raices = sorted(glob.glob('/content/zip_asrn/**/asrn/nucleo.py', recursive=True), key=len)
            if not raices:
                raise RuntimeError('Ese ZIP no tiene asrn/nucleo.py: descarga el ZIP del repositorio ASRN')
            shutil.copytree(os.path.dirname(os.path.dirname(raices[0])), tmp)   # raices[0] = <raiz>/asrn/nucleo.py
    if not _tiene(tmp):
        raise RuntimeError('Ese código no tiene la carpeta examples/train_100m: descarga la versión más nueva del '
                           'repositorio')
    shutil.rmtree(CODIGO, ignore_errors=True)
    shutil.copytree(tmp, CODIGO, ignore=shutil.ignore_patterns('.git', '__pycache__', 'results'))
os.chdir(CODIGO)
sys.path[:0] = [CODIGO, os.path.join(CODIGO, 'examples', 'train_100m')]
print('Código en', CODIGO)
''')

md(r'''
## 3. Instalar lo necesario

`tiktoken` (el tokenizador de GPT-2) y, con `COMPILAR = True`, PyTorch 2.14: `torch.compile` falla con este modelo en
versiones anteriores, y Colab trae una anterior. Instalar PyTorch tarda entre 5 y 10 minutos (una vez por sesión).
''')

code(r'''
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tiktoken', 'pyarrow', 'huggingface_hub', 'plotly'],
               check=True)
if COMPILAR:
    version = subprocess.run([sys.executable, '-c', 'import torch; print(torch.__version__)'], capture_output=True,
                             text=True).stdout.strip()
    if tuple(int(n) for n in version.split('+')[0].split('.')[:2]) < (2, 14):
        print(f'PyTorch {version} -> 2.14 (son varios gigas: entre 5 y 10 minutos)...', flush=True)
        p = subprocess.Popen([sys.executable, '-m', 'pip', 'install', '--progress-bar', 'off', 'torch==2.14.*',
                              '--index-url', 'https://download.pytorch.org/whl/cu130'],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for linea in p.stdout:
            if linea.startswith(('Collecting', 'Downloading', 'Installing', 'Successfully', 'ERROR')):
                print(linea, end='', flush=True)
        if p.wait() != 0:
            print('No se pudo instalar PyTorch 2.14: se entrena sin compilar (algo más lento)')
            COMPILAR = False
import torch
print('PyTorch', torch.__version__, '| compilar:', COMPILAR)
''')

md(r'''
## 4. El modelo

Cuenta los parámetros sin armar el modelo y revisa que pase de 100 millones.
''')

code(r'''
import asrn                     # el paquete: capa estable por omision
cfg = asrn.ConfigASRN(V=50257, d=ANCHO, n_capas=CAPAS)
des = cfg.contar_parametros(desglose=True)
total = des['total']
print(f'{CAPAS} capas de ancho {ANCHO}, una capa ASRN cada {cfg.largo_cada} bloques '
      f'({len(range(0, CAPAS - 1, cfg.largo_cada))} capas ASRN)')
for k, v in des.items():
    print(f'  {k:>16}: {v:>13,}')
assert total >= 100e6, f'{total:,} parámetros: sube ANCHO o CAPAS para pasar de 100 millones'
por_paso = LOTE * ACUMULAR * LARGO
pasos = -(-PIEZAS // por_paso)
print(f'\n{pasos:,} pasos de {por_paso:,} piezas = {pasos * por_paso / 1e9:.2f} mil millones de piezas '
      f'({pasos * por_paso / total:.0f} por parámetro)')
print(f'A ~100 mil piezas por segundo serían unas {pasos * por_paso / 1e5 / 3600:.1f} horas (la velocidad real se ve '
      'al entrenar)')
''')

md(r'''
## 5. Preparar los datos de FineWeb

Baja los archivos de FineWeb uno por uno, los tokeniza y guarda las piezas en `asrn_100m/datos` (archivos `.bin` de
16 bits). La validación sale del último archivo de la fuente, que el entrenamiento nunca usa. Si se corta, al volver
a correr esta celda sigue desde el primer archivo sin terminar; si ya está todo, no hace nada.
''')

code(r'''
def correr(argumentos):
    """Corre un script mostrando su salida en vivo. Si detienes la celda, le avisa (Ctrl+C) para que guarde y salga."""
    import signal
    p = subprocess.Popen([sys.executable, '-u'] + argumentos, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    try:
        for linea in p.stdout:
            if 'Warning' not in linea and 'warnings.warn' not in linea:
                print(linea, end='', flush=True)
    except KeyboardInterrupt:
        print('\nDeteniendo (guarda antes de salir)...', flush=True)
        p.send_signal(signal.SIGINT)
        for linea in p.stdout:
            print(linea, end='', flush=True)
    return p.wait()

codigo = correr(['examples/train_100m/preparar_fineweb.py', '--salida', DATOS, '--piezas', str(PIEZAS),
                 '--val', str(VAL), '--fuente', FUENTE, '--temporal', '/content/fineweb_tmp'])
if codigo != 0:
    raise RuntimeError('La preparación de los datos terminó con un error (arriba está el detalle)')
''')

md(r'''
## 6. Entrenar

Imprime cada 10 pasos la pérdida, la tasa de aprendizaje, la velocidad y cuánto falta, y cada 250 pasos la perplejidad
de validación (menos es mejor). Guarda cada 100 pasos en `asrn_100m/modelo/modelo.pt`, el mejor en validación en
`modelo.mejor.pt` y todo lo impreso en `registro.txt`. Si la celda se detiene o Colab se desconecta, volver a correrla
sigue desde el último punto guardado.
''')

code(r'''
argumentos = ['examples/train_100m/entrenar.py', '--datos', DATOS, '--salida', MODELO, '--ancho', str(ANCHO),
              '--capas', str(CAPAS), '--largo', str(LARGO), '--lote', str(LOTE), '--acumular', str(ACUMULAR),
              '--piezas', str(PIEZAS), '--lr', str(LR)] + (['--compilar'] if COMPILAR else [])
if correr(argumentos) != 0:
    raise RuntimeError('El entrenamiento terminó con un error (arriba está el detalle)')
''')

md(r'''
## 7. Cómo va: gráficas interactivas

Lee el historial del último punto guardado. Se puede correr en cualquier momento, también en otra sesión mientras no
haya entrenamiento. Pasa el cursor sobre las curvas para ver los valores.
''')

code(r'''
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

punto = os.path.join(MODELO, 'modelo.pt')
estado = asrn.nucleo._leer(punto)
hist = estado['historial']
por_paso = LOTE * ACUMULAR * LARGO
tren = [h for h in hist if 'perdida' in h]
val = [h for h in hist if 'val_perdida' in h]
print(f"paso {estado['paso']:,}: {estado['paso'] * por_paso / 1e9:.2f} mil millones de piezas vistas")

def suave(xs, k=10):
    out, acc = [], []
    for x in xs:
        acc.append(x)
        acc = acc[-k:]
        out.append(sum(acc) / len(acc))
    return out

fig = make_subplots(rows=2, cols=2, subplot_titles=('Pérdida de entrenamiento', 'Perplejidad de validación',
                                                    'Tasa de aprendizaje', 'Piezas por segundo'))
x = [h['paso'] * por_paso / 1e9 for h in tren]
fig.add_trace(go.Scatter(x=x, y=[h['perdida'] for h in tren], mode='lines', name='por paso', opacity=0.35,
                         line=dict(color='#0d7465', width=1)), 1, 1)
fig.add_trace(go.Scatter(x=x, y=suave([h['perdida'] for h in tren]), mode='lines', name='promedio',
                         line=dict(color='#0d7465', width=2.5)), 1, 1)
fig.add_trace(go.Scatter(x=[h['paso'] * por_paso / 1e9 for h in val], y=[h['val_perplejidad'] for h in val],
                         mode='lines+markers', name='validación', line=dict(color='#c0562a', width=2.5)), 1, 2)
fig.add_trace(go.Scatter(x=x, y=[h['lr'] for h in tren], mode='lines', name='tasa',
                         line=dict(color='#7a4fbf', width=2)), 2, 1)
fig.add_trace(go.Scatter(x=x, y=[h['piezas_s'] for h in tren], mode='lines', name='piezas/s',
                         line=dict(color='#2a6fc0', width=1.5)), 2, 2)
fig.update_yaxes(type='log', row=1, col=2)
for c in (1, 2):
    for r in (1, 2):
        fig.update_xaxes(title_text='miles de millones de piezas vistas', row=r, col=c)
fig.update_layout(height=700, template='plotly_white', showlegend=False,
                  title=f'ASRN {total / 1e6:.1f} M en {FUENTE}: paso {estado["paso"]:,}')
fig.write_html(os.path.join(MODELO, 'curvas.html'))
fig.show()
if val:
    print(f"última perplejidad de validación: {val[-1]['val_perplejidad']:.2f}; la mejor: "
          f"{min(h['val_perplejidad'] for h in val):.2f}")
''')

md(r'''
## 8. Evaluar el mejor modelo

Perplejidad en todo el conjunto de validación con textos de 1 024 piezas (como al entrenar) y de 4 096 (4 veces más
largos de lo que vio: ASRN suele mejorar con más contexto).
''')

code(r'''
from preparar_fineweb import leer_datos
disp = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo = asrn.cargar(os.path.join(MODELO, 'modelo.mejor.pt'), disp)
_, val_piezas, _ = leer_datos(DATOS)

@torch.no_grad()
def perplejidad(largo, max_piezas=4_000_000, lote=8):
    n = min(val_piezas.numel() - 1, max_piezas) // largo
    total, cuenta = 0.0, 0
    for i in range(0, n, lote):
        idx = torch.arange(i, min(n, i + lote)).unsqueeze(1) * largo + torch.arange(largo + 1)
        w = val_piezas[idx].to(disp).long()
        with torch.autocast(disp.type, dtype=torch.bfloat16, enabled=disp.type == 'cuda'):
            lg = modelo(w[:, :-1])
        total += torch.nn.functional.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), w[:, 1:].reshape(-1),
                                                   reduction='sum').item()
        cuenta += w[:, 1:].numel()
    return math.exp(total / cuenta)

for L in (1024, 4096):
    print(f'perplejidad con textos de {L:>5,}: {perplejidad(L, lote=8 if L == 1024 else 2):.2f}')
''')

md(r'''
## 9. Escribir

El modelo continúa unos textos de ejemplo (FineWeb está en inglés). Cambia `INICIOS` para probar los tuyos.
''')

code(r'''
import tiktoken
enc = tiktoken.get_encoding('gpt2')
INICIOS = ['The history of the city began', 'In machine learning, a neural network',
           'The recipe calls for two cups of flour and']
for texto in INICIOS:
    x = torch.tensor([enc.encode_ordinary(texto)], device=disp)
    salida, _ = asrn.escribir(modelo, x, n_nuevos=80, temperatura=0.8, top_k=40, semilla=0, devolver_logits=False)
    print('>>', enc.decode(salida[0].tolist()).replace('\n', ' '), '\n')
''')

md(r'''
## 10. El modelo final

Al terminar el entrenamiento queda `asrn_100m/modelo/asrn_100m.pt` (solo los pesos y la configuración, ~0.4 GB).
Se carga con `asrn.cargar('asrn_100m.pt')`. `modelo.mejor.pt` es el de mejor validación y `modelo.pt` el punto
de control para seguir entrenando (con el optimizador, ~1.2 GB).

**Si algo falla**

- *Out of memory* al entrenar: `LOTE = 8` y `ACUMULAR = 64` (mismas piezas por paso). Con un punto de control ya
  guardado, esto se puede cambiar y seguir.
- La descarga de FineWeb se corta: vuelve a correr la celda 5; sigue donde quedó.
- Quieres empezar de cero: borra `asrn_100m/modelo` en Drive (los datos se conservan).
- Quieres más o menos piezas: cambia `PIEZAS` antes de preparar los datos. Si ya están preparados con otro número,
  usa otra `CARPETA_DRIVE` o borra `asrn_100m/datos`.
''')

code(r'''
final = os.path.join(MODELO, 'asrn_100m.pt')
for nombre in ('asrn_100m.pt', 'modelo.mejor.pt', 'modelo.pt', 'registro.txt', 'curvas.html'):
    ruta = os.path.join(MODELO, nombre)
    print(f'{nombre:>16}: ' + (f'{os.path.getsize(ruta) / 1e9:.2f} GB' if os.path.exists(ruta) else 'todavía no'))
''')


def construir(ruta):
    nb = nbformat.v4.new_notebook()
    nb.metadata = {'accelerator': 'GPU', 'colab': {'gpuType': 'A100', 'machine_shape': 'hm', 'provenance': [],
                                                   'name': 'entrenar_asrn_100m.ipynb'},
                   'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}}
    nb.cells = CELDAS
    for i, c in enumerate(nb.cells):
        c.id = f'asrn100m-{i}'
    nbformat.validate(nb)
    nbformat.write(nb, ruta)
    print(f'{ruta}: {len(nb.cells)} celdas')


if __name__ == '__main__':
    construir(sys.argv[1] if len(sys.argv) > 1 else SALIDA_POR_OMISION)
