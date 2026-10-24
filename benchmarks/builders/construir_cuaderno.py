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
"""Arma benchmarks/asrn_colab.ipynb (sin salidas): el cuaderno unico de Google Colab que corre, en una A100 con
PyTorch 2.14, las pruebas del repositorio, el banco de ASRN 1.0 contra los originales y el experimento de citas. Correr
desde la raiz del repositorio:
    python benchmarks/builders/construir_cuaderno.py benchmarks/asrn_colab.ipynb
(sin argumento escribe ese mismo archivo). Las celdas son delgadas: todo lo que hacen esta en benchmarks/banco_asrn.py,
banco_gpu.py, modelos_oficiales.py y experimento_citas.py, que se prueban en CPU."""
import os
import sys

import nbformat

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # benchmarks/builders -> raiz
SALIDA_POR_OMISION = os.path.join(RAIZ, 'benchmarks', 'asrn_colab.ipynb')

CELDAS = []


def md(texto):
    CELDAS.append(('md', texto.strip('\n')))


def code(texto):
    CELDAS.append(('code', texto.strip('\n')))


# ------------------------------------------------------------------------------------------------------------------
md(r'''
# ASRN 1.0 en una A100 con PyTorch 2.14: todas las pruebas en un cuaderno

Este cuaderno corre, en una **A100 de Google Colab** y con **PyTorch 2.14** (la primera versión con la que
`torch.compile` compila la capa ASRN y sus kernels Triton), todo lo que respalda el lanzamiento de ASRN 1.0: las pruebas
del repositorio, las protecciones numéricas en la GPU de verdad, el costo por capa, la velocidad de entrenar (también
compilado) y de escribir, la calidad en tres tareas contra los modelos originales, la ablación y el experimento de
citas acotadas. Lo explican el informe técnico
([`docs/TECHNICAL_REPORT_es.md`](https://github.com/KIROKOWA/ASRN/blob/main/docs/TECHNICAL_REPORT_es.md); en inglés,
[`docs/TECHNICAL_REPORT.md`](https://github.com/KIROKOWA/ASRN/blob/main/docs/TECHNICAL_REPORT.md)) y
[`benchmarks/README.md`](https://github.com/KIROKOWA/ASRN/blob/main/benchmarks/README.md).
**Nada viene de corridas anteriores.**

**Los rivales, en su versión más rápida disponible** (Mamba-2 y RWKV-7 originales; el Transformer++ es nuestro):

| Modelo | Implementación | Kernels |
|---|---|---|
| **ASRN** | el paquete `asrn` (`asrn/estable.py`: la capa de v89 con las protecciones del grupo A; mismos pesos y cuentas) | Triton propios en modo `ESTABLE`; además, compilado por partes (`compilar_partes`) |
| **Transformer++** | La receta de Llama (RMSNorm, SwiGLU, RoPE), la referencia de los artículos de Mamba y RWKV; escrita por nosotros en `modelos_oficiales.py` | `scaled_dot_product_attention` = FlashAttention-2 en A100; caché de llaves para escribir; además, compilado con `torch.compile` |
| **Mamba-2** | `mamba_ssm` oficial (Dao y Gu): `MambaLMHeadModel` con capas `Mamba2` y sus valores por omisión | SSD de Triton de los autores; con `causal-conv1d`, el camino fusionado |
| **RWKV-7** | `flash-linear-attention` (fla), la implementación de referencia en Hugging Face | Triton de fla (por bloques al entrenar, recurrente al escribir) |

Si una librería no se puede instalar, ese modelo usa la reimplementación de `benchmarks/rivales_lineales.py` (con
los kernels de fla) y el resumen lo dice en cada fila (`implementacion`).

| Sección | Qué responde | A100, `completo` (medido) | `rapido` |
|---|---|---|---|
| 0. PyTorch 2.14 | Instala PyTorch 2.14.1 si Colab trae otra versión (la sesión se reinicia una vez) | 2–3 min | igual |
| 1. Preparación | Código, librerías originales y una prueba de que cada modelo corre y escribe bien | 13 min | igual |
| 2. Pruebas | Las 55 pruebas del núcleo (con las de `torch.compile`), las 18 de la capa estable y las 7 comprobaciones en GPU | 10 min | 10 min |
| 3. Estabilidad en GPU | ¿El reloj de 32 bits de CUDA se degrada? ¿El kernel `ESTABLE` da lo mismo que PyTorch? ¿La fuga y el reloj corrido se ven en la GPU? Una capa con 1 M de piezas | 1 min | 1 min |
| 4. Una capa | Tiempo y memoria de ida y vuelta de 1 024 a 262 144 piezas | 10 min | 3 min |
| 5. Entrenar ~50 M | Piezas por segundo y memoria con 1 a 32 mil piezas por texto, sin compilar y compilado (ASRN y Transformer++) | 10 min | 10 min |
| 6. Escribir | Milisegundos por pieza nueva con 1 a 64 mil piezas de contexto | 9 min | 3 min |
| 7. Copia | Copiar desde 512 atrás; con textos 4 y 16 veces más largos | 32 min | 8 min |
| 8. MQAR | Recordar 64 pares llave-valor (lo más difícil de Zoology); con 2 y 4 veces más pares | 31 min | 8 min |
| 9. WikiText-103 | Perplejidad con el mismo presupuesto (~45 M de parámetros, 20 M de piezas), con 1 024 y 4 096 | 51 min | 15 min |
| 10. Ablación | ASRN quitando una pieza a la vez, en copia y MQAR | 81 min | 15 min |
| 11. Citas acotadas | ¿La puerta y las señales de repetición enseñan cuándo dejar de copiar? 5 variantes de la capa, 6 000 pasos | 37 min (3 semillas) | 10–20 min (1 semilla) |
| 12. Resumen | Tablas, gráficas y un ZIP para descargar | 1 min | 1 min |
| **Total** | | **≈ 4 h 50 min** (~62 unidades) | **≈ 1 h 45 min** (~23 unidades) |

(Unidades de Colab con una A100 a ~13 por hora; tu panel de Colab muestra la tasa exacta. Los tiempos de `completo` son
los medidos en la corrida del 17 de octubre de 2026 (`results/banco_a100/`, sin contar la sección 0 ni el resumen); los
de `rapido`, estimados.)

**Cómo correrlo**

1. **Entorno de ejecución → Cambiar tipo de entorno de ejecución → A100 GPU** (y "RAM alta" si aparece).
2. Revisa la celda de configuración (`MODO`, `SECCIONES`). **Entorno de ejecución → Ejecutar todas**.
3. La sección 0 instala PyTorch 2.14 y **reinicia la sesión una vez**: cuando Colab vuelva, usa otra vez
   **Ejecutar todas** (la segunda vez ya no se reinicia).
4. Acepta montar Google Drive: cada resultado se guarda ahí al terminar. **Si Colab se desconecta**, vuelve a
   *Ejecutar todas*: lo que ya terminó no se repite.
5. Al final se descarga `asrn_colab.zip`: `resultados.json`, `RESUMEN.md`, las gráficas y la carpeta `citas/` con los
   datos crudos de cada semilla.

**Protocolo** (el mismo de `banco_asrn.py`): mismo ancho y casi los mismos parámetros para todos; mismos lotes en cada
semilla; AdamW (decaimiento 0.1, betas 0.9/0.95), calentamiento 2% y coseno hasta lr/10, recorte en 1, bf16. En cada
tarea, la mejor de dos tasas (1e-3 y 3e-3) con la semilla 0 y esa tasa con las semillas 1 y 2: media ± desviación.
''')

code(r'''
# Configuración
MODO = 'completo'           # 'completo' (~4 h 50 min) o 'rapido' (~1 h 45 min: menos pasos y una semilla)
SECCIONES = ['pruebas', 'estabilidad', 'capa', 'modelo', 'escritura', 'copia', 'mqar', 'wikitext', 'ablacion', 'citas']
PYTORCH = '2.14.1'          # la version de PyTorch de la corrida (>= 2.14: torch.compile de la capa ASRN)
GUARDAR_EN_DRIVE = True     # cada resultado se guarda en Drive al terminar; si Colab se desconecta, se sigue de ahí
CARPETA_DRIVE = 'ASRN/colab_pt214'
MAMBA_COMPILADO = True      # intenta causal-conv1d compilado (el camino fusionado de Mamba-2); si no hay rueda para
                            # esta version de PyTorch y compilarlo tarda mas de 8 min, Mamba-2 usa conv1d de PyTorch
REPO = 'https://github.com/KIROKOWA/ASRN.git'
RAMA = 'main'
# Para CONTINUAR una corrida (los resultados en Drive se conservan; lo ya hecho no se repite):
SEMILLAS = None             # None: las del perfil (3). 1: solo la semilla 0 (con sus dos tasas) de lo que falte
TASAS_EXTRA = {}            # tasas de mas para un modelo en una tarea, p. ej. {'wikitext/Transformer++': (3e-4, 6e-4)}
REHACER = []                # resultados a borrar antes de correr, para rehacerlos: 'seccion', 'seccion/modelo' o
                            # 'seccion/modelo/tarea'; secciones: calidad, ablacion, capa_gpu, modelo_gpu, escritura_gpu
''')

md(r'''
## 0. PyTorch 2.14

`torch.compile` de la capa ASRN necesita PyTorch ≥ 2.14 (con versiones anteriores Inductor falla con este modelo). Si
Colab trae otra versión, esta celda instala PyTorch 2.14.1 y reinicia la sesión; cuando vuelva, **Ejecutar todas** otra
vez. `torchvision` y `torchaudio` de Colab quedan atados a la versión anterior y se desinstalan (este cuaderno no los
usa).
''')

code(r'''
import importlib.metadata, os, subprocess, sys
if 'PYTORCH' not in globals():
    PYTORCH = '2.14.1'
def _version(paquete):
    try:
        return importlib.metadata.version(paquete)
    except importlib.metadata.PackageNotFoundError:
        return None
_actual = _version('torch')
if _actual is None or _actual.split('+')[0] != PYTORCH:
    print(f'PyTorch {_actual} -> {PYTORCH} (unos 2 minutos)...')
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', f'torch=={PYTORCH}'], capture_output=True,
                       text=True)
    if r.returncode != 0:
        raise RuntimeError(f'No se pudo instalar PyTorch {PYTORCH}: {r.stderr[-1500:]}')
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchvision', 'torchaudio'],
                   capture_output=True)
    if 'google.colab' in sys.modules or 'torch' in sys.modules:
        print(f'PyTorch {_version("torch")} instalado. La sesión se reinicia ahora: cuando vuelva, '
              'usa otra vez "Ejecutar todas".')
        os.kill(os.getpid(), 9)
print('PyTorch', _version('torch'), '| Triton', _version('triton'))
''')

md(r'''
## 1. Preparación
''')

code(r'''
import os, subprocess, sys
for _n, _v in dict(MODO='completo', GUARDAR_EN_DRIVE=True, CARPETA_DRIVE='ASRN/colab_pt214', MAMBA_COMPILADO=True,
                   REPO='https://github.com/KIROKOWA/ASRN.git', RAMA='main', SEMILLAS=None,
                   REHACER=[], TASAS_EXTRA={},
                   SECCIONES=['pruebas', 'estabilidad', 'capa', 'modelo', 'escritura', 'copia', 'mqar', 'wikitext',
                              'ablacion', 'citas']).items():
    if _n not in globals():
        globals()[_n] = _v
        print(f'(configuración: {_n} = {_v!r}, el valor por omisión)')
try:
    print(subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True).stdout)
except (OSError, subprocess.CalledProcessError):
    print('No hay GPU: el cuaderno corre en modo mínimo (solo revisa que todo funcione; sus números no sirven).\n'
          'En Colab: Entorno de ejecución → Cambiar tipo de entorno de ejecución → A100 GPU.')

EN_COLAB = 'google.colab' in sys.modules
SALIDA = os.path.abspath('resultados_asrn_colab')
if EN_COLAB and GUARDAR_EN_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        SALIDA = os.path.join('/content/drive/MyDrive', CARPETA_DRIVE)
    except Exception as e:                                      # noqa: BLE001
        print('No se montó Drive (los resultados quedan solo en esta sesión):', e)
os.makedirs(SALIDA, exist_ok=True)
print('Resultados en', SALIDA)
''')

code(r'''
# El código de ASRN: desde GitHub si el repositorio es público; si es privado, desde el ZIP que subas.
import glob, shutil, zipfile
CARPETA = '/content/asrn' if EN_COLAB else os.path.abspath('asrn_codigo')

def _tiene_codigo(carpeta):     # con la organizacion actual (tests/, benchmarks/): una copia anterior se reemplaza
    return all(os.path.exists(os.path.join(carpeta, *p)) for p in (('asrn', 'estable.py'), ('tests',), ('benchmarks',)))

if not _tiene_codigo(CARPETA):
    shutil.rmtree(CARPETA, ignore_errors=True)
    if os.environ.get('ASRN_DIR'):                             # para revisar el cuaderno fuera de Colab
        shutil.copytree(os.environ['ASRN_DIR'], CARPETA, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    else:
        r = subprocess.run(['git', 'clone', '-q', '--depth', '1', '-b', RAMA, REPO, CARPETA], capture_output=True,
                           text=True, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
        if r.returncode != 0:
            if not EN_COLAB:
                raise RuntimeError(f'No se pudo descargar {REPO} ({RAMA}): {r.stderr.strip()}')
            print(f'No se pudo clonar la rama {RAMA} (¿repositorio privado?). Sube el ZIP de esa rama '
                  '(en GitHub: elige la rama, botón verde Code → Download ZIP).')
            from google.colab import files
            subido = files.upload()
            if not subido:
                raise RuntimeError('No se subió ningún archivo')
            shutil.rmtree('/content/zip_asrn', ignore_errors=True)
            with zipfile.ZipFile(next(iter(subido))) as z:
                z.extractall('/content/zip_asrn')
            raices = sorted(glob.glob('/content/zip_asrn/**/asrn/estable.py', recursive=True), key=len)
            if not raices:
                raise RuntimeError('Ese ZIP no tiene asrn/estable.py: descarga el ZIP de la rama ' + RAMA)
            shutil.copytree(os.path.dirname(os.path.dirname(raices[0])), CARPETA)   # raices[0] = <raiz>/asrn/...py
os.chdir(CARPETA)
sys.path[:0] = [CARPETA, os.path.join(CARPETA, 'benchmarks')]
import hashlib
_huella = hashlib.sha256(b''.join(open(f, 'rb').read() for f in sorted(
    glob.glob('asrn/*.py') + glob.glob('*.py') + glob.glob('benchmarks/*.py')))).hexdigest()[:12]   # el codigo exacto, aunque venga de un ZIP
VERSION_CODIGO = ((subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
                   if os.path.isdir('.git') else 'ZIP') + f' | sha256 {_huella}')
print('Código listo en', CARPETA, '|', VERSION_CODIGO)
''')

code(r'''
# Librerías de los originales (PyTorch ya quedó en la sección 0).
import time
def pip(*args, timeout=None, env=None):
    t = time.time()
    try:
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *args], capture_output=True, text=True,
                           timeout=timeout, env={**os.environ, **(env or {})})
        ok, msg = r.returncode == 0, r.stderr[-400:]
    except subprocess.TimeoutExpired:
        ok, msg = False, f'más de {timeout} s'
    print(f"pip {' '.join(args)}: {'listo' if ok else 'FALLÓ: ' + msg} ({time.time() - t:.0f} s)")
    return ok

pip('-r', 'requirements.txt')
pip('flash-linear-attention', 'datasets', 'tiktoken', 'einops', 'ninja', 'packaging', 'wheel', 'matplotlib')
INSTALACION_MAMBA = 'no'
if os.environ.get('ASRN_SIN_MAMBA') != '1':
    # mamba-ssm sin compilar su extension de Mamba-1 (Mamba-2 no la usa: sus kernels SSD son Triton); con sus
    # dependencias, que mamba_ssm importa al cargarse (Mamba-3)
    if pip('mamba-ssm', '--no-build-isolation', env={'MAMBA_SKIP_CUDA_BUILD': 'TRUE'}):
        INSTALACION_MAMBA = 'kernels Triton de Mamba-2'
        # causal-conv1d compilado: el camino fusionado (use_mem_eff_path), el mas rapido. Rueda lista o <= 8 min
        if MAMBA_COMPILADO and pip('causal-conv1d', '--no-build-isolation', timeout=480):
            INSTALACION_MAMBA += ' + causal-conv1d (camino fusionado)'
import importlib
importlib.invalidate_caches()
import modelos_oficiales as mo
print('Mamba-2:', INSTALACION_MAMBA)
for k, v in mo.disponibles().items():
    print(f'  {k:14} {v}')
''')

code(r'''
# Herramientas: perfil del banco, resultados (en Drive) y cómo correr cada sección.
import datetime, json, platform, traceback
import torch
import asrn.nucleo as asrn
import banco_asrn as banco
import banco_gpu as bg

GPU = torch.cuda.is_available()
if GPU:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
MODO_EFECTIVO = MODO if GPU else 'minimo'
AJUSTES = {
    'completo': {},
    'rapido': dict(semillas=1, tareas={'copia': dict(pasos=1000), 'mqar': dict(pasos=1000),
                                       'wikitext': dict(pasos=610)}),
    'minimo': dict(semillas=1, tasas=(3e-3,), d=32, n_capas=2,
                   tareas={'copia': dict(pasos=3, lote=2, evaluar=(1024,)), 'mqar': dict(pasos=3, lote=2, evaluar=(512,)),
                           'wikitext': dict(pasos=3, lote=2, d=64, parametros='4m', evaluar=(1024,))}),
}[MODO_EFECTIVO]
if SEMILLAS is not None:
    AJUSTES = dict(AJUSTES, semillas=SEMILLAS)
if TASAS_EXTRA:
    AJUSTES = dict(AJUSTES, tasas_extra=dict(TASAS_EXTRA))
banco.usar_perfil('a100', **AJUSTES)
print('Modo', MODO_EFECTIVO, '| tareas:', {t: (c['pasos'], c['lote'], c['largo']) for t, c in banco.TAREAS.items()})

ARCHIVO = os.path.join(SALIDA, 'resultados.json')
RES = json.load(open(ARCHIVO)) if os.path.exists(ARCHIVO) else {}
if RES.get('entorno', {}).get('modo') not in (None, MODO_EFECTIVO):
    print(f"OJO: {ARCHIVO} es de otro modo ({RES['entorno']['modo']}); se sigue agregando a ese archivo")
if 'entorno' in RES:    # al continuar una corrida, el JSON mezcla filas de varias ejecuciones: se guardan todas
    RES.setdefault('entornos_anteriores', []).append(RES['entorno'])
RES['entorno'] = {
    'fecha': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
    'gpu': torch.cuda.get_device_name(0) if GPU else 'sin GPU', 'torch': torch.__version__,
    'cuda': torch.version.cuda, 'python': platform.python_version(), 'codigo': VERSION_CODIGO,
    'modo': MODO_EFECTIVO, 'mamba': INSTALACION_MAMBA, 'originales': mo.disponibles(),
    'compila': bg.COMPILA}
RES.setdefault('estado', {})

def _rehacer(pedido):
    """Borra de RES lo que pide 'seccion[/modelo[/tarea]]' para que se vuelva a correr."""
    partes = pedido.split('/')
    seccion, modelo, tarea = (partes + [None, None])[:3]
    sec = RES.get(seccion)
    if not sec:
        return 0
    clave = 'corridas' if 'corridas' in sec else 'filas'
    antes = len(sec[clave])
    sec[clave] = [f for f in sec[clave] if not ((modelo is None or f.get('modelo') == modelo)
                                                and (tarea is None or f.get('tarea') == tarea))]
    if seccion == 'calidad' and modelo:                      # que vuelva a elegir su tasa
        for k in [k for k in sec.get('mejor_lr', {}) if k.endswith('/' + modelo) and (tarea is None or k.startswith(tarea + '/'))]:
            del sec['mejor_lr'][k]
    return antes - len(sec[clave])
for _p in REHACER:
    print(f'REHACER {_p}: {_rehacer(_p)} resultados borrados')

def guardar():
    tmp = ARCHIVO + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(RES, f, indent=1)
    os.replace(tmp, ARCHIVO)

class FallaPruebas(Exception):
    """La sección terminó, pero algo de lo que comprueba no pasó (p. ej. una suite de pruebas con fallas)."""

def seccion(nombre, funcion, siempre=False):
    """Corre una sección (si está en SECCIONES), guarda y anota cuánto tardó o el error, sin detener el cuaderno."""
    if nombre not in SECCIONES and not siempre:
        print(f'(sección {nombre}: no está en SECCIONES)')
        return
    t = time.time()
    try:
        funcion()
        RES['estado'][nombre] = {'ok': True, 'minutos': round((time.time() - t) / 60, 1)}
    except FallaPruebas as e:                                   # sin traceback: el detalle ya se imprimio arriba
        RES['estado'][nombre] = {'ok': False, 'error': str(e), 'minutos': round((time.time() - t) / 60, 1)}
    except Exception as e:                                      # noqa: BLE001
        traceback.print_exc()
        RES['estado'][nombre] = {'ok': False, 'error': f'{type(e).__name__}: {e}', 'minutos': round((time.time() - t) / 60, 1)}
    finally:
        guardar()
        if GPU:
            torch.cuda.empty_cache()
    print(f"[{nombre}: {'listo' if RES['estado'][nombre]['ok'] else 'FALLÓ'} en {RES['estado'][nombre]['minutos']} min]")
guardar()
''')

code(r'''
# Cada original, de verdad en esta GPU: ida y vuelta de un modelo chico, y escribir paso a paso = el texto entero.
# Si uno falla aquí, el resto del cuaderno usa su reimplementación (y lo anota).
def prueba_originales():
    out = {}
    for nombre in ('Transformer++', 'Mamba-2', 'RWKV-7'):
        if not mo.disponibles().get(nombre, 'no').startswith('si'):
            out[nombre] = 'no disponible'
            continue
        try:
            torch.manual_seed(0)
            m = mo.CLASES[nombre](512, 128, 2).to(banco.DISP)
            x = torch.randint(0, 512, (2, 300), device=banco.DISP)
            lg = m(x)
            lg.float().square().mean().backward()
            assert torch.isfinite(lg).all() and all(torch.isfinite(p.grad).all() for p in m.parameters()
                                                    if p.grad is not None)
        except Exception as e:                                  # noqa: BLE001
            out[nombre] = f'FALLÓ al entrenar (se usa la reimplementación): {type(e).__name__}: {e}'
            mo.DESACTIVADOS.add(nombre)
            continue
        try:
            m.eval()
            with torch.no_grad():
                entero = m(x).float()
                estado, l0 = m.prefijo(x[:, :200], 320)
                pasos = [l0] + [m.paso(x[:, t], estado) for t in range(200, 300)]
            dif = ((torch.stack(pasos, 1).float() - entero[:, 199:]).abs().max() / entero.abs().max()).item()
            out[nombre] = f'ok; escribir paso a paso vs el texto entero: {dif:.1e}'
            if not dif < 5e-2:
                out[nombre] += ' (DEMASIADO distinto: no se mide su escritura)'
                mo.SIN_ESCRITURA.add(nombre)
        except Exception as e:                                  # noqa: BLE001
            out[nombre] = f'ok al entrenar; FALLÓ al escribir (no se mide su escritura): {type(e).__name__}: {e}'
            mo.SIN_ESCRITURA.add(nombre)
    RES['prueba_originales'] = out
    for k, v in out.items():
        print(f'  {k:14} {v}')
seccion('originales', prueba_originales, siempre=True)
''')

md(r'''
## 2. Pruebas del repositorio

Las 55 pruebas del núcleo (`tests/test_nucleo.py`, la capa de v89; con PyTorch 2.14 también las dos de
`torch.compile`) y las 18 de la capa estable (`tests/test_estable.py`), en CPU (no necesitan GPU; los kernels Triton se
prueban con el intérprete) y, al final, las 7 comprobaciones en GPU de `tests/verificar_gpu.py` (bf16, checkpoint,
FSDP con NCCL, `torch.compile`, GPU = CPU, causalidad, escribir paso a paso). Si alguna termina con fallas, la sección queda como **FALLÓ** en
el resumen, con cada archivo que falló (los detalles quedan en `resultados.json`, en `pruebas`).
''')

code(r'''
def pruebas():
    out, malas = {}, []
    for archivo in ('tests/test_nucleo.py', 'tests/test_estable.py'):
        t = time.time()
        r = subprocess.run([sys.executable, archivo], capture_output=True, text=True)
        lineas = r.stdout.strip().splitlines()
        out[archivo] = {'resumen': lineas[-1] if lineas else r.stderr[-300:], 'codigo': r.returncode,
                        'fallas': [l for l in lineas if l.startswith('FALLA')], 'minutos': round((time.time() - t) / 60, 1)}
        print(f"{archivo}: {out[archivo]['resumen']} ({out[archivo]['minutos']} min)")
        for l in out[archivo]['fallas']:
            print('   ', l[:300])
        if r.returncode != 0:
            malas.append(f"{archivo} ({' '.join(out[archivo]['resumen'].split())[:200]})")
    if GPU:
        t = time.time()
        r = subprocess.run([sys.executable, 'tests/verificar_gpu.py', '--json', '/tmp/verif.json'],
                           capture_output=True, text=True)
        print(r.stdout[-2500:])
        out['verificar_gpu'] = json.load(open('/tmp/verif.json')) if os.path.exists('/tmp/verif.json') else r.stderr[-500:]
        if r.returncode != 0:
            malas.append(f'tests/verificar_gpu.py (código {r.returncode})')
    RES['pruebas'] = out
    if malas:                   # lo medido ya quedo en RES['pruebas']; la seccion queda como FALLO en el resumen
        raise FallaPruebas('; '.join(malas))
seccion('pruebas', pruebas)
''')

md(r'''
## 3. Las protecciones, en la GPU de verdad

- El reloj B_t = cumsum(β) con el `cumsum` de CUDA en 32 bits, contra 64 bits, de 1 a 32 millones de piezas.
- El kernel Triton en modo `ESTABLE` (reloj en dos partes y máscara exacta) compilado para esta GPU, contra la versión
  de PyTorch: fp32 casi igual; bf16 no más lejos de fp32 que PyTorch en bf16.
- La fuga causal con γ = 10⁹ y el reloj corrido (10⁶, 1.6·10⁷, 10⁹), con el kernel de v89 y con el estable.
- Una capa con 2²⁰ piezas: tiempo, memoria, y cuánto se separan v89 y la estable.
''')

code(r'''
seccion('estabilidad', lambda: bg.seccion_estabilidad_gpu(RES))
''')

md(r'''
## 4. Una capa, textos cada vez más largos

Una capa de mezcla sola (d = 512, lote 1, bf16), ida y vuelta: ASRN (v89 y estable), la atención del Transformer++
(FlashAttention-2), Mamba-2 (`mamba_ssm.Mamba2`) y RWKV-7 (`fla.layers.RWKV7Attention`).
''')

code(r'''
seccion('capa', lambda: bg.seccion_capa_gpu(RES, largos=None if MODO_EFECTIVO == 'completo' else
                                           ((1024, 16384, 262144) if GPU else None)))
''')

md(r'''
## 5. Entrenar modelos de ~50 millones

32 768 piezas por paso (bf16, AdamW fusionado), con textos de 1 024 a 32 768. Con PyTorch 2.14, ASRN (la capa
estable) y el Transformer++ también compilados (`compilar_partes` y `torch.compile`): filas aparte en la tabla.
''')

code(r'''
seccion('modelo', lambda: bg.seccion_modelo_gpu(RES, compilar=GPU))
''')

md(r'''
## 6. Escribir

Lote 1, después de un prefijo de 1 024 a 65 536 piezas: cuánto tarda el prefijo y cada pieza nueva. ASRN con su
escritura actual (O(t) por pieza, ver la sección 2.4 del informe), el Transformer++ con caché de llaves, Mamba-2 y
RWKV-7 con su estado.
''')

code(r'''
seccion('escritura', lambda: bg.seccion_escritura_gpu(RES))
''')

md(r'''
## 7–9. Calidad: copia, MQAR y WikiText-103

Cada modelo con la mejor de dos tasas (semilla 0) y esa tasa con las semillas 1 y 2. Cada corrida se guarda al
terminar: si Colab se corta, al volver a ejecutar sigue con la siguiente.
''')

code(r'''
seccion('copia', lambda: banco.seccion_calidad(RES, guardar, tareas=('copia',)))
''')

code(r'''
seccion('mqar', lambda: banco.seccion_calidad(RES, guardar, tareas=('mqar',)))
''')

code(r'''
def wikitext():
    t = time.time()
    c = banco.TAREAS['wikitext']
    banco.corpus_wikitext(c['pasos'] * c['lote'] * c['largo'])
    print(f"WikiText-103 listo en {time.time() - t:.0f} s ({banco._WIKI['entrenar'].numel():,} piezas de entrenamiento, "
          f"{banco._WIKI['validar'].numel():,} de validación)")
    banco.seccion_calidad(RES, guardar, tareas=('wikitext',))
seccion('wikitext', wikitext)
''')

md(r'''
## 10. Ablación

ASRN (capa estable, grupo A) quitando una pieza a la vez, con su mejor tasa: copia y MQAR. `estable_AB` agrega el
grupo B (tope de la puerta y piso de κ); `mezcla_lse` prueba la propuesta de pesar las tablas por su logsumexp.
''')

code(r'''
seccion('ablacion', lambda: banco.seccion_ablacion(RES, guardar, tareas=('copia', 'mqar')))
''')

md(r'''
## 11. El experimento de citas acotadas

¿La puerta y las señales de repetición (Hawkes y racha) le enseñan al modelo cuándo dejar de copiar? Texto sintético con
citas de 6 a 12 piezas; la capa completa de 1.0 contra la misma sin las señales de Hawkes y de racha, con una puerta que
solo ve el estado, sin puerta y sin el reloj
([`benchmarks/experimento_citas.py`](https://github.com/KIROKOWA/ASRN/blob/main/benchmarks/experimento_citas.py); la
corrida en CPU está en `results/citas/`). Aquí corre en la GPU. Cada semilla se guarda al terminar cada modelo y se retoma si Colab
se corta.
''')

code(r'''
def citas():
    import experimento_citas as ec
    carpeta = os.path.join(SALIDA, 'citas')
    pasos = 20 if MODO_EFECTIVO == 'minimo' else 6000
    semillas = (0, 1, 2) if MODO_EFECTIVO == 'completo' else (0,)
    for s in semillas:
        ec.correr_semilla(s, pasos, carpeta, dispositivo=banco.DISP)
    ec.resumir(carpeta)
    RES['citas'] = json.load(open(os.path.join(carpeta, 'resultados.json')))
seccion('citas', citas)
''')

md(r'''
## 12. Resumen
''')

code(r'''
from IPython.display import Image, Markdown, display
_citas = os.path.join(SALIDA, 'citas', 'RESUMEN.md')
texto = '\n\n'.join(t for t in (bg.resumir_gpu(RES), banco.resumir(RES),
                                 open(_citas).read() if os.path.exists(_citas) else '') if t)
estado = '\n'.join(f"- {k}: {'listo' if v['ok'] else 'FALLÓ — ' + v.get('error', '')} ({v['minutos']} min)"
                   for k, v in RES['estado'].items())
texto = (f"## ASRN 1.0 en {RES['entorno']['gpu']} con PyTorch {RES['entorno']['torch']} ({RES['entorno']['fecha']}, "
         f"modo {MODO_EFECTIVO})\n\n{estado}\n\n{texto}")
with open(os.path.join(SALIDA, 'RESUMEN.md'), 'w') as f:
    f.write(texto + '\n')
display(Markdown(texto))
for ruta in bg.graficas(RES, SALIDA):
    display(Image(ruta))
guardar()
archivo_zip = shutil.make_archive('/tmp/asrn_colab', 'zip', SALIDA)
print('Todo en', SALIDA, '| ZIP:', archivo_zip)
if EN_COLAB:
    from google.colab import files
    files.download(archivo_zip)
''')


def construir(ruta):
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        'accelerator': 'GPU',
        'colab': {'gpuType': 'A100', 'machine_shape': 'hm', 'provenance': [], 'name': 'asrn_colab.ipynb'},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        'language_info': {'name': 'python'}}
    nb.cells = [nbformat.v4.new_markdown_cell(t) if k == 'md' else nbformat.v4.new_code_cell(t) for k, t in CELDAS]
    for i, c in enumerate(nb.cells):        # identificadores fijos: el mismo programa da el mismo archivo, byte a byte
        c.id = f'asrn-{i}'
    nbformat.validate(nb)
    nbformat.write(nb, ruta)
    print(f'{ruta}: {len(nb.cells)} celdas')


if __name__ == '__main__':
    construir(sys.argv[1] if len(sys.argv) > 1 else SALIDA_POR_OMISION)
