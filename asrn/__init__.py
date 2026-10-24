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
"""ASRN 1.0 -- Adaptive Sparse Recurrence Network.

El paquete de produccion. Junta dos modulos:
  - asrn.nucleo   la capa ASRN, sus kernels Triton (con el modo ESTABLE), el modelo, entrenar(), escribir(),
                  guardar()/cargar(), FSDP y la linea de comandos (es el codigo de v89)
  - asrn.estable  la capa con las protecciones numericas (mascara exacta, reloj de 64 bits, topes) y el recorte
                  dinamico del gradiente

Lo que se importa de aqui usa la capa estable por omision: crear_modelo(), crear_desde_config() y cargar() devuelven
un ModeloASRN cuyas capas ASRN son CapaASRNEstable con el grupo A de protecciones (mismos parametros y mismas cuentas
que la capa del nucleo salvo el redondeo; los archivos guardados con cualquiera de las dos cargan en la otra).
protecciones=Protecciones.textos_largos(L) agrega el grupo B; protecciones=Protecciones.ninguna() da los mismos bits
que la capa del nucleo.

Uso:
    import asrn
    modelo = asrn.crear_modelo(V=8000, d=256, n_capas=8)
    asrn.entrenar(modelo, piezas, validacion=piezas_val, guardar_en='modelo.pt',
                  recorte=asrn.RecorteDinamico(modelo))
    texto = asrn.escribir(asrn.cargar('modelo.pt'), inicio, n_nuevos=200)

Linea de comandos:
    python -m asrn --datos piezas.pt --V 8000 --guardar modelo.pt
    torchrun --nproc_per_node=8 -m asrn --fsdp --talla 1b ...
"""
from . import estable, nucleo
from .estable import CapaASRNEstable, Protecciones, RecorteDinamico, convertir
from .nucleo import (TALLAS, BloqueLocal, CapaASRN, ConfigASRN, ModeloASRN, PasoGrafado, compilar_partes,
                     crear_optimizador, entrenar, escaneo, escribir, esta_repartido, evaluar, firmas_de_piezas,
                     guardar, inicializar, leer_cantidad, leer_piezas, materializar, pesos_completos, repartir_fsdp,
                     tasa_aprendizaje)

__version__ = '1.0'

__all__ = ['CapaASRN', 'CapaASRNEstable', 'BloqueLocal', 'ModeloASRN', 'ConfigASRN', 'TALLAS', 'Protecciones',
           'RecorteDinamico', 'PasoGrafado', 'crear_modelo', 'crear_desde_config', 'cargar', 'guardar', 'convertir',
           'inicializar', 'materializar', 'repartir_fsdp', 'esta_repartido', 'pesos_completos', 'compilar_partes',
           'crear_optimizador', 'tasa_aprendizaje', 'entrenar', 'evaluar', 'escribir', 'leer_piezas',
           'leer_cantidad', 'firmas_de_piezas', 'escaneo', 'nucleo', 'estable', '__version__',
           # English aliases (same objects; see the end of this file)
           'ASRNLayer', 'ASRNCoreLayer', 'ASRNModel', 'ASRNConfig', 'LocalBlock', 'Protections', 'DynamicClip',
           'create_model', 'create_from_config', 'convert', 'train', 'evaluate', 'generate', 'save', 'load']


def crear_modelo(V, d=64, n_capas=2, largo_cada=2, semilla=None, protecciones=None, **kw):
    """ModeloASRN con la receta final y la capa estable. Los mismos argumentos que nucleo.crear_modelo (los pesos son
    los mismos para la misma semilla); protecciones: Protecciones (por omision, el grupo A)."""
    modelo = nucleo.crear_modelo(V, d=d, n_capas=n_capas, largo_cada=largo_cada, semilla=semilla, **kw)
    return convertir(modelo, protecciones)


def crear_desde_config(cfg, semilla=None, dispositivo=None, protecciones=None):
    """ModeloASRN desde una ConfigASRN (o el nombre de una de TALLAS), con la capa estable. dispositivo='meta': solo
    formas (para llenarlo despues con inicializar() o materializar(), por ejemplo ya repartido con repartir_fsdp())."""
    return convertir(nucleo.crear_desde_config(cfg, semilla=semilla, dispositivo=dispositivo), protecciones)


def cargar(archivo, dispositivo=None, confiar=False, protecciones=None):
    """Carga un modelo guardado con guardar() o entrenar() (de v83 en adelante), con la capa estable."""
    return convertir(nucleo.cargar(archivo, dispositivo=dispositivo, confiar=confiar), protecciones)


# ----------------------------------------------------------------------------------------------------------------
# English aliases: the same objects under English names, so the package can be used without the Spanish identifiers.
# They are plain assignments (asrn.train is asrn.entrenar); arguments and keywords keep their Spanish names.
# ----------------------------------------------------------------------------------------------------------------

ASRNLayer = CapaASRNEstable           # the ASRN layer with group-A protections (the default layer of the package)
ASRNCoreLayer = CapaASRN              # the unprotected core layer (v89 numerics)
ASRNModel = ModeloASRN
ASRNConfig = ConfigASRN
LocalBlock = BloqueLocal
Protections = Protecciones
DynamicClip = RecorteDinamico
create_model = crear_modelo
create_from_config = crear_desde_config
convert = convertir
train = entrenar
evaluate = evaluar
generate = escribir
save = guardar
load = cargar
