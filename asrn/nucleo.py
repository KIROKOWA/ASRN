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
"""ASRN 1.0 (v89) -- Adaptive Sparse Recurrence Network.

Modelo de lenguaje con la capa ASRN y lo necesario para entrenarlo y hacerlo escribir:
  - CapaASRN, ModeloASRN, crear_modelo: la capa y el modelo (firmas por numero de pieza: exactas en cualquier
    dispositivo y tamano de lote)
  - ConfigASRN y TALLAS: todo el modelo descrito en un solo objeto (de ~3 mil a ~27 mil millones de parametros)
  - entrenar(): AdamW, calentamiento + coseno, bf16/fp16, acumulacion, recorte, salto de pasos con NaN, validacion,
    guardado del mejor, reanudar exacto, Ctrl+C guarda; en una GPU o repartido entre varias (FSDP). evaluar()
  - escribir(): escritura rapida con "memoria de lo ajeno"; top_k, top_p, frenos opcionales
  - guardar() / cargar(): guardado atomico, carga segura; carga modelos de v83/v84/v85/v86/v87 (dan los mismos numeros)
  - linea de comandos: python -m asrn --datos piezas.pt --V 8000 --guardar modelo.pt
                       torchrun --nproc_per_node=8 -m asrn --fsdp --talla 1b ...   (varias GPUs)
                       (python -m asrn usa la capa estable; python -m asrn.nucleo, la capa sin protecciones)

Lo de v86/v87 (todo arranca neutro: con los pesos de v85, v89 calcula lo mismo):
  - reloj de tiempo propio por pieza (beta >= 0) y nucleo continuo alfa_s exp(-gamma (B_t - B_s)) en la busqueda
  - intensidad de Hawkes de cada grupo de 1..vistos palabras y racha de repeticion en la puerta, con un escaneo
    asociativo paralelo O(N) (escaneo()); la respuesta a la racha se aprende por incrementos (d_racha)
  - flujo de tensores de bajo consumo: ~45% menos memoria guardada para la vuelta que v85, O(N)
  - escribir(conservar_cita=True); torch.compile ve el modelo como un solo grafo

v89 (las cuentas del modelo no cambian):
  - ConfigASRN.para_parametros('50m', V=...): arma la configuracion del tamano que quieras (3k, 50m, 1.5b, 7e9...),
    con lo que fijes (d, n_capas, ...) respetado; en la linea de comandos --parametros 50m. TALLAS quedan solo como
    ejemplos (la de ~12 millones se llama ahora '12m', mismo modelo)

Lo nuevo de v88 / ASRN 1.0 (mismas cuentas que v87: los modelos de v87 cargan y dan los mismos numeros):
  - ConfigASRN: un solo objeto con todo lo que define al modelo (validar(), contar_parametros() sin gastar memoria,
    a_dict()/desde_dict()) y TALLAS con configuraciones listas, de 'nano-3k' a '27b'.
    crear_desde_config(cfg, semilla) = crear_modelo con esa configuracion.
  - checkpoint='asrn' (lo de v87: recalcula solo las capas ASRN en la vuelta) o 'todo' (tambien los bloques
    locales): menos memoria, mismas cuentas
  - inicializar(modelo, semilla): llena los pesos de un modelo creado vacio (dispositivo 'meta') o ya repartido,
    con exactamente los mismos numeros que crear_modelo(semilla=...). Asi un modelo gigante nunca existe entero en
    una sola maquina
  - repartir_fsdp(modelo): reparte los pesos entre procesos (FSDP2: cada bloque local y cada capa ASRN por separado).
    entrenar() lo detecta: cada proceso toma lotes distintos, la perdida y la decision de saltar un paso son comunes,
    solo el proceso 0 escribe; guardar() junta los pesos completos y un archivo guardado en 1 GPU sirve para seguir
    en varias, y al reves
  - linea de comandos: --talla, --checkpoint {no,asrn,todo}, --fsdp

Uso:
    import asrn.nucleo as asrn       # el nucleo; el paquete `asrn` da lo mismo con la capa estable por omision
    modelo = asrn.crear_modelo(V=8000, d=64, n_capas=2)          # modelo de lenguaje con ASRN
    cfg = asrn.TALLAS['125m'].con(V=8000); print(cfg.contar_parametros())
    modelo = asrn.crear_desde_config(cfg, semilla=0)            # el mismo modelo desde una configuracion
    logits = modelo(x)                                            # x: (lote, largo) numeros de pieza
    asrn.entrenar(modelo, piezas, validacion=piezas_val, guardar_en='modelo.pt')
    texto, _ = asrn.escribir(modelo, inicio, n_nuevos=200)        # escribe con "memoria de lo ajeno"
    capa = asrn.CapaASRN(d)                                       # la capa sola: h = capa(E, h[, piezas])
    asrn.guardar(modelo, 'modelo.pt'); modelo = asrn.cargar('modelo.pt')
"""
import dataclasses
import math
import numbers
import os
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

try:
    import triton
    import triton.language as tl
except ImportError:                                 # sin Triton todo funciona igual, con PyTorch puro
    triton = None

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

_PROHIBIDO = -1e9


def _prohibido(dtype):
    """Puntaje de 'no permitido'. -1e9 como en v83; en fp16 (que no llega a 1e9) el minimo que cabe."""
    return max(_PROHIBIDO, torch.finfo(dtype).min)


# ----------------------------------------------------------------------------------------------------------------
# Piezas chicas
# ----------------------------------------------------------------------------------------------------------------

def _largo_seguido(a, b):
    """Cuantas piezas seguidas coinciden, empezando por la primera (a y b: (..., K), se difunden). Recorre las K
    columnas (K = sufijo, 4) con un solo tensor booleano vivo: menos memoria pico que un cumprod de (..., K).
    Cuenta en enteros de 1 byte (si K < 256): el resultado tiene el tamano de los puntajes (ventanas x C x 2C), y en
    64 bits era ~7% de todos los bytes que mueve la capa."""
    sigue = a[..., 0] == b[..., 0]
    largo = sigue.to(torch.uint8 if a.shape[-1] < 256 else torch.int32)
    for j in range(1, a.shape[-1]):
        sigue = sigue & (a[..., j] == b[..., j])
        largo = largo + sigue
    return largo


def _indice(largo):
    """Enteros chicos (uint8, int16) como indice: w[x] con uint8 seria una mascara, no un indice."""
    return largo if largo.dtype in (torch.int32, torch.int64) else largo.int()


class _DesordenarSumar(torch.autograd.Function):
    """Salida de las ventanas (W*C, k), en el orden por casillero de cada tabla -> (B, L, k): de vuelta al orden del
    texto y sumada sobre las T tablas. fila (B*T*L,): la fila de cada (lote, tabla, posicion); fuente (W*C,): la
    posicion del texto (b*L + t) de cada fila, o B*L si es relleno. Ida: index_select y suma, como siempre. Vuelta:
    cada fila recibe el gradiente de su posicion (lo mismo que 'ordenar'), sin pasar por un tensor (B, T, L, k) ni
    sumas atomicas: menos bytes y siempre el mismo resultado."""

    @staticmethod
    def forward(ctx, x, fila, fuente, B, T, L):
        ctx.save_for_backward(fuente)
        return x.index_select(0, fila).view(B, T, L, x.shape[-1]).sum(1)

    @staticmethod
    def backward(ctx, g):
        (fuente,) = ctx.saved_tensors
        k = g.shape[-1]
        g = torch.cat([g.reshape(-1, k), g.new_zeros(1, k)])                  # + una fila de ceros para el relleno
        return g.index_select(0, fuente), None, None, None, None, None


class _Tomar(torch.autograd.Function):
    """w[i] para un w chico (el peso del sufijo, 5 numeros; la respuesta a la racha, 16) y millones de posiciones i.
    La vuelta suma el gradiente de las posiciones de cada entrada: una reduccion por entrada (n pasadas). La vuelta
    de PyTorch (y antes index_add_) suma con operaciones atomicas sobre esas pocas direcciones, que la GPU hace una
    por una: en una A100 era la mitad del tiempo de la capa entera. Forma fija (n), asi torch.compile no corta el
    grafo como con bincount, y siempre el mismo resultado."""

    @staticmethod
    def forward(ctx, i, w):
        ctx.save_for_backward(i)                       # tal como llega (1 byte por posicion desde _largo_seguido)
        ctx.n, ctx.tipo = w.shape[0], w.dtype
        return w[_indice(i)]

    @staticmethod
    def backward(ctx, g):
        (i,) = ctx.saved_tensors
        g = g.float()
        if i.numel() * ctx.n <= 2 ** 22:               # pocas posiciones (la racha): todas las entradas de una vez
            k = torch.arange(ctx.n, device=g.device)
            gw = torch.where(i.unsqueeze(-1) == k, g.unsqueeze(-1), 0.0).reshape(-1, ctx.n).sum(0)
        else:                                          # millones (el sufijo): una pasada por entrada, sin copias
            gw = torch.stack([torch.where(i == k, g, 0.0).sum() for k in range(ctx.n)])
        return None, gw.to(ctx.tipo)


def firmas_de_piezas(piezas):
    """Firma de cada pieza para la coincidencia exacta y los 'vistos': su propio numero. Solo importa que piezas
    iguales den firmas iguales y distintas den distintas, y asi es exacto en cualquier dispositivo y tamano de lote."""
    return piezas.to(torch.float32)


def _anteriores(sig, n, desde=1):
    """(B, L, n): para cada posicion t, las firmas de t-desde, t-desde-1, ..., t-desde-n+1 (NaN antes del inicio).
    Una ventana deslizante (unfold) sobre el texto con relleno: sin bucles ni copias por desplazamiento."""
    L = sig.shape[1]
    previo = n + desde - 1
    return F.pad(sig, (previo, 0), value=float('nan')).unfold(1, n, 1)[:, :L].flip(-1)


def _claves_de(sig, c, n):
    """(B, L, n) claves (64 bits) de los grupos de 1..n piezas que terminan en cada posicion: sig (B, L) firmas, c (n,)
    coeficientes de 64 bits en el mismo dispositivo. Las mismas operaciones, en el mismo orden, en CPU y en GPU: la
    escritura calcula en la CPU las claves de las piezas nuevas y salen los mismos bits que en el texto entero."""
    with torch.autocast(device_type=sig.device.type, enabled=False):
        previas = _anteriores(sig.double(), n, desde=0)                      # (B, L, n): t, t-1, ...
        clave, claves = torch.zeros_like(previas[..., 0]), []
        for k in range(n):
            clave = clave + previas[..., k] * c[k]
            claves.append(clave)
        return torch.stack(claves, -1)


def escaneo(a, b):
    """S_i = a_i * S_{i-1} + b_i sobre la ultima dimension. Con gradiente, la vuelta es otro escaneo, de derecha a
    izquierda (_Escaneo). En GPU con Triton, la ida y la vuelta son un kernel cada una (_escaneo_triton); si no, el
    arbol par-impar de PyTorch (_escaneo_ida), ~140 operaciones por escaneo con 1 024 piezas."""
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        return _Escaneo.apply(*torch.broadcast_tensors(a, b))
    if _usar_triton(a):
        return _escaneo_triton(*torch.broadcast_tensors(a, b))
    return _escaneo_ida(a, b)


def _usar_triton(x):
    """Triton en GPU (o en CPU con su interprete, TRITON_INTERPRET=1, para probarlo), en 16 o 32 bits y fuera de
    torch.compile (que arma sus propios kernels). ASRN_TRITON=0 lo apaga."""
    return (triton is not None and os.environ.get('ASRN_TRITON', '1') != '0'
            and (x.is_cuda or os.environ.get('TRITON_INTERPRET') == '1')
            and x.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and not torch.compiler.is_compiling())


if triton is not None:
    @triton.jit
    def _componer(a1, b1, a2, b2):
        """Dos pasos seguidos de S -> a S + b son un solo paso: (a1, b1) y luego (a2, b2) = (a1 a2, a2 b1 + b2)."""
        return a1 * a2, a2 * b1 + b2

    @triton.jit
    def _escaneo_kernel(A, B, S, SI, GA, N, VUELTA: tl.constexpr, BLOQUE: tl.constexpr):
        """Una fila por programa, en bloques de BLOQUE elementos con lo acumulado del bloque anterior (acarreo);
        dentro del bloque, tl.associative_scan en la SRAM. Las cuentas en 32 bits.
          ida (VUELTA=False):  S_i = A_i S_{i-1} + B_i, con A = a y B = b
          vuelta (VUELTA=True): de derecha a izquierda, l_i = a_{i+1} l_{i+1} + g_i (B = g): escribe dL/db = l en S
                                y dL/da_i = l_i S_{i-1} en GA (SI = S de la ida)"""
        base = tl.program_id(0).to(tl.int64) * N
        pos = tl.arange(0, BLOQUE)
        acarreo = tl.sum(tl.zeros([BLOQUE], tl.float32), 0)
        for inicio in range(0, N, BLOQUE):
            m = inicio + pos < N
            if VUELTA:
                p = N - 1 - (inicio + pos)
                x = tl.load(A + base + p + 1, mask=m & (p + 1 < N), other=0.0).to(tl.float32)
                x = tl.where(m, x, 1.0)
                s_previo = tl.load(SI + base + p - 1, mask=m & (p >= 1), other=0.0).to(tl.float32)
            else:
                p = inicio + pos
                x = tl.load(A + base + p, mask=m, other=1.0).to(tl.float32)
            y = tl.load(B + base + p, mask=m, other=0.0).to(tl.float32)
            pa, pb = tl.associative_scan((x, y), 0, _componer)
            s = pa * acarreo + pb
            if VUELTA:
                tl.store(GA + base + p, s * s_previo, mask=m)
            tl.store(S + base + p, s, mask=m)
            acarreo = tl.sum(tl.where(pos == BLOQUE - 1, s, 0.0), 0)


def _escaneo_triton(a, b, vuelta=False, S_ida=None):
    """Ida: S. Vuelta (b = g, S_ida = S de la ida): (dL/da, dL/db). Todo (..., N), del tipo de a."""
    N = a.shape[-1]
    forma, tipo = a.shape, a.dtype
    a2, b2 = a.reshape(-1, N).contiguous(), b.reshape(-1, N).contiguous()
    filas = a2.shape[0]
    S = torch.empty((filas, N), dtype=torch.float32, device=a.device)
    SI = S_ida.reshape(-1, N).contiguous() if vuelta else S
    GA = torch.empty_like(S) if vuelta else S
    bloque = min(1024, triton.next_power_of_2(max(N, 16)))
    if filas and N:
        _escaneo_kernel[(filas,)](a2, b2, S, SI, GA, N, VUELTA=vuelta, BLOQUE=bloque)
    if vuelta:
        return GA.view(forma).to(tipo), S.view(forma).to(tipo)
    return S.view(forma).to(tipo)


# ----------------------------------------------------------------------------------------------------------------
# Atencion por casilleros fusionada (Triton): lo mismo que CapaASRN._acumular, sin escribir en la memoria de la GPU
# las copias ordenadas ni los puntajes. La version de PyTorch (_acumular) es la referencia y el respaldo.
# ----------------------------------------------------------------------------------------------------------------

if triton is not None:
    @triton.jit
    def _punto(a, b, IEEE: tl.constexpr, A32: tl.constexpr):
        """tl.dot con acumulacion en 32 bits. IEEE: entradas de 32 bits sin TF32 (las mismas cuentas que PyTorch
        en fp32). A32: pasa las entradas a 32 bits antes (solo para el interprete de Triton, que multiplica mal
        en bf16)."""
        if A32:
            c = tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="ieee")
        elif IEEE:
            c = tl.dot(a, b, input_precision="ieee")
        else:
            c = tl.dot(a, b)
        return c

    @triton.jit
    def _cas_puntajes(PHI, QC, KC, PF, BT_, BLO, LA, ORDEN, IDS, W, GAMMA, r, j, L, T,
                      P, CF, escala_phi, escala_c,
                      C: tl.constexpr, PP: tl.constexpr, CFP: tl.constexpr, K: tl.constexpr, IEEE: tl.constexpr,
                      A32: tl.constexpr, ESTABLE: tl.constexpr):
        """Puntajes del bloque j (C preguntas) de la fila r = b*T + tabla contra sus 2C llaves (el bloque anterior
        y el actual en el orden por casillero), con todo lo que hace _acumular: phi, confirmacion, sufijo, nucleo
        de tiempo y mascara (mismo casillero, entre las C anteriores, llave real).
        ESTABLE (asrn/estable.py): el reloj llega en dos partes de 32 bits, BT_ (alta) y BLO (baja), de un reloj de 64
        bits, y el nucleo de tiempo usa la diferencia B_k - B_q tomada ANTES de multiplicar por la tasa (sin
        cancelar dos numeros grandes). Sin ESTABLE, BLO no se lee y todo es como siempre."""
        b = r // T
        tau = r % T
        iq = tl.arange(0, C)
        ik = tl.arange(0, 2 * C)
        s = j * C + iq                                       # posicion ordenada de cada pregunta
        t = j * C - C + ik                                   # y de cada llave
        qm = s < L
        km = (t >= 0) & (t < L)
        qpos = tl.load(ORDEN + r.to(tl.int64) * L + s, mask=qm, other=0)
        kpos = tl.load(ORDEN + r.to(tl.int64) * L + t, mask=km, other=0)
        qid = tl.load(IDS + r.to(tl.int64) * L + s, mask=qm, other=-1)
        kid = tl.load(IDS + r.to(tl.int64) * L + t, mask=km, other=-2)
        filq = b.to(tl.int64) * L + qpos                     # fila de cada una en (B*L, ...)
        filk = b.to(tl.int64) * L + kpos
        pc = tl.arange(0, PP)
        phi_q = tl.load(PHI + filq[:, None] * P + pc[None, :], mask=qm[:, None] & (pc < P)[None, :], other=0.0)
        phi_k = tl.load(PHI + filk[:, None] * P + pc[None, :], mask=km[:, None] & (pc < P)[None, :], other=0.0)
        cc = tl.arange(0, CFP)
        qc = tl.load(QC + filq[:, None] * CF + cc[None, :], mask=qm[:, None] & (cc < CF)[None, :], other=0.0)
        kc = tl.load(KC + filk[:, None] * CF + cc[None, :], mask=km[:, None] & (cc < CF)[None, :], other=0.0)
        s1 = _punto(phi_q, tl.trans(phi_k), IEEE, A32)
        s2 = _punto(qc, tl.trans(kc), IEEE, A32)
        sc = s1 * escala_phi + s2 * escala_c                 # (C, 2C)
        # largo del sufijo comun (piezas anteriores iguales seguidas) y su peso w_sufijo[largo]
        sigue = qm[:, None] & km[None, :]
        largo = tl.zeros([C, 2 * C], tl.int32)
        for c in tl.static_range(K):
            pq = tl.load(PF + filq * K + c, mask=qm, other=0.0)
            pk = tl.load(PF + filk * K + c, mask=km, other=0.0)
            sigue = sigue & (pq[:, None] == pk[None, :])
            largo += sigue.to(tl.int32)
        peso = tl.zeros([C, 2 * C], tl.float32) + tl.load(W)
        for c in tl.static_range(K):
            peso = tl.where(largo == c + 1, tl.load(W + c + 1), peso)
        tasa = tl.load(GAMMA + tau) / C
        bq = tl.load(BT_ + filq, mask=qm, other=0.0)
        bk = tl.load(BT_ + filk, mask=km, other=0.0)
        lak = tl.load(LA + filk, mask=km, other=0.0)
        if ESTABLE:
            # la parte alta se resta exacta entre piezas cercanas (Sterbenz) y la baja corrige el redondeo
            bq_bajo = tl.load(BLO + filq, mask=qm, other=0.0)
            bk_bajo = tl.load(BLO + filk, mask=km, other=0.0)
            dt = (bk[None, :] - bq[:, None]) + (bk_bajo[None, :] - bq_bajo[:, None])
            sc = sc + peso + lak[None, :] + tasa * dt
        else:
            dt = bk[None, :] - bq[:, None]
            sc = sc + peso + (lak + tasa * bk)[None, :] - (tasa * bq)[:, None]
        valido = ((ik[None, :] >= iq[:, None]) & (ik[None, :] < C + iq[:, None]) & (qid[:, None] == kid[None, :])
                  & km[None, :] & qm[:, None])
        sc = tl.where(valido, sc, -1e9)
        return (sc, valido, largo, qm, km, filq, filk, phi_q, phi_k, qc, kc, bq, bk, tasa, qpos, tau, dt)

    @triton.jit
    def _cas_softmax(sc, valido, ESTABLE: tl.constexpr):
        """softmax(sc) * hay, como _acumular (sin llaves validas: todo 0). Devuelve tambien hay y el maximo.
        ESTABLE: el maximo se toma solo entre las llaves validas y las no validas pesan 0 exacto, por bajo que sea el
        puntaje de las validas. Sin eso, si todas las validas quedan bajo -1e9 (nucleo de tiempo enorme), la masa
        se va a las no validas: la propia pregunta y piezas posteriores (el futuro)."""
        hay = tl.max(valido.to(tl.int32), 1) > 0
        if ESTABLE:
            m = tl.where(hay, tl.max(tl.where(valido, sc, float('-inf')), 1), 0.0)
            e = tl.where(valido, tl.exp(sc - m[:, None]), 0.0)
            att = e / tl.where(hay, tl.sum(e, 1), 1.0)[:, None]
        else:
            m = tl.max(sc, 1)
            e = tl.exp(sc - m[:, None])
            att = e / tl.sum(e, 1)[:, None] * hay.to(tl.float32)[:, None]
        return att, hay, m

    @triton.jit
    def _casilleros_ida(PHI, QC, KC, V, PF, BT_, BLO, LA, ORDEN, IDS, W, GAMMA, OUT, SEN, L, T, P, CF, D,
                        escala_phi, escala_c, C: tl.constexpr, PP: tl.constexpr, CFP: tl.constexpr,
                        BD: tl.constexpr, K: tl.constexpr, IEEE: tl.constexpr,
                        A32: tl.constexpr, ESTABLE: tl.constexpr):
        """Un programa por (fila r, bloque j): salida de cada pregunta, escrita en su posicion del texto en
        OUT (B*T, L, D), y las senales (hay, maximo, pico) en SEN (B*T, L, 3)."""
        r = tl.program_id(0)
        j = tl.program_id(1)
        (sc, valido, largo, qm, km, filq, filk, phi_q, phi_k, qc, kc, bq, bk, tasa, qpos,
         tau, dt) = _cas_puntajes(PHI, QC, KC, PF, BT_, BLO, LA, ORDEN, IDS, W, GAMMA, r, j, L, T, P, CF, escala_phi,
                                  escala_c, C, PP, CFP, K, IEEE, A32, ESTABLE)
        att, hay, m = _cas_softmax(sc, valido, ESTABLE)
        salida = r.to(tl.int64) * L + qpos                   # (C,) fila de cada pregunta en OUT
        dc = tl.arange(0, BD)
        for d0 in range(0, D, BD):
            dm = d0 + dc < D
            v = tl.load(V + filk[:, None] * D + d0 + dc[None, :], mask=km[:, None] & dm[None, :], other=0.0)
            o = _punto(att.to(v.dtype), v, IEEE, A32)
            tl.store(OUT + salida[:, None] * D + d0 + dc[None, :], o.to(OUT.dtype.element_ty),
                     mask=qm[:, None] & dm[None, :])
        tl.store(SEN + salida * 3, hay.to(tl.float32), mask=qm)
        tl.store(SEN + salida * 3 + 1, m, mask=qm)
        tl.store(SEN + salida * 3 + 2, tl.max(att, 1), mask=qm)

    @triton.jit
    def _casilleros_vuelta(PHI, QC, KC, V, PF, BT_, BLO, LA, ORDEN, IDS, W, GAMMA, DO,
                           GPHI_Q, GQC, GB_Q, GPHI_K, GKC, GV, GLA, GB_K, PARTES, L, T, BT, P, CF, D, NB,
                           escala_phi, escala_c, C: tl.constexpr, PP: tl.constexpr, CFP: tl.constexpr,
                           BD: tl.constexpr, K: tl.constexpr, IEEE: tl.constexpr, A32: tl.constexpr,
                           ESTABLE: tl.constexpr):
        """La vuelta del mismo bloque, recalculando los puntajes (como FlashAttention: nada guardado de la ida).
        DO (B, L, D): gradiente de la salida sumada sobre las tablas, el mismo para todas.

        Sin atomicas: cada contribucion tiene su propio lugar y se suma despues, en orden. Cada posicion es pregunta
        una vez por tabla: G*_Q (B*T, L, .). Cada llave aparece en dos ventanas de su tabla: como bloque actual
        (lugar 1, la escribe el programa de su bloque) y como bloque anterior (lugar 0, la escribe el programa del
        bloque siguiente; si no hay siguiente, el ultimo programa escribe ceros): G*_K (2, B*T, L, .). Lo de gamma y
        w_sufijo va a PARTES, un renglon por programa."""
        r = tl.program_id(0)
        j = tl.program_id(1)
        (sc, valido, largo, qm, km, filq, filk, phi_q, phi_k, qc, kc, bq, bk, tasa, qpos,
         tau, dt) = _cas_puntajes(PHI, QC, KC, PF, BT_, BLO, LA, ORDEN, IDS, W, GAMMA, r, j, L, T, P, CF, escala_phi,
                                  escala_c, C, PP, CFP, K, IEEE, A32, ESTABLE)
        att, hay, m = _cas_softmax(sc, valido, ESTABLE)
        kpos = filk - (r // T).to(tl.int64) * L
        ik = tl.arange(0, 2 * C)
        lugar_k = ((ik >= C).to(tl.int64) * BT + r) * L + kpos         # (2C,) fila de cada llave en G*_K
        cero_k = (r.to(tl.int64)) * L + kpos                             # su fila en el lugar 0
        ultimo = km & (ik >= C) & (j == NB - 1)                          # llaves sin bloque siguiente
        lugar_q = r.to(tl.int64) * L + qpos
        dc = tl.arange(0, BD)
        dp = tl.zeros([C, 2 * C], tl.float32)
        for d0 in range(0, D, BD):
            dm = d0 + dc < D
            v = tl.load(V + filk[:, None] * D + d0 + dc[None, :], mask=km[:, None] & dm[None, :], other=0.0)
            do = tl.load(DO + filq[:, None] * D + d0 + dc[None, :], mask=qm[:, None] & dm[None, :], other=0.0)
            do = do.to(v.dtype)
            dp += _punto(do, tl.trans(v), IEEE, A32)
            dv = _punto(tl.trans(att.to(v.dtype)), do, IEEE, A32)
            tl.store(GV + lugar_k[:, None] * D + d0 + dc[None, :], dv.to(GV.dtype.element_ty),
                     mask=km[:, None] & dm[None, :])
            tl.store(GV + cero_k[:, None] * D + d0 + dc[None, :], tl.zeros([2 * C, BD], GV.dtype.element_ty),
                     mask=ultimo[:, None] & dm[None, :])
        ds = att * (dp - tl.sum(att * dp, 1)[:, None])        # gradiente del puntaje (0 donde no es valido)
        dsp = ds.to(phi_q.dtype)
        dsc = ds.to(qc.dtype)
        pc = tl.arange(0, PP)
        cc = tl.arange(0, CFP)
        pm, cm = (pc < P)[None, :], (cc < CF)[None, :]
        tl.store(GPHI_Q + lugar_q[:, None] * P + pc[None, :],
                 (_punto(dsp, phi_k, IEEE, A32) * escala_phi).to(GPHI_Q.dtype.element_ty), mask=qm[:, None] & pm)
        tl.store(GPHI_K + lugar_k[:, None] * P + pc[None, :],
                 (_punto(tl.trans(dsp), phi_q, IEEE, A32) * escala_phi).to(GPHI_K.dtype.element_ty),
                 mask=km[:, None] & pm)
        tl.store(GQC + lugar_q[:, None] * CF + cc[None, :],
                 (_punto(dsc, kc, IEEE, A32) * escala_c).to(GQC.dtype.element_ty), mask=qm[:, None] & cm)
        tl.store(GKC + lugar_k[:, None] * CF + cc[None, :],
                 (_punto(tl.trans(dsc), qc, IEEE, A32) * escala_c).to(GKC.dtype.element_ty), mask=km[:, None] & cm)
        col = tl.sum(ds, 0)                                  # (2C,) por llave
        tl.store(GLA + lugar_k, col, mask=km)
        tl.store(GB_K + lugar_k, tasa * col, mask=km)
        tl.store(GB_Q + lugar_q, -tasa * tl.sum(ds, 1), mask=qm)
        tl.store(GPHI_K + cero_k[:, None] * P + pc[None, :], tl.zeros([2 * C, PP], GPHI_K.dtype.element_ty),
                 mask=ultimo[:, None] & pm)
        tl.store(GKC + cero_k[:, None] * CF + cc[None, :], tl.zeros([2 * C, CFP], GKC.dtype.element_ty),
                 mask=ultimo[:, None] & cm)
        tl.store(GLA + cero_k, tl.zeros([2 * C], tl.float32), mask=ultimo)
        tl.store(GB_K + cero_k, tl.zeros([2 * C], tl.float32), mask=ultimo)
        renglon = PARTES + (r.to(tl.int64) * NB + j) * (K + 2)
        tl.store(renglon, tl.sum(tl.sum(ds * dt, 1), 0))                                # d tasa
        for c in tl.static_range(K + 1):
            tl.store(renglon + 1 + c, tl.sum(tl.sum(tl.where(largo == c, ds, 0.0), 1), 0))   # d w_sufijo[c]


def _usar_triton_casilleros(E, C, P, CF):
    """La atencion por casilleros en Triton: como _usar_triton, con bloques que tl.dot acepta (C, P y la
    confirmacion de al menos 16). ASRN_TRITON_CASILLEROS=0 apaga solo esta parte."""
    return (_usar_triton(E) and os.environ.get('ASRN_TRITON_CASILLEROS', '1') != '0'
            and C >= 16 and C & (C - 1) == 0 and P >= 16 and CF >= 16)


class _CasillerosTriton(torch.autograd.Function):
    """Ida y vuelta de la atencion por casilleros con los kernels de arriba. Devuelve la salida sumada sobre las
    tablas (B, L, d) y las senales (B*T, L, 3), sin gradiente. Para la vuelta guarda solo sus entradas."""

    @staticmethod
    def forward(ctx, phi, V, Qc, Kc, Bt, log_alfa, w, gamma, Pf, orden, ids_o, T, C, escala_phi, escala_c,
                Bt_bajo=None):
        """Bt_bajo (opcional, asrn/estable.py): la parte baja (B, L) de un reloj de 64 bits cuya parte alta es Bt.
        Con ella los kernels corren en modo ESTABLE (ver _cas_puntajes y _cas_softmax); sin ella, como siempre."""
        B, L, P = phi.shape
        D, CF, K = V.shape[-1], Qc.shape[-1], Pf.shape[-1]
        ent = [x.contiguous() for x in (phi, Qc, Kc, V, Pf, Bt.float(), log_alfa.float())]
        bajo = ent[5] if Bt_bajo is None else Bt_bajo.detach().float().contiguous()
        orden, ids_o = orden.contiguous(), ids_o.contiguous()
        w, gamma = w.detach().float().contiguous(), gamma.detach().float().contiguous()
        OUT = torch.empty((B * T, L, D), dtype=V.dtype, device=V.device)
        SEN = torch.empty((B * T, L, 3), dtype=torch.float32, device=V.device)
        NB = triton.cdiv(L, C)
        cfg = dict(C=C, PP=max(16, triton.next_power_of_2(P)), CFP=max(16, triton.next_power_of_2(CF)), BD=64, K=K,
                   IEEE=V.dtype == torch.float32, A32=os.environ.get('TRITON_INTERPRET') == '1',
                   ESTABLE=Bt_bajo is not None)
        _casilleros_ida[(B * T, NB)](ent[0], ent[1], ent[2], ent[3], ent[4], ent[5], bajo, ent[6], orden, ids_o, w,
                                     gamma, OUT, SEN, L, T, P, CF, D, escala_phi, escala_c, **cfg)
        ctx.save_for_backward(*ent, bajo, orden, ids_o, w, gamma)
        ctx.datos = (B, L, P, D, CF, K, T, C, NB, escala_phi, escala_c, cfg,
                     [x.dtype for x in (phi, Qc, Kc, V, Bt, log_alfa)])
        ctx.mark_non_differentiable(SEN)
        return OUT.view(B, T, L, D).sum(1), SEN

    @staticmethod
    def backward(ctx, dsalida, _dsen):
        phi, Qc, Kc, V, Pf, Bt, la, bajo, orden, ids_o, w, gamma = ctx.saved_tensors
        B, L, P, D, CF, K, T, C, NB, escala_phi, escala_c, cfg, tipos = ctx.datos
        BT, dv = B * T, V.device
        f32 = dict(dtype=torch.float32, device=dv)
        G = dict(GPHI_Q=torch.empty((BT, L, P), dtype=phi.dtype, device=dv),
                 GQC=torch.empty((BT, L, CF), dtype=Qc.dtype, device=dv),
                 GB_Q=torch.empty((BT, L), **f32),
                 GPHI_K=torch.empty((2, BT, L, P), dtype=phi.dtype, device=dv),
                 GKC=torch.empty((2, BT, L, CF), dtype=Kc.dtype, device=dv),
                 GV=torch.empty((2, BT, L, D), dtype=V.dtype, device=dv),
                 GLA=torch.empty((2, BT, L), **f32), GB_K=torch.empty((2, BT, L), **f32))
        PARTES = torch.empty((BT, NB, K + 2), **f32)
        _casilleros_vuelta[(BT, NB)](phi, Qc, Kc, V, Pf, Bt, bajo, la, orden, ids_o, w, gamma, dsalida.contiguous(),
                                     *G.values(), PARTES, L, T, BT, P, CF, D, NB, escala_phi, escala_c, **cfg)

        def sumar(q, k):                # tablas (y los dos lugares de cada llave) -> (B, L, ...), en 32 bits
            total = k.view(2, B, T, *k.shape[2:]).sum((0, 2), dtype=torch.float32)
            return total if q is None else total + q.view(B, T, *q.shape[1:]).sum(1, dtype=torch.float32)
        partes = PARTES.view(B, T, NB, K + 2).sum((0, 2))   # (T, K+2)
        tp, tq, tk, tv, tb, tl_ = tipos
        return (sumar(G['GPHI_Q'], G['GPHI_K']).to(tp), sumar(None, G['GV']).to(tv),
                G['GQC'].view(B, T, L, CF).sum(1, dtype=torch.float32).to(tq), sumar(None, G['GKC']).to(tk),
                sumar(G['GB_Q'], G['GB_K']).to(tb), sumar(None, G['GLA']).to(tl_), partes[:, 1:].sum(0),
                partes[:, 0] / C, None, None, None, None, None, None, None, None)


class _Escaneo(torch.autograd.Function):
    """Vuelta de S_i = a_i S_{i-1} + b_i. Con g_i = dL/dS_i, el adjunto l_i = g_i + a_{i+1} l_{i+1} es la misma
    recurrencia al reves; dL/db_i = l_i y dL/da_i = l_i S_{i-1}. Guarda a y S: 2 numeros por elemento."""

    @staticmethod
    def forward(ctx, a, b):
        S = _escaneo_triton(a, b) if _usar_triton(a) else _escaneo_ida(a, b)
        S = S.clone() if S is b else S
        ctx.save_for_backward(a, S)
        return S

    @staticmethod
    def backward(ctx, g):
        a, S = ctx.saved_tensors
        if _usar_triton(a):
            ga, gb = _escaneo_triton(a, g, vuelta=True, S_ida=S)
            return (ga if ctx.needs_input_grad[0] else None), (gb if ctx.needs_input_grad[1] else None)
        lam = _escaneo_ida(F.pad(a[..., 1:], (0, 1)).flip(-1), g.flip(-1)).flip(-1)
        ga = lam * F.pad(S[..., :-1], (1, 0)) if ctx.needs_input_grad[0] else None
        return ga, (lam if ctx.needs_input_grad[1] else None)


def _escaneo_ida(a, b):
    """Escaneo asociativo paralelo de la recurrencia lineal S_i = a_i * S_{i-1} + b_i (S_{-1} = 0), sobre la ultima
    dimension. a, b: (..., N) -> S: (..., N).

    Los mapas afines x -> a x + b forman un monoide: (a1, b1) o (a2, b2) = (a1 a2, a2 b1 + b2). Par-impar: se combinan
    los pares vecinos (N/2 elementos), se resuelven recursivamente los impares y cada par sale de un paso desde el impar
    anterior. Trabajo N + N/2 + ... = O(N), profundidad O(log N), y autograd guarda O(N) (no O(N log N) como
    Hillis-Steele). Con a en [0, 1] solo hay productos de numeros en [0, 1] y sumas: estable en fp32 y bf16. a_i = 0
    reinicia la recurrencia: asi se escanean muchos segmentos independientes en una sola pasada."""
    N = a.shape[-1]
    if N == 1:
        return b
    if N % 2:                                                    # relleno con el neutro (a=1, b=0)
        return _escaneo_ida(F.pad(a, (0, 1), value=1.0), F.pad(b, (0, 1)))[..., :N]
    a_par, a_impar = a[..., 0::2], a[..., 1::2]                   # (..., N/2)
    b_par, b_impar = b[..., 0::2], b[..., 1::2]
    impar = _escaneo_ida(a_par * a_impar, a_impar * b_par + b_impar)   # S en 1, 3, 5, ...
    par = a_par * F.pad(impar[..., :-1], (1, 0)) + b_par          # S en 0, 2, 4, ...: un paso desde el impar anterior
    return torch.stack([par, impar], -1).flatten(-2)


# ----------------------------------------------------------------------------------------------------------------
# La capa ASRN
# ----------------------------------------------------------------------------------------------------------------

def _a_autocast(x):
    """x en el tipo de 16 bits de autocast si autocast esta activo (si no, igual). Sirve para convertir una vez un
    tensor que varias multiplicaciones van a leer: autocast lo convertiria en cada una."""
    disp = x.device.type
    if torch.is_autocast_enabled(disp) and x.is_floating_point():
        return x.to(torch.get_autocast_dtype(disp))
    return x


class CapaASRN(nn.Module):
    """Encuentra lo que ya aparecio antes en el texto y lo copia cuando conviene. Costo proporcional al largo.

    Piezas (receta final):
      - n_tablas tablas de casilleros (hash); las ultimas `tablas_contexto` usan tambien la palabra anterior
      - confirmacion con contexto sobre el estado interno NORMALIZADO
      - coincidencia exacta hacia atras (hasta `sufijo` palabras seguidas)
      - puerta con confianza + senales de si lo reciente ya salio antes (grupos de 1 a `vistos` palabras)
    Entrada: E (vectores de palabra) y h (estado), ambos (lote, largo, d); opcional piezas (lote, largo), los numeros
    de pieza (si no se dan, la firma sale de E . r como en v83). Salida: h + lo copiado."""

    def __init__(self, d, n_tablas=8, n_bits=14, afin_oculto=128, chunk=32, tablas_contexto=4, confirmar=16,
                 sufijo=4, vistos=4, dropout=0.0, nodos_racha=16, racha_max=256):
        super().__init__()
        assert 0 < tablas_contexto < n_tablas and nodos_racha >= 2
        self.n_tablas, self.n_bits, self.chunk = n_tablas, n_bits, chunk
        self.tablas_contexto, self.confirmar, self.sufijo, self.vistos = tablas_contexto, confirmar, sufijo, vistos
        mitad = max(8, afin_oculto // 2)
        self.afin_palabra = nn.Sequential(nn.Linear(d, mitad), nn.GELU(),
                                          nn.Linear(mitad, (n_tablas - tablas_contexto) * n_bits))
        self.afin_contexto = nn.Sequential(nn.Linear(2 * d, mitad), nn.GELU(),
                                           nn.Linear(mitad, tablas_contexto * n_bits))
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.g = nn.Linear(d, 1)
        self.temperatura = (n_tablas * n_bits) ** 0.5
        self.register_buffer('pot', 2 ** torch.arange(n_bits))
        self.g_conf = nn.Linear(4, 1)
        nn.init.zeros_(self.g_conf.weight)
        nn.init.zeros_(self.g_conf.bias)
        self.cq = nn.Linear(d, confirmar, bias=False)
        nn.init.zeros_(self.cq.weight)
        self.ck = nn.Linear(d, confirmar, bias=False)
        self.ln_ctx = nn.LayerNorm(d)
        self.w_vistos = nn.Parameter(torch.zeros(vistos))
        g_v = torch.Generator().manual_seed(12345)
        self.register_buffer('c_vistos', torch.rand(vistos, generator=g_v, dtype=torch.float64) + 0.5)
        self.w_sufijo = nn.Parameter(torch.zeros(sufijo + 1))
        self.register_buffer('r_firma', torch.randn(d))
        self.drop = nn.Dropout(dropout)
        # --- v86: tiempo propio y nucleo continuo. Todo neutro al inicio (y sin usar el azar): la capa calcula lo
        # mismo que v85, y los modelos de v85 cargan y dan los mismos numeros.
        self.w_reloj = nn.Parameter(torch.zeros(2, d))                       # [beta, alfa] desde el estado normalizado
        self.b_reloj = nn.Parameter(torch.tensor([math.log(math.e - 1), 0.0]))   # beta = 1: el reloj cuenta piezas
        self.gamma = nn.Parameter(torch.zeros(n_tablas))                      # tasa de olvido de cada tabla
        # memoria de lo repetido: Hawkes de cada grupo de 1..vistos palabras y racha. Tasas en escala log, en un
        # espectro geometrico (los grupos cortos olvidan rapido, los largos, que son raros, recuerdan mas)
        self.log_kappa = nn.Parameter(-torch.log(chunk * 4.0 ** torch.arange(vistos)))
        self.log_kappa_racha = nn.Parameter(torch.tensor([-math.log(8.0 * chunk)]))
        self.u_hawkes = nn.Parameter(torch.zeros(vistos))                     # cuanto empujan a la puerta (0: nada)
        # respuesta de la puerta a la racha: lineal por tramos sobre log1p(racha), en una grilla fija de nodos hasta
        # racha_max. No monotona: puede subir mientras una cita es normal y caer en seco cuando ya es demasiado larga.
        # Se aprende por INCREMENTOS entre nodos vecinos (valor en el nodo i = d_0 + ... + d_i): un nodo al que los
        # datos nunca llegan no recibe gradiente y su incremento queda en 0, asi que hereda el valor del ultimo nodo
        # con datos (extrapolacion plana) en vez de volver a 0. El freno aprendido sigue frenando pasado lo visto.
        self.racha_paso = math.log1p(racha_max) / (nodos_racha - 1)
        self.d_racha = nn.Parameter(torch.zeros(nodos_racha))                 # incremento en cada nodo (0: nada)

    # --- lo que entra a la busqueda ---------------------------------------------------------------------------

    def _firma(self, E):
        """Firma desde el vector (v83): un numero por palabra (E . r), en 32 bits. Solo si no hay numeros de pieza."""
        with torch.autocast(device_type=E.device.type, enabled=False):
            return E.float() @ self.r_firma.float()

    def _phi(self, E):
        """(B, L, T*n_bits) salida del hash de todas las tablas: primero las de palabra sola, despues las de contexto."""
        llave = torch.cat([E, F.pad(E, (0, 0, 1, 0))[:, :E.shape[1]]], dim=-1)   # (B, L, 2d) [palabra, anterior]
        return torch.cat([self.afin_palabra(E), self.afin_contexto(llave)], dim=-1)

    def _phi_y_valores(self, E):
        """_phi(E) y _valores(E) con una multiplicacion menos: la primera capa del hash de palabra y los valores
        leen E juntos (E se pasa a 16 bits una vez menos). Cada numero sale con el mismo redondeo que antes: es el
        mismo producto, solo en otra columna. Los valores: v(E_{t+1}) es v(E) desplazado una pieza hacia atras (v no
        tiene sesgo: la ultima queda en 0). El hash de contexto sigue con [palabra, anterior]: partirlo en dos
        productos suma un redondeo mas, y cerca de cero eso cambia casilleros; pero se arma con E ya en 16 bits, que
        da los mismos numeros que convertir [palabra, anterior] entero (convertir y juntar = juntar y convertir)."""
        d, L = E.shape[-1], E.shape[1]
        pal = self.afin_palabra
        m = pal[0].out_features
        E = _a_autocast(E)          # una sola conversion a 16 bits (autocast la haria en cada multiplicacion)
        y = F.linear(E, torch.cat([pal[0].weight, self.v.weight]), F.pad(pal[0].bias, (0, d)))
        hp, v = y.split([m, d], dim=-1)
        llave = torch.cat([E, F.pad(E, (0, 0, 1, 0))[:, :L]], dim=-1)
        phi = torch.cat([pal[2](pal[1](hp)), self.afin_contexto(llave)], dim=-1)
        return phi, F.pad(v, (0, 0, 0, 1))[:, 1:]

    def _ids(self, phi):
        """(..., T) casillero de cada tabla: los signos de sus n_bits salidas leidos como un numero."""
        bits = phi.reshape(*phi.shape[:-1], self.n_tablas, self.n_bits) > 0
        return (bits.int() * self.pot.int()).sum(-1, dtype=torch.int32)    # n_bits <= 30: cabe en 32 bits

    def _contexto(self, h):
        """Desde el estado normalizado ĥ: la confirmacion (Qc, Kc), cada uno (B, L, confirmar), y el reloj por pieza,
        cada uno (B, L), que sale en 32 bits (con autocast la proyeccion w . ĥ se calcula en 16 bits y despues se pasa
        a 32; la suma B_t = cumsum(beta) si es en 32 bits):
          beta = softplus(w_b . ĥ + b_b) >= 0   cuanto tiempo propio pasa en esta pieza (B_t = suma de beta hasta t)
          log alfa, alfa = 2 sigma(w_a . ĥ) en (0, 2)   cuan memorable es esta aparicion (1 al inicio)
        El nucleo de memoria es K(t, s) = alfa_s exp(-kappa (B_t - B_s)): W(n) = alfa e^(-beta n) con n medido en
        tiempo propio, que depende de los datos."""
        hh = _a_autocast(self.ln_ctx(h))
        cf = self.confirmar          # una sola multiplicacion para cq, ck y el reloj: hh se lee una vez
        y = F.linear(hh, torch.cat([self.cq.weight, self.ck.weight, self.w_reloj]),
                     F.pad(self.b_reloj, (2 * cf, 0)))
        u = y[..., 2 * cf:].float()                                                  # (B, L, 2)
        return y[..., :cf], y[..., cf:2 * cf], F.softplus(u[..., 0]), F.logsigmoid(u[..., 1]) + math.log(2)

    def _valores(self, E):
        """(B, L, d): lo que se copia desde la posicion s es la pieza que vino DESPUES (s+1); la ultima no tiene."""
        return self.v(F.pad(E, (0, 0, 0, 1))[:, 1:])

    def _peso_sufijo(self, largo):
        return _Tomar.apply(largo, self.w_sufijo)

    def _claves_vistos(self, sig):
        """(B, L, vistos) claves (64 bits) de los grupos de 1..vistos palabras que terminan en cada posicion. Las
        claves se comparan por igualdad exacta: se suman siempre en el mismo orden (un paso por palabra del grupo)."""
        return _claves_de(sig, self.c_vistos, self.vistos)

    def _ordenar_claves(self, claves):
        """claves (B, L, n) -> orden (B, n, L) estable (por clave y, dentro de cada clave, por posicion) y repetido
        (B, n, L), en ese orden: la aparicion anterior de la misma clave esta justo antes."""
        k = claves.transpose(1, 2)
        orden = torch.argsort(k, dim=-1, stable=True)
        ks = torch.gather(k, -1, orden)
        # el relleno se hace en enteros: con la comparacion en 64 bits y un relleno booleano, Inductor (torch.compile
        # en CPU) genera C++ que no compila (mezcla mascaras de double y de float)
        return orden, F.pad((ks[..., 1:] == ks[..., :-1]).to(torch.uint8), (1, 0)).bool()

    def _vistos(self, sig):
        """(B, L, vistos): 1.0 si el grupo de 1, 2, ... palabras que termina en t ya salio antes."""
        with torch.no_grad():
            orden, repetido = self._ordenar_claves(self._claves_vistos(sig))
            return torch.zeros_like(repetido).scatter(-1, orden, repetido).transpose(1, 2).float()

    def _repeticion(self, sig, beta, Bt, log_alfa):
        """Memoria de lo repetido en el texto entero (sig, beta, Bt, log_alfa: (B, L)). Devuelve
          vistos (B, L, n): 1 si el grupo de n palabras que termina en t ya salio antes (la senal de v85)
          hawkes (B, L, n): intensidad de Hawkes de ese grupo, lambda_n(t) = suma sobre sus apariciones anteriores s de
                 alfa_s exp(-kappa_n (B_t - B_s)): cuantas veces, cuan recientes y cuan memorables. 1[lambda > 0] es
                 exactamente 'vistos'. Escaneo segmentado sobre el orden por clave que ya da 'vistos' (ningun
                 ordenamiento nuevo): en ese orden la aparicion anterior esta justo antes, y
                 S_i = a_i (S_{i-1} + alfa_{i-1}), a_i = exp(-kappa (B_i - B_{i-1})) si sigue la misma clave, 0 si no.
          racha (B, L): l_t = r_t (exp(-kappa_r beta_t) l_{t-1} + 1), con r_t = 'vistos' del grupo mas largo: cuanto
                 lleva el texto repitiendose. Escaneo sobre el texto; con kappa_r = 0 es el largo exacto de la racha.
        Rompen el aliasing de v85, donde 'vistos' en {0,1} y el sufijo <= 4 confunden una cita de 5 piezas con una
        de 50: la puerta puede aprender cuando termina una cita."""
        n = self.vistos
        with torch.no_grad():
            orden, repetido = self._ordenar_claves(self._claves_vistos(sig))           # (B, n, L)
            vistos = torch.zeros_like(repetido).scatter(-1, orden, repetido)         # (B, n, L) en orden del texto
        kappa = self.log_kappa.exp().unsqueeze(-1)                                   # (n, 1)
        B_o = torch.gather(Bt.unsqueeze(1).expand(-1, n, -1), -1, orden)            # (B, n, L) reloj, orden por clave
        alfa_o = torch.gather(log_alfa.exp().unsqueeze(1).expand(-1, n, -1), -1, orden)
        a = repetido * torch.exp(-kappa * F.pad((B_o[..., 1:] - B_o[..., :-1]).clamp(min=0), (1, 0)))
        lam = escaneo(a, a * F.pad(alfa_o[..., :-1], (1, 0)))                       # (B, n, L)
        hawkes = torch.zeros_like(lam).scatter(-1, orden, lam).transpose(1, 2)      # (B, L, n) en orden del texto
        r = vistos[:, -1].float()                                                    # (B, L)
        racha = escaneo(r * torch.exp(-self.log_kappa_racha.exp() * beta), r)       # (B, L)
        return {'vistos': vistos.transpose(1, 2).float(), 'hawkes': hawkes, 'racha': racha}

    def _repeticion_paso(self, claves_mem, claves_t, B_mem, B_t, log_alfa_mem, beta_t, racha_previa):
        """Lo mismo para una pieza nueva t contra la memoria 0..t-1: claves_mem (B, t, n), claves_t (B, n),
        B_mem y log_alfa_mem (B, t), B_t y beta_t (B,), racha_previa (B,). La racha es un estado de un numero."""
        igual = claves_mem == claves_t.unsqueeze(1)                                  # (B, t, n)
        nucleo = (log_alfa_mem.unsqueeze(-1)
                  - self.log_kappa.exp() * (B_t.unsqueeze(1) - B_mem).unsqueeze(-1)).exp()   # (B, t, n)
        vistos = igual.any(1).float()                                                # (B, n)
        r = vistos[:, -1]
        racha = r * (torch.exp(-self.log_kappa_racha.exp() * beta_t) * racha_previa + 1)
        return {'vistos': vistos.unsqueeze(1), 'hawkes': (igual * nucleo).sum(1).unsqueeze(1),
                'racha': racha.unsqueeze(1)}

    # --- entrenamiento y evaluacion: todas las tablas a la vez --------------------------------------------------

    def _acumular(self, E, Qc, Kc, Bt, log_alfa, sig):
        """Todas las tablas a la vez, en el espacio ordenado por casillero.

        Cada tabla ordena las posiciones por casillero (estable: dentro de un casillero, por posicion) y las corta en
        bloques de `chunk`. Cada pregunta mira las `chunk` posiciones ordenadas anteriores: sus companeros de casillero
        mas recientes. Las llaves (bloque anterior + actual) son una VISTA solapada (unfold) de una sola copia
        ordenada, con un bloque de ceros delante de cada fila: nada se duplica y autograd guarda una sola copia.
        Ventanas: W = BT*R - 1 (R = bloques por fila + 1); las que cruzan de una fila a la siguiente se descartan.

        Nucleo continuo: la tabla tau suma log alfa_s - (gamma_tau / chunk) (B_t - B_s) a la puntuacion. Se arma como
        (llave) - (pregunta), dos vectores que se difunden: nunca se guarda un tensor (C, 2C) para la vuelta. La forma
        relativa (B_t - B_s) deja estacionaria la senal 'mejor' de la puerta."""
        if _usar_triton_casilleros(E, self.chunk, self.n_tablas * self.n_bits, self.confirmar):
            return self._acumular_triton(E, Qc, Kc, Bt, log_alfa, sig)
        B, L, _ = E.shape
        T, C, cf = self.n_tablas, self.chunk, self.confirmar
        n_pad = (-L) % C
        R = (L + n_pad) // C + 1                       # bloques por fila, con el de ceros delante
        BT = B * T
        dv = E.device

        phi, V = self._phi_y_valores(E)                                          # (B, L, T*nb), (B, L, d)
        ids = self._ids(phi.detach()).transpose(1, 2).reshape(BT, L)             # (BT, L) fila = b*T + tabla
        orden = ids.argsort(dim=-1, stable=True)                                 # (BT, L) posicion ordenada -> texto
        nulo = B * L                                                             # fila de ceros al final de la fuente
        fuente = F.pad(orden + torch.arange(0, nulo, L, device=dv).repeat_interleave(T).unsqueeze(1),
                       (C, n_pad), value=nulo)                                   # (BT, R*C) fila de la fuente
        real = fuente != nulo                                                    # (BT, R*C)
        ids_o = F.pad(torch.gather(ids, 1, orden), (C, n_pad), value=-1)         # (BT, R*C)

        def ordenar(x):            # (B, L, k) -> (BT, R*C, k): orden por casillero, ceros en los rellenos
            k = x.shape[-1]        # index_select: la vuelta solo guarda los indices, no x
            return torch.cat([x.reshape(nulo, k), x.new_zeros(1, k)]).index_select(0, fuente.view(-1)).view(BT, -1, k)

        def preguntas(x):          # (BT, R*C, ...) -> (W, C, ...): el bloque que pregunta en cada ventana
            return x.reshape(BT * R, C, *x.shape[2:])[1:]

        def llaves(x):             # (BT, R*C, ...) -> (W, 2C, ...): bloque anterior + actual, vista sin copiar
            return x.reshape(BT * R * C, *x.shape[2:]).unfold(0, 2 * C, C).movedim(-1, 1)

        # de cada posicion del texto, su fila en la salida de las ventanas (W*C filas): la fila r empieza en r*R*C
        # (su bloque de ceros no pregunta) y la posicion ordenada s esta en r*R*C + s
        en_orden = torch.empty_like(orden).scatter_(1, orden, torch.arange(L, device=dv).expand(BT, L))
        fila = (en_orden + torch.arange(0, BT * R * C, R * C, device=dv).unsqueeze(1)).reshape(-1)  # (BT*L,)

        def desordenar(x):         # (W, C, k) -> (B, T, L, k) en el orden del texto (para las senales, sin vuelta):
            k = x.shape[-1]        # filas enteras con un indice por fila, no uno de 64 bits por numero
            return x.reshape(-1, k).index_select(0, fila).view(B, T, L, k)

        phi_o, V_o = ordenar(phi), ordenar(V)                                    # (BT, R*C, T*nb), (BT, R*C, d)
        Qc_o, Kc_o, Pf_o = ordenar(Qc), ordenar(Kc), ordenar(_anteriores(sig, self.sufijo))

        sc = torch.bmm(preguntas(phi_o), llaves(phi_o).transpose(1, 2)) / self.temperatura      # (W, C, 2C)
        sc = sc + torch.bmm(preguntas(Qc_o), llaves(Kc_o).transpose(1, 2)) * cf ** -0.5
        sc = sc + self._peso_sufijo(_largo_seguido(preguntas(Pf_o).unsqueeze(2), llaves(Pf_o).unsqueeze(1)))
        tasa = (self.gamma / C).repeat(B).unsqueeze(1)                           # (BT, 1) la de la tabla de cada fila
        B_o = ordenar(Bt.unsqueeze(-1)).squeeze(-1)                              # (BT, R*C) reloj, ordenado
        llave = ordenar(log_alfa.unsqueeze(-1)).squeeze(-1) + tasa * B_o         # (BT, R*C)
        sc = sc + llaves(llave).unsqueeze(1) - preguntas(tasa * B_o).unsqueeze(2)                # (W, C, 2C)

        i = torch.arange(C, device=dv).unsqueeze(1)
        j = torch.arange(2 * C, device=dv)
        anteriores = (j >= i) & (j < C + i)                # (C, 2C): las C posiciones ordenadas antes de la pregunta
        valido = (preguntas(ids_o).unsqueeze(2) == llaves(ids_o).unsqueeze(1)) & llaves(real).unsqueeze(1) & anteriores
        sc = sc.masked_fill(~valido, _prohibido(sc.dtype))                       # (W, C, 2C)
        hay = valido.any(-1, keepdim=True)                                       # (W, C, 1)
        att = torch.softmax(sc, dim=-1) * hay
        out = torch.bmm(att, llaves(V_o))                                        # (W, C, d)

        salida = _DesordenarSumar.apply(out.reshape(-1, out.shape[-1]), fila, fuente.view(-1)[C:], B, T, L)  # (B, L, d)
        senal = desordenar(torch.cat([hay.float(), sc.detach().amax(-1, keepdim=True).float(),
                                      att.detach().amax(-1, keepdim=True).float()], dim=-1))   # (B, T, L, 3)
        hay_t, sc_max, pico_t = senal[..., 0:1], senal[..., 1:2], senal[..., 2:3]
        mejor = torch.where(hay_t > 0, sc_max, torch.full_like(sc_max, _PROHIBIDO)).amax(1).clamp(min=_PROHIBIDO)
        return {'salida': salida, 'peso_total': hay_t.sum(1), 'mejor': mejor,
                'pico': (pico_t * hay_t).sum(1), 'peso_ctx': hay_t[:, T - self.tablas_contexto:].sum(1)}

    def _acumular_triton(self, E, Qc, Kc, Bt, log_alfa, sig):
        """Lo mismo que _acumular con el kernel fusionado (_CasillerosTriton): las mismas cuentas, salvo el redondeo
        (en el kernel los puntajes se suman en 32 bits)."""
        B, L, _ = E.shape
        T, C = self.n_tablas, self.chunk
        phi, V = self._phi_y_valores(E)
        ids = self._ids(phi.detach()).transpose(1, 2).reshape(B * T, L)
        orden = ids.argsort(dim=-1, stable=True)
        salida, senal = _CasillerosTriton.apply(phi, V, Qc, Kc, Bt, log_alfa, self.w_sufijo,
                                                self.gamma, _anteriores(sig, self.sufijo), orden,
                                                torch.gather(ids, 1, orden), T, C, 1.0 / self.temperatura,
                                                self.confirmar ** -0.5)
        senal = senal.view(B, T, L, 3)
        hay_t, sc_max, pico_t = senal[..., 0:1], senal[..., 1:2], senal[..., 2:3]
        mejor = torch.where(hay_t > 0, sc_max, torch.full_like(sc_max, _PROHIBIDO)).amax(1).clamp(min=_PROHIBIDO)
        return {'salida': salida, 'peso_total': hay_t.sum(1), 'mejor': mejor,
                'pico': (pico_t * hay_t).sum(1), 'peso_ctx': hay_t[:, T - self.tablas_contexto:].sum(1)}

    def _combinar(self, acc, h):
        """Promedio entre tablas, puerta y aporte (igual para entrenar y para escribir)."""
        salida, pt = acc['salida'], acc['peso_total'].detach()
        salida = salida / acc['peso_total'].clamp(min=1)
        senales = torch.cat([pt / self.n_tablas,
                             torch.where(pt > 0, acc['mejor'], torch.zeros_like(pt)),
                             acc['pico'] / pt.clamp(min=1),
                             acc['peso_ctx'].detach() / self.tablas_contexto], dim=-1)          # (B, L, 4)
        # g(h) en la precision de h (32 bits al entrenar con autocast): es un solo numero por pieza, y asi h no se
        # copia entera a 16 bits (ni su gradiente de vuelta a 32)
        with torch.autocast(device_type=h.device.type, enabled=False):
            logit = F.linear(h, self.g.weight.to(h.dtype), self.g.bias.to(h.dtype))
        logit = logit + self.g_conf(senales).to(logit.dtype)
        logit = logit + (acc['vistos'].to(logit.dtype) * self.w_vistos.to(logit.dtype)).sum(-1, keepdim=True)
        # v86: cuanto y cuan recientemente se repitio lo que acaba de salir, y cuanto lleva repitiendose (log1p:
        # rendimientos decrecientes; 0 si nunca salio)
        logit = logit + (torch.log1p(acc['hawkes']).to(logit.dtype) * self.u_hawkes.to(logit.dtype)).sum(-1, keepdim=True)
        logit = logit + self._respuesta_racha(acc['racha']).unsqueeze(-1).to(logit.dtype)
        aporte = self.o(salida)
        return h + torch.sigmoid(logit).to(aporte.dtype) * self.drop(aporte)

    @property
    def u_racha(self):
        """(nodos,) valor de la respuesta a la racha en cada nodo: la suma acumulada de los incrementos d_racha."""
        return self.d_racha.cumsum(0)

    def _respuesta_racha(self, racha):
        """(B, L) -> (B, L): interpolacion lineal de u_racha en x = log1p(racha) / paso (constante pasado el ultimo
        nodo). Equivale a una base de funciones sombrero, pero para la vuelta guarda solo el nodo y la fraccion de
        cada pieza, no la base (B, L, nodos). La pendiente del tramo i es d_racha[i+1]."""
        x = (torch.log1p(racha) / self.racha_paso).clamp(max=self.d_racha.shape[0] - 1)
        # nodo de la izquierda; un NaN (pesos rotos) va al nodo 0 en vez de dar un indice basura: la salida sale NaN
        # igual y entrenar() salta el paso, en vez de caerse (en GPU, un indice fuera de rango mata el proceso)
        i = torch.nan_to_num(x.detach(), nan=0.0).floor().long().clamp(0, self.d_racha.shape[0] - 2)
        return _Tomar.apply(i, self.u_racha) + (x - i) * _Tomar.apply(i, self.d_racha[1:])    # un indice guardado

    def forward(self, E, h, piezas=None):
        sig = self._firma(E) if piezas is None else firmas_de_piezas(piezas)
        Qc, Kc, beta, log_alfa = self._contexto(h)
        Bt = beta.cumsum(1)                                                          # (B, L) reloj
        acc = self._acumular(E, Qc, Kc, Bt, log_alfa, sig)
        acc.update(self._repeticion(sig, beta, Bt, log_alfa))
        return self._combinar(acc, h)

    # --- escritura: una pregunta contra toda la memoria --------------------------------------------------------

    def _caracteristicas(self, E, h):
        phi = self._phi(E)
        Qc, Kc, beta, log_alfa = self._contexto(h)
        return phi, self._ids(phi), self._valores(E), Qc, Kc, beta, log_alfa

    def _atender(self, phi_q, ids_q, Qc_q, Pf_q, B_q, mem, perm_k):
        """La pregunta (B, 1, .) en la posicion t contra la memoria (B, t, .) de las posiciones 0..t-1. En cada
        casillero, las `chunk` apariciones anteriores mas recientes (lo mismo que ve el texto entero).
        perm_k: (B, t) llaves permitidas (None = todas). B_q: (B, 1) reloj de la pregunta. Todas las tablas a la vez:
        (B, 1, T, t)."""
        W = self.chunk
        base_sc = torch.einsum('bqp,bkp->bqk', phi_q, mem['phi']) / self.temperatura            # (B, 1, t)
        base_sc = base_sc + torch.einsum('bqp,bkp->bqk', Qc_q, mem['Kc']) * self.confirmar ** -0.5
        base_sc = base_sc + self._peso_sufijo(_largo_seguido(Pf_q.unsqueeze(2), mem['Pf'].unsqueeze(1)))
        valido = ids_q.unsqueeze(-1) == mem['ids'].transpose(1, 2).unsqueeze(1)                 # (B, 1, T, t)
        if perm_k is not None:
            valido = valido & perm_k[:, None, None, :]
        cs = valido.cumsum(-1)
        valido = valido & (cs[..., -1:] - cs + valido <= W)               # a lo sumo W llaves, las mas recientes
        nucleo = (mem['log_alfa'][:, None, None, :]                                              # (B, 1, T, t)
                  - (self.gamma / W)[None, None, :, None] * (B_q[:, :, None, None] - mem['B'][:, None, None, :]))
        sc = (base_sc.unsqueeze(2) + nucleo).masked_fill(~valido, _prohibido(base_sc.dtype))
        hay = valido.any(-1, keepdim=True)
        att = torch.softmax(sc, dim=-1) * hay
        hay = hay.to(mem['V'].dtype)
        out = torch.einsum('bqtk,bkd->bqtd', att, mem['V'])                                      # (B, 1, T, d)
        sc_max = sc.detach().amax(-1, keepdim=True)
        mejor = torch.where(hay > 0, sc_max, torch.full_like(sc_max, _PROHIBIDO)).amax(2).clamp(min=_PROHIBIDO)
        return {'salida': out.sum(2), 'peso_total': hay.sum(2), 'mejor': mejor.to(mem['V'].dtype),
                'pico': (att.detach().amax(-1, keepdim=True) * hay).sum(2),
                'peso_ctx': hay[:, :, self.n_tablas - self.tablas_contexto:].sum(2)}

    # --- escritura en O(1) por pieza: solo los candidatos de cada tabla ------------------------------------------

    def _kappas(self):
        """(n,) tasas de Hawkes y (1,) la de la racha."""
        return self.log_kappa.exp(), self.log_kappa_racha.exp()

    def _atender_cand(self, phi_q, Qc_q, Pf_q, B_q, cand, valido):
        """La pregunta (B, 1, .) contra sus candidatos: en cada tabla, sus `chunk` companeros de casillero mas
        recientes (los mismos que _atender elige entre toda la memoria), que el escritor lleva en un anillo por
        casillero. cand: phi, Kc, Pf, B, log_alfa, V, cada uno (B, T, C, .) o (B, T, C); valido (B, T, C) las ranuras
        ocupadas. Mismas cuentas que _atender, sin recorrer la memoria: O(T C) por pieza."""
        W = self.chunk
        sc = torch.einsum('bqp,btkp->btk', phi_q, cand['phi']) / self.temperatura               # (B, T, C)
        sc = sc + torch.einsum('bqp,btkp->btk', Qc_q, cand['Kc']) * self.confirmar ** -0.5
        sc = sc + self._peso_sufijo(_largo_seguido(Pf_q.unsqueeze(2), cand['Pf']))
        sc = sc + (cand['log_alfa'] - (self.gamma / W)[None, :, None] * (B_q.unsqueeze(-1) - cand['B']))
        sc = sc.masked_fill(~valido, _prohibido(sc.dtype))
        hay = valido.any(-1, keepdim=True)                                                      # (B, T, 1)
        att = torch.softmax(sc, dim=-1) * hay
        hay = hay.to(cand['V'].dtype)
        out = torch.einsum('btk,btkd->btd', att, cand['V'])                                     # (B, T, d)
        sc_max = sc.detach().amax(-1, keepdim=True)
        mejor = torch.where(hay > 0, sc_max, torch.full_like(sc_max, _PROHIBIDO)).amax(1, keepdim=True)
        return self._acc_cand(out, att, hay, mejor.clamp(min=_PROHIBIDO), cand['V'].dtype)

    def _acc_cand(self, out, att, hay, mejor, dtype):
        """Las salidas de _atender_cand con las formas de _atender: (B, 1, d) y (B, 1, 1)."""
        return {'salida': out.sum(1, keepdim=True), 'peso_total': hay.sum(1, keepdim=True),
                'mejor': mejor.to(dtype), 'pico': (att.detach().amax(-1, keepdim=True) * hay).sum(1, keepdim=True),
                'peso_ctx': hay[:, self.n_tablas - self.tablas_contexto:].sum(1, keepdim=True)}

    def _repeticion_o1(self, memoria, claves_t, B_t, log_alfa_t, beta_t, racha_previa):
        """vistos, hawkes y racha de la pieza nueva en O(1) (memoria: _MemoriaHawkes de la capa). claves_t (B, n) en
        la CPU; B_t, log_alfa_t, beta_t, racha_previa (B,)."""
        kappa, kappa_r = self._kappas()
        vistos, hawkes = memoria.paso(claves_t, B_t, log_alfa_t, kappa)                         # (B, n) cada uno
        r = vistos[:, -1]
        racha = r * (torch.exp(-kappa_r * beta_t.float()) * racha_previa + 1)
        return {'vistos': vistos.unsqueeze(1), 'hawkes': hawkes.unsqueeze(1), 'racha': racha.unsqueeze(1)}

# ----------------------------------------------------------------------------------------------------------------
# El modelo de lenguaje
# ----------------------------------------------------------------------------------------------------------------

_MEZCLAS = ('conv', 'memoria')


class BloqueLocal(nn.Module):
    """Contexto local causal: convolucion sobre las k-1 palabras anteriores y la actual, mas una capa densa.

    Variantes (las de por omision son la receta final; las otras se midieron en history/docs/RENDIMIENTO.md):
      mezcla='conv'     convolucion densa (d x d x k): la mitad de los parametros del bloque, para ver k piezas
      mezcla='memoria'  convolucion por canal (d x k, casi gratis: el presupuesto liberado va a la capa densa) mas una
                        memoria con olvido por canal: m_t = f_t m_{t-1} + (1 - f_t) u_t, con u y f
                        desde el estado (f depende de la pieza). f queda por debajo de 1 - 1/memoria_max: la memoria
                        nunca recuerda mas alla de ~memoria_max piezas (lo lejano es trabajo de la capa ASRN), asi su
                        estado se ve igual con cualquier largo de texto. Cada canal arranca con su propia escala de
                        tiempo, de 1 a memoria_max piezas en un espectro geometrico (como las tasas de Hawkes de la
                        capa ASRN). Usa el mismo escaneo que la capa ASRN.
    (Se probaron y se quitaron, history/results/bloques_1: la convolucion por canal sin memoria, que da lo mismo que la
    receta, y la capa densa con puerta, que empeora.)"""

    def __init__(self, d, k=4, ff=128, dropout=0.0, mezcla='conv', memoria_max=1024):
        super().__init__()
        assert mezcla in _MEZCLAS and memoria_max >= 2, (mezcla, memoria_max)
        self.k, self.mezcla, self.f_max = k, mezcla, 1.0 - 1.0 / memoria_max
        self.ln1 = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, k, groups=1 if mezcla == 'conv' else d)
        if mezcla == 'memoria':
            self.mem = nn.Linear(d, 2 * d)                    # u y el olvido f
            self.mem_o = nn.Linear(d, d, bias=False)
            tau = torch.logspace(0, math.log10(memoria_max), d)   # escalas de tiempo, de 1 a memoria_max piezas
            with torch.no_grad():                                 # f_max sigmoid(b) = 1 - 1/tau (aprox.)
                q = ((1 - 1 / tau) / self.f_max).clamp(1e-3, 1 - 1e-3)
                self.mem.bias[d:] = torch.log(q / (1 - q))
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def _memoria(self, z):
        """(B, L, d) -> (B, L, d): la memoria con olvido de cada canal, en 32 bits."""
        u, f = self.mem(z).float().chunk(2, dim=-1)
        f = torch.sigmoid(f) * self.f_max
        m = escaneo(f.transpose(1, 2).contiguous(), ((1 - f) * u).transpose(1, 2).contiguous())
        return self.mem_o(m.transpose(1, 2).to(z.dtype))

    def forward(self, h):
        z = self.ln1(h)
        c = F.gelu(self.conv(F.pad(z.transpose(1, 2), (self.k - 1, 0))).transpose(1, 2))
        if self.mezcla == 'memoria':
            c = c + self._memoria(z)
        h = h + self.drop(c)
        return h + self.drop(self.ff(self.ln2(h)))


def _modo_checkpoint(valor):
    """False/'no' -> 'no'; True/'asrn' -> 'asrn' (lo de v87); 'todo' -> 'todo'."""
    modo = {False: 'no', None: 'no', True: 'asrn'}.get(valor, valor) if not isinstance(valor, str) else valor
    if modo not in ('no', 'asrn', 'todo'):
        raise ValueError(f"checkpoint: 'no', 'asrn' o 'todo' (llego {valor!r})")
    return modo


def _revisar_precision(modelo):
    """Los pesos van en 32 bits. bf16/fp16 se usan con autocast (precision='bf16' en entrenar(), o
    torch.autocast al usar el modelo), no convirtiendo el modelo con .to(torch.bfloat16) o .half(): la capa mezcla
    cuentas en 32 y 64 bits (reloj, firmas, claves de 'vistos') que con pesos de 16 bits fallan o pierden exactitud."""
    if modelo.emb.weight.dtype in (torch.float16, torch.bfloat16):
        raise ValueError(f"el modelo tiene los pesos en {modelo.emb.weight.dtype}: ASRN necesita pesos en 32 bits. "
                         "Para bf16/fp16 usa autocast (entrenar(..., precision='bf16') o "
                         "with torch.autocast('cuda', dtype=torch.bfloat16): ...) y deja el modelo en float32")


def _escalar_profundo(modulo, n_capas):
    """Arranque para modelos profundos (v80): las salidas que se suman al estado se achican por 1/sqrt(2 n_capas)."""
    f = 1.0 / (2 * n_capas) ** 0.5
    if isinstance(modulo, BloqueLocal):
        modulo.conv.weight.mul_(f)
        modulo.ff[2].weight.mul_(f)
        if modulo.mezcla == 'memoria':
            modulo.mem_o.weight.mul_(f)
    else:
        modulo.o.weight.mul_(f)


class ModeloASRN(nn.Module):
    """Palabras -> n_capas bloques locales, con una CapaASRN despues del bloque i cuando i % largo_cada == 0 (sin
    contar el ultimo) -> salida (atada a los vectores de palabra). No tiene tabla de posiciones.
    init_profundo=None: arranque para modelos profundos si n_capas >= 8. checkpoint: menos memoria al entrenar
    (mismas cuentas, algo mas de tiempo): 'asrn' (o True, como en v87) recalcula en la vuelta solo las capas ASRN,
    'todo' tambien los bloques locales, 'no' (o False) nada. dropout: solo al entrenar (0 = receta original)."""

    def __init__(self, V, d=64, n_capas=2, largo_cada=2, ff=None, k=4, init_profundo=None, checkpoint=False,
                 dropout=0.0, mezcla='conv', memoria_max=1024, **opciones_capa):
        super().__init__()
        assert n_capas >= 2 and largo_cada >= 1
        assert V <= 2 ** 24, "las firmas por numero de pieza son exactas hasta 2**24 piezas"
        self.config = dict(V=V, d=d, n_capas=n_capas, largo_cada=largo_cada, ff=ff or 2 * d, k=k, dropout=dropout,
                           mezcla=mezcla, memoria_max=memoria_max,
                           opciones_capa=dict(opciones_capa))
        self.n_capas, self.checkpoint = n_capas, checkpoint
        self.init_profundo = n_capas >= 8 if init_profundo is None else bool(init_profundo)
        self.pos_largo = [i for i in range(n_capas - 1) if i % largo_cada == 0]
        self.emb = nn.Embedding(V, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight
        self.bloques = nn.ModuleList([BloqueLocal(d, k, ff or 2 * d, dropout, mezcla, memoria_max)
                                      for _ in range(n_capas)])
        self.largos = nn.ModuleList([CapaASRN(d, dropout=dropout, **opciones_capa) for _ in self.pos_largo])
        if self.init_profundo:
            with torch.no_grad():
                for b in self.bloques:
                    _escalar_profundo(b, n_capas)
                for c in self.largos:
                    _escalar_profundo(c, n_capas)

    @property
    def checkpoint(self):
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, valor):
        self._checkpoint = _modo_checkpoint(valor)

    def forward(self, x):
        _revisar_precision(self)
        E = self.emb(x)
        h, j = E, 0
        recalcular = self.training and torch.is_grad_enabled()
        ck_bloques = recalcular and self.checkpoint == 'todo'
        ck_asrn = recalcular and self.checkpoint in ('asrn', 'todo')
        for i, blq in enumerate(self.bloques):
            h = torch.utils.checkpoint.checkpoint(blq, h, use_reentrant=False) if ck_bloques else blq(h)
            if i in self.pos_largo:
                if ck_asrn:
                    h = torch.utils.checkpoint.checkpoint(self.largos[j], E, h, x, use_reentrant=False)
                else:
                    h = self.largos[j](E, h, x)
                j += 1
        return self.head(self.ln_f(h))

    def perdida(self, x, y, ignorar=-100):
        """Entropia cruzada media (en 32 bits) de predecir y a partir de x, ambos (lote, largo)."""
        logits = self(x)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=ignorar)

    def num_parametros(self):
        return sum(p.numel() for p in self.parameters())


def crear_modelo(V, d=64, n_capas=2, largo_cada=2, semilla=None, **kw):
    """ModeloASRN con la receta final. semilla: fija el azar justo antes de construir (el mismo arranque que tuvo
    esa semilla en las pruebas). kw: ff, k, init_profundo, checkpoint, dropout u opciones de CapaASRN."""
    if semilla is not None:
        torch.manual_seed(semilla)
    return ModeloASRN(V, d, n_capas, largo_cada, **kw)


# ----------------------------------------------------------------------------------------------------------------
# Configuracion: todo el modelo en un solo objeto
# ----------------------------------------------------------------------------------------------------------------

_OPCIONES_CAPA = ('n_tablas', 'n_bits', 'afin_oculto', 'chunk', 'tablas_contexto', 'confirmar', 'sufijo', 'vistos',
                  'nodos_racha', 'racha_max')


@dataclasses.dataclass(frozen=True)
class ConfigASRN:
    """Todo lo que define a un ModeloASRN. Los valores por defecto son la receta final (la de v87).

    Modelo: V (vocabulario), d (ancho), n_capas (bloques locales; hay una capa ASRN despues de cada `largo_cada`
    bloques, sin contar el ultimo), ff (capa densa del bloque; None = 2d), k (ventana de la convolucion local),
    dropout, init_profundo (None = si n_capas >= 8), checkpoint ('no', 'asrn' o 'todo'), mezcla y memoria_max (el
    bloque local: ver BloqueLocal).
    Capa ASRN: n_tablas, n_bits, afin_oculto, chunk, tablas_contexto, confirmar, sufijo, vistos, nodos_racha, racha_max
    (ver CapaASRN)."""
    V: int
    d: int = 64
    n_capas: int = 2
    largo_cada: int = 2             # una capa ASRN cada dos bloques: misma calidad, 16-18% mas rapido que cada uno
    ff: int = None                  # None = 2d (sigue a d si se cambia d con con())
    k: int = 4
    dropout: float = 0.0
    init_profundo: bool = None
    checkpoint: str = 'no'
    mezcla: str = 'conv'
    memoria_max: int = 1024
    n_tablas: int = 8
    n_bits: int = 14
    afin_oculto: int = 128
    chunk: int = 32
    tablas_contexto: int = 4
    confirmar: int = 16
    sufijo: int = 4
    vistos: int = 4
    nodos_racha: int = 16
    racha_max: int = 256

    def __post_init__(self):
        object.__setattr__(self, 'checkpoint', _modo_checkpoint(self.checkpoint))
        self.validar()

    def validar(self):
        """Revisa que la configuracion tenga sentido y explica que esta mal si no (ValueError)."""
        def falla(msg):
            raise ValueError(f"ConfigASRN: {msg}")
        enteros = ('V', 'd', 'n_capas', 'largo_cada', 'k') + _OPCIONES_CAPA
        for n in enteros:
            if not isinstance(getattr(self, n), int) or isinstance(getattr(self, n), bool):
                falla(f"{n} debe ser un numero entero (llego {getattr(self, n)!r})")
        if not 2 <= self.V <= 2 ** 24:
            falla(f"V={self.V}: entre 2 y 2**24 piezas (las firmas por numero de pieza son exactas hasta 2**24)")
        if self.d < 1 or self.k < 1:
            falla("d y k deben ser >= 1")
        if self.n_capas < 2:
            falla(f"n_capas={self.n_capas}: hacen falta al menos 2 (la capa ASRN va entre bloques)")
        if self.largo_cada < 1:
            falla("largo_cada debe ser >= 1")
        if self.ff is not None and (not isinstance(self.ff, int) or self.ff < 1):
            falla(f"ff={self.ff!r}: None (= 2d) o un entero >= 1")
        if not 0.0 <= self.dropout < 1.0:
            falla(f"dropout={self.dropout}: entre 0 y 1 (sin llegar a 1)")
        if not 0 < self.tablas_contexto < self.n_tablas:
            falla(f"tablas_contexto={self.tablas_contexto}: entre 1 y n_tablas-1 (n_tablas={self.n_tablas})")
        if not 1 <= self.n_bits <= 30:
            falla(f"n_bits={self.n_bits}: entre 1 y 30")
        for n in ('afin_oculto', 'chunk', 'confirmar', 'sufijo', 'vistos', 'racha_max'):
            if getattr(self, n) < 1:
                falla(f"{n} debe ser >= 1")
        if self.nodos_racha < 2:
            falla("nodos_racha debe ser >= 2")
        if self.mezcla not in _MEZCLAS:
            falla(f"mezcla={self.mezcla!r}: una de {', '.join(_MEZCLAS)} (ver BloqueLocal)")
        if not isinstance(self.memoria_max, int) or isinstance(self.memoria_max, bool) or self.memoria_max < 2:
            falla(f"memoria_max={self.memoria_max!r}: un entero >= 2 (piezas que recuerda la memoria del bloque)")
        return self

    def con(self, **cambios):
        """Copia con algunos valores cambiados (y revisada): TALLAS['125m'].con(V=8000)."""
        return dataclasses.replace(self, **cambios)

    def con_mismos_parametros(self, **cambios):
        """Copia con `cambios` y el ancho de la capa densa (ff, multiplo de 8) que deja el total de parametros lo
        mas cerca posible del de esta configuracion: para comparar variantes del bloque con el mismo presupuesto
        (p. ej. cfg.con_mismos_parametros(mezcla='memoria'))."""
        objetivo = _contar_formula(self)
        base = dataclasses.replace(self, **cambios)
        return min((dataclasses.replace(base, ff=f) for f in range(8, 16 * base.d + 1, 8)),
                   key=lambda c: abs(_contar_formula(c) - objetivo))

    def opciones_capa(self):
        return {n: getattr(self, n) for n in _OPCIONES_CAPA}

    def a_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def desde_dict(cls, dic):
        desconocidos = set(dic) - {c.name for c in dataclasses.fields(cls)}
        if desconocidos:
            raise ValueError(f"ConfigASRN: campos desconocidos {sorted(desconocidos)}")
        return cls(**dic)

    @classmethod
    def desde_modelo(cls, modelo):
        """La configuracion de un ModeloASRN ya creado (o cargado)."""
        base = _base(modelo)
        return cls.desde_config_guardada(base.config, init_profundo=base.init_profundo, checkpoint=base.checkpoint)

    @classmethod
    def desde_config_guardada(cls, c, **extra):
        """Desde el diccionario 'config' que guardan guardar() y entrenar() (sin cargar los pesos)."""
        ff = None if c['ff'] == 2 * c['d'] else c['ff']
        return cls(V=c['V'], d=c['d'], n_capas=c['n_capas'], largo_cada=c['largo_cada'], ff=ff, k=c['k'],
                   dropout=c.get('dropout', 0.0), mezcla=c.get('mezcla', 'conv'),
                   memoria_max=c.get('memoria_max', 1024), **extra, **{**ConfigASRN(V=2).opciones_capa(), **c['opciones_capa']})

    @classmethod
    def para_parametros(cls, objetivo, V, **fijos):
        """La configuracion con la receta final mas cercana a `objetivo` parametros: un numero (50e6) o un texto
        ('3k', '50m', '1.5b'). V es el vocabulario de tu tokenizador. fijos: cualquier campo que quieras dejar fijo
        (d=512, n_capas=12, largo_cada=2, ...); la busqueda solo mueve lo demas.

        Busca el ancho d y el numero de capas: entre las formas que quedan a menos de 3% del objetivo elige la de
        proporcion d / capas mas cercana a 64 (ni muy ancha ni muy profunda); si ninguna llega a 3%, la mas cercana con
        un aviso, y si ni esa queda a menos de 50% (V muy grande para el objetivo, o valores fijados que no dejan
        llegar), ValueError explicando por que.
        Si la capa ASRN normal no cabe (objetivos de pocos miles), usa la capa reducida (2 tablas de 6 bits).
        Es una regla de forma razonable, no una forma optima medida: eso lo diran las curvas de escalado."""
        return _para_parametros(objetivo, V, fijos)

    def contar_parametros(self, desglose=False):
        """Parametros exactos, sin gastar memoria (el modelo se arma en el dispositivo 'meta', que solo guarda
        formas). desglose=True: {'vocabulario', 'bloques_locales', 'capas_asrn', 'otros', 'total'} (el vocabulario
        de entrada y la salida son la misma matriz: se cuenta una vez)."""
        with torch.device('meta'):
            m = ModeloASRN(self.V, self.d, self.n_capas, self.largo_cada, self.ff, self.k,
                           init_profundo=False, dropout=self.dropout, mezcla=self.mezcla,
                           memoria_max=self.memoria_max, **self.opciones_capa())
        total = m.num_parametros()
        if not desglose:
            return total
        voc = m.emb.weight.numel()
        blq = sum(p.numel() for p in m.bloques.parameters())
        asrn = sum(p.numel() for p in m.largos.parameters())
        return {'vocabulario': voc, 'bloques_locales': blq, 'capas_asrn': asrn,
                'otros': total - voc - blq - asrn, 'total': total}


def _cfg_chica(V, d, n_capas, ff, k):
    """Capa ASRN reducida para los modelos mas chicos: 2 tablas de 6 bits, hash de 16 de ancho oculto."""
    return ConfigASRN(V=V, d=d, n_capas=n_capas, ff=ff, k=k, n_tablas=2, tablas_contexto=1, n_bits=6, afin_oculto=16,
                      confirmar=4, vistos=2, nodos_racha=4)


# Ejemplos de configuraciones (para cualquier otro tamano: ConfigASRN.para_parametros('50m', V=...)).
# Los numeros de parametros son exactos con el V que trae cada una (cambiar V los cambia):
#   print({n: c.contar_parametros() for n, c in TALLAS.items()})
# Hasta hoy solo se entrenaron modelos de hasta ~300 millones, con versiones anteriores. 'nano-3k' y '12m' nunca se
# entrenaron con esta receta; de '1b' para arriba son configuraciones sin entrenar: el codigo las crea y las reparte,
# pero nadie ha medido como aprenden.
TALLAS = {
    'nano-3k':   _cfg_chica(V=48, d=10, n_capas=2, ff=24, k=3),           # ~3 mil, letras
    '12m':       ConfigASRN(V=8000, d=256, n_capas=16),                    # ~12 millones
    '30m':       ConfigASRN(V=32768, d=384, n_capas=13),
    '125m':      ConfigASRN(V=32768, d=768, n_capas=18),
    '350m':      ConfigASRN(V=32768, d=1024, n_capas=31),
    '1b':        ConfigASRN(V=32768, d=2048, n_capas=27),
    '7b':        ConfigASRN(V=32768, d=4096, n_capas=44),
    '27b':       ConfigASRN(V=32768, d=8192, n_capas=44),
}


_SUFIJOS = {'k': 1e3, 'm': 1e6, 'b': 1e9, 't': 1e12}


def leer_cantidad(texto):
    """'3k' -> 3000, '50m' -> 50 000 000, '1.5b' -> 1 500 000 000; un numero se devuelve tal cual (entero)."""
    if isinstance(texto, (int, float)):
        valor = float(texto)
    else:
        t = str(texto).strip().lower().replace('_', '').replace(',', '')
        try:
            valor = float(t[:-1]) * _SUFIJOS[t[-1]] if t and t[-1] in _SUFIJOS else float(t)
        except ValueError:
            raise ValueError(f"cantidad de parametros no entendida: {texto!r} (ejemplos: 3k, 50m, 1.5b, 7e9)")
    if not (math.isfinite(valor) and valor >= 1):
        raise ValueError(f"cantidad de parametros: hace falta un numero positivo (llego {texto!r})")
    return int(round(valor))


def _contar_formula(cfg):
    """El mismo numero que cfg.contar_parametros(), con una formula (sin armar el modelo): para buscar entre miles de
    formas al instante. La prueba test_formula_de_parametros_exacta la compara con el conteo real."""
    d, V, ff, k = cfg.d, cfg.V, cfg.ff or 2 * cfg.d, cfg.k
    conv = k * d * d + d if cfg.mezcla == 'conv' else k * d + d
    memoria = (d * 2 * d + 2 * d) + d * d if cfg.mezcla == 'memoria' else 0
    densa = d * ff + ff + ff * d + d
    bloque = 2 * d + conv + memoria + 2 * d + densa
    T, Tc, nb, m = cfg.n_tablas, cfg.tablas_contexto, cfg.n_bits, max(8, cfg.afin_oculto // 2)
    capa = ((d * m + m) + (m * (T - Tc) * nb + (T - Tc) * nb)            # hash de palabra sola
            + (2 * d * m + m) + (m * Tc * nb + Tc * nb)                   # hash de contexto
            + 2 * d * d + (d + 1) + 5 + 2 * d * cfg.confirmar + 2 * d     # v, o, g, g_conf, cq, ck, ln_ctx
            + cfg.vistos + (cfg.sufijo + 1) + (2 * d + 2) + T             # w_vistos, w_sufijo, reloj, gamma
            + cfg.vistos + 1 + cfg.vistos + cfg.nodos_racha)              # kappas, u_hawkes, d_racha
    n_asrn = len([i for i in range(cfg.n_capas - 1) if i % cfg.largo_cada == 0])
    return V * d + 2 * d + cfg.n_capas * bloque + n_asrn * capa


_REDUCIDA = dict(n_tablas=2, tablas_contexto=1, n_bits=6, afin_oculto=16, confirmar=4, vistos=2, nodos_racha=4)


def _para_parametros(objetivo, V, fijos):
    objetivo = leer_cantidad(objetivo)
    fijos = dict(fijos)
    validos = {c.name for c in dataclasses.fields(ConfigASRN)} - {'V'}
    if set(fijos) - validos:
        raise ValueError(f"para_parametros: campos desconocidos {sorted(set(fijos) - validos)} "
                         f"(se pueden fijar: {', '.join(sorted(validos))})")
    d_fijo, n_fijo = fijos.pop('d', None), fijos.pop('n_capas', None)
    anchos = ([d_fijo] if d_fijo is not None
              else list(range(4, 64, 2)) + list(range(64, 2048, 8)) + list(range(2048, 32769, 64)))
    mejor = {}
    for reducida in (False, True):
        if reducida and any(n in fijos for n in _REDUCIDA):
            break                                   # el usuario fijo la capa: no se cambia por la reducida
        extra = {**(_REDUCIDA if reducida else {}), **fijos}
        plantilla = ConfigASRN(V=V, d=anchos[0], n_capas=2 if n_fijo is None else n_fijo, **extra)   # revisa lo fijo
        cands = []
        for d in anchos:
            for n in ([n_fijo] if n_fijo is not None else range(2, 513)):
                c = dataclasses.replace(plantilla, d=d, n_capas=n)
                total = _contar_formula(c)
                cands.append((abs(total - objetivo) / objetivo, abs(math.log(d / n / 64)), d, n, total))
                if total > 1.5 * objetivo:
                    break
            if _contar_formula(dataclasses.replace(plantilla, d=d, n_capas=n_fijo or 2)) > 1.5 * objetivo:
                break
        cerca = [c for c in cands if c[0] <= 0.03]
        elegido = min(cerca, key=lambda c: (c[1], c[0])) if cerca else min(cands, key=lambda c: c[0])
        mejor[reducida] = (elegido, extra)
        if cerca:
            break                                   # la receta normal alcanza: no hace falta la reducida
    (err, _, d, n, total), extra = min(mejor.values(), key=lambda par: par[0][0])
    if err > 0.03:                                  # lo pedido no se puede: decirlo en vez de devolver otra cosa
        fijado = {k: v for k, v in dict(d=d_fijo, n_capas=n_fijo, **fijos).items() if v is not None}
        msg = (f"para_parametros: lo mas cercano a {objetivo:,} parametros con V={V:,}"
               + (f" y lo fijado ({', '.join(f'{k}={v}' for k, v in fijado.items())})" if fijado else "")
               + f" es d={d}, n_capas={n}: {total:,} parametros ({100 * (total / objetivo - 1):+.0f}%).")
        if total > objetivo and V * d > objetivo / 2:
            msg += (f" Solo el vocabulario (V x d = {V * d:,}) ocupa {100 * V * d / objetivo:.0f}% de lo pedido: pide "
                    "mas parametros o usa un V menor" + (" o un d menor." if d_fijo is not None else "."))
        else:
            msg += " Suelta alguno de los valores fijados o pide otro tamano."
        if err > 0.5:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=3)
    return ConfigASRN(V=V, d=d, n_capas=n, **extra)


def crear_desde_config(cfg, semilla=None, dispositivo=None):
    """ModeloASRN desde una ConfigASRN (o el nombre de una de TALLAS). Con semilla, el mismo arranque que
    crear_modelo(..., semilla=semilla). dispositivo='meta': solo formas, sin memoria (para llenarlo despues con
    inicializar(), por ejemplo ya repartido con repartir_fsdp())."""
    if isinstance(cfg, str):
        cfg = TALLAS[cfg]
    cfg.validar()
    kw = dict(ff=cfg.ff, k=cfg.k, init_profundo=cfg.init_profundo, checkpoint=cfg.checkpoint, dropout=cfg.dropout,
              mezcla=cfg.mezcla, memoria_max=cfg.memoria_max, **cfg.opciones_capa())
    if dispositivo is not None and torch.device(dispositivo).type == 'meta':
        with torch.device('meta'):
            return ModeloASRN(cfg.V, cfg.d, cfg.n_capas, cfg.largo_cada, **kw)
    m = crear_modelo(cfg.V, cfg.d, cfg.n_capas, cfg.largo_cada, semilla=semilla, **kw)
    return m if dispositivo is None else m.to(dispositivo)


def _copiar_a(destino, origen):
    """Copia los tensores de `origen` (completos, en CPU) en los de `destino`, que pueden estar repartidos (DTensor
    de FSDP): cada proceso se queda solo con su pedazo, sin comunicacion."""
    fuente = dict(origen.named_parameters())
    fuente.update(origen.named_buffers())
    for nombre, t in list(destino.named_parameters()) + list(destino.named_buffers()):
        completo = fuente[nombre]
        with torch.no_grad():
            if _es_dtensor(t):
                from torch.distributed.tensor import distribute_tensor
                t.copy_(distribute_tensor(completo.to(t.device, t.dtype), t.device_mesh, t.placements,
                                          src_data_rank=None))
            else:
                t.copy_(completo.to(t.device, t.dtype))


def _es_dtensor(t):
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    return isinstance(t, DTensor)


def inicializar(modelo, semilla=0):
    """Llena todos los pesos y buffers de `modelo` con exactamente los mismos numeros que tendria
    crear_modelo(..., semilla=semilla). Sirve para un modelo creado en 'meta' (despues de to_empty) o ya repartido
    con FSDP: se arma pieza por pieza en CPU (un bloque o una capa a la vez, en el mismo orden que el constructor, asi
    el azar sale igual) y se copia; nunca existe el modelo entero en una sola maquina. Todos los procesos deben usar la
    misma semilla. No cambia el azar global."""
    base = _base(modelo)
    if base.head.weight is not base.emb.weight:
        if esta_repartido(base):
            raise ValueError("la salida y el vocabulario ya no son la misma matriz: usa materializar() (o "
                             "to_empty antes de repartir_fsdp)")
        base.head.weight = base.emb.weight                # to_empty() separa las matrices atadas: se vuelven a atar
    c = base.config
    d, V, n = c['d'], c['V'], c['n_capas']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(semilla)
        with torch.device('cpu'):
            emb = nn.Embedding(V, d)                      # el mismo orden de consumo del azar que ModeloASRN
            nn.init.normal_(emb.weight, std=0.02)
            _copiar_a(base.emb, emb)
            _copiar_a(base.ln_f, nn.LayerNorm(d))
            del emb
            nn.Linear(d, V, bias=False)                   # la salida consume azar aunque luego se ata a emb
            for blq in base.bloques:
                nuevo = BloqueLocal(d, c['k'], c['ff'], c['dropout'], c.get('mezcla', 'conv'),
                                    c.get('memoria_max', 1024))
                if base.init_profundo:
                    with torch.no_grad():
                        _escalar_profundo(nuevo, n)
                _copiar_a(blq, nuevo)
            for capa in base.largos:
                nueva = CapaASRN(d, dropout=c['dropout'], **c['opciones_capa'])
                if base.init_profundo:
                    with torch.no_grad():
                        _escalar_profundo(nueva, n)
                _copiar_a(capa, nueva)
    return modelo


def materializar(modelo, dispositivo, semilla=0):
    """Modelo creado en 'meta' (repartido o no) -> pesos reales en `dispositivo`, iguales a crear_modelo(semilla)."""
    base = _base(modelo)
    base.to_empty(device=dispositivo)
    if not esta_repartido(base):
        base.head.weight = base.emb.weight
    return inicializar(modelo, semilla)


# ----------------------------------------------------------------------------------------------------------------
# Repartir entre procesos (FSDP)
# ----------------------------------------------------------------------------------------------------------------

def _distribuido():
    """(rango, n_procesos) si hay un grupo de procesos de torch.distributed iniciado; (0, 1) si no."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def esta_repartido(modelo):
    return any(_es_dtensor(p) for p in _base(modelo).parameters())


def repartir_fsdp(modelo, malla=None, precision_pesos=None, liberar_tras_ida=True):
    """Reparte los pesos de `modelo` entre los procesos del grupo (FSDP2, torch.distributed.fsdp.fully_shard): cada
    bloque local y cada capa ASRN es una unidad, y lo demas (vocabulario y salida, que son la misma matriz) queda en
    la raiz. Cada proceso guarda 1/n de los pesos, de los gradientes y del estado de AdamW; en la ida se juntan los de
    una unidad a la vez.

    precision_pesos=None: los pesos se juntan en 32 bits y las cuentas van en bf16/fp16 por autocast, como sin
    repartir (el reloj, Hawkes y las firmas siguen en 32 bits donde la capa lo pide). Juntar en 16 bits
    (torch.bfloat16) todavia no funciona: la capa necesita pesos de 32 bits (ver _revisar_precision) y se rechaza.
    El modelo puede estar en 'meta' (creado con crear_desde_config(..., dispositivo='meta')): despues hace falta
    modelo.to_empty(device=...) e inicializar(modelo, semilla). Devuelve el mismo modelo, repartido."""
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    if precision_pesos not in (None, torch.float32):
        raise ValueError(f"precision_pesos={precision_pesos}: todavia no funciona (la capa ASRN necesita pesos de 32 "
                         "bits). Deja precision_pesos=None: las cuentas van igual en bf16 con precision='bf16'")
    base = _base(modelo)
    kw = {'reshard_after_forward': liberar_tras_ida}
    if malla is None:
        malla = _malla_por_defecto()
    if malla is not None:
        kw['mesh'] = malla
    if precision_pesos is not None:
        kw['mp_policy'] = MixedPrecisionPolicy(param_dtype=precision_pesos, reduce_dtype=torch.float32)
    for blq in base.bloques:
        fully_shard(blq, **kw)
    for capa in base.largos:
        fully_shard(capa, **kw)
    fully_shard(base, **kw)
    return modelo


def _malla_por_defecto():
    """La malla (DeviceMesh: como se ordenan los procesos) que usa fully_shard si no se le da una. Sin malla,
    fully_shard la arma en la GPU si hay GPU, aunque el grupo use gloo (comunicacion por CPU): los pesos quedan en la
    GPU y la entrada en la CPU, y la ida falla. Con gloo la malla va en la CPU; con nccl, fully_shard decide."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized() and dist.get_backend() == 'gloo':
        from torch.distributed.device_mesh import init_device_mesh
        return init_device_mesh('cpu', (dist.get_world_size(),))
    return None


def pesos_completos(modelo):
    """state_dict completo en CPU, aunque el modelo este repartido. Repartido: TODOS los procesos deben llamarla
    (juntar es comunicacion colectiva) y solo el proceso 0 recibe los pesos; los demas, un diccionario vacio (asi no
    hay n copias del modelo entero en la memoria de la maquina)."""
    base = _base(modelo)
    if esta_repartido(base):
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        pesos = get_model_state_dict(base, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if 'emb.weight' in pesos:
            pesos['head.weight'] = pesos['emb.weight']          # la misma matriz: torch.save la guarda una vez
        return pesos
    return {k: v.detach().cpu() for k, v in base.state_dict().items()}


_PARTES_COMPILABLES = ('_phi_y_valores', '_contexto', '_repeticion', '_combinar')


def compilar_partes(modelo):
    """torch.compile por partes: cada bloque local y, en cada capa ASRN, sus partes chicas (hash, valores, reloj y
    confirmacion, Hawkes y racha, puerta). Junta sus operaciones y conversiones de tipo en pocos kernels; la
    atencion por casilleros y el escaneo siguen con sus kernels Triton (compilar el modelo entero los cambiaria por
    las versiones de PyTorch). Mismas cuentas salvo el redondeo. Necesita PyTorch >= 2.14 (antes, Inductor falla
    con este modelo). Compila para formas fijas (dynamic=False): cada forma nueva de lote compila de nuevo, unas 10
    veces mas rapido que la compilacion para formas variables (que con este modelo pasaba de una hora en una A100).
    Las 9 capas y los modelos nuevos comparten lo compilado. Devuelve el mismo modelo; llamarla dos veces no hace
    nada."""
    base = _base(modelo)
    if torch.__version__ < _TORCH_COMPILE_MINIMO:
        raise RuntimeError(f"compilar_partes necesita PyTorch {_TORCH_COMPILE_MINIMO} o mas nuevo (hay "
                           f"{torch.__version__}): antes, torch.compile falla con este modelo")
    for m in base.modules():
        if getattr(m, '_partes_compiladas', False):
            continue
        if isinstance(m, BloqueLocal):
            m.forward = torch.compile(m.forward, dynamic=False)
            m._partes_compiladas = True
        elif isinstance(m, CapaASRN):
            for nombre in _PARTES_COMPILABLES:
                setattr(m, nombre, torch.compile(getattr(m, nombre), dynamic=False))
            m._partes_compiladas = True
    return modelo


def _base(modelo):
    """El modelo de verdad aunque venga envuelto por torch.compile."""
    return getattr(modelo, '_orig_mod', modelo)


# ----------------------------------------------------------------------------------------------------------------
# Escribir
# ----------------------------------------------------------------------------------------------------------------

_ANILLO_MAX_BYTES = 2 * 1024 ** 3     # tope de los anillos de la escritura O(1) (todas las capas); si no caben, O(t)


def _anillo_desde(ids, n, C, nc):
    """Los C companeros de casillero mas recientes de cada casillero de cada tabla, entre las posiciones 0..n-1.
    ids (B, L, T) casilleros. Devuelve anillo (B, T, nc, C) int32 con posiciones (-1: ranura vacia) y cuenta
    (B, T, nc) int64: cuantas posiciones entraron en cada casillero. La posicion de lugar r dentro de su casillero
    esta en la ranura r % C; la proxima en entrar va a la ranura cuenta % C y pisa a la mas vieja."""
    B, L, T = ids.shape
    dv = ids.device
    anillo = torch.full((B, T, nc, C), -1, dtype=torch.int32, device=dv)
    cuenta = torch.zeros(B, T, nc, dtype=torch.long, device=dv)
    if n <= 0:
        return anillo, cuenta
    x = ids[:, :n].transpose(1, 2).long()                                    # (B, T, n)
    cuenta.scatter_add_(-1, x, torch.ones_like(x))
    orden = x.argsort(dim=-1, stable=True)                                   # por casillero; dentro, por posicion
    xs = torch.gather(x, -1, orden)
    i = torch.arange(n, device=dv).expand_as(xs)
    nuevo = torch.ones_like(xs, dtype=torch.bool)
    nuevo[..., 1:] = xs[..., 1:] != xs[..., :-1]
    lugar = i - torch.where(nuevo, i, torch.zeros_like(i)).cummax(-1).values     # lugar dentro de su casillero
    queda = lugar >= torch.gather(cuenta, -1, xs) - C                        # solo las C mas recientes
    ranura = (torch.arange(B * T, device=dv).view(B, T, 1) * nc + xs) * C + lugar % C
    anillo.view(-1)[ranura[queda]] = orden[queda].int()
    return anillo, cuenta


def _anillo_meter(anillo, cuenta, ids_t, t):
    """Mete la posicion t, de casilleros ids_t (B, T), en el anillo de cada tabla."""
    B, T = ids_t.shape
    b = torch.arange(B, device=ids_t.device).unsqueeze(1)
    tau = torch.arange(T, device=ids_t.device).unsqueeze(0)
    k = ids_t.long()
    c = cuenta[b, tau, k]                                                    # (B, T)
    anillo[b, tau, k, c % anillo.shape[-1]] = t
    cuenta[b, tau, k] = c + 1


def _candidatos(cache, anillo, ids_t):
    """Lo que hace falta de los candidatos de la pieza nueva: en cada tabla, las posiciones del anillo de su casillero.
    Devuelve (cand, valido): cand phi, Kc, Pf, B, log_alfa, V, cada uno (B, T, C, .) o (B, T, C); valido (B, T, C)."""
    B, T = ids_t.shape
    dv = ids_t.device
    pos = anillo[torch.arange(B, device=dv).unsqueeze(1), torch.arange(T, device=dv).unsqueeze(0),
                 ids_t.long()].long()                                        # (B, T, C)
    valido = pos >= 0
    pos = pos.clamp(min=0)
    b = torch.arange(B, device=dv).view(B, 1, 1)
    return {n: cache[n][b, pos] for n in ('phi', 'Kc', 'Pf', 'B', 'log_alfa', 'V')}, valido


class _MemoriaHawkes:
    """La intensidad de Hawkes de una capa al escribir, en O(1) por pieza. Por cada clave (grupo de 1..n piezas de
    cada texto del lote) guarda el reloj de su ultima aparicion, B_u, y A = lambda + alfa en ese momento. Entonces
        lambda_n(t) = exp(-kappa_n (B_t - B_u)) A,     y despues     A <- lambda_n(t) + alfa_t,   B_u <- B_t:
    la suma de la definicion (sum_s alfa_s exp(-kappa_n (B_t - B_s))) llevada como recurrencia, el mismo paso del
    escaneo que usa el texto entero. Las claves se buscan en diccionarios de Python, con claves calculadas en la CPU
    (las piezas ya estan ahi al escribir: sin esperar a la GPU); los estados quedan en el dispositivo."""

    def __init__(self, B, n, dv, dtype_reloj, capacidad=1024):
        self.n = n
        self.mapas = [{} for _ in range(B * n)]                              # (texto, grupo) -> {clave: ranura}
        self.Bu = torch.zeros(capacidad, dtype=dtype_reloj, device=dv)
        self.A = torch.zeros(capacidad, dtype=torch.float32, device=dv)
        self.usadas = 0

    def _ranuras(self, k):
        while self.usadas + k > self.Bu.shape[0]:                            # crece al doble: O(1) amortizado
            self.Bu = torch.cat([self.Bu, torch.zeros_like(self.Bu)])
            self.A = torch.cat([self.A, torch.zeros_like(self.A)])
        self.usadas += k
        return range(self.usadas - k, self.usadas)

    def cargar(self, claves, hawkes, Bt, log_alfa):
        """El estado al final de la entrada. claves (B, L, n) en la CPU; hawkes (B, L, n), Bt y log_alfa (B, L) del
        texto entero, en el dispositivo. De cada clave cuenta su ultima aparicion: A = hawkes + alfa ahi."""
        B, L, n = claves.shape
        k = claves.transpose(1, 2)                                           # (B, n, L)
        orden = k.argsort(dim=-1, stable=True)
        ks = torch.gather(k, -1, orden)
        ultima = torch.ones_like(ks, dtype=torch.bool)
        ultima[..., :-1] = ks[..., 1:] != ks[..., :-1]
        ultima &= ks == ks                                                   # las claves con NaN no cuentan
        ranuras, bs, js, ps = [], [], [], []
        for b in range(B):
            for j in range(n):
                m = ultima[b, j]
                cl, pos = ks[b, j][m].tolist(), orden[b, j][m].tolist()
                r = self._ranuras(len(cl))
                self.mapas[b * n + j] = dict(zip(cl, r))
                ranuras += list(r)
                bs += [b] * len(cl)
                js += [j] * len(cl)
                ps += pos
        if ranuras:
            dv = self.A.device
            r, b, j, p = (torch.tensor(v, device=dv) for v in (ranuras, bs, js, ps))
            self.Bu[r] = Bt[b, p].to(self.Bu.dtype)
            self.A[r] = (hawkes[b, p, j] + log_alfa[b, p].exp()).float()

    def paso(self, claves_t, B_t, log_alfa_t, kappa):
        """claves_t (B, n) en la CPU (64 bits); B_t, log_alfa_t (B,) y kappa (n,) en el dispositivo. Devuelve vistos y
        hawkes (B, n) en el dispositivo y guarda el estado de la pieza nueva."""
        B, n = claves_t.shape
        valores = claves_t.tolist()
        leer = [[-1] * n for _ in range(B)]
        bs, js, rs = [], [], []
        for b in range(B):
            for j in range(n):
                v = valores[b][j]
                if v != v:                                                   # NaN: el grupo empieza antes del texto
                    continue
                mapa = self.mapas[b * n + j]
                r = mapa.get(v)
                if r is None:
                    r = mapa[v] = self._ranuras(1)[0]
                else:
                    leer[b][j] = r
                bs.append(b)
                js.append(j)
                rs.append(r)
        dv = self.A.device
        leer = torch.tensor(leer, device=dv)                                 # (B, n)
        visto = leer >= 0
        r = leer.clamp(min=0)
        dB = (B_t.unsqueeze(1) - self.Bu[r]).float()                         # resta en el tipo del reloj
        hawkes = torch.where(visto, torch.exp(-kappa * dB) * self.A[r], torch.zeros_like(dB))
        if rs:
            b, j, r = (torch.tensor(v, device=dv) for v in (bs, js, rs))
            self.Bu[r] = B_t[b].to(self.Bu.dtype)
            self.A[r] = hawkes[b, j] + log_alfa_t[b].float().exp()
        return visto.float(), hawkes


class _Escritor:
    """Escritura paso a paso: guarda lo necesario de cada capa y procesa solo la pieza nueva. La memoria se reserva
    de antemano (capacidad) y se escribe en su lugar: cada paso no copia todo lo anterior.

    o1=True (por omision, si los anillos caben en _ANILLO_MAX_BYTES): O(1) por pieza nueva, sin importar cuanto texto
    haya. Cada tabla lleva un anillo por casillero con sus `chunk` posiciones mas recientes (las que _atender
    elegiria entre toda la memoria; con memoria_de_lo_ajeno, solo las de la entrada), la intensidad de Hawkes se lleva
    por clave (_MemoriaHawkes) y la racha es un numero. La memoria sigue creciendo con el texto (lo que se copia vive
    ahi), pero cada pieza lee una cantidad fija: T x chunk candidatos por capa. o1=False: la version O(t) de antes, que
    compara la pieza con toda la memoria (para verificar que las dos dan lo mismo).
    Las piezas que se pasan a paso() pueden estar en la CPU (escribir() las pasa asi): las claves de Hawkes salen de
    ahi y la GPU no tiene que esperar."""

    def __init__(self, modelo, memoria_de_lo_ajeno=True, o1=None):
        self.m, self.ajeno, self.o1 = _base(modelo), memoria_de_lo_ajeno, o1

    @staticmethod
    def _reservar(x, cap):
        buf = x.new_zeros((x.shape[0], cap) + tuple(x.shape[2:]))
        buf[:, :x.shape[1]] = x
        return buf

    def _crecer(self):
        cap = 2 * self.cap
        self.sig = self._reservar(self.sig, cap)
        for c in self.cache:
            for nombre in c:
                c[nombre] = self._reservar(c[nombre], cap)
        self.cap = cap

    def _bytes_anillos(self, B):
        return sum(B * c.n_tablas * 2 ** c.n_bits * (4 * c.chunk + 8) for c in self.m.largos)

    @torch.no_grad()
    def inicio(self, x0, capacidad=0):
        m = self.m
        B, L0 = x0.shape
        if self.o1 is None:
            self.o1 = self._bytes_anillos(B) <= _ANILLO_MAX_BYTES
            if not self.o1:
                warnings.warn(f'escritura O(t): los anillos de la escritura O(1) ocuparian '
                              f'{self._bytes_anillos(B) / 2 ** 30:.1f} GiB (n_bits grande o lote grande)')
        self.L0, self.t, self.cap = L0, L0, max(L0 + 1, capacidad)
        E = m.emb(x0)
        sig = firmas_de_piezas(x0)
        self.sig = self._reservar(sig, self.cap)
        self.e_ult = E[:, -1:]
        h, j = E, 0
        self.buf, self.cache, self.racha, self.anillos, self.hawkes, self.c_cpu = [], [], [], [], [], []
        if self.o1:
            n_max = max(c.vistos for c in m.largos) if len(m.largos) else 1
            sig_cpu = firmas_de_piezas(x0.cpu())                                     # (B, L0)
            self.ventana = F.pad(sig_cpu, (n_max, 0), value=float('nan'))[:, -n_max:].clone()   # (B, n_max)
        for i, blq in enumerate(m.bloques):
            u = blq.ln1(h)
            self.buf.append(F.pad(u, (0, 0, blq.k - 1, 0))[:, u.shape[1]:])
            h = blq(h)
            if i in m.pos_largo:
                capa = m.largos[j]
                phi, ids, V, Qc, Kc, beta, log_alfa = capa._caracteristicas(E, h)
                Bt = beta.cumsum(1)
                rep = capa._repeticion(sig, beta, Bt, log_alfa)
                guardar = [('phi', phi), ('ids', ids), ('V', V), ('Kc', Kc), ('Pf', _anteriores(sig, capa.sufijo)),
                           ('B', Bt), ('log_alfa', log_alfa)]
                if not self.o1:
                    guardar.append(('claves', capa._claves_vistos(sig)))
                self.cache.append({n: self._reservar(x, self.cap) for n, x in guardar})
                self.racha.append(rep['racha'][:, -1])
                if self.o1:
                    self.anillos.append(_anillo_desde(ids, L0 - 1 if self.ajeno else L0, capa.chunk,
                                                      2 ** capa.n_bits))
                    self.c_cpu.append(capa.c_vistos.cpu())
                    mem = _MemoriaHawkes(B, capa.vistos, E.device, Bt.dtype)
                    mem.cargar(_claves_de(sig_cpu, self.c_cpu[-1], capa.vistos), rep['hawkes'], Bt, log_alfa)
                    self.hawkes.append(mem)
                h = capa(E, h, x0)
                j += 1
        return m.head(m.ln_f(h[:, -1:]))[:, -1]

    @torch.no_grad()
    def paso(self, pieza):
        m = self.m
        if self.t >= self.cap:
            self._crecer()
        t = self.t
        dv = self.sig.device
        pieza_cpu = pieza.view(-1).cpu() if self.o1 else None
        pieza = pieza.view(-1).to(dv)
        e_t = m.emb(pieza.view(-1, 1))                                           # (B, 1, d)
        B = e_t.shape[0]
        self.sig[:, t] = firmas_de_piezas(pieza)
        if self.o1:
            self.ventana = torch.cat([self.ventana[:, 1:], firmas_de_piezas(pieza_cpu).unsqueeze(1)], 1)
            perm = None
        else:
            perm = (torch.arange(t, device=dv) + 1 < self.L0).unsqueeze(0).expand(B, -1) if self.ajeno else None
        h, j = e_t, 0
        for i, blq in enumerate(m.bloques):
            ventana = torch.cat([self.buf[i], blq.ln1(h)], 1)
            z = blq.conv(ventana.transpose(1, 2)).transpose(1, 2)
            self.buf[i] = ventana[:, 1:]
            h = h + F.gelu(z)
            h = h + blq.ff(blq.ln2(h))
            if i in m.pos_largo:
                capa, c = m.largos[j], self.cache[j]
                c['V'][:, t - 1] = capa.v(e_t[:, 0])
                phi_t = capa._phi(torch.cat([self.e_ult, e_t], 1))[:, -1:]                  # (B, 1, T*nb)
                ids_t = capa._ids(phi_t)                                                    # (B, 1, T)
                Qc_t, Kc_t, beta_t, log_alfa_t = capa._contexto(h)                         # (B, 1, .)
                c['B'][:, t] = c['B'][:, t - 1] + beta_t[:, 0]                              # el reloj avanza
                c['log_alfa'][:, t] = log_alfa_t[:, 0]
                if self.o1:
                    # firmas previas de la posicion t (sufijo) y, en la CPU, las claves de 'vistos'
                    c['Pf'][:, t] = _anteriores(self.sig[:, max(0, t - capa.sufijo):t + 1], capa.sufijo)[:, -1]
                    cand, valido = _candidatos(c, self.anillos[j][0], ids_t[:, 0])
                    acc = capa._atender_cand(phi_t, Qc_t, c['Pf'][:, t:t + 1], c['B'][:, t:t + 1], cand, valido)
                    claves = _claves_de(self.ventana[:, -capa.vistos:], self.c_cpu[j], capa.vistos)[:, -1]
                    acc.update(capa._repeticion_o1(self.hawkes[j], claves, c['B'][:, t], log_alfa_t[:, 0],
                                                   beta_t[:, 0], self.racha[j]))
                else:
                    # firmas previas y claves de 'vistos' de la posicion t: las mismas cuentas que para el texto entero
                    reciente = self.sig[:, max(0, t + 1 - max(capa.sufijo + 1, capa.vistos)):t + 1]
                    c['Pf'][:, t] = _anteriores(reciente, capa.sufijo)[:, -1]
                    claves = capa._claves_vistos(reciente)[:, -1]                            # (B, vistos)
                    c['claves'][:, t] = claves
                    mem = {n: c[n][:, :t] for n in ('phi', 'ids', 'V', 'Kc', 'Pf', 'B', 'log_alfa')}
                    acc = capa._atender(phi_t, ids_t, Qc_t, c['Pf'][:, t:t + 1], c['B'][:, t:t + 1], mem, perm)
                    acc.update(capa._repeticion_paso(c['claves'][:, :t], claves, c['B'][:, :t], c['B'][:, t],
                                                     c['log_alfa'][:, :t], beta_t[:, 0], self.racha[j]))
                self.racha[j] = acc['racha'][:, 0]
                h = capa._combinar(acc, h)
                c['phi'][:, t], c['ids'][:, t], c['Kc'][:, t] = phi_t[:, 0], ids_t[:, 0], Kc_t[:, 0]
                if self.o1 and not self.ajeno:
                    _anillo_meter(*self.anillos[j], ids_t[:, 0], t)                 # t es llave desde la pieza t+1
                j += 1
        self.e_ult = e_t
        self.t += 1
        return m.head(m.ln_f(h))[:, -1]


class _Ngramas:
    """De cada texto del lote: para cada grupo de n-1 piezas, las piezas que lo siguieron. Que piezas cierran un grupo
    de n que ya salio (siguiendo el final del texto) se consulta en O(1) por pieza escrita, sin recorrer el texto; es
    lo mismo que _continuaciones_de_cita sobre el texto entero."""

    def __init__(self, textos, n):
        self.n = n
        self.textos = [list(t) for t in textos]
        self.mapas = []
        for t in self.textos:
            mapa = {}
            for i in range(n - 1, len(t)):
                mapa.setdefault(tuple(t[i - n + 1:i]), set()).add(t[i])
            self.mapas.append(mapa)

    def agregar(self, piezas):
        for t, mapa, p in zip(self.textos, self.mapas, piezas):
            t.append(p)
            if len(t) >= self.n:
                mapa.setdefault(tuple(t[len(t) - self.n:-1]), set()).add(p)

    def siguientes(self, V, dv):
        """(B, V) bool: las piezas que, escritas ahora, cierran un grupo de n piezas que ya salio."""
        filas, cols = [], []
        for b, (t, mapa) in enumerate(zip(self.textos, self.mapas)):
            if len(t) >= self.n:
                s = mapa.get(tuple(t[len(t) - self.n + 1:]), ())
                filas += [b] * len(s)
                cols += list(s)
        mascara = torch.zeros(len(self.textos), V, dtype=torch.bool)
        if cols:
            mascara[filas, cols] = True
        return mascara.to(dv)


def _frenos(lg, x, L0, pena_repeticion, ventana_pena, no_repetir_ngramas):
    """Pena de repeticion sobre las ultimas `ventana_pena` piezas y prohibicion de repetir n-gramas ya escritos."""
    B = x.shape[0]
    if pena_repeticion != 1.0:
        vis = torch.zeros_like(lg, dtype=torch.bool).scatter_(1, x[:, -ventana_pena:], True)
        lg = torch.where(vis, torch.where(lg > 0, lg / pena_repeticion, lg * pena_repeticion), lg)
    n = no_repetir_ngramas
    if n and x.shape[1] - L0 >= n:
        gen = x[:, L0:]
        if n == 1:
            lg = lg.scatter(1, gen, -float('inf'))
        else:
            grupos = gen.unfold(1, n, 1)                                                     # (B, G, n)
            coincide = (grupos[..., :-1] == gen[:, -(n - 1):].unsqueeze(1)).all(-1)          # (B, G)
            prohibidas = torch.zeros(B, lg.shape[-1], device=lg.device).scatter_add_(
                1, grupos[..., -1], coincide.float()) > 0
            lg = lg.masked_fill(prohibidas, -float('inf'))
    return lg


def _frenar(lg, reciente, pena_repeticion, prohibidas):
    """_frenos en O(1) por pieza: pena de repeticion sobre `reciente` (B, ventana) y las piezas `prohibidas` (B, V)
    bool que cerrarian un grupo de n ya escrito (de _Ngramas sobre lo escrito), o None."""
    if pena_repeticion != 1.0:
        vis = torch.zeros_like(lg, dtype=torch.bool).scatter_(1, reciente, True)
        lg = torch.where(vis, torch.where(lg > 0, lg / pena_repeticion, lg * pena_repeticion), lg)
    if prohibidas is not None:
        lg = lg.masked_fill(prohibidas, -float('inf'))
    return lg


def _continuaciones_de_cita(x, V, n=4):
    """(B, V) bool: las piezas que, escritas ahora, cierran un grupo de n piezas que ya salio en x (la senal
    'vistos' del grupo mas largo, la que alimenta la racha): seguir una cita (o un bucle) en vez de escribir algo nuevo."""
    cuenta = torch.zeros(x.shape[0], V, device=x.device)
    if x.shape[1] < n:
        return cuenta > 0
    grupos = x.unfold(1, n, 1)                                                       # (B, G, n)
    coincide = (grupos[..., :-1] == x[:, x.shape[1] - (n - 1):].unsqueeze(1)).all(-1)   # (B, G)
    return cuenta.scatter_add_(1, grupos[..., -1], coincide.float()) > 0


def _conservar_cita(q, p, cita):
    """Reparte q (lo que dejaron la temperatura y los recortes) de modo que la masa de seguir una cita sea la del
    modelo a temperatura 1, p[cita]. Dentro de cada parte (citar / escribir algo nuevo) queda la forma de q.
    Si q dejo en cero una de las partes (top_k o top_p las recortaron), q no se toca."""
    masa_p = (p * cita).sum(-1, keepdim=True)
    masa_q = (q * cita).sum(-1, keepdim=True)
    ok = (masa_q > 0) & (masa_q < 1)
    escala = torch.where(cita, masa_p / masa_q.clamp(min=1e-30), (1 - masa_p) / (1 - masa_q).clamp(min=1e-30))
    return torch.where(ok, q * escala, q)


def _muestrear(lg, temperatura, top_k, g, top_p=1.0, cita=None):
    """cita: (B, V) bool o None. Si se da, la temperatura y los recortes solo eligen DENTRO de 'seguir la cita' y
    DENTRO de 'escribir algo nuevo'; cuanto se cita queda como a temperatura 1 (ver _conservar_cita)."""
    p1 = torch.softmax(lg, -1) if cita is not None else None
    lg = lg / max(temperatura, 1e-6)
    if top_k and top_k < lg.shape[-1]:
        corte = lg.topk(top_k, dim=-1).values[:, -1:]
        lg = lg.masked_fill(lg < corte, -float('inf'))
    if top_p < 1.0:
        orden, idx = lg.sort(dim=-1, descending=True)
        acumulada = torch.softmax(orden, -1).cumsum(-1)
        fuera = acumulada - torch.softmax(orden, -1) >= top_p
        fuera[..., 0] = False                                        # siempre queda al menos la mas probable
        lg = lg.scatter(1, idx, orden.masked_fill(fuera, -float('inf')))
    prob = torch.softmax(lg, -1)
    if cita is not None:
        prob = _conservar_cita(prob, p1, cita)
    return torch.multinomial(prob.cpu(), 1, generator=g).to(lg.device)


@torch.no_grad()
def escribir(modelo, inicio, n_nuevos=200, temperatura=0.8, top_k=40, semilla=0, memoria_de_lo_ajeno=True,
             pena_repeticion=1.0, ventana_pena=64, no_repetir_ngramas=0, parar_en=None, top_p=1.0,
             devolver_logits=True, conservar_cita=True):
    """Escribe n_nuevos piezas despues de `inicio` (lote, largo) o (largo,). Por defecto con "memoria de lo ajeno":
    ASRN copia solo del texto de entrada, no de lo que el mismo escribio. parar_en: pieza que termina cada texto (se
    rellena con ella). top_p < 1: muestreo por nucleo (siempre queda al menos la pieza mas probable).
    no_repetir_ngramas=n: no escribe un grupo de n piezas que ya escribio (si en un paso eso prohibe todas las piezas
    de un texto, en ese paso no se aplica). Devuelve (secuencias completas, logits de cada paso); con
    devolver_logits=False, (secuencias, None) sin guardar (lote, pasos, V) numeros.
    El modelo escribe tal como fue entrenado: no hay perillas que cambien la puerta al escribir (v85 tenia
    escala_vistos y ajuste_puerta). Cuanto copiar lo deciden la intensidad de Hawkes y la racha que la puerta aprendio.
    conservar_cita=True: la temperatura y top_k/top_p no cambian la probabilidad de SEGUIR una cita (cerrar un grupo de
    4 piezas que ya salio), que queda la del modelo a temperatura 1; solo afinan la eleccion dentro de 'seguir la cita'
    y dentro de 'escribir algo nuevo'. Sin esto, T < 1 agudiza la moda, seguir una cita suele serla, y las citas se
    alargan mas alla de lo que el modelo aprendio. Con temperatura 1 y sin recortes no cambia nada."""
    modelo = _base(modelo)
    estaba = modelo.training
    try:
        return _escribir(modelo, inicio, n_nuevos, temperatura, top_k, semilla, memoria_de_lo_ajeno, pena_repeticion,
                         ventana_pena, no_repetir_ngramas, parar_en, top_p, devolver_logits, conservar_cita)
    finally:
        modelo.train(estaba)


@torch.no_grad()
def _escribir(modelo, inicio, n_nuevos, temperatura, top_k, semilla, memoria_de_lo_ajeno, pena_repeticion,
              ventana_pena, no_repetir_ngramas, parar_en, top_p, devolver_logits, conservar_cita=True):
    modelo.eval()
    _revisar_precision(modelo)
    disp = next(modelo.parameters()).device
    g = torch.Generator(device='cpu').manual_seed(semilla)
    esc = _Escritor(modelo, memoria_de_lo_ajeno)
    x = inicio.to(disp)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    L0 = x.shape[1]
    assert L0 >= 1, "hace falta al menos una pieza de inicio"
    V = modelo.config['V']
    if int(x.min()) < 0 or int(x.max()) >= V:
        raise ValueError(f"inicio: piezas fuera de [0, {V}) (minimo {int(x.min())}, maximo {int(x.max())}): "
                         "revisa que el tokenizador sea el mismo del modelo")
    lg = esc.inicio(x, capacidad=L0 + n_nuevos)
    # todo lo que se consulta en cada paso cuesta O(1): el texto va en un bufer reservado (sin copiarlo entero en cada
    # pieza) y los grupos de n piezas, en diccionarios que crecen con cada pieza (_Ngramas), no se recorre el texto
    texto = x.new_empty(x.shape[0], L0 + n_nuevos)
    texto[:, :L0] = x
    largo = L0
    cita_ng = _Ngramas(x.cpu().tolist(), modelo.largos[0].vistos) if conservar_cita else None
    escritos = _Ngramas([[] for _ in range(x.shape[0])], no_repetir_ngramas) if no_repetir_ngramas else None
    pasos = []
    terminado = torch.zeros(x.shape[0], dtype=torch.bool, device=disp)
    for k in range(n_nuevos):
        lg = lg.float().clone()
        if devolver_logits:
            pasos.append(lg.cpu().clone())
        frenado = _frenar(lg, texto[:, max(0, largo - ventana_pena):largo],
                          pena_repeticion, escritos.siguientes(V, disp) if escritos else None)
        todo_prohibido = torch.isneginf(frenado).all(-1, keepdim=True)   # no_repetir_ngramas no dejo ninguna pieza
        lg = torch.where(todo_prohibido, lg, frenado)
        cita = cita_ng.siguientes(V, disp) if conservar_cita else None
        sig = _muestrear(lg, temperatura, top_k, g, top_p, cita)
        if parar_en is not None:
            sig = torch.where(terminado.unsqueeze(1), torch.full_like(sig, parar_en), sig)
            terminado = terminado | (sig[:, 0] == parar_en)
        texto[:, largo] = sig[:, 0]
        largo += 1
        sig_cpu = sig[:, 0].cpu()
        for ng in (cita_ng, escritos):
            if ng is not None:
                ng.agregar(sig_cpu.tolist())
        if parar_en is not None and bool(terminado.all()):
            break
        if k < n_nuevos - 1:
            lg = esc.paso(sig_cpu)
    x = texto[:, :largo]
    if not devolver_logits:
        return x.cpu(), None
    return x.cpu(), (torch.stack(pasos, 1) if pasos else torch.zeros(x.shape[0], 0, modelo.config['V']))


# ----------------------------------------------------------------------------------------------------------------
# Guardar y cargar
# ----------------------------------------------------------------------------------------------------------------

def _guardar_atomico(obj, archivo):
    """Escribe a un archivo temporal, lo baja al disco (fsync) y lo renombra: si se corta a la mitad (o se va la luz),
    el archivo anterior queda sano. Si algo falla, borra el temporal."""
    carpeta = os.path.dirname(os.path.abspath(archivo))
    os.makedirs(carpeta, exist_ok=True)
    tmp = archivo + '.tmp'
    try:
        with open(tmp, 'wb') as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, archivo)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def guardar(modelo, archivo):
    """Guarda configuracion y pesos (los que carga cargar()). Con el modelo repartido, TODOS los procesos deben
    llamarla: se juntan los pesos completos y solo el proceso 0 escribe."""
    base = _base(modelo)
    pesos = pesos_completos(base) if esta_repartido(base) else base.state_dict()
    if _distribuido()[0] == 0:
        _guardar_atomico({'config': base.config, 'pesos': pesos}, archivo)


def _leer(archivo, confiar=False, mmap=False):
    """Lee un archivo de torch.save en modo seguro (weights_only). mmap=True: los tensores se leen del disco a medida
    que se usan, y los procesos de una misma maquina comparten una sola copia en memoria (para reanudar repartido)."""
    if not os.path.isfile(archivo):
        raise FileNotFoundError(f"no existe el archivo {archivo}")
    try:
        return torch.load(archivo, map_location='cpu', weights_only=True, mmap=mmap)
    except Exception as e:
        if mmap:
            return _leer(archivo, confiar)              # los archivos de formato viejo (sin zip) no se pueden mapear
        if not confiar:
            raise ValueError(f"{archivo}: no se pudo leer en modo seguro ({type(e).__name__}). Si el archivo es tuyo, "
                             "usa confiar=True (ejecuta codigo guardado en el archivo).") from e
        return torch.load(archivo, map_location='cpu', weights_only=False)


_NUEVOS_V87 = ('w_reloj', 'b_reloj', 'gamma', 'log_kappa', 'log_kappa_racha', 'u_hawkes', 'd_racha')


def _migrar_racha(pesos):
    """Los archivos de v86 guardaban la respuesta a la racha como valores por nodo ('u_racha'); v87 guarda sus
    incrementos ('d_racha'). La conversion es exacta: el modelo calcula lo mismo. Devuelve (pesos, nombres
    convertidos)."""
    viejos = [n for n in pesos if n.rsplit('.', 1)[-1] == 'u_racha']
    if not viejos:
        return pesos, []
    pesos = dict(pesos)
    for n in viejos:
        u = pesos.pop(n)
        pesos[n[:-len('u_racha')] + 'd_racha'] = torch.cat([u[:1], u[1:] - u[:-1]])
    return pesos, [n[:-len('u_racha')] + 'd_racha' for n in viejos]


def _cargar_pesos(modelo, pesos):
    """Carga pesos de v87, de v86 (convertidos con _migrar_racha) o de v83/v84/v85: a esos les faltan solo los
    parametros nuevos desde v86, que quedan en su valor inicial (neutro), asi que el modelo calcula exactamente lo
    mismo que en su version."""
    pesos, _ = _migrar_racha(pesos)
    faltan, sobran = modelo.load_state_dict(pesos, strict=False)
    raros = [n for n in faltan if n.rsplit('.', 1)[-1] not in _NUEVOS_V87]
    if sobran or raros:
        raise ValueError(f"pesos que no son de ASRN v83-v87: faltan {raros}, sobran {list(sobran)}")
    return modelo


def cargar(archivo, dispositivo=None, confiar=False):
    """Carga un modelo guardado con guardar() o entrenar() (de v83 a v89).
    Lee en modo seguro; confiar=True solo para archivos propios que no se puedan leer asi."""
    r = _leer(archivo, confiar)
    if not isinstance(r, dict) or 'config' not in r or 'pesos' not in r:
        raise ValueError(f"{archivo}: no es un modelo guardado con guardar() o entrenar() de ASRN")
    c = dict(r['config'])
    m = ModeloASRN(c['V'], c['d'], c['n_capas'], c['largo_cada'], c['ff'], c['k'], init_profundo=False,
                   dropout=c.get('dropout', 0.0), mezcla=c.get('mezcla', 'conv'),
                   memoria_max=c.get('memoria_max', 1024), **c['opciones_capa'])
    _cargar_pesos(m, r['pesos'])
    return m.to(dispositivo or dev).eval()


# ----------------------------------------------------------------------------------------------------------------
# Entrenar
# ----------------------------------------------------------------------------------------------------------------

_TORCH_COMPILE_MINIMO = '2.14'      # antes, Inductor falla con este modelo (CPU: C++ invalido; A100 con 2.11: Triton)


def _elegir_precision(precision, disp):
    if precision != 'auto':
        assert precision in ('fp32', 'bf16', 'fp16'), precision
        return precision
    if disp.type == 'cuda':
        return 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    return 'fp32'


def _autocast(precision, disp, cache=True):
    tipo = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(precision)
    return torch.autocast(device_type=disp.type, dtype=tipo, enabled=tipo is not None, cache_enabled=cache)


def _preparar_datos(datos, V, largo, disp):
    """Tensor 1D de numeros de pieza (se revisa que esten en [0, V)). Si cabe con holgura, va a la GPU."""
    datos = torch.as_tensor(datos)
    if datos.dim() != 1 or datos.is_floating_point() or datos.dtype == torch.bool:
        raise ValueError(f"datos: hace falta un tensor 1D de numeros de pieza (llego {datos.dtype} {tuple(datos.shape)})")
    if datos.numel() <= largo + 1:
        raise ValueError(f"hacen falta mas de {largo + 1} piezas (hay {datos.numel()})")
    if datos.dtype not in (torch.int32, torch.int64):
        datos = datos.to(torch.int64)
    lo, hi = int(datos.min()), int(datos.max())
    if lo < 0 or hi >= V:
        raise ValueError(f"piezas fuera de [0, {V}): minimo {lo}, maximo {hi}")
    datos = datos.to(torch.int32)
    if disp.type == 'cuda' and datos.numel() * 4 < 2 ** 31:
        datos = datos.to(disp)
    return datos


def _lote(datos, largo, lote, g, disp):
    """Ventanas al azar de largo+1 piezas: x = las primeras largo, y = las mismas corridas una."""
    ini = torch.randint(0, datos.numel() - largo - 1 + 1, (lote,), generator=g)
    idx = (ini.unsqueeze(1) + torch.arange(largo + 1)).to(datos.device)
    w = datos[idx].to(disp, non_blocking=True).long()
    return w[:, :-1], w[:, 1:]


def _nombres_en_optimizador(modelo):
    """Nombres de los parametros en el orden en que crear_optimizador() los numera (matrices, luego el resto)."""
    con, sin = [], []
    for n, p in _base(modelo).named_parameters():
        if p.requires_grad:
            (con if p.dim() >= 2 else sin).append(n)
    return con + sin


def crear_optimizador(modelo, lr=1e-3, decaimiento=0.1, betas=(0.9, 0.95), capturable=False):
    """AdamW; decaimiento solo en matrices (no en sesgos, normas ni vectores de la puerta). capturable=True: se puede
    grabar en un CUDA Graph (PasoGrafado)."""
    modelo = _base(modelo)
    con, sin = [], []
    for p in modelo.parameters():
        if p.requires_grad:
            (con if p.dim() >= 2 else sin).append(p)
    grupos = [{'params': con, 'weight_decay': decaimiento}, {'params': sin, 'weight_decay': 0.0}]
    extra = {'capturable': True} if capturable else {}
    if next(modelo.parameters()).device.type == 'cuda' and not esta_repartido(modelo):
        extra['fused'] = True
    try:
        return torch.optim.AdamW(grupos, lr=lr, betas=betas, **extra)
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(grupos, lr=lr, betas=betas)


class PasoGrafado:
    """Un paso de entrenamiento entero (ida, vuelta, recorte del gradiente y AdamW) grabado como CUDA Graph: la GPU
    lo repite con una sola llamada, sin que Python lance uno por uno los ~3 000 kernels del paso. Ayuda cuando la GPU
    pasa tiempo esperando a Python (textos cortos, modelos chicos). Las cuentas son las mismas que sin grabar.

        opt = crear_optimizador(modelo, lr, capturable=True)
        paso = PasoGrafado(modelo, opt, lote, largo)
        for x, y in lotes:                                  # x, y: (lote, largo), siempre la misma forma
            perdida, norma, salto = paso(x, y, lr=...)      # tensores en la GPU: leerlos solo al registrar

    Los primeros `calentar` pasos van sin grabar (son pasos de verdad: compilan los kernels y crean el estado de
    AdamW); el siguiente graba y desde ahi se repite. Si la perdida o la norma no son finitas, AdamW no cambia nada
    en ese paso (salto = 1), como entrenar(). Limites: una GPU, bf16 o fp32 (fp16 necesita el escalador), sin
    checkpoint (recalcular dentro de un grafo no es seguro), sin FSDP ni torch.compile, sin acumulacion; el modelo
    no puede cambiar de forma ni de parametros despues de grabar. grabar=False: lo mismo sin grafo (para comparar, o
    en CPU)."""

    def __init__(self, modelo, optimizador, lote, largo, precision='bf16', recorte=1.0, calentar=3, grabar=None):
        base = _base(modelo)
        disp = next(base.parameters()).device
        self.grabar = disp.type == 'cuda' if grabar is None else grabar
        if self.grabar and disp.type != 'cuda':
            raise ValueError("PasoGrafado: CUDA Graphs necesita el modelo en una GPU (o grabar=False)")
        if precision == 'fp16':
            raise ValueError("PasoGrafado: fp16 necesita el escalador de gradientes; usa 'bf16' o 'fp32'")
        if esta_repartido(base) or getattr(base, 'checkpoint', 'no') != 'no' or modelo is not base:
            raise ValueError("PasoGrafado: sin FSDP, sin checkpoint y sin torch.compile")
        if self.grabar and not all(g.get('capturable') and g.get('fused') for g in optimizador.param_groups):
            raise ValueError("PasoGrafado: el optimizador tiene que ser crear_optimizador(..., capturable=True) en GPU")
        self.modelo, self.opt, self.precision, self.disp = base, optimizador, precision, disp
        self.recorte, self.calentar, self.hechos, self.grafo = recorte if recorte else float('inf'), calentar, 0, None
        self.params = [p for p in base.parameters() if p.requires_grad]
        self.x = torch.zeros((lote, largo), dtype=torch.long, device=disp)
        self.y = torch.zeros((lote, largo), dtype=torch.long, device=disp)
        if self.grabar:                 # la tasa vive en la GPU: cambiarla no obliga a grabar de nuevo
            self.lr = torch.tensor(float(optimizador.param_groups[0]['lr']), device=disp)
            for g in optimizador.param_groups:
                g['lr'] = self.lr

    def _paso(self):
        with _autocast(self.precision, self.disp, cache=not self.grabar):
            logits = self.modelo(self.x)
        perdida = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), self.y.reshape(-1))
        perdida.backward()
        norma = torch.nn.utils.clip_grad_norm_(self.params, self.recorte)
        salto = (~(torch.isfinite(perdida) & torch.isfinite(norma))).float()      # forma (), como en GradScaler
        if self.opt.param_groups[0].get('fused'):
            self.opt.found_inf = salto          # AdamW fusionado no cambia nada si found_inf = 1, sin leerlo en Python
            try:
                self.opt.step()
            finally:
                del self.opt.found_inf
        elif not salto.item():
            self.opt.step()
        return perdida.detach(), norma.detach(), salto

    def __call__(self, x, y, lr=None):
        self.x.copy_(x, non_blocking=True)
        self.y.copy_(y, non_blocking=True)
        if lr is not None:
            if self.grabar:
                self.lr.fill_(lr)
            else:
                for g in self.opt.param_groups:
                    g['lr'] = lr
        if not self.grabar:
            self.opt.zero_grad(set_to_none=True)
            return self._paso()
        if self.hechos < self.calentar:     # sin grabar, en otra cola (como pide PyTorch antes de grabar)
            cola = torch.cuda.Stream()
            cola.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(cola):
                self.opt.zero_grad(set_to_none=True)
                resultado = self._paso()
            torch.cuda.current_stream().wait_stream(cola)
            self.hechos += 1
            return resultado
        if self.grafo is None:              # gradientes en None: dentro del grafo se asignan, no se suman
            self.opt.zero_grad(set_to_none=True)
            self.grafo = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.grafo):
                self.salida = self._paso()
        self.grafo.replay()
        return self.salida


def tasa_aprendizaje(paso, lr, pasos, calentamiento, lr_min):
    """Subida lineal durante `calentamiento` pasos y despues bajada en coseno hasta lr_min en `pasos`."""
    if paso < calentamiento:
        return lr * (paso + 1) / calentamiento
    avance = min(1.0, (paso - calentamiento) / max(1, pasos - calentamiento))
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * avance))


@torch.no_grad()
def evaluar(modelo, datos, largo=512, lote=16, max_lotes=50, precision='auto'):
    """Perdida media y perplejidad sobre ventanas seguidas (sin solaparse) desde el inicio de `datos`."""
    _revisar_enteros(largo=largo, lote=lote, max_lotes=max_lotes)
    base = _base(modelo)
    disp = next(base.parameters()).device
    datos = _preparar_datos(datos, base.config['V'], largo, disp)
    precision = _elegir_precision(precision, disp)
    estaba = base.training
    base.eval()
    inicios = torch.arange(0, datos.numel() - largo - 1 + 1, largo)[:lote * max_lotes]
    total, n = 0.0, 0
    for i in range(0, inicios.numel(), lote):
        idx = (inicios[i:i + lote].unsqueeze(1) + torch.arange(largo + 1)).to(datos.device)
        w = datos[idx].to(disp).long()
        with _autocast(precision, disp):
            logits = base(w[:, :-1])
        total += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), w[:, 1:].reshape(-1),
                                 reduction='sum').item()
        n += w[:, 1:].numel()
    base.train(estaba)
    total = _promedio_procesos(total, disp)              # repartido: todos ven las mismas ventanas; mismo numero
    perdida = total / max(n, 1)
    return {'perdida': perdida, 'perplejidad': math.exp(min(perdida, 50)), 'piezas': n}


def _revisar_enteros(minimo=1, **valores):
    """ValueError claro si alguno de los valores no es un entero >= minimo (en vez de un error raro mas adelante)."""
    for nombre, valor in valores.items():
        if isinstance(valor, bool) or not isinstance(valor, numbers.Integral) or valor < minimo:
            raise ValueError(f"{nombre}={valor!r}: hace falta un numero entero >= {minimo}")


def _promedio_procesos(valor, disp):
    """Promedio de un numero entre todos los procesos (el mismo numero si no hay varios)."""
    import torch.distributed as dist
    rango, n = _distribuido()
    if n == 1:
        return valor
    t = torch.tensor([float(valor)], dtype=torch.float64, device=disp if dist.get_backend() != 'gloo' else 'cpu')
    dist.all_reduce(t)
    return float(t) / n


def _opt_a_nombres(estado, nombres):
    """Estado de AdamW numerado (opt.state_dict()) -> por nombre de parametro (el formato de FSDP)."""
    return {'state': {nombres[i]: st for i, st in estado['state'].items()},
            'param_groups': [{**pg, 'params': [nombres[i] for i in pg['params']]} for pg in estado['param_groups']]}


def _opt_a_numeros(estado, nombres):
    """El inverso de _opt_a_nombres."""
    numero = {n: i for i, n in enumerate(nombres)}
    return {'state': {numero[n]: st for n, st in estado['state'].items()},
            'param_groups': [{**pg, 'params': [numero[n] for n in pg['params']]} for pg in estado['param_groups']]}


def _estado(base, opt, escalador, paso, mejor_val, g, historial, ajustes=None):
    """Todo lo necesario para reanudar. Repartido: colectiva (todos los procesos la llaman); el estado de AdamW va
    por nombre de parametro y el azar de cada proceso por separado. ajustes: los de entrenar() (pasos, lr, ...), para
    avisar al reanudar si cambiaron."""
    estado = {'config': base.config, 'escalador': escalador.state_dict(), 'paso': paso, 'mejor_val': mejor_val,
              'azar': g.get_state(), 'historial': historial, 'version': 'asrn_v89', 'ajustes': ajustes or {}}
    if esta_repartido(base):
        import torch.distributed as dist
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, StateDictOptions
        estado['pesos'] = pesos_completos(base)
        estado['optimizador_nombres'] = get_optimizer_state_dict(
            base, opt, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        azares = [None] * _distribuido()[1]
        dist.all_gather_object(azares, g.get_state())
        estado['azar_procesos'] = azares
    else:
        estado['pesos'] = base.state_dict()
        estado['optimizador'] = opt.state_dict()
    return estado


def _reanudar_estado(base, opt, r, migrados):
    """Carga pesos y AdamW de un archivo de entrenar(), guardado en un proceso o repartido, en un modelo en un
    proceso o repartido (las cuatro combinaciones)."""
    nombres = _nombres_en_optimizador(base)
    if 'optimizador_nombres' in r:
        por_nombre = r['optimizador_nombres']
    else:
        por_nombre = _opt_a_nombres(r['optimizador'], nombres)      # el mismo modelo: los mismos nombres
    if migrados:                # los momentos de AdamW de los valores por nodo no valen para los incrementos
        por_nombre = {**por_nombre, 'state': {n: st for n, st in por_nombre['state'].items() if n not in migrados}}
    if esta_repartido(base):
        from torch.distributed.checkpoint.state_dict import (set_model_state_dict, set_optimizer_state_dict,
                                                             StateDictOptions)
        opciones = StateDictOptions(full_state_dict=True)
        set_model_state_dict(base, r['pesos'], options=opciones)
        set_optimizer_state_dict(base, opt, por_nombre, options=opciones)
    else:
        base.load_state_dict(r['pesos'])
        opt.load_state_dict(_opt_a_numeros(por_nombre, nombres))


def entrenar(modelo, datos, validacion=None, largo=512, lote=16, pasos=10000, lr=1e-3, lr_min=None,
             calentamiento=None, decaimiento=0.1, betas=(0.9, 0.95), recorte=1.0, acumular=1, precision='auto',
             compilar=False, evaluar_cada=500, lotes_eval=50, guardar_en=None, guardar_cada=1000, reanudar=None,
             semilla=0, registrar_cada=50, registro=print, max_saltos=20):
    """Entrena `modelo` (en su dispositivo) con ventanas al azar de `datos` (tensor 1D de numeros de pieza).

    Por paso: `acumular` lotes de `lote` ventanas de `largo` piezas. lr: calentamiento lineal (por defecto 2% de los
    pasos, maximo 2000) y coseno hasta lr_min (por defecto lr/10). precision: 'auto' (bf16 en GPU que lo soporta, fp16
    con escalado si no, fp32 en CPU), 'fp32', 'bf16' o 'fp16'. compilar: False, 'partes' (compilar_partes: lo mas
    rapido en GPU, necesita PyTorch >= 2.14) o True (el modelo entero; deja de usar los kernels Triton). Un paso con
    perdida NaN/inf se salta (mas de
    `max_saltos` seguidos detiene todo). recorte: norma maxima del gradiente (0/None: sin recorte) o un objeto que se
    llama sin argumentos despues de cada backward, recorta y devuelve la norma (asrn.estable.RecorteDinamico).
    validacion: tensor 1D aparte, evaluado cada `evaluar_cada` pasos.
    guardar_en='modelo.pt': guarda cada `guardar_cada` pasos y al final (archivo que carga cargar() y que sirve para
    reanudar=), y el mejor en validacion en 'modelo.mejor.pt'. Ctrl+C guarda y termina. Devuelve el historial.

    Repartido (repartir_fsdp, varios procesos): todos los procesos llaman a entrenar() con los mismos argumentos.
    Cada uno toma lotes distintos (azar semilla + 1000003*rango; el proceso 0 usa los mismos lotes que sin repartir),
    asi que cada paso ve n_procesos * lote * acumular ventanas. La perdida y la decision de saltar un paso son
    comunes, solo el proceso 0 escribe registros y archivos. fp16 no esta soportado repartido (usar bf16)."""
    base = _base(modelo)
    disp = next(base.parameters()).device
    rango, n_proc = _distribuido()
    repartido = esta_repartido(base)
    if n_proc > 1 and not repartido:
        raise ValueError("hay varios procesos pero el modelo no esta repartido: usa repartir_fsdp(modelo) antes "
                         "(sin eso cada proceso entrenaria su propia copia)")
    if rango != 0:
        registro = lambda *a, **k: None                 # noqa: E731  solo el proceso 0 escribe
    _revisar_enteros(largo=largo, lote=lote, acumular=acumular, evaluar_cada=evaluar_cada, lotes_eval=lotes_eval,
                     guardar_cada=guardar_cada, registrar_cada=registrar_cada)
    _revisar_enteros(minimo=0, pasos=pasos, max_saltos=max_saltos)
    V = base.config['V']
    datos = _preparar_datos(datos, V, largo, disp)
    if validacion is not None:
        validacion = _preparar_datos(validacion, V, largo, disp)
    precision = _elegir_precision(precision, disp)
    if repartido and precision == 'fp16':
        raise ValueError("fp16 con el modelo repartido no esta soportado: usa precision='bf16' (A100, L4, H100)")
    if disp.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    lr_min = lr / 10 if lr_min is None else lr_min
    calentamiento = min(2000, max(1, pasos // 50)) if calentamiento is None else max(1, calentamiento)
    opt = crear_optimizador(base, lr, decaimiento, betas)
    escalador = torch.amp.GradScaler(disp.type, enabled=(precision == 'fp16'))
    g = torch.Generator(device='cpu').manual_seed(semilla + 1_000_003 * rango)
    paso0, mejor_val, historial = 0, float('inf'), []
    ajustes = dict(pasos=pasos, lr=lr, lr_min=lr_min, calentamiento=calentamiento, largo=largo, lote=lote,
                   acumular=acumular, decaimiento=decaimiento)
    if reanudar:
        r = _leer(reanudar, mmap=n_proc > 1)            # repartido: una sola copia del archivo en la memoria
        if 'optimizador' not in r and 'optimizador_nombres' not in r:
            raise ValueError(f"{reanudar} tiene solo los pesos (lo guardo guardar(), como el '.mejor.pt'): para "
                             "reanudar hace falta el archivo de guardar_en, que trae tambien AdamW y el paso. Para "
                             "empezar un entrenamiento nuevo desde esos pesos: entrenar(cargar(archivo), ...) sin "
                             "reanudar")
        r['pesos'], migrados = _migrar_racha(r['pesos'])
        _reanudar_estado(base, opt, r, migrados)
        if r.get('escalador'):                  # vacio si se guardo sin fp16: el escalado de fp16 arranca de cero
            escalador.load_state_dict(r['escalador'])
        paso0, mejor_val, historial = r['paso'], r['mejor_val'], list(r.get('historial', []))
        cambios = {k: (r['ajustes'][k], v) for k, v in ajustes.items()
                   if k in r.get('ajustes', {}) and r['ajustes'][k] != v}
        if cambios:
            registro("aviso: estos ajustes no son los del archivo (antes -> ahora): "
                     + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in cambios.items())
                     + ". La tasa de aprendizaje sigue la curva con los de ahora")
        if paso0 >= pasos:
            registro(f"aviso: el archivo ya va en el paso {paso0} y pasos={pasos}: no queda nada que entrenar "
                     "(sube pasos, o --pasos en la linea de comandos)")
        azares = r.get('azar_procesos')
        if azares is not None and len(azares) == n_proc:
            g.set_state(azares[rango])
        elif rango == 0:
            g.set_state(r['azar'])
        if n_proc > 1 and (azares is None or len(azares) != n_proc):
            registro(f"aviso: el archivo es de {len(azares) if azares else 1} proceso(s) y ahora hay {n_proc}: los "
                     "lotes de los procesos nuevos arrancan de su semilla (el proceso 0 sigue igual)")
        registro(f"reanudando desde el paso {paso0} ({reanudar})")
        del r                                           # suelta el archivo (y su mapeo en memoria)
    if compilar and torch.__version__ < _TORCH_COMPILE_MINIMO:
        registro(f"aviso: con PyTorch {torch.__version__}, torch.compile falla con este modelo (en CPU de "
                 f"2.6 a 2.13, y en una A100 con 2.11; en CPU funciona desde {_TORCH_COMPILE_MINIMO})")
    if compilar == 'partes':
        if torch.__version__ >= _TORCH_COMPILE_MINIMO:
            compilar_partes(base)
        else:
            registro("aviso: se sigue sin compilar")
        adelante = base
    else:
        adelante = torch.compile(base) if compilar else base
    registro(f"ASRN 1.0 (v89): {base.num_parametros():,} parametros | {datos.numel():,} piezas | precision "
             f"{precision} | {lote * acumular * largo * n_proc:,} piezas por paso | {disp}"
             + (f" | repartido en {n_proc} procesos" if repartido else ""))

    def guardar_estado(archivo, paso_):
        if archivo:
            estado = _estado(base, opt, escalador, paso_, mejor_val, g, historial, ajustes)   # colectiva si repartido
            if rango == 0:
                _guardar_atomico(estado, archivo)

    mejor_archivo = None
    if guardar_en:
        raiz, ext = os.path.splitext(guardar_en)
        mejor_archivo = raiz + '.mejor' + (ext or '.pt')

    base.train()
    saltos, t0, piezas_vistas, suma, cuenta = 0, time.time(), 0, 0.0, 0
    paso = paso0
    try:
        while paso < pasos:
            lr_t = tasa_aprendizaje(paso, lr, pasos, calentamiento, lr_min)
            for grupo in opt.param_groups:
                grupo['lr'] = lr_t
            perdida = torch.zeros((), device=disp)
            for _ in range(acumular):
                x, y = _lote(datos, largo, lote, g, disp)
                with _autocast(precision, disp):
                    logits = adelante(x)
                p = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1))
                escalador.scale(p / acumular).backward()
                perdida += p.detach() / acumular
            perdida = _promedio_procesos(perdida.item(), disp)     # comun a todos: todos saltan o ninguno
            norma = float('nan')
            if math.isfinite(perdida):
                escalador.unscale_(opt)
                if callable(recorte):                   # recorte propio, p. ej. asrn.RecorteDinamico(modelo)
                    norma = recorte()
                else:
                    norma = torch.nn.utils.clip_grad_norm_(base.parameters(), recorte if recorte else float('inf'))
                norma = float(norma.full_tensor() if _es_dtensor(norma) else norma)
            # con fp16 el escalador ya salta los pasos con gradiente inf; en fp32/bf16 lo hacemos aqui
            if not math.isfinite(perdida) or (precision != 'fp16' and not math.isfinite(norma)):
                opt.zero_grad(set_to_none=True)
                saltos += 1
                registro(f"paso {paso}: perdida {perdida}, norma {norma}: salto el paso ({saltos} seguidos)")
                if saltos > max_saltos:
                    raise RuntimeError("demasiados pasos con NaN/inf seguidos: baja lr o revisa los datos")
                paso += 1
                continue
            saltos = 0
            escalador.step(opt)
            escalador.update()
            opt.zero_grad(set_to_none=True)
            paso += 1
            piezas_vistas += lote * acumular * largo * n_proc
            suma, cuenta = suma + perdida, cuenta + 1

            if paso % registrar_cada == 0 or paso == pasos:
                dt = time.time() - t0
                fila = {'paso': paso, 'perdida': suma / cuenta, 'lr': lr_t, 'norma': norma,
                        'piezas_s': piezas_vistas / max(dt, 1e-9)}
                historial.append(fila)
                registro(f"paso {paso:>7} | perdida {fila['perdida']:.4f} | lr {lr_t:.2e} | norma {fila['norma']:.2f}"
                         f" | {fila['piezas_s']:,.0f} piezas/s")
                t0, piezas_vistas, suma, cuenta = time.time(), 0, 0.0, 0
            if validacion is not None and (paso % evaluar_cada == 0 or paso == pasos):
                ev = evaluar(base, validacion, largo, lote, lotes_eval, precision)
                historial.append({'paso': paso, 'val_perdida': ev['perdida'], 'val_perplejidad': ev['perplejidad']})
                nuevo = ev['perdida'] < mejor_val
                if nuevo:
                    mejor_val = ev['perdida']
                    if mejor_archivo:
                        guardar(base, mejor_archivo)
                registro(f"   validacion: perdida {ev['perdida']:.4f} | perplejidad {ev['perplejidad']:.2f}"
                         + ("  (mejor)" if nuevo else ""))
            if guardar_en and paso % guardar_cada == 0:
                guardar_estado(guardar_en, paso)
    except KeyboardInterrupt:
        if n_proc > 1:
            raise                       # repartido no se puede juntar con procesos detenidos: vale lo ultimo guardado
        registro(f"interrumpido en el paso {paso}" + (f": guardado en {guardar_en}" if guardar_en else ""))
    guardar_estado(guardar_en, paso)
    base.eval()
    return historial


# ----------------------------------------------------------------------------------------------------------------
# Linea de comandos
# ----------------------------------------------------------------------------------------------------------------

_FORMATOS_PIEZAS = {'.bin': (torch.uint16, 2), '.u16': (torch.uint16, 2), '.u32': (torch.int32, 4),
                    '.i32': (torch.int32, 4)}


def leer_piezas(archivo):
    """Numeros de pieza desde .pt (tensor 1D), .bin/.u16 (uint16 crudo) o .u32/.i32 (int32 crudo)."""
    ext = os.path.splitext(archivo)[1].lower()
    if ext != '.pt' and ext not in _FORMATOS_PIEZAS:
        raise ValueError(f"{archivo}: formato '{ext}' no reconocido. Se aceptan .pt (tensor 1D), .bin o .u16 "
                         "(uint16 crudo) y .i32 o .u32 (int32 crudo)")
    if not os.path.isfile(archivo):
        raise FileNotFoundError(f"no existe el archivo {archivo}")
    if ext == '.pt':
        t = torch.load(archivo, map_location='cpu', weights_only=True)
        return torch.as_tensor(t).reshape(-1)
    tipo, tam = _FORMATOS_PIEZAS[ext]
    n = os.path.getsize(archivo) // tam
    return torch.from_file(archivo, shared=False, size=n, dtype=tipo).to(torch.int32)


def _principal(args=None, preparar=None):
    import argparse
    a = argparse.ArgumentParser(description="Entrenar un modelo ASRN 1.0 (v89) con numeros de pieza ya tokenizados. "
                                "Varias GPUs: torchrun --nproc_per_node=N -m asrn --fsdp ...")
    a.add_argument('--datos', required=True, help='piezas de entrenamiento (.pt, .bin uint16, .i32)')
    a.add_argument('--validacion', help='piezas de validacion; si no se da, se aparta --fraccion_val del final')
    a.add_argument('--fraccion_val', type=float, default=0.01)
    a.add_argument('--V', type=int, help='tamano del vocabulario (obligatorio salvo al reanudar)')
    a.add_argument('--parametros', help="tamano que quieras, p. ej. 3k, 50m, 1.5b (necesita --V; --d/--capas lo fijan)")
    a.add_argument('--talla', choices=sorted(TALLAS), help='uno de los ejemplos de TALLAS (--d/--capas lo ajustan)')
    a.add_argument('--ancho', '--d', dest='d', type=int, help='d (con torchrun usa --ancho: --d lo confunde)')
    a.add_argument('--capas', type=int)
    a.add_argument('--largo_cada', type=int)
    a.add_argument('--mezcla', choices=_MEZCLAS, help="bloque local: 'conv' (receta final) o 'memoria' (en prueba)")
    a.add_argument('--memoria_max', type=int, help="con --mezcla memoria: piezas que recuerda (por omision 1024)")
    a.add_argument('--dropout', type=float)
    a.add_argument('--checkpoint', nargs='?', const='asrn', default='no', choices=['no', 'asrn', 'todo'],
                   help="menos memoria, algo mas lento: 'asrn' (solo con --checkpoint) o 'todo'")
    a.add_argument('--fsdp', action='store_true', help='repartir el modelo entre los procesos de torchrun')
    a.add_argument('--largo', type=int, default=512)
    a.add_argument('--lote', type=int, default=16)
    a.add_argument('--acumular', type=int, default=1)
    a.add_argument('--pasos', type=int, default=20000)
    a.add_argument('--lr', type=float, default=1e-3)
    a.add_argument('--calentamiento', type=int)
    a.add_argument('--decaimiento', type=float, default=0.1)
    a.add_argument('--precision', default='auto', choices=['auto', 'fp32', 'bf16', 'fp16'])
    a.add_argument('--compilar', nargs='?', const='partes', default=None, choices=['partes', 'todo'],
                   help="torch.compile: 'partes' (sin valor; lo mas rapido en GPU) o 'todo' (el modelo entero)")
    a.add_argument('--evaluar_cada', type=int, default=500)
    a.add_argument('--guardar', default='modelo.pt')
    a.add_argument('--guardar_cada', type=int, default=1000)
    a.add_argument('--reanudar', help='archivo guardado por entrenar() para seguir donde quedo')
    a.add_argument('--semilla', type=int, default=0)
    o = a.parse_args(args)
    for nombre in ('largo', 'lote', 'acumular', 'evaluar_cada', 'guardar_cada'):
        if getattr(o, nombre) < 1:
            a.error(f"--{nombre} debe ser 1 o mas (llego {getattr(o, nombre)})")
    if o.pasos < 0 or not 0 <= o.fraccion_val < 1:
        a.error("--pasos debe ser 0 o mas y --fraccion_val estar entre 0 y 1 (sin llegar a 1)")

    import torch.distributed as dist
    n_env, rango_env = int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', 0))
    if n_env > 1 and not o.fsdp:
        a.error(f"hay {n_env} procesos (torchrun) pero falta --fsdp: sin eso cada proceso entrenaria su propia copia "
                "del modelo y todos escribirian el mismo archivo")
    if o.fsdp and 'RANK' not in os.environ and not dist.is_initialized():
        a.error("--fsdp se usa lanzando con torchrun: torchrun --nproc_per_node=N -m asrn --fsdp ...")

    def aviso(texto):                                   # con torchrun, solo el proceso 0 escribe
        if rango_env == 0:
            print(texto)

    disp = torch.device(dev)
    if o.fsdp:
        rango_local = int(os.environ.get('LOCAL_RANK', 0))
        if disp.type == 'cuda':
            torch.cuda.set_device(rango_local)
            disp = torch.device('cuda', rango_local)
        if not dist.is_initialized():
            dist.init_process_group('nccl' if disp.type == 'cuda' else 'gloo')

    try:
        if o.reanudar:
            ignoradas = [n for n, v in (('--V', o.V), ('--parametros', o.parametros), ('--talla', o.talla),
                                        ('--ancho', o.d), ('--capas', o.capas), ('--largo_cada', o.largo_cada),
                                        ('--mezcla', o.mezcla), ('--memoria_max', o.memoria_max),
                                        ('--dropout', o.dropout))
                         if v is not None]
            if ignoradas:
                aviso(f"aviso: al reanudar, el modelo sale del archivo; se ignoran {', '.join(ignoradas)}")
            cfg = ConfigASRN.desde_config_guardada(_leer(o.reanudar, mmap=True)['config'], checkpoint=o.checkpoint)
        else:
            if o.parametros and o.talla:
                a.error('usa --parametros o --talla, no los dos')
            if o.parametros and o.V is None:
                a.error('--parametros necesita --V (el vocabulario de tu tokenizador)')
            if o.V is None and not o.talla:
                a.error('--V es obligatorio (o --talla, o --reanudar)')
            if o.parametros:
                fijos = {k: v for k, v in dict(d=o.d, n_capas=o.capas, largo_cada=o.largo_cada, mezcla=o.mezcla,
                                               memoria_max=o.memoria_max).items() if v is not None}
                cfg = ConfigASRN.para_parametros(o.parametros, o.V, **fijos)
                o.d = o.capas = o.largo_cada = o.mezcla = o.memoria_max = None
                aviso(f"configuracion para {o.parametros}: d={cfg.d}, capas={cfg.n_capas}, "
                      f"{cfg.contar_parametros():,} parametros")
            else:
                cfg = TALLAS[o.talla] if o.talla else ConfigASRN(V=o.V, d=256, n_capas=8)
            cambios = {k: v for k, v in dict(V=o.V, d=o.d, n_capas=o.capas, largo_cada=o.largo_cada,
                                             mezcla=o.mezcla, memoria_max=o.memoria_max, dropout=o.dropout, checkpoint=o.checkpoint).items() if v is not None}
            cfg = cfg.con(**cambios)

        datos = leer_piezas(o.datos)
        val = leer_piezas(o.validacion) if o.validacion else None
    except (ValueError, OSError) as e:                  # configuracion o archivos: mensaje corto, sin traza
        a.error(str(e))
    if val is None:
        corte = int(datos.numel() * (1 - o.fraccion_val))
        datos, val = datos[:corte], datos[corte:]
        if val.numel() <= o.largo + 1:
            if o.fraccion_val > 0:
                aviso(f"aviso: la validacion ({val.numel()} piezas, --fraccion_val {o.fraccion_val}) no alcanza para "
                      f"una ventana de --largo {o.largo}: se entrena sin validacion")
            val = None
    if o.fsdp:
        modelo = crear_desde_config(cfg, dispositivo='meta')
        modelo = repartir_fsdp(preparar(modelo) if preparar else modelo)
        materializar(modelo, disp, o.semilla)          # nunca existe entero en un solo proceso
    else:
        modelo = crear_desde_config(cfg, semilla=o.semilla)
        modelo = (preparar(modelo) if preparar else modelo).to(disp)
    try:
        return entrenar(modelo, datos, validacion=val, largo=o.largo, lote=o.lote, pasos=o.pasos, lr=o.lr,
                        calentamiento=o.calentamiento, decaimiento=o.decaimiento, acumular=o.acumular,
                        precision=o.precision, compilar={'partes': 'partes', 'todo': True}.get(o.compilar, False),
                        evaluar_cada=o.evaluar_cada,
                        guardar_en=o.guardar, guardar_cada=o.guardar_cada, reanudar=o.reanudar, semilla=o.semilla)
    finally:
        if o.fsdp and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    _principal()
