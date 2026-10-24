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
"""ASRN estable: la capa ASRN con protecciones numericas para textos muy largos (la de omision de `asrn`).

La capa calcula lo mismo que CapaASRN de asrn.nucleo (mismos parametros y mismos nombres: los modelos guardados cargan
tal cual), reescrita parte por parte con la forma de cada tensor y con estas protecciones
(docs/TECHNICAL_REPORT.md, seccion del reloj, explica cada una y lo que se midio):

  Grupo A, sin efecto en el regimen normal (mismas cuentas que v89 salvo el redondeo), encendidas por omision:
    mascara_exacta  las llaves no validas pesan 0 exacto. En v89 la mascara es un puntaje de -1e9: si todos los
                    puntajes validos de una pregunta quedan por debajo (nucleo de tiempo enorme), el softmax manda la
                    masa a las no validas, que incluyen la propia pregunta (copia E_{t+1}) y piezas posteriores: el
                    futuro. Medido: con gamma = 1e9 cambiar el futuro cambia el pasado, en PyTorch y en el kernel.
    reloj_64        el reloj B_t = suma de beta se acumula en 64 bits y el nucleo de tiempo usa la diferencia
                    B_t - B_s tomada antes de multiplicar (en el kernel, en dos partes de 32 bits). Con 32 bits la
                    diferencia entre piezas vecinas pierde 2% con 1 M de piezas, 10% con 4 M, y desde 16 M el reloj se
                    detiene (B + beta == B). La proyeccion del reloj (w . h_norm) va en 32 bits tambien con autocast.
    beta_max        tope de beta (por omision 1e4): un beta enorme (pesos rotos) no vuelve infinito al reloj.
    log_alfa_min    piso de log alfa (por omision -30): log alfa nunca llega a -inf.
    mejor_max       |mejor| <= mejor_max en la senal de la puerta (por omision 1e4): un -1e9 nunca entra a la puerta.
  Grupo B, cambian la funcion (apagadas por omision; para entrenar desde cero con textos largos):
    tope_puerta     logit de la puerta -> c tanh(logit / c): la puerta nunca se satura del todo
    kappa_min       piso de las tasas de Hawkes y racha: con kappa -> 0 la intensidad crece con el largo del texto

Ademas:
  - convertir(modelo): cambia las capas ASRN de un ModeloASRN por capas estables con los mismos pesos
  - RecorteDinamico: recorte del gradiente por grupos (reloj, puerta, resto) con umbral adaptativo; se pasa como
    recorte= a asrn.entrenar()

Uso (asrn.crear_modelo y asrn.cargar ya devuelven capas estables con el grupo A):
    import asrn
    modelo = asrn.crear_modelo(V=8000, d=256, n_capas=8)
    asrn.entrenar(modelo, datos, recorte=asrn.RecorteDinamico(modelo))
    modelo_v89 = asrn.convertir(asrn.nucleo.crear_modelo(V=8000))     # convertir un modelo de la capa sin protecciones
"""
import dataclasses
import math

import torch
import torch.nn.functional as F

from . import nucleo as v89
from .nucleo import _PROHIBIDO, _DesordenarSumar, _a_autocast, _anteriores, _largo_seguido, _prohibido


@dataclasses.dataclass(frozen=True)
class Protecciones:
    """Que protecciones usa la capa (ver el docstring del modulo). Protecciones() = las del grupo A."""
    mascara_exacta: bool = True
    reloj_64: bool = True
    beta_max: float = 1e4
    log_alfa_min: float = -30.0
    mejor_max: float = 1e4
    tope_puerta: float = None
    kappa_min: float = None

    @classmethod
    def ninguna(cls):
        """Sin protecciones: la capa calcula exactamente lo mismo que v89 (para comparar)."""
        return cls(mascara_exacta=False, reloj_64=False, beta_max=None, log_alfa_min=None, mejor_max=None)

    @classmethod
    def textos_largos(cls, largo_entrenamiento=1024):
        """Grupo A + grupo B, para entrenar desde cero: puerta con tope 10 y tasas de Hawkes que olvidan al menos
        en ~4 veces el largo de entrenamiento (asi la intensidad no crece con textos mas largos que esos)."""
        return cls(tope_puerta=10.0, kappa_min=1.0 / (4 * largo_entrenamiento))


class _ProyeccionIEEE(torch.autograd.Function):
    """u = x . w^T + b con cuentas de 32 bits de verdad, multiplicando y sumando elemento a elemento. En GPU un producto
    de matrices de 32 bits va en TF32 (10 bits de mantisa) si torch.backends.cuda.matmul.allow_tf32 esta encendido, y
    entrenar() lo enciende: el reloj perderia la precision que reloj_64 promete. Para k chico (el reloj: k = 2).
    x (..., d) fp32, w (k, d), b (k,) -> (..., k). Para la vuelta guarda x y w (la vuelta si puede ir en TF32)."""

    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(x, w)
        return torch.stack([(x * w[i]).sum(-1) for i in range(w.shape[0])], -1) + b

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        g2 = g.reshape(-1, g.shape[-1])
        gx = (g @ w) if ctx.needs_input_grad[0] else None
        gw = (g2.t() @ x.reshape(-1, x.shape[-1])) if ctx.needs_input_grad[1] else None
        gb = g2.sum(0) if ctx.needs_input_grad[2] else None
        return gx, gw, gb


class _TiempoRelativo(torch.autograd.Function):
    """Nucleo de tiempo de cada (pregunta, llave) de cada ventana: tasa * (B_k - B_q), con la resta en 64 bits.

    Bq (W, C) y Bk (W, 2C): el reloj (64 bits) de las preguntas y llaves de cada ventana; tasa (W, 1, 1) = gamma / C de
    la tabla de la ventana. Salida (W, C, 2C) en el tipo de tasa. Para la vuelta guarda solo sus entradas (vistas del
    reloj ordenado, sin copiar): nada de (W, C, 2C), como v89."""

    @staticmethod
    def forward(ctx, Bq, Bk, tasa):
        ctx.save_for_backward(Bq, Bk, tasa)
        return (Bk.unsqueeze(1) - Bq.unsqueeze(2)).to(tasa.dtype) * tasa

    @staticmethod
    def backward(ctx, g):
        Bq, Bk, tasa = ctx.saved_tensors
        gt = g * tasa                                                              # (W, C, 2C)
        dtasa = (g * (Bk.unsqueeze(1) - Bq.unsqueeze(2)).to(g.dtype)).sum((1, 2), keepdim=True)
        return (-gt.sum(2)).to(Bq.dtype), gt.sum(1).to(Bk.dtype), dtasa


def _softmax_valido(sc, valido, exacta):
    """softmax de sc sobre la ultima dimension, solo entre las llaves validas. sc (..., n), valido (..., n) bool.
    Devuelve (att, hay, maximo): att (..., n) con 0 en las no validas y en las filas sin llaves validas; hay (..., 1);
    maximo (..., 1) el mayor puntaje valido (sin gradiente; _PROHIBIDO si no hay).

    exacta=False: lo de v89 (mascara de -1e9). exacta=True: se resta antes el maximo de las VALIDAS (sin gradiente:
    el softmax no cambia con una constante por fila), asi la mejor valida queda en 0 y una no valida, en -1e9, nunca
    gana. En el regimen normal da los mismos bits que v89: el softmax ya resta ese mismo maximo por dentro."""
    hay = valido.any(-1, keepdim=True)
    prohibido = _prohibido(sc.dtype)
    if not exacta:
        sc = sc.masked_fill(~valido, prohibido)
        maximo = sc.detach().amax(-1, keepdim=True)
        return torch.softmax(sc, dim=-1) * hay, hay, torch.where(hay, maximo, torch.full_like(maximo, _PROHIBIDO))
    maximo = sc.detach().masked_fill(~valido, float('-inf')).amax(-1, keepdim=True)
    ref = torch.where(hay, maximo, torch.zeros_like(maximo))
    att = torch.softmax((sc - ref).masked_fill(~valido, prohibido), dim=-1) * hay
    return att, hay, torch.where(hay, maximo, torch.full_like(maximo, _PROHIBIDO))


class CapaASRNEstable(v89.CapaASRN):
    """CapaASRN con protecciones (ver Protecciones). Mismos argumentos que CapaASRN mas `protecciones`.

    Notacion de formas: B lote, L largo, d ancho, T = n_tablas, nb = n_bits, P = T*nb, C = chunk, cf = confirmar,
    K = sufijo, n = vistos, R = L/C + 1 bloques por fila (con el de ceros delante), W = B*T*R - 1 ventanas."""

    def __init__(self, d, protecciones=None, **kw):
        super().__init__(d, **kw)
        self.protecciones = protecciones or Protecciones()

    # --- reloj propio y estado normalizado -------------------------------------------------------------------

    def _contexto(self, h):
        """h (B, L, d) -> Qc, Kc (B, L, cf), beta (B, L) y log_alfa (B, L).

          h_norm   = LN(h)                                  (B, L, d), en 32 bits (LayerNorm no baja de 32 con autocast)
          Qc, Kc   = h_norm W_q, h_norm W_k                  (B, L, cf), en 16 bits con autocast
          u        = h_norm w_reloj + b_reloj                (B, L, 2), en 32 bits
          beta     = min(softplus(u_0), beta_max)            >= 0: tiempo propio de la pieza
          log_alfa = max(logsigmoid(u_1) + log 2, minimo)    alfa en (0, 2): cuan memorable es la pieza
        Con reloj_64, beta sale en 64 bits: la suma B_t = cumsum(beta) (forward y la escritura paso a paso) se acumula
        asi en 64 bits sin cambiar nada mas.
        Por que u en 32 bits: con fp16, h_norm w (d sumandos) puede pasar de 65 504 y dar inf, y beta = inf rompe el
        reloj entero; con bf16, u tiene 3 cifras y beta lleva un sesgo que se acumula en B. v89 lo calcula en la misma
        multiplicacion que Qc y Kc (en 16 bits con autocast). En GPU, ademas, sin TF32 (_ProyeccionIEEE): la primera
        corrida en A100 tenia TF32 encendido y un producto de matrices de 32 bits pierde ahi 13 de sus 23 bits."""
        p = self.protecciones
        hh32 = self.ln_ctx(h)                                                         # (B, L, d)
        hh = _a_autocast(hh32)
        cf = self.confirmar
        if p.reloj_64 and h.is_cuda:
            # en GPU: Qc, Kc con la multiplicacion de siempre (16 bits con autocast, TF32 si esta permitido); el reloj
            # aparte, en 32 bits de verdad aunque TF32 este encendido (_ProyeccionIEEE; 2 columnas, ~2d cuentas)
            y = F.linear(hh, torch.cat([self.cq.weight, self.ck.weight]))            # (B, L, 2cf)
            u = _ProyeccionIEEE.apply(hh32.float(), self.w_reloj.float(), self.b_reloj.float())   # (B, L, 2)
        elif p.reloj_64 and hh.dtype != hh32.dtype:
            # CPU con autocast: Qc, Kc en 16 bits; el reloj aparte, en 32 (la CPU no usa TF32)
            y = F.linear(hh, torch.cat([self.cq.weight, self.ck.weight]))            # (B, L, 2cf)
            with torch.autocast(device_type=h.device.type, enabled=False):
                u = F.linear(hh32.float(), self.w_reloj.float(), self.b_reloj.float())   # (B, L, 2)
        else:
            # sin autocast todo ya es de 32 bits: una sola multiplicacion, como v89 (mismos bits)
            y = F.linear(hh, torch.cat([self.cq.weight, self.ck.weight, self.w_reloj]),
                         F.pad(self.b_reloj, (2 * cf, 0)))
            u = y[..., 2 * cf:].float()
        beta = F.softplus(u[..., 0])                                                  # (B, L)
        log_alfa = F.logsigmoid(u[..., 1]) + math.log(2)                              # (B, L)
        if p.beta_max is not None:
            beta = beta.clamp(max=p.beta_max)
        if p.log_alfa_min is not None:
            log_alfa = log_alfa.clamp(min=p.log_alfa_min)
        if p.reloj_64:
            beta = beta.double()
        return y[..., :cf], y[..., cf:2 * cf], beta, log_alfa

    def _kappas(self):
        """(n,) tasas de Hawkes y (1,) la de la racha, con el piso kappa_min si esta puesto."""
        k, kr = self.log_kappa.exp(), self.log_kappa_racha.exp()
        m = self.protecciones.kappa_min
        return (k, kr) if m is None else (k.clamp(min=m), kr.clamp(min=m))

    # --- donde copiar: atencion por casilleros ---------------------------------------------------------------

    def _acumular(self, E, Qc, Kc, Bt, log_alfa, sig):
        """Todas las tablas a la vez, en el espacio ordenado por casillero (ver CapaASRN._acumular para la idea).

        E (B, L, d), Qc y Kc (B, L, cf), Bt (B, L) reloj (64 bits con reloj_64), log_alfa (B, L), sig (B, L) firmas.
        Devuelve salida (B, L, d) sumada sobre las tablas y las senales de la puerta (B, L, 1) cada una."""
        if v89._usar_triton_casilleros(E, self.chunk, self.n_tablas * self.n_bits, self.confirmar):
            return self._acumular_triton(E, Qc, Kc, Bt, log_alfa, sig)
        p = self.protecciones
        B, L, _ = E.shape
        T, C, cf = self.n_tablas, self.chunk, self.confirmar
        n_pad = (-L) % C
        R = (L + n_pad) // C + 1
        BT = B * T
        dv = E.device

        phi, V = self._phi_y_valores(E)                                          # (B, L, P), (B, L, d)
        ids = self._ids(phi.detach()).transpose(1, 2).reshape(BT, L)             # (BT, L) casillero; fila = b*T + tabla
        orden = ids.argsort(dim=-1, stable=True)                                 # (BT, L) posicion ordenada -> texto
        nulo = B * L
        fuente = F.pad(orden + torch.arange(0, nulo, L, device=dv).repeat_interleave(T).unsqueeze(1),
                       (C, n_pad), value=nulo)                                   # (BT, R*C) fila de la fuente
        real = fuente != nulo                                                    # (BT, R*C)
        ids_o = F.pad(torch.gather(ids, 1, orden), (C, n_pad), value=-1)         # (BT, R*C)

        def ordenar(x):            # (B, L, k) -> (BT, R*C, k)
            k = x.shape[-1]
            return torch.cat([x.reshape(nulo, k), x.new_zeros(1, k)]).index_select(0, fuente.view(-1)).view(BT, -1, k)

        def preguntas(x):          # (BT, R*C, ...) -> (W, C, ...)
            return x.reshape(BT * R, C, *x.shape[2:])[1:]

        def llaves(x):             # (BT, R*C, ...) -> (W, 2C, ...): bloque anterior + actual, vista sin copiar
            return x.reshape(BT * R * C, *x.shape[2:]).unfold(0, 2 * C, C).movedim(-1, 1)

        en_orden = torch.empty_like(orden).scatter_(1, orden, torch.arange(L, device=dv).expand(BT, L))
        fila = (en_orden + torch.arange(0, BT * R * C, R * C, device=dv).unsqueeze(1)).reshape(-1)  # (BT*L,)

        def desordenar(x):         # (W, C, k) -> (B, T, L, k) en el orden del texto
            k = x.shape[-1]
            return x.reshape(-1, k).index_select(0, fila).view(B, T, L, k)

        phi_o, V_o = ordenar(phi), ordenar(V)                                    # (BT, R*C, P), (BT, R*C, d)
        Qc_o, Kc_o, Pf_o = ordenar(Qc), ordenar(Kc), ordenar(_anteriores(sig, self.sufijo))   # (BT, R*C, cf|K)

        # puntaje de contenido: hash + confirmacion + sufijo exacto                                    (W, C, 2C)
        sc = torch.bmm(preguntas(phi_o), llaves(phi_o).transpose(1, 2)) / self.temperatura
        sc = sc + torch.bmm(preguntas(Qc_o), llaves(Kc_o).transpose(1, 2)) * cf ** -0.5
        sc = sc + self._peso_sufijo(_largo_seguido(preguntas(Pf_o).unsqueeze(2), llaves(Pf_o).unsqueeze(1)))
        # nucleo de tiempo: log alfa_s - (gamma_tau / C)(B_t - B_s)
        tasa = (self.gamma / C).repeat(B).unsqueeze(1)                           # (BT, 1)
        B_o = ordenar(Bt.unsqueeze(-1)).squeeze(-1)                              # (BT, R*C), en el tipo del reloj
        la_o = ordenar(log_alfa.unsqueeze(-1)).squeeze(-1)                       # (BT, R*C)
        if p.reloj_64:
            tasa_w = preguntas(tasa.expand(BT, R * C))[:, :1].unsqueeze(-1)       # (W, 1, 1) la de la tabla
            sc = sc + llaves(la_o).unsqueeze(1) + _TiempoRelativo.apply(preguntas(B_o), llaves(B_o), tasa_w)
        else:                      # v89: (log alfa_k + tasa B_k) - tasa B_q, dos numeros grandes que se cancelan
            sc = sc + llaves(la_o + tasa * B_o).unsqueeze(1) - preguntas(tasa * B_o).unsqueeze(2)

        i = torch.arange(C, device=dv).unsqueeze(1)
        j = torch.arange(2 * C, device=dv)
        anteriores = (j >= i) & (j < C + i)                                      # (C, 2C) las C anteriores en el orden
        valido = (preguntas(ids_o).unsqueeze(2) == llaves(ids_o).unsqueeze(1)) & llaves(real).unsqueeze(1) & anteriores
        att, hay, sc_max = _softmax_valido(sc, valido, p.mascara_exacta)        # (W, C, 2C), (W, C, 1), (W, C, 1)
        out = torch.bmm(att, llaves(V_o))                                        # (W, C, d)

        salida = _DesordenarSumar.apply(out.reshape(-1, out.shape[-1]), fila, fuente.view(-1)[C:], B, T, L)  # (B, L, d)
        senal = desordenar(torch.cat([hay.float(), sc_max.float(),
                                      att.detach().amax(-1, keepdim=True).float()], dim=-1))   # (B, T, L, 3)
        return self._senales(salida, senal)

    def _acumular_triton(self, E, Qc, Kc, Bt, log_alfa, sig):
        """Lo mismo con el kernel fusionado de v89 en modo ESTABLE: el reloj de 64 bits se pasa en dos partes de 32
        (alta, con gradiente, y baja = B - alta, sin gradiente: la derivada de tasa (B_k - B_q) no depende de como se
        parta B) y el kernel usa la mascara exacta. Sin reloj_64 ni mascara_exacta, el kernel de siempre."""
        p = self.protecciones
        B, L, _ = E.shape
        T, C = self.n_tablas, self.chunk
        phi, V = self._phi_y_valores(E)
        ids = self._ids(phi.detach()).transpose(1, 2).reshape(B * T, L)
        orden = ids.argsort(dim=-1, stable=True)
        alto = Bt.float()                                                        # (B, L) con gradiente
        bajo = None
        if p.reloj_64 or p.mascara_exacta:
            bajo = (Bt.detach().double() - alto.detach().double()).float()       # (B, L), |bajo| <= ulp(alto) / 2
        salida, senal = v89._CasillerosTriton.apply(phi, V, Qc, Kc, alto, log_alfa, self.w_sufijo,
                                                    self.gamma, _anteriores(sig, self.sufijo), orden,
                                                    torch.gather(ids, 1, orden), T, C, 1.0 / self.temperatura,
                                                    self.confirmar ** -0.5, bajo)
        return self._senales(salida, senal.view(B, T, L, 3))

    def _senales(self, salida, senal):
        """senal (B, T, L, 3) = (hay, maximo, pico) de cada tabla -> lo que usa la puerta, cada uno (B, L, 1)."""
        T = self.n_tablas
        hay_t, sc_max, pico_t = senal[..., 0:1], senal[..., 1:2], senal[..., 2:3]
        mejor = torch.where(hay_t > 0, sc_max, torch.full_like(sc_max, _PROHIBIDO)).amax(1).clamp(min=_PROHIBIDO)
        return {'salida': salida, 'peso_total': hay_t.sum(1), 'mejor': mejor,
                'pico': (pico_t * hay_t).sum(1), 'peso_ctx': hay_t[:, T - self.tablas_contexto:].sum(1)}

    # --- memoria de lo repetido: vistos, Hawkes y racha ------------------------------------------------------

    def _repeticion(self, sig, beta, Bt, log_alfa):
        """sig, beta, Bt, log_alfa (B, L) -> vistos (B, L, n), hawkes (B, L, n), racha (B, L). Ver CapaASRN._repeticion.
        Las diferencias del reloj entre apariciones seguidas de un grupo se toman en el tipo del reloj (64 bits con
        reloj_64) y recien despues se pasan a 32: exp(-kappa dB) no pierde la diferencia entre dos B grandes."""
        n = self.vistos
        kappa, kappa_r = self._kappas()
        with torch.no_grad():
            orden, repetido = self._ordenar_claves(self._claves_vistos(sig))     # (B, n, L), orden por clave
            vistos = torch.zeros_like(repetido).scatter(-1, orden, repetido)     # (B, n, L) orden del texto
        B_o = torch.gather(Bt.unsqueeze(1).expand(-1, n, -1), -1, orden)        # (B, n, L) reloj en orden por clave
        alfa_o = torch.gather(log_alfa.exp().unsqueeze(1).expand(-1, n, -1), -1, orden)
        dB = F.pad((B_o[..., 1:] - B_o[..., :-1]).clamp(min=0), (1, 0)).float()  # (B, n, L) >= 0
        a = repetido * torch.exp(-kappa.unsqueeze(-1) * dB)                      # (B, n, L) en [0, 1]
        lam = v89.escaneo(a, a * F.pad(alfa_o[..., :-1], (1, 0)))               # (B, n, L)
        hawkes = torch.zeros_like(lam).scatter(-1, orden, lam).transpose(1, 2)  # (B, L, n)
        r = vistos[:, -1].float()                                                # (B, L)
        racha = v89.escaneo(r * torch.exp(-kappa_r * beta.float()), r)          # (B, L)
        return {'vistos': vistos.transpose(1, 2).float(), 'hawkes': hawkes, 'racha': racha}

    def _repeticion_paso(self, claves_mem, claves_t, B_mem, B_t, log_alfa_mem, beta_t, racha_previa):
        """Una pieza nueva contra la memoria (escritura): claves_mem (B, t, n), claves_t (B, n), B_mem y log_alfa_mem
        (B, t), B_t y beta_t (B,), racha_previa (B,)."""
        kappa, kappa_r = self._kappas()
        igual = claves_mem == claves_t.unsqueeze(1)                              # (B, t, n)
        dB = (B_t.unsqueeze(1) - B_mem).float()                                  # (B, t) resta en el tipo del reloj
        nucleo = (log_alfa_mem.unsqueeze(-1) - kappa * dB.unsqueeze(-1)).exp()   # (B, t, n)
        vistos = igual.any(1).float()                                            # (B, n)
        r = vistos[:, -1]
        racha = r * (torch.exp(-kappa_r * beta_t.float()) * racha_previa + 1)
        return {'vistos': vistos.unsqueeze(1), 'hawkes': (igual * nucleo).sum(1).unsqueeze(1),
                'racha': racha.unsqueeze(1)}

    def _atender(self, phi_q, ids_q, Qc_q, Pf_q, B_q, mem, perm_k):
        """Escritura: la pregunta (B, 1, .) contra la memoria (B, t, .), todas las tablas a la vez (B, 1, T, t).
        Mismas protecciones que _acumular: resta del reloj en su tipo y mascara exacta."""
        W = self.chunk
        p = self.protecciones
        base_sc = torch.einsum('bqp,bkp->bqk', phi_q, mem['phi']) / self.temperatura            # (B, 1, t)
        base_sc = base_sc + torch.einsum('bqp,bkp->bqk', Qc_q, mem['Kc']) * self.confirmar ** -0.5
        base_sc = base_sc + self._peso_sufijo(_largo_seguido(Pf_q.unsqueeze(2), mem['Pf'].unsqueeze(1)))
        valido = ids_q.unsqueeze(-1) == mem['ids'].transpose(1, 2).unsqueeze(1)                 # (B, 1, T, t)
        if perm_k is not None:
            valido = valido & perm_k[:, None, None, :]
        cs = valido.cumsum(-1)
        valido = valido & (cs[..., -1:] - cs + valido <= W)                                    # las W mas recientes
        dB = (B_q[:, :, None, None] - mem['B'][:, None, None, :]).float()                       # (B, 1, 1, t)
        nucleo = mem['log_alfa'][:, None, None, :] - (self.gamma / W)[None, None, :, None] * dB  # (B, 1, T, t)
        att, hay, sc_max = _softmax_valido(base_sc.unsqueeze(2) + nucleo, valido, p.mascara_exacta)
        hay = hay.to(mem['V'].dtype)
        out = torch.einsum('bqtk,bkd->bqtd', att, mem['V'])                                      # (B, 1, T, d)
        mejor = sc_max.amax(2).clamp(min=_PROHIBIDO)
        return {'salida': out.sum(2), 'peso_total': hay.sum(2), 'mejor': mejor.to(mem['V'].dtype),
                'pico': (att.detach().amax(-1, keepdim=True) * hay).sum(2),
                'peso_ctx': hay[:, :, self.n_tablas - self.tablas_contexto:].sum(2)}

    def _atender_cand(self, phi_q, Qc_q, Pf_q, B_q, cand, valido):
        """Escritura en O(1): la pregunta (B, 1, .) contra los `chunk` candidatos de cada tabla (B, T, C, .), los que
        _atender elegiria entre toda la memoria. Mismas protecciones: resta del reloj en su tipo y mascara exacta."""
        W = self.chunk
        p = self.protecciones
        sc = torch.einsum('bqp,btkp->btk', phi_q, cand['phi']) / self.temperatura               # (B, T, C)
        sc = sc + torch.einsum('bqp,btkp->btk', Qc_q, cand['Kc']) * self.confirmar ** -0.5
        sc = sc + self._peso_sufijo(_largo_seguido(Pf_q.unsqueeze(2), cand['Pf']))
        dB = (B_q.unsqueeze(-1) - cand['B']).float()                                            # (B, T, C)
        nucleo = cand['log_alfa'] - (self.gamma / W)[None, :, None] * dB
        att, hay, sc_max = _softmax_valido(sc + nucleo, valido, p.mascara_exacta)               # (B, T, C), (B, T, 1)
        hay = hay.to(cand['V'].dtype)
        out = torch.einsum('btk,btkd->btd', att, cand['V'])                                     # (B, T, d)
        return self._acc_cand(out, att, hay, sc_max.amax(1, keepdim=True).clamp(min=_PROHIBIDO), cand['V'].dtype)

    # --- si copiar: la puerta --------------------------------------------------------------------------------

    def _combinar(self, acc, h):
        """h' = h + g * W_o c (B, L, d), con c = salida / tablas con candidatos y la puerta g (B, L, 1):

          logit = w_g . h + w_conf . [tablas con candidatos / T, mejor, pico, tablas de contexto / Tc]
                  + w_vistos . vistos + u_hawkes . log1p(hawkes) + f(racha)
          g     = sigmoid(logit)            (con tope_puerta: sigmoid(c tanh(logit / c)))
        Todo el logit en 32 bits. 'mejor' se acota a +-mejor_max: es el unico termino de la puerta que no esta acotado
        por construccion (un puntaje); hawkes entra con log1p y la racha por una funcion constante pasado su ultimo
        nodo."""
        p = self.protecciones
        salida, pt = acc['salida'], acc['peso_total'].detach()                   # (B, L, d), (B, L, 1)
        salida = salida / acc['peso_total'].clamp(min=1)
        mejor = torch.where(pt > 0, acc['mejor'], torch.zeros_like(pt))
        if p.mejor_max is not None:
            mejor = mejor.clamp(-p.mejor_max, p.mejor_max)
        senales = torch.cat([pt / self.n_tablas, mejor, acc['pico'] / pt.clamp(min=1),
                             acc['peso_ctx'].detach() / self.tablas_contexto], dim=-1)            # (B, L, 4)
        with torch.autocast(device_type=h.device.type, enabled=False):
            logit = F.linear(h, self.g.weight.to(h.dtype), self.g.bias.to(h.dtype))           # (B, L, 1)
        logit = logit + self.g_conf(senales).to(logit.dtype)
        logit = logit + (acc['vistos'].to(logit.dtype) * self.w_vistos.to(logit.dtype)).sum(-1, keepdim=True)
        hawkes = torch.log1p(acc['hawkes']).to(logit.dtype)
        logit = logit + (hawkes * self.u_hawkes.to(logit.dtype)).sum(-1, keepdim=True)
        logit = logit + self._respuesta_racha(acc['racha']).unsqueeze(-1).to(logit.dtype)
        if p.tope_puerta is not None:
            logit = p.tope_puerta * torch.tanh(logit / p.tope_puerta)
        aporte = self.o(salida)                                                  # (B, L, d)
        return h + torch.sigmoid(logit).to(aporte.dtype) * self.drop(aporte)

    # --- todo junto ------------------------------------------------------------------------------------------

    def forward(self, E, h, piezas=None):
        """E (B, L, d) vectores de pieza, h (B, L, d) estado, piezas (B, L) numeros de pieza (o None) -> h' (B, L, d).

          sig            (B, L)      firma exacta de cada pieza (su numero, en 32 bits)
          Qc, Kc         (B, L, cf)  confirmacion, desde el estado normalizado
          beta, log_alfa (B, L)      reloj propio (beta en 64 bits con reloj_64)
          Bt             (B, L)      reloj acumulado B_t = sum_{u<=t} beta_u
          acc            salida (B, L, d) de las T tablas y senales (B, L, 1); vistos, hawkes (B, L, n); racha (B, L)
        """
        sig = self._firma(E) if piezas is None else v89.firmas_de_piezas(piezas)
        Qc, Kc, beta, log_alfa = self._contexto(h)
        Bt = beta.cumsum(1)
        acc = self._acumular(E, Qc, Kc, Bt, log_alfa, sig)
        acc.update(self._repeticion(sig, beta, Bt, log_alfa))
        return self._combinar(acc, h)


def convertir(modelo, protecciones=None):
    """Cambia (en su lugar) cada CapaASRN de un ModeloASRN por una CapaASRNEstable con los mismos pesos y la devuelve.
    Antes de repartir_fsdp() y de compilar. Las capas ya estables solo cambian de protecciones."""
    base = v89._base(modelo)
    if v89.esta_repartido(base):
        raise ValueError("convertir() va antes de repartir_fsdp(): el modelo ya esta repartido")
    cfg = base.config
    for j, vieja in enumerate(base.largos):
        if isinstance(vieja, CapaASRNEstable):
            vieja.protecciones = protecciones or Protecciones()
            continue
        disp = vieja.v.weight.device
        with torch.device('meta'):
            nueva = CapaASRNEstable(cfg['d'], protecciones, dropout=cfg['dropout'], **cfg['opciones_capa'])
        nueva = nueva.to_empty(device=disp)
        nueva.load_state_dict(vieja.state_dict())
        nueva.train(vieja.training)
        base.largos[j] = nueva
    return modelo


# ----------------------------------------------------------------------------------------------------------------
# Recorte dinamico del gradiente
# ----------------------------------------------------------------------------------------------------------------

GRUPOS_POR_OMISION = {
    'reloj': ('w_reloj', 'b_reloj', 'gamma', 'log_kappa', 'log_kappa_racha'),
    'puerta': ('g.weight', 'g.bias', 'g_conf', 'w_vistos', 'u_hawkes', 'd_racha', 'w_sufijo'),
}


class RecorteDinamico:
    """Recorte del gradiente por grupos con un umbral que se adapta a la historia de cada grupo.

    Cada grupo g (por omision: el reloj y las tasas, la puerta, y el resto) tiene su norma n_g. Con una media movil
    exponencial de la norma (mu_g) y de su cuadrado:
        umbral_g = min(tope, mu_g + z * sigma_g)      (antes de `calentar` pasos: solo tope)
        si n_g > umbral_g, los gradientes del grupo se multiplican por umbral_g / n_g
    Las medias se actualizan con la norma ya recortada (un pico no sube el umbral del paso siguiente) y no se
    actualizan si la norma no es finita (ese paso lo salta entrenar()).
    Por que por grupos: con un solo recorte global, un pico en unos pocos numeros (gamma, el reloj) achica en ese paso
    el gradiente de todo el modelo; por grupos, solo el del grupo con el pico.

    Sin sincronizar con la GPU (todo en tensores) y sin estado fuera de este objeto. Uso: entrenar(..., recorte=r), o
    a mano: norma = r() despues de backward() y antes de opt.step(). Devuelve la norma total ANTES de recortar.
    Probado en un solo proceso; con el modelo repartido (FSDP) todavia no. Su historia no se guarda con el modelo: al
    reanudar, vuelve a calentar."""

    def __init__(self, modelo, tope=1.0, z=3.0, beta=0.98, calentar=50, grupos=None):
        assert tope > 0 and z > 0 and 0 < beta < 1 and calentar >= 0
        self.tope, self.z, self.beta, self.calentar = float(tope), float(z), float(beta), int(calentar)
        grupos = GRUPOS_POR_OMISION if grupos is None else grupos
        self.grupos = {k: [] for k in list(grupos) + ['resto']}
        for nombre, p in v89._base(modelo).named_parameters():
            if not p.requires_grad:
                continue
            partes = nombre.split('.')
            # cada patron es el nombre del parametro ('gamma'), el de su modulo ('g_conf') o los dos ('g.weight')
            claves = {partes[-1], '.'.join(partes[-2:])} | ({partes[-2]} if len(partes) > 1 else set())
            g = next((k for k, cl in grupos.items() if claves & set(cl)), 'resto')
            self.grupos[g].append(p)
        self.grupos = {k: v for k, v in self.grupos.items() if v}
        self.pasos = 0
        self.media = self.media2 = None
        self.ultimo = {}

    def __call__(self):
        normas = []
        for ps in self.grupos.values():
            gs = [p.grad for p in ps if p.grad is not None]
            normas.append(torch.nn.utils.get_total_norm(gs) if gs else torch.zeros((), device=ps[0].device))
        normas = torch.stack([n.full_tensor() if v89._es_dtensor(n) else n for n in normas]).float()   # (G,)
        if self.media is None:
            self.media, self.media2 = torch.zeros_like(normas), torch.zeros_like(normas)
        tope = torch.full_like(normas, self.tope)
        if self.pasos >= self.calentar and self.pasos > 0:
            corr = 1 - self.beta ** self.pasos                                    # correccion de sesgo de la media
            mu, mu2 = self.media / corr, self.media2 / corr
            sigma = (mu2 - mu.square()).clamp(min=0).sqrt()
            umbral = torch.minimum(tope, mu + self.z * sigma)
        else:
            umbral = tope
        factor = (umbral / (normas + 1e-6)).clamp(max=1.0)                        # (G,)
        finito = torch.isfinite(normas)
        for (k, ps), f in zip(self.grupos.items(), factor):
            gs = [p.grad for p in ps if p.grad is not None]
            if gs:
                torch._foreach_mul_(gs, torch.where(torch.isfinite(f), f, 1.0).to(gs[0].dtype))
        recortada = torch.where(finito, torch.minimum(normas, umbral), self.media)
        self.media = torch.where(finito, self.beta * self.media + (1 - self.beta) * recortada, self.media)
        self.media2 = torch.where(finito, self.beta * self.media2 + (1 - self.beta) * recortada.square(), self.media2)
        self.pasos += 1
        self.ultimo = {'normas': normas, 'umbral': umbral, 'factor': factor}
        return normas.square().sum().sqrt()

    def resumen(self):
        """{grupo: (norma, umbral, factor)} del ultimo paso (lee de la GPU: para registrar, no en cada paso)."""
        if not self.ultimo:
            return {}
        return {k: tuple(float(self.ultimo[c][i]) for c in ('normas', 'umbral', 'factor'))
                for i, k in enumerate(self.grupos)}
