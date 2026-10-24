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
"""Pruebas del nucleo de ASRN 1.0 (asrn/nucleo.py, el codigo de v89). Correr desde la carpeta del repositorio con:
    python tests/test_nucleo.py [parte_del_nombre ...]
(usa benchmarks/rivales_lineales.py para las pruebas de los rivales de costo lineal)

Cubren la capa y sus kernels Triton (contra el camino de PyTorch), el modelo, la configuracion y las tallas, inicializar
en meta, el checkpoint por niveles, la precision mixta, FSDP con 2 procesos en CPU, la causalidad, los casos limite y los
mensajes de error claros al guardar, reanudar y en la linea de comandos. Ademas: escribir paso a paso da lo mismo que el
modelo entero, la memoria guardada para la vuelta es proporcional al largo (O(N)), y entrenar aprende a copiar.

Con PyTorch anterior a 2.14, las dos pruebas de torch.compile se informan como 'salta' (ver su explicacion), no como
'ok'; sin Triton, tambien las de sus kernels."""
import math
import os
import sys
import tempfile

import torch

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_AQUI)
for _ruta in (_RAIZ, os.path.join(_RAIZ, 'benchmarks')):
    # el paquete asrn (en la raiz) y los rivales de costo lineal
    if _ruta not in sys.path:
        sys.path.insert(0, _ruta)

import asrn.nucleo as v89


class Saltada(Exception):
    """La prueba no se puede correr aqui (falta Triton, o la version de PyTorch no la permite): el corredor la cuenta
    aparte, no como 'ok'."""

TOL = 1e-4


def _abrir(capas):
    """La puerta de confirmacion, el sufijo y los 'vistos' arrancan en cero: se abren un poco para que todo cuente."""
    with torch.no_grad():
        for c in capas:
            torch.nn.init.normal_(c.cq.weight, std=0.3)
            c.w_sufijo.normal_(std=0.5)
            c.w_vistos.normal_(std=0.5)
            c.g_conf.weight.normal_(std=0.3)


def _modelo(V=60, d=32, n_capas=3, semilla=1, **kw):
    """Un modelo chico con una capa ASRN tras cada bloque, con la confirmacion, el sufijo y los 'vistos' abiertos (los
    parametros que se agregaron a la capa en v86-v89 quedan en su valor inicial neutro)."""
    m = v89.crear_modelo(V, d=d, n_capas=n_capas, largo_cada=1, semilla=semilla, **kw)
    _abrir(m.largos)
    return m


def _mover_lo_nuevo(capas, semilla=0):
    """Saca de su valor neutro los parametros nuevos de v89 (como despues de entrenar), para probarlos de verdad."""
    g = torch.Generator().manual_seed(semilla)
    with torch.no_grad():
        for c in capas:
            for n, p in c.named_parameters():
                if n in v89._NUEVOS_V87:
                    p.add_(torch.randn(p.shape, generator=g) * 0.5)


def _texto_con_repeticiones(B, L, V, semilla=0):
    """Texto con trozos repetidos, para que haya casilleros llenos, sufijos y 'vistos'."""
    g = torch.Generator().manual_seed(semilla)
    x = torch.randint(0, V, (B, L), generator=g)
    trozo = x[:, :L // 4].clone()
    x[:, L // 2:L // 2 + L // 4] = trozo
    return x


def test_sin_perillas_de_escritura():
    """v89 escribe tal como fue entrenado: sin escala_vistos ni ajuste_puerta, y el modelo queda como estaba."""
    import inspect
    m = _modelo(chunk=8, n_bits=3)
    _mover_lo_nuevo(m.largos)
    assert 'escala_vistos' not in inspect.signature(v89.escribir).parameters
    assert 'ajuste_puerta' not in inspect.signature(v89.escribir).parameters
    assert not any(hasattr(c, 'escala_vistos') or hasattr(c, 'ajuste_puerta') for c in m.largos)
    antes = {k: v.clone() for k, v in m.state_dict().items()}
    m.train()
    v89.escribir(m, _texto_con_repeticiones(2, 30, 60), n_nuevos=10)
    assert m.training and all(torch.equal(v, antes[k]) for k, v in m.state_dict().items())
    try:
        v89.escribir(m, torch.zeros(1, 0, dtype=torch.long), n_nuevos=5)
    except AssertionError:
        pass
    assert m.training


def test_escribir_paso_a_paso_igual_que_entero():
    for mover in (False, True):                       # neutro y con todo lo nuevo de v89 actuando
        m = _modelo(chunk=8, n_bits=3)
        if mover:
            _mover_lo_nuevo(m.largos)
        m.eval()
        x0 = _texto_con_repeticiones(2, 40, 60)
        sec, pasos = v89.escribir(m, x0, n_nuevos=60, memoria_de_lo_ajeno=False, top_k=0)
        with torch.no_grad():
            entero = m(sec[:, :-1])[:, 40 - 1:]
        assert (entero - pasos).abs().max() < TOL, (mover, (entero - pasos).abs().max())


def test_escritura_o1_igual_que_o_t():
    """La escritura O(1) (anillos por casillero, Hawkes por clave) da lo mismo que la O(t) (cada pieza contra toda la
    memoria) y que el modelo con el texto entero: con casilleros que se llenan (el anillo pisa a los viejos), lotes de
    varios textos, entradas de 1 y 2 piezas, con y sin memoria de lo ajeno, y memoria que tiene que crecer."""
    for mover in (False, True):
        m = _modelo(chunk=4, n_bits=2)                # 4 casilleros por tabla, 4 por anillo: se llenan enseguida
        if mover:
            _mover_lo_nuevo(m.largos)
        m.eval()
        for B, L0, n in ((2, 40, 50), (1, 2, 30), (3, 1, 20)):
            x = _texto_con_repeticiones(B, L0 + n, 60)
            for ajeno in (False, True):
                salidas = {}
                for o1 in (True, False):
                    esc = v89._Escritor(m, memoria_de_lo_ajeno=ajeno, o1=o1)
                    lgs = [esc.inicio(x[:, :L0], capacidad=L0 + 3)]
                    lgs += [esc.paso(x[:, t]) for t in range(L0, L0 + n - 1)]
                    salidas[o1] = torch.stack(lgs, 1)
                assert (salidas[True] - salidas[False]).abs().max() < TOL, (mover, B, L0, ajeno)
                if not ajeno:
                    with torch.no_grad():
                        entero = m(x[:, :L0 + n - 1])[:, L0 - 1:]
                    assert (salidas[True] - entero).abs().max() < TOL, (mover, B, L0)


def test_anillo_y_ngramas_como_su_definicion():
    """_anillo_desde guarda en cada casillero las C posiciones mas recientes; _anillo_meter sigue igual; _Ngramas da lo
    mismo que _continuaciones_de_cita sobre el texto entero, pieza a pieza."""
    g = torch.Generator().manual_seed(0)
    B, L, T, nc, C = 2, 70, 3, 5, 4
    ids = torch.randint(0, nc, (B, L, T), generator=g, dtype=torch.int32)
    anillo, cuenta = v89._anillo_desde(ids, 50, C, nc)
    for t in range(50, L):
        v89._anillo_meter(anillo, cuenta, ids[:, t], t)
    for b in range(B):
        for tau in range(T):
            for k in range(nc):
                pos = [p for p in range(L) if ids[b, p, tau] == k]
                assert cuenta[b, tau, k] == len(pos)
                assert sorted(p for p in anillo[b, tau, k].tolist() if p >= 0) == pos[-C:], (b, tau, k)
    x = torch.randint(0, 6, (2, 40), generator=g)
    for n in (1, 2, 4):
        ng = v89._Ngramas(x[:, :3].tolist(), n)
        for t in range(3, 40):
            assert torch.equal(ng.siguientes(6, 'cpu'), v89._continuaciones_de_cita(x[:, :t], 6, n)), (n, t)
            ng.agregar(x[:, t].tolist())


def test_rivales_lineales_nucleos_y_causalidad():
    """Mamba-2 y RWKV-7 (rivales_lineales.py, los rivales de las pruebas en GPU): la recurrencia paso a paso de cada
    uno da lo mismo que su definicion escrita de otra forma; los modelos son causales, aprenden (gradientes finitos en
    todo) y rival_parecido llega cerca del tamano pedido."""
    import rivales_lineales as R
    g = torch.Generator().manual_seed(0)
    B, L, H, D = 2, 9, 2, 4
    q, k, v = (torch.randn(B, L, H, D, generator=g) for _ in range(3))
    gg = -torch.rand(B, L, H, generator=g)
    # SSD: o_t = suma_{s<=t} (q_t . k_s) exp(G_t - G_s) v_s, con G la suma acumulada de g
    G = gg.cumsum(1)
    esperado = torch.zeros(B, L, H, D)
    for t in range(L):
        for s_ in range(t + 1):
            peso = (q[:, t] * k[:, s_]).sum(-1) * (G[:, t] - G[:, s_]).exp()
            esperado[:, t] += peso[..., None] * v[:, s_]
    assert (R.nucleo_ssd_lento(q, k, v, gg) - esperado).abs().max() < 1e-4
    # RWKV-7: S_t = S_{t-1} (diag(exp(w_t)) + a_t b_t^T) + v_t k_t^T, o_t = S_t r_t, con matrices enteras
    r, w = torch.randn(B, L, H, D, generator=g), -0.6 * torch.rand(B, L, H, D, generator=g)
    a, b = torch.randn(B, L, H, D, generator=g) * 0.3, torch.randn(B, L, H, D, generator=g) * 0.3
    S = torch.zeros(B, H, D, D)
    esperado = torch.zeros(B, L, H, D)
    for t in range(L):
        M = torch.diag_embed(w[:, t].exp()) + a[:, t, :, :, None] * b[:, t, :, None, :]
        S = S @ M + v[:, t, :, :, None] * k[:, t, :, None, :]
        esperado[:, t] = (S @ r[:, t, :, :, None]).squeeze(-1)
    assert (R.nucleo_rwkv7_lento(r, w, k, v, a, b) - esperado).abs().max() < 1e-4
    for tipo in ('mamba2', 'rwkv7'):
        m = R.ModeloLineal(tipo, 50, 32, 2)
        x = torch.randint(0, 50, (2, 24), generator=g)
        y = m(x)
        y.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()), tipo
        with torch.no_grad():
            x2 = x.clone()
            x2[:, 12:] = (x2[:, 12:] + 1) % 50
            assert (m(x2)[:, :12] - y[:, :12]).abs().max() < 1e-5, tipo
        c = R.rival_parecido(tipo, 3e5, 500, 64)
        assert abs(c.num_parametros() - 3e5) / 3e5 < 0.15, (tipo, c.num_parametros())


def test_variantes_del_bloque_local():
    """mezcla ('conv', 'memoria'): la formula de parametros es exacta, son causales, aprenden, guardan y cargan; la
    memoria es su recurrencia m_t = f_t m_{t-1} + (1 - f_t) u_t con f <= 1 - 1/memoria_max; con_mismos_parametros
    queda a menos de 1% del presupuesto."""
    g = torch.Generator().manual_seed(0)
    base = v89.ConfigASRN(V=300, d=64, n_capas=3)
    for mezcla in v89._MEZCLAS:
        c = base.con(mezcla=mezcla)
        m = v89.crear_desde_config(c, semilla=0)
        assert c.contar_parametros() == v89._contar_formula(c) == m.num_parametros(), mezcla
        x = torch.randint(0, 300, (2, 50), generator=g)
        y = m(x)
        y.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        with torch.no_grad():
            x2 = x.clone()
            x2[:, 30:] = (x2[:, 30:] + 1) % 300
            assert (m(x2)[:, :30] - y[:, :30]).abs().max() < 1e-5, mezcla
        with tempfile.TemporaryDirectory() as t:
            f = os.path.join(t, 'm.pt')
            v89.guardar(m, f)
            m2 = v89.cargar(f, 'cpu')
            c2 = v89.ConfigASRN.desde_modelo(m2)
            assert torch.equal(m.eval()(x), m2(x)) and c2.mezcla == mezcla
        v = base.con_mismos_parametros(mezcla=mezcla)
        assert abs(v.contar_parametros() - base.contar_parametros()) < 0.01 * base.contar_parametros()
    b = v89.BloqueLocal(16, mezcla='memoria', memoria_max=8)
    z = torch.randn(2, 20, 16, generator=g)
    u, f = b.mem(z).chunk(2, dim=-1)
    f = torch.sigmoid(f) * (1 - 1 / 8)                   # el olvido no pasa de 1 - 1/memoria_max
    mem, salida = torch.zeros(2, 16), []
    for t in range(20):
        mem = f[:, t] * mem + (1 - f[:, t]) * u[:, t]
        salida.append(mem)
    assert (b._memoria(z) - b.mem_o(torch.stack(salida, 1))).abs().max() < 1e-5
    c = base.con(mezcla='memoria', memoria_max=16)
    assert v89.ConfigASRN.desde_modelo(v89.crear_desde_config(c)).memoria_max == 16
    for mal in (dict(mezcla='otra'), dict(mezcla='canal'), dict(memoria_max=1)):
        try:
            base.con(**mal)
        except ValueError:
            pass
        else:
            raise AssertionError(mal)


def test_lo_nuevo_cambia_la_salida_y_aprende():
    """Fuera del valor neutro, lo nuevo cambia la salida; en el valor neutro ya recibe gradiente (puede aprender)."""
    m = _modelo(chunk=8, n_bits=3)
    x = _texto_con_repeticiones(2, 64, 60)
    with torch.no_grad():
        neutro = m(x)
    m(x).square().mean().backward()
    for c in m.largos:
        for n, p in c.named_parameters():
            if n in ('gamma', 'w_reloj', 'u_hawkes', 'd_racha'):
                assert p.grad is not None and p.grad.abs().max() > 0, n
    _mover_lo_nuevo(m.largos)
    with torch.no_grad():
        assert (m(x) - neutro).abs().max() > 1e-3


def test_escribir_crece_la_memoria():
    m = _modelo(chunk=8, n_bits=3)
    _mover_lo_nuevo(m.largos)
    m.eval()
    esc = v89._Escritor(m, memoria_de_lo_ajeno=False)
    x0 = _texto_con_repeticiones(1, 10, 60)
    esc.inicio(x0, capacidad=12)                      # menos de lo necesario: tiene que crecer sola
    sec = [x0]
    lgs = []
    for k in range(20):
        p = torch.tensor([(7 * k) % 60])
        sec.append(p.view(1, 1))
        lgs.append(esc.paso(p))
    with torch.no_grad():
        entero = m(torch.cat(sec, 1))[:, 10:]
    assert (entero - torch.stack(lgs, 1)).abs().max() < TOL


def test_hawkes_y_racha_igual_que_su_definicion():
    """hawkes: la suma O(N^2) de la definicion; su soporte es 'vistos' bit a bit. racha: la recurrencia paso a paso."""
    torch.manual_seed(0)
    capa = v89.CapaASRN(16)
    _mover_lo_nuevo([capa])
    L = 150
    x = _texto_con_repeticiones(3, L, 6)
    sig = v89.firmas_de_piezas(x)
    beta = torch.rand(3, L) * 0.5
    Bt, log_alfa = beta.cumsum(1), torch.randn(3, L) * 0.3
    r = capa._repeticion(sig, beta, Bt, log_alfa)
    claves = capa._claves_vistos(sig)                                                   # (B, L, n)
    antes = torch.ones(L, L, dtype=torch.bool).tril(-1)                                  # [t, s]: s < t
    for k in range(capa.vistos):
        igual = (claves[..., k].unsqueeze(-1) == claves[..., k].unsqueeze(-2)) & antes
        nucleo = log_alfa.exp().unsqueeze(1) * torch.exp(
            -capa.log_kappa[k].exp() * (Bt.unsqueeze(-1) - Bt.unsqueeze(-2)).clamp(min=0))
        assert torch.allclose(r['hawkes'][..., k], (igual * nucleo).sum(-1), atol=1e-4), k
    assert torch.equal((r['hawkes'] > 0).float(), r['vistos'])
    assert torch.equal(r['vistos'], capa._vistos(sig))
    racha, l = [], torch.zeros(3)
    for t in range(L):
        v = r['vistos'][:, t, -1]
        l = v * (torch.exp(-capa.log_kappa_racha.exp() * beta[:, t]) * l + 1)
        racha.append(l)
    assert torch.allclose(r['racha'], torch.stack(racha, 1), atol=1e-4)
    assert r['racha'].max() > 3                                                          # hay rachas largas


def test_respuesta_racha_lineal_por_tramos_y_no_monotona():
    capa = v89.CapaASRN(16, nodos_racha=5, racha_max=80)
    racha = torch.tensor([[0.0, 1.0, 5.0, 9.0, 12.0, 80.0, 500.0]])
    assert torch.equal(capa._respuesta_racha(racha), torch.zeros_like(racha))         # neutra al inicio
    with torch.no_grad():
        capa.d_racha.copy_(torch.tensor([0.0, 1.0, 1.0, -6.0, 0.0]))                     # sube y cae en seco
    y = capa._respuesta_racha(racha)
    nodos = torch.expm1(torch.arange(5) * capa.racha_paso)
    assert torch.allclose(capa._respuesta_racha(nodos.unsqueeze(0)), capa.u_racha.detach().unsqueeze(0), atol=1e-5)
    assert y[0, 2] > y[0, 1] > y[0, 0] and y[0, 4] < 0 and y[0, -1] == -4               # no monotona; plana al final
    x = torch.rand(1, 40, requires_grad=True) * 30
    capa._respuesta_racha(x).sum().backward()
    assert capa.d_racha.grad.abs().sum() > 0


def test_freno_de_racha_se_extiende_pasado_los_datos():
    """Un nodo al que los datos nunca llegan no recibe gradiente: su incremento queda en 0 y hereda el valor del
    ultimo nodo con datos. Antes (valores por nodo) esos nodos quedaban en 0 y una racha muy larga ya no se frenaba."""
    capa = v89.CapaASRN(16, nodos_racha=16, racha_max=256)
    opt = torch.optim.Adam([capa.d_racha], lr=0.05)
    racha = torch.rand(4096) * 12                                         # los datos: rachas de a lo sumo 12
    objetivo = torch.where(racha < 6, 3.0, -4.0)                          # copiar al empezar, frenar pasado 6
    for _ in range(300):
        opt.zero_grad()
        (capa._respuesta_racha(racha.unsqueeze(0))[0] - objetivo).square().mean().backward()
        opt.step()
    ultimo_con_datos = int(math.log1p(12) / capa.racha_paso) + 1
    assert torch.all(capa.d_racha.detach()[ultimo_con_datos + 1:] == 0)    # sin gradiente: incremento intacto
    lejos = capa._respuesta_racha(torch.tensor([[12.0, 18.0, 85.0, 500.0]]))[0].detach()
    assert torch.all(lejos < -3), lejos                                   # una racha de 85 sigue frenada


def test_migrar_racha_de_archivos_viejos_es_exacta():
    """Los archivos de v86 guardaban 'u_racha' (valor por nodo): cargan como incrementos sin cambiar nada."""
    capa = v89.CapaASRN(16, nodos_racha=6, racha_max=50)
    with torch.no_grad():
        capa.d_racha.normal_()
    racha = torch.rand(1, 100) * 80
    antes = capa._respuesta_racha(racha)
    pesos = {k: v for k, v in capa.state_dict().items() if k != 'd_racha'}
    pesos['u_racha'] = capa.u_racha.detach().clone()
    pesos, migrados = v89._migrar_racha(pesos)
    nueva = v89.CapaASRN(16, nodos_racha=6, racha_max=50)
    nueva.load_state_dict(pesos)
    assert migrados == ['d_racha'] and torch.allclose(nueva._respuesta_racha(racha), antes, atol=1e-5)


def test_racha_no_agranda_la_memoria():
    capa = v89.CapaASRN(64)
    racha = torch.rand(2, 1024, requires_grad=True) * 40
    por_pieza = bytes_guardados(lambda: capa._respuesta_racha(racha)) / 2048
    assert por_pieza <= 24, por_pieza                       # nodo (int64) + fraccion + pendiente: 6 floats


# ----------------------------------------------------------------------------------------------------------------
# Escaneo asociativo
# ----------------------------------------------------------------------------------------------------------------

def _recurrencia_secuencial(a, b):
    S, out = torch.zeros_like(b[..., 0]), []
    for i in range(a.shape[-1]):
        S = a[..., i] * S + b[..., i]
        out.append(S)
    return torch.stack(out, -1)


def test_escaneo_igual_que_la_recurrencia():
    g = torch.Generator().manual_seed(0)
    for N in (1, 2, 3, 7, 64, 1000, 4097):
        a = torch.rand(3, 5, N, generator=g, dtype=torch.float64)
        b = torch.randn(3, 5, N, generator=g, dtype=torch.float64)
        a[..., ::7] = 0                                   # reinicios de segmento
        assert torch.allclose(v89.escaneo(a, b), _recurrencia_secuencial(a, b), atol=1e-12), N
    a = torch.rand(2, 33, generator=g, dtype=torch.float64, requires_grad=True)
    b = torch.randn(2, 33, generator=g, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(v89.escaneo, (a, b))


_PRUEBA_TRITON = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
import torch, asrn.nucleo as v89
assert v89._usar_triton(torch.zeros(1))
g = torch.Generator().manual_seed(0)
for N in (1, 2, 5, 300, 2500):                           # 2500: varios bloques de 1024 con acarreo
    a = torch.rand(3, 4, N, generator=g); b = torch.randn(3, 4, N, generator=g); a[..., ::7] = 0
    ar, br = a.double().requires_grad_(), b.double().requires_grad_()
    at, bt = a.clone().requires_grad_(), b.clone().requires_grad_()
    w = torch.randn(3, 4, N, generator=g)
    (v89._escaneo_ida(ar, br) * w.double()).sum().backward()
    st = v89.escaneo(at, bt)
    (st * w).sum().backward()
    ga = ar.grad if ar.grad is not None else torch.zeros_like(ar)
    assert torch.allclose(st.double(), v89._escaneo_ida(a.double(), b.double()), atol=1e-5), N
    assert torch.allclose(at.grad.double(), ga, atol=1e-5) and torch.allclose(bt.grad.double(), br.grad, atol=1e-5), N
with torch.no_grad():
    assert torch.allclose(v89.escaneo(a, b).double(), v89._escaneo_ida(a.double(), b.double()), atol=1e-5)
print('ok')
"""


def test_escaneo_triton_igual_que_pytorch():
    """El kernel Triton del escaneo (ida y vuelta) da lo mismo que el arbol de PyTorch. Corre en CPU con el
    interprete de Triton (TRITON_INTERPRET=1), que ejecuta el mismo kernel que la GPU."""
    import subprocess
    try:
        import triton  # noqa: F401
    except ImportError:
        raise Saltada('Triton no esta instalado')
    r = subprocess.run([sys.executable, '-c', _PRUEBA_TRITON, _RAIZ],
                       capture_output=True, text=True, env={**os.environ, 'TRITON_INTERPRET': '1'}, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith('ok'), r.stderr[-3000:]


_PRUEBA_CASILLEROS = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
import torch, asrn.nucleo as v89
assert v89._usar_triton_casilleros(torch.zeros(1), 16, 32, 16)


def gradientes(capa, x, E0, h0, triton, bf16=False):
    os.environ['ASRN_TRITON_CASILLEROS'] = '1' if triton else '0'
    capa.zero_grad()
    E, h = E0.clone().requires_grad_(), h0.clone().requires_grad_()
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
        y = capa(E, h, x)
    (y.float() * torch.randn(y.shape, generator=torch.Generator().manual_seed(1))).sum().backward()
    return y.detach().float(), {'E': E.grad, 'h': h.grad, **{n: p.grad.clone() for n, p in capa.named_parameters()
                                                             if p.grad is not None}}


def caso(d, kw, B, L, bf16=False):
    torch.manual_seed(0)
    capa = v89.CapaASRN(d, **kw)
    with torch.no_grad():
        for n, p in capa.named_parameters():
            p.add_(torch.randn_like(p) * (0.5 if n in v89._NUEVOS_V87 else 0.05))
    x = torch.randint(0, 12, (B, L))
    x[:, L // 2:L // 2 + L // 4] = x[:, :L // 4]
    E0, h0 = torch.randn(12, d)[x], torch.randn(B, L, d)
    if not bf16:                                   # fp32: lo mismo que PyTorch salvo el redondeo
        (y0, g0), (y1, g1) = gradientes(capa, x, E0, h0, False), gradientes(capa, x, E0, h0, True)
        assert (y0 - y1).abs().max() <= 1e-5 * y0.abs().max(), (d, kw, L)
        for k in g0:
            assert (g0[k] - g1[k]).abs().max() <= 1e-4 * (g0[k].abs().max() + 1e-3), (d, kw, L, k)
    else:                                          # bf16: no mas lejos de fp32 que la version de PyTorch en bf16
        _, verdad = gradientes(capa, x, E0, h0, False)
        _, ref = gradientes(capa, x, E0, h0, False, True)
        _, tri = gradientes(capa, x, E0, h0, True, True)
        for k in verdad:
            n = verdad[k].abs().max() + 1e-3
            assert (tri[k] - verdad[k]).abs().max() / n <= 1.5 * (ref[k] - verdad[k]).abs().max() / n + 1e-2, k


caso(32, dict(chunk=16, n_bits=4), 2, 90)          # largo que no es multiplo del bloque
caso(32, dict(chunk=16, n_bits=4), 1, 5)           # mas corto que un bloque
caso(32, dict(chunk=16, n_bits=4), 1, 1)
caso(96, dict(n_bits=14), 2, 150)                  # la configuracion de siempre: phi de 112, d no multiplo de 64
caso(48, dict(chunk=32, n_bits=6, confirmar=20, sufijo=6), 2, 200)
caso(64, dict(), 2, 160, bf16=True)
print('ok')
"""


def test_casilleros_triton_igual_que_pytorch():
    """El kernel Triton de la atencion por casilleros (ida y vuelta) da lo mismo que _acumular de PyTorch: en fp32
    salvo el redondeo, y en bf16 no mas lejos de fp32 que la version de PyTorch en bf16. Corre en CPU con el
    interprete de Triton (TRITON_INTERPRET=1)."""
    import subprocess
    try:
        import triton  # noqa: F401
    except ImportError:
        raise Saltada('Triton no esta instalado')
    r = subprocess.run([sys.executable, '-c', _PRUEBA_CASILLEROS, _RAIZ],
                       capture_output=True, text=True, env={**os.environ, 'TRITON_INTERPRET': '1'}, timeout=1800)
    assert r.returncode == 0 and r.stdout.strip().endswith('ok'), r.stderr[-3000:]


def test_paso_grafado_mismas_cuentas_y_limites():
    """PasoGrafado sin grabar (en CPU) da exactamente los mismos pesos que el paso de siempre, salta los pasos con
    NaN sin tocar nada y rechaza lo que no se puede grabar. Grabar de verdad (CUDA Graphs) solo se puede en GPU:
    lo mide history/experiments/medir_velocidad_v89.py."""
    V, lote, largo = 50, 3, 40
    a = v89.crear_modelo(V, d=32, n_capas=3, semilla=0)
    b = v89.crear_modelo(V, d=32, n_capas=3, semilla=0)
    oa, ob = v89.crear_optimizador(a, 1e-2), v89.crear_optimizador(b, 1e-2)
    paso = v89.PasoGrafado(b, ob, lote, largo, precision='fp32', recorte=0.5)
    g = torch.Generator().manual_seed(0)
    for i in range(3):
        x, y = torch.randint(0, V, (lote, largo), generator=g), torch.randint(0, V, (lote, largo), generator=g)
        lr = 1e-2 * (i + 1) / 3
        for grupo in oa.param_groups:
            grupo['lr'] = lr
        oa.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(a(x).reshape(-1, V), y.reshape(-1)).backward()
        torch.nn.utils.clip_grad_norm_(a.parameters(), 0.5)
        oa.step()
        perdida, norma, salto = paso(x, y, lr=lr)
        assert salto.item() == 0 and torch.isfinite(perdida) and torch.isfinite(norma)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    antes = [p.detach().clone() for p in b.parameters()]
    with torch.no_grad():
        b.emb.weight[0, 0] = float('nan')             # el paso siguiente da NaN: no tiene que tocar nada
    antes[0][0, 0] = float('nan')
    _, _, salto = paso(torch.zeros(lote, largo, dtype=torch.long), torch.zeros(lote, largo, dtype=torch.long))
    assert salto.item() == 1
    assert all(torch.equal(p.nan_to_num(), q.nan_to_num()) for p, q in zip(b.parameters(), antes))
    # con AdamW fusionado (el de la GPU; en CPU tambien existe) el salto va por found_inf, sin leer nada en Python:
    # el mismo camino que dentro del CUDA Graph
    c = v89.crear_modelo(V, d=32, n_capas=3, semilla=0)
    oc = torch.optim.AdamW(c.parameters(), lr=1e-2, fused=True)
    paso_c = v89.PasoGrafado(c, oc, lote, largo, precision='fp32', recorte=0.5)
    inicio = [p.detach().clone() for p in c.parameters()]
    x = torch.randint(0, V, (lote, largo), generator=g)
    _, _, salto = paso_c(x, x)
    assert salto.item() == 0 and not torch.equal(c.emb.weight, inicio[0])      # un paso normal si cambia los pesos
    with torch.no_grad():
        c.emb.weight[0, 0] = float('nan')
    antes = [p.detach().clone() for p in c.parameters()]
    _, _, salto = paso_c(torch.zeros(lote, largo, dtype=torch.long), torch.zeros(lote, largo, dtype=torch.long))
    assert salto.item() == 1 and all(torch.equal(p.nan_to_num(), q.nan_to_num()) for p, q in zip(c.parameters(), antes))
    assert all(st['step'].item() == 1 for st in oc.state.values())      # el paso saltado no cuenta
    for mal in (dict(precision='fp16'), dict(grabar=True)):
        try:
            v89.PasoGrafado(a, oa, lote, largo, **mal)
        except ValueError:
            pass
        else:
            raise AssertionError(mal)


def test_escaneo_memoria_lineal():
    por_elemento = []
    for N in (1024, 16384):
        a = torch.rand(4, N, requires_grad=True)
        b = torch.rand(4, N, requires_grad=True)
        por_elemento.append(bytes_guardados(lambda: v89.escaneo(a, b)) / (4 * N))
    assert max(por_elemento) <= 8.5, por_elemento               # O(N): guarda a y S, 2 numeros de 4 bytes


# ----------------------------------------------------------------------------------------------------------------
# Memoria: lo que la capa guarda para la vuelta
# ----------------------------------------------------------------------------------------------------------------

def bytes_guardados(fn):
    """Bytes de los tensores que autograd guarda para la vuelta al calcular fn() (cada almacenamiento una vez)."""
    total, vistos = [0], set()

    def guardar(t):
        k = t.untyped_storage().data_ptr()
        if k not in vistos:
            vistos.add(k)
            total[0] += t.untyped_storage().nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(guardar, lambda t: t):
        fn()
    return total[0]


def _por_pieza(modulo, L, B=2, d=64):
    torch.manual_seed(0)
    capa = modulo.CapaASRN(d)
    E = torch.randn(B, L, d, requires_grad=True)
    h = torch.randn(B, L, d, requires_grad=True)
    x = torch.randint(0, 500, (B, L))
    return bytes_guardados(lambda: capa(E, h, x)) / (B * L)


def test_memoria_lineal():
    por_pieza = {L: _por_pieza(v89, L) for L in (256, 1024, 2048)}
    # O(N): lo guardado por pieza no crece con el largo (baja un poco: el bloque de ceros de cada fila pesa 1/bloques)
    assert por_pieza[256] >= por_pieza[1024] >= por_pieza[2048] > 0.9 * por_pieza[256], por_pieza


# ----------------------------------------------------------------------------------------------------------------
# Lo demas
# ----------------------------------------------------------------------------------------------------------------

def test_top_p():
    g = torch.Generator().manual_seed(0)
    lg = torch.tensor([[5.0, 4.0, 0.0, -1.0, -2.0]])
    vistos = {int(v89._muestrear(lg, 1.0, 0, g, top_p=0.5)) for _ in range(50)}
    assert vistos == {0}
    vistos = {int(v89._muestrear(lg, 1.0, 0, g, top_p=0.9)) for _ in range(200)}
    assert vistos == {0, 1}


def test_continuaciones_de_cita_como_su_definicion():
    """Las piezas que cerrarian un grupo de 4 que ya salio, contra la busqueda directa."""
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 5, (3, 60), generator=g)                       # vocabulario chico: muchas repeticiones
    cita = v89._continuaciones_de_cita(x, 5)
    for b, fila in enumerate(x.tolist()):
        esperado = {fila[s + 3] for s in range(len(fila) - 3) if fila[s:s + 3] == fila[-3:]}
        assert set(torch.nonzero(cita[b]).view(-1).tolist()) == esperado
    assert not v89._continuaciones_de_cita(x[:, :3], 5).any()


def test_conservar_cita_deja_la_masa_de_citar_como_a_temperatura_1():
    g = torch.Generator().manual_seed(0)
    lg = torch.randn(4, 50, generator=g) * 3
    cita = torch.rand(4, 50, generator=g) < 0.1
    p1 = torch.softmax(lg, -1)
    q = torch.softmax(lg / 0.6, -1)
    r = v89._conservar_cita(q, p1, cita)
    assert torch.allclose((r * cita).sum(-1), (p1 * cita).sum(-1), atol=1e-6)            # cuanto se cita: como T=1
    assert torch.allclose(r.sum(-1), torch.ones(4), atol=1e-6)
    adentro = torch.where(cita, r, 0) / (r * cita).sum(-1, keepdim=True)                  # dentro: la forma de q
    assert torch.allclose(adentro, torch.where(cita, q, 0) / (q * cita).sum(-1, keepdim=True), atol=1e-6)
    # sin cita, o si los recortes dejaron en cero una de las partes, no cambia nada
    assert torch.equal(v89._conservar_cita(q, p1, torch.zeros_like(cita)), q)
    recortada = torch.where(cita, 0.0, q)
    assert torch.equal(v89._conservar_cita(recortada, p1, cita), recortada)


def test_conservar_cita_no_cambia_temperatura_1():
    """A temperatura 1 sin recortes, conservar_cita escribe lo mismo que sin el."""
    m = _modelo(chunk=8, n_bits=3)
    _mover_lo_nuevo(m.largos)
    x0 = _texto_con_repeticiones(2, 30, 60)
    a, _ = v89.escribir(m, x0, n_nuevos=30, temperatura=1.0, top_k=0, semilla=5, conservar_cita=True)
    b, _ = v89.escribir(m, x0, n_nuevos=30, temperatura=1.0, top_k=0, semilla=5, conservar_cita=False)
    assert torch.equal(a, b)


def test_tomar_igual_que_bincount():
    """La vuelta de w[largo] (una reduccion por entrada, forma fija para torch.compile) da lo mismo que el conteo."""
    g = torch.Generator().manual_seed(0)
    largo = torch.randint(0, 5, (7, 9), generator=g)
    w = torch.randn(5, requires_grad=True)
    peso = torch.randn(7, 9, generator=g)
    (v89._Tomar.apply(largo, w) * peso).sum().backward()
    assert torch.allclose(w.grad, torch.bincount(largo.view(-1), weights=peso.view(-1), minlength=5).float(), atol=1e-6)
    chico = largo.to(torch.uint8)                     # lo que da _largo_seguido: un indice, no una mascara
    w2 = w.detach().clone().requires_grad_()
    out = v89._Tomar.apply(chico, w2)
    (out * peso).sum().backward()
    assert torch.equal(out, w[largo]) and torch.equal(w2.grad, w.grad)


def test_piezas_rapidas_igual_que_su_definicion():
    """Las versiones que mueven menos bytes dan lo mismo que su definicion directa."""
    g = torch.Generator().manual_seed(0)
    # multiplicaciones juntadas (_phi_y_valores, _contexto) y la puerta en 32 bits: lo mismo salvo el redondeo, en 32
    # bits y con autocast en 16 (ahi, mismos numeros en 16 bits: cada producto se redondea una vez, como antes)
    capa = v89.CapaASRN(48)
    with torch.no_grad():
        for p in capa.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.1)
    E, h = torch.randn(2, 30, 48, generator=g), torch.randn(2, 30, 48, generator=g)
    for bf16 in (False, True):
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
            phi, V = capa._phi_y_valores(E)
            ctx = capa._contexto(h)
            hh = capa.ln_ctx(h)
            u = torch.nn.functional.linear(hh, capa.w_reloj, capa.b_reloj).float()
            ctx0 = (capa.cq(hh), capa.ck(hh), torch.nn.functional.softplus(u[..., 0]),
                    torch.nn.functional.logsigmoid(u[..., 1]) + math.log(2))
            pares = [(phi, capa._phi(E)), (V, capa._valores(E))] + list(zip(ctx, ctx0))
        for a, b in pares:
            assert a.dtype == b.dtype and a.shape == b.shape
            assert (a.float() - b.float()).abs().max() <= (1e-2 if bf16 else 1e-5) * (b.float().abs().max() + 1e-6)
    for K in (4, 300):                                # 300: no cabe en 1 byte, cuenta en 32 bits
        a = torch.randint(0, 2, (6, 1, K), generator=g).float()
        b = torch.randint(0, 2, (1, 5, K), generator=g).float()
        a[0, 0], b[0, 0] = 1.0, 1.0                   # una pareja que coincide entera
        definicion = (a == b).long().cumprod(-1).sum(-1)
        largo = v89._largo_seguido(a, b)
        assert largo.dtype == (torch.uint8 if K < 256 else torch.int32) and torch.equal(largo.long(), definicion)
    capa = v89.CapaASRN(16, n_bits=30)
    phi = torch.randn(3, 7, capa.n_tablas * 30, generator=g)
    bits = phi.reshape(3, 7, capa.n_tablas, 30) > 0
    assert torch.equal(capa._ids(phi).long(), (bits.long() * 2 ** torch.arange(30)).sum(-1))
    # desordenar y sumar las tablas: la vuelta propia es la derivada (B=2, T=3, L=4; 30 filas, 6 de relleno)
    B, T, L, k = 2, 3, 4, 5
    orden = torch.stack([torch.randperm(L, generator=g) for _ in range(B * T)])
    fila = (torch.empty_like(orden).scatter_(1, orden, torch.arange(L).expand(B * T, L))
            + torch.arange(0, B * T * 5, 5).unsqueeze(1)).reshape(-1)
    fuente = torch.full((B * T * 5,), B * L)
    fuente[fila] = (torch.arange(B).repeat_interleave(T * L) * L + torch.arange(L).repeat(B * T))
    x = torch.randn(B * T * 5, k, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: v89._DesordenarSumar.apply(x, fila, fuente, B, T, L), (x,))


def test_checkpoint_mismas_cuentas():
    m = _modelo(n_capas=4)
    _mover_lo_nuevo(m.largos)
    x = _texto_con_repeticiones(2, 64, 60)
    m.train()
    m(x).square().mean().backward()
    g1 = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    m.checkpoint = True
    m(x).square().mean().backward()
    for a, b in zip(g1, (p.grad for p in m.parameters())):
        assert (a - b).abs().max() < 1e-5


def test_precision_mixta():
    m = _modelo()
    _mover_lo_nuevo(m.largos)
    x = _texto_con_repeticiones(2, 64, 60)
    for tipo in (torch.bfloat16, torch.float16):
        with torch.autocast('cpu', dtype=tipo):
            p = m.perdida(x[:, :-1], x[:, 1:])
        p.backward()
        assert torch.isfinite(p)
        assert all(torch.isfinite(q.grad).all() for q in m.parameters() if q.grad is not None)
        m.zero_grad()


def test_guardar_cargar():
    m = _modelo(dropout=0.1)
    x = _texto_con_repeticiones(2, 40, 60)
    with tempfile.TemporaryDirectory() as d:
        v89.guardar(m, os.path.join(d, 'm.pt'))
        m2 = v89.cargar(os.path.join(d, 'm.pt'), 'cpu')
        m.eval()
        assert torch.equal(m(x), m2(x))
        assert m2.config['dropout'] == 0.1


def _tarea_copia(n, L, V, semilla):
    """Cada ventana: una mitad al azar y la misma mitad otra vez. La segunda mitad solo se acierta copiando."""
    g = torch.Generator().manual_seed(semilla)
    mitad = torch.randint(1, V, (n, L // 2), generator=g)
    return torch.cat([mitad, mitad, torch.zeros(n, 1, dtype=torch.long)], 1).reshape(-1)


def test_entrenar_aprende_a_copiar_y_reanuda():
    V, L = 64, 64
    datos = _tarea_copia(3000, L, V, 0)
    val = _tarea_copia(20, L, V, 1)
    with tempfile.TemporaryDirectory() as d:
        arch = os.path.join(d, 'modelo.pt')
        m = v89.crear_modelo(V, d=64, n_capas=2, semilla=0, chunk=16)
        # lr 6e-3 y no 3e-3: con 3e-3 la salida de la meseta inicial (log 63) dependia del orden de las sumas (numero de
        # hilos, instrucciones de MKL, version de PyTorch) y en algunas maquinas no llegaba en 600 pasos (4.19 > 1.5).
        # Con 6e-3 sale antes del paso 250 en todas las semillas, hilos y versiones probadas, con el mismo costo
        v89.entrenar(m, datos, validacion=val, largo=L + 1, lote=16, pasos=600, lr=6e-3, evaluar_cada=300,
                     guardar_en=arch, guardar_cada=300, registrar_cada=100, registro=lambda *_: None)
        # la convolucion solo ve 4 piezas atras: la segunda mitad (32 atras) solo se acierta si ASRN copia.
        # sin copiar cuesta log(63) = 4.14
        x = val.view(20, L + 1)
        with torch.no_grad():
            p = torch.nn.functional.cross_entropy(m(x[:, :-1]).transpose(1, 2), x[:, 1:], reduction='none')
        copia = p[:, L // 2:L - 1].mean().item()
        assert copia < 1.5, copia
        assert os.path.exists(os.path.join(d, 'modelo.mejor.pt'))
        # reanudar desde el paso 3 da exactamente lo mismo que seguir de corrido
        m_a = v89.crear_modelo(V, d=32, n_capas=2, semilla=0, chunk=16)
        v89.entrenar(m_a, datos, largo=L + 1, lote=4, pasos=6, guardar_en=arch, guardar_cada=3,
                     registro=lambda *_: None)
        corrido = {k: v.clone() for k, v in m_a.state_dict().items()}
        m_b = v89.crear_modelo(V, d=32, n_capas=2, semilla=0, chunk=16)

        def ctrl_c(texto):                                  # Ctrl+C justo despues del paso 3: guarda y termina
            if texto.startswith('paso'):
                raise KeyboardInterrupt
        v89.entrenar(m_b, datos, largo=L + 1, lote=4, pasos=6, guardar_en=arch, registrar_cada=3, registro=ctrl_c)
        assert v89._leer(arch)['paso'] == 3
        m_c = v89.cargar(arch, 'cpu')
        v89.entrenar(m_c, datos, largo=L + 1, lote=4, pasos=6, reanudar=arch, registro=lambda *_: None)
        for k, v in m_c.state_dict().items():
            assert torch.allclose(v, corrido[k], atol=1e-6), k


def test_datos_malos_avisan():
    m = v89.crear_modelo(10, d=16, n_capas=2)
    for malo in (torch.tensor([1, 2, 10] * 50), torch.tensor([-1, 2, 3] * 50), torch.rand(150), torch.arange(5)):
        try:
            v89.entrenar(m, malo, largo=8, lote=2, pasos=1, registro=lambda *_: None)
        except ValueError:
            continue
        raise AssertionError(f"no aviso: {malo[:5]}")
    v89.entrenar(m, torch.tensor([1, 2, 9] * 50, dtype=torch.uint8), largo=8, lote=2, pasos=1,
                 registro=lambda *_: None)


def test_leer_piezas():
    with tempfile.TemporaryDirectory() as d:
        t = torch.tensor([0, 5, 65535, 7], dtype=torch.int32)
        arch = os.path.join(d, 'p.bin')
        with open(arch, 'wb') as f:
            f.write(bytes(b for v in t.tolist() for b in int(v).to_bytes(2, 'little')))
        assert v89.leer_piezas(arch).tolist() == t.tolist()
        torch.save(t, os.path.join(d, 'p.pt'))
        assert v89.leer_piezas(os.path.join(d, 'p.pt')).tolist() == t.tolist()
        with open(os.path.join(d, 'p.i32'), 'wb') as f:
            f.write(b''.join(int(v).to_bytes(4, 'little') for v in t.tolist()))
        assert v89.leer_piezas(os.path.join(d, 'p.i32')).tolist() == t.tolist()


def test_torch_compile_un_solo_grafo_y_mismas_cuentas():
    """torch.compile ve el modelo entero como un solo grafo (sin cortes) y da los mismos logits y gradientes.
    Con PyTorch 2.6 a 2.13, Inductor falla al compilar este modelo en CPU (C++ invalido, o Python se cae por una
    corrupcion de memoria que no se puede atrapar): con esas versiones la prueba se salta en vez de tumbar las demas."""
    if torch.__version__ < v89._TORCH_COMPILE_MINIMO:
        raise Saltada(f'torch.compile en CPU falla con PyTorch {torch.__version__}; comprobado desde '
                      f'{v89._TORCH_COMPILE_MINIMO}')
    m = _modelo(chunk=8, n_bits=3)
    _mover_lo_nuevo(m.largos)
    x = _texto_con_repeticiones(2, 64, 60)
    torch._dynamo.reset()
    ex = torch._dynamo.explain(m)(x)
    assert ex.graph_count == 1 and ex.graph_break_count == 0, [r.reason for r in ex.break_reasons]
    torch._dynamo.reset()
    m(x).square().mean().backward()
    g_e = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    lg_e = m(x).detach()
    m.zero_grad(set_to_none=True)
    mc = torch.compile(m)
    lg_c = mc(x)
    lg_c.square().mean().backward()
    assert (lg_c.detach() - lg_e).abs().max() < TOL
    for n, p in m.named_parameters():
        if n in g_e:
            assert torch.allclose(p.grad, g_e[n], atol=1e-4, rtol=1e-3), n



# ================================================================================================================
# ASRN 1.0 (v89)
# ================================================================================================================

def test_compilar_partes_mismas_cuentas():
    """compilar_partes (bloques locales y partes chicas de cada capa ASRN) da las mismas salidas y gradientes que sin
    compilar, salvo el redondeo, y llamarla dos veces no compila de nuevo."""
    if torch.__version__ < v89._TORCH_COMPILE_MINIMO:
        raise Saltada(f'torch.compile necesita PyTorch {v89._TORCH_COMPILE_MINIMO} (hay {torch.__version__})')
    ref = v89.crear_modelo(60, d=32, n_capas=3, semilla=0)
    _mover_lo_nuevo(ref.largos)
    comp = v89.crear_modelo(60, d=32, n_capas=3)
    comp.load_state_dict(ref.state_dict())
    v89.compilar_partes(comp)
    forward = comp.bloques[0].forward
    v89.compilar_partes(comp)
    assert comp.bloques[0].forward is forward
    x = _texto_con_repeticiones(2, 96, 60)
    a, b = ref(x), comp(x)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
    a.square().mean().backward()
    b.square().mean().backward()
    for (n, p), q in zip(ref.named_parameters(), comp.parameters()):
        if p.grad is not None:
            assert torch.allclose(p.grad, q.grad, atol=1e-5, rtol=1e-3), n


def test_config_valida_y_explica_errores():
    malas = [dict(V=1), dict(V=2 ** 24 + 1), dict(V=50, n_capas=1), dict(V=50, tablas_contexto=8),
             dict(V=50, tablas_contexto=0), dict(V=50, n_bits=31), dict(V=50, dropout=1.0), dict(V=50, d=0),
             dict(V=50, checkpoint='a veces'), dict(V=50, nodos_racha=1), dict(V=50, ff=0), dict(V=50.5)]
    for kw in malas:
        try:
            v89.ConfigASRN(**kw)
        except ValueError as e:
            assert 'ConfigASRN' in str(e) or 'checkpoint' in str(e)
        else:
            raise AssertionError(f"debio fallar: {kw}")
    c = v89.ConfigASRN(V=50, d=16, n_capas=3, checkpoint=True)
    assert c.checkpoint == 'asrn'
    assert v89.ConfigASRN.desde_dict(c.a_dict()) == c
    assert c.con(d=24).d == 24 and c.d == 16
    try:
        v89.ConfigASRN.desde_dict({**c.a_dict(), 'capas': 3})
    except ValueError:
        pass
    else:
        raise AssertionError("campo desconocido")


def test_contar_parametros_exacto_y_sin_memoria():
    for cfg in [v89.TALLAS['nano-3k'], v89.ConfigASRN(V=70, d=24, n_capas=5, largo_cada=2, ff=40, k=3),
                v89.ConfigASRN(V=90, d=16, n_capas=3, n_tablas=4, tablas_contexto=2, n_bits=8)]:
        real = v89.crear_desde_config(cfg, semilla=0).num_parametros()
        assert cfg.contar_parametros() == real
        dg = cfg.contar_parametros(desglose=True)
        assert dg['total'] == real == dg['vocabulario'] + dg['bloques_locales'] + dg['capas_asrn'] + dg['otros']
    esperado = {'nano-3k': (2e3, 4e3), '12m': (10e6, 14e6), '30m': (25e6, 35e6), '125m': (110e6, 140e6),
                '350m': (300e6, 380e6), '1b': (0.9e9, 1.2e9), '7b': (6e9, 8e9), '27b': (25e9, 29e9)}
    assert set(esperado) == set(v89.TALLAS)
    for n, (lo, hi) in esperado.items():
        assert lo <= v89.TALLAS[n].contar_parametros() <= hi, n     # 27b: sin reservar un solo peso


def test_config_igual_que_crear_modelo_y_desde_modelo():
    cfg = v89.ConfigASRN(V=60, d=24, n_capas=9, dropout=0.1)
    a = v89.crear_desde_config(cfg, semilla=3)
    b = v89.crear_modelo(60, 24, 9, semilla=3, dropout=0.1)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    vuelta = v89.ConfigASRN.desde_modelo(a)
    assert vuelta.con(init_profundo=None) == cfg.con(init_profundo=None) and vuelta.init_profundo is True
    nano = v89.crear_desde_config('nano-3k', semilla=0)
    lg = nano(torch.randint(0, 48, (2, 64)))
    assert lg.shape == (2, 64, 48) and torch.isfinite(lg).all()


def test_materializar_igual_que_crear_modelo_bit_a_bit():
    for cfg in [v89.TALLAS['nano-3k'], v89.ConfigASRN(V=100, d=32, n_capas=9),
                v89.ConfigASRN(V=100, d=24, n_capas=4, largo_cada=2, dropout=0.1)]:
        for semilla in (0, 7):
            ref = v89.crear_desde_config(cfg, semilla=semilla)
            antes = torch.random.get_rng_state()
            m = v89.materializar(v89.crear_desde_config(cfg, dispositivo='meta'), 'cpu', semilla)
            assert torch.equal(antes, torch.random.get_rng_state())          # no toca el azar global
            r, q = ref.state_dict(), m.state_dict()
            assert r.keys() == q.keys() and all(torch.equal(r[k], q[k]) for k in r)
            assert m.head.weight is m.emb.weight
            x = torch.randint(0, cfg.V, (2, 40))
            assert torch.equal(ref.eval()(x), m.eval()(x))


def test_checkpoint_por_niveles_mismas_cuentas_menos_memoria():
    m = v89.crear_modelo(80, 32, 4, semilla=0)
    _abrir(m.largos)
    x = torch.randint(0, 80, (2, 128))
    y = torch.randint(0, 80, (2, 128))
    m.train()
    res = {}
    for modo in ('no', 'asrn', 'todo'):
        m.checkpoint = modo
        m.zero_grad(set_to_none=True)
        guardado = [0]

        def empacar(t):
            guardado[0] += t.numel() * t.element_size()
            return t
        with torch.autograd.graph.saved_tensors_hooks(empacar, lambda t: t):
            p = m.perdida(x, y)
        p.backward()
        res[modo] = (p.item(), {n: q.grad.clone() for n, q in m.named_parameters()}, guardado[0])
    for modo in ('asrn', 'todo'):
        assert abs(res[modo][0] - res['no'][0]) < 1e-6
        for n, gr in res['no'][1].items():
            assert torch.allclose(res[modo][1][n], gr, atol=1e-6, rtol=1e-5), (modo, n)
    assert res['todo'][2] < res['asrn'][2] < res['no'][2], {k: v[2] for k, v in res.items()}
    m.checkpoint = True
    assert m.checkpoint == 'asrn'


def test_estado_de_adamw_por_nombre_ida_y_vuelta():
    m = v89.crear_modelo(40, 16, 3, semilla=0)
    opt = v89.crear_optimizador(m)
    m.perdida(torch.randint(0, 40, (2, 32)), torch.randint(0, 40, (2, 32))).backward()
    opt.step()
    nombres = v89._nombres_en_optimizador(m)
    ida = v89._opt_a_nombres(opt.state_dict(), nombres)
    assert set(ida['state']) <= set(nombres) and 'emb.weight' in ida['state']
    vuelta = v89._opt_a_numeros(ida, nombres)
    original = opt.state_dict()
    assert vuelta['param_groups'] == original['param_groups']
    for i, st in original['state'].items():
        assert all(torch.equal(st[k], vuelta['state'][i][k]) for k in st)


# --- FSDP con 2 procesos en CPU (gloo) ---------------------------------------------------------------------------

def _puerto_libre():
    import socket
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _en_procesos(funcion, n=2, *args):
    """Corre funcion(rango, n, *args) en n procesos con un grupo gloo (CPU). Falla si algun proceso falla."""
    import torch.multiprocessing as mp
    puerto = _puerto_libre()
    mp.spawn(_arrancar_proceso, args=(n, puerto, funcion, args), nprocs=n, join=True)


def _arrancar_proceso(rango, n, puerto, funcion, args):
    import torch.distributed as dist
    os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = '127.0.0.1', str(puerto)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rango, world_size=n)
    try:
        funcion(rango, n, *args)
    finally:
        dist.destroy_process_group()


_CFG_FSDP = dict(V=50, d=32, n_capas=4)


def _fsdp_ida_vuelta_y_paso(rango, n, carpeta):
    """Repartido = sin repartir: misma ida, mismo gradiente (el promedio de los lotes de cada proceso) y el mismo
    paso de AdamW con recorte."""
    import torch.distributed as dist
    cfg = v89.ConfigASRN(**_CFG_FSDP)
    ref = v89.crear_desde_config(cfg, semilla=5)
    _abrir(ref.largos)
    m = v89.repartir_fsdp(v89.crear_desde_config(cfg, dispositivo='meta'))
    v89.materializar(m, 'cpu', 5)
    v89._copiar_a(m, ref)                       # con las puertas abiertas, como ref
    assert v89.esta_repartido(m) and m.emb.weight.to_local().shape[0] < cfg.V
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, cfg.V, (2 * n, 64), generator=g)
    y = torch.randint(0, cfg.V, (2 * n, 64), generator=g)
    m.eval(), ref.eval()
    assert torch.allclose(m(x), ref(x), atol=1e-5)
    m.train(), ref.train()
    opt_m, opt_r = v89.crear_optimizador(m, lr=1e-2), v89.crear_optimizador(ref, lr=1e-2)
    m.perdida(x[2 * rango:2 * rango + 2], y[2 * rango:2 * rango + 2]).backward()
    ref.perdida(x, y).backward()
    nm = torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5).full_tensor()
    nr = torch.nn.utils.clip_grad_norm_(ref.parameters(), 0.5)
    assert abs(float(nm) - float(nr)) < 1e-4 * max(1.0, float(nr)), (float(nm), float(nr))
    opt_m.step(), opt_r.step()
    pesos = v89.pesos_completos(m)
    if rango == 0:
        for k, v in ref.state_dict().items():
            assert torch.allclose(pesos[k], v, atol=2e-5), k
    else:
        assert pesos == {}
    if rango == 0:
        with open(os.path.join(carpeta, 'ok_paso'), 'w') as f:
            f.write('ok')
    dist.barrier()


def _fsdp_entrenar_guardar_reanudar(rango, n, carpeta):
    """entrenar() repartido: aprende, guarda un archivo que cargar() abre en un proceso y que da lo mismo que el
    modelo repartido, y reanuda (de un archivo repartido y de uno de un solo proceso)."""
    import torch.distributed as dist
    cfg = v89.ConfigASRN(**_CFG_FSDP, checkpoint='todo')
    datos = _texto_que_se_repite(40_000)
    archivo = os.path.join(carpeta, 'rep.pt')
    m = v89.repartir_fsdp(v89.crear_desde_config(cfg, dispositivo='meta'))
    v89.materializar(m, 'cpu', 0)
    h = v89.entrenar(m, datos, validacion=datos[:4000], largo=64, lote=4, pasos=60, lr=3e-3, evaluar_cada=30,
                     lotes_eval=4, guardar_en=archivo, guardar_cada=30, registrar_cada=30, registro=lambda *a: None)
    perdidas = [f['perdida'] for f in h if 'perdida' in f]
    assert perdidas[-1] < perdidas[0], perdidas
    x = torch.randint(0, cfg.V, (2, 64), generator=torch.Generator().manual_seed(1))
    lg = m.eval()(x)
    dist.barrier()
    unico = v89.cargar(archivo, dispositivo='cpu')
    assert torch.allclose(unico(x), lg, atol=1e-5)
    r = torch.load(archivo, weights_only=True)
    assert r['paso'] == 60 and 'optimizador_nombres' in r and len(r['azar_procesos']) == n
    # reanudar repartido desde el archivo repartido
    m2 = v89.repartir_fsdp(v89.crear_desde_config(cfg, dispositivo='meta'))
    v89.materializar(m2, 'cpu', 123)
    h2 = v89.entrenar(m2, datos, largo=64, lote=4, pasos=70, lr=3e-3, reanudar=archivo,
                      guardar_en=os.path.join(carpeta, 'rep2.pt'), registro=lambda *a: None, registrar_cada=10)
    assert h2[-1]['paso'] == 70
    # reanudar repartido desde un archivo de un solo proceso (guardado por el proceso 0 antes)
    m3 = v89.repartir_fsdp(v89.crear_desde_config(cfg, dispositivo='meta'))
    v89.materializar(m3, 'cpu', 0)
    h3 = v89.entrenar(m3, datos, largo=64, lote=4, pasos=25, lr=3e-3, reanudar=os.path.join(carpeta, 'unico.pt'),
                      registro=lambda *a: None, registrar_cada=5)
    assert h3[-1]['paso'] == 25
    if rango == 0:
        with open(os.path.join(carpeta, 'ok_entrenar'), 'w') as f:
            f.write('ok')
    dist.barrier()


def _texto_que_se_repite(n, V=50, semilla=0):
    """Piezas al azar con trozos que se repiten: se aprende copiando."""
    g = torch.Generator().manual_seed(semilla)
    trozos = torch.randint(0, V, (20, 16), generator=g)
    return trozos[torch.randint(0, 20, (n // 16,), generator=g)].reshape(-1)


def test_fsdp_igual_que_sin_repartir():
    with tempfile.TemporaryDirectory() as tmp:
        _en_procesos(_fsdp_ida_vuelta_y_paso, 2, tmp)
        assert os.path.exists(os.path.join(tmp, 'ok_paso'))


def test_fsdp_entrenar_guardar_y_reanudar():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = v89.ConfigASRN(**_CFG_FSDP)
        unico = v89.crear_desde_config(cfg, semilla=0)          # archivo de un solo proceso, para reanudar repartido
        v89.entrenar(unico, _texto_que_se_repite(40_000), largo=64, lote=4, pasos=20, lr=3e-3,
                     guardar_en=os.path.join(tmp, 'unico.pt'), registro=lambda *a: None)
        _en_procesos(_fsdp_entrenar_guardar_reanudar, 2, tmp)
        assert os.path.exists(os.path.join(tmp, 'ok_entrenar'))
        # y al reves: reanudar en un solo proceso desde el archivo repartido
        m = v89.crear_desde_config(cfg, semilla=9)
        h = v89.entrenar(m, _texto_que_se_repite(40_000), largo=64, lote=4, pasos=65, lr=3e-3,
                         reanudar=os.path.join(tmp, 'rep.pt'), registro=lambda *a: None, registrar_cada=5)
        assert h[-1]['paso'] == 65


def test_varios_procesos_sin_repartir_avisa():
    with tempfile.TemporaryDirectory() as tmp:
        _en_procesos(_sin_repartir_falla, 2, tmp)


def _sin_repartir_falla(rango, n, carpeta):
    m = v89.crear_modelo(50, 16, 3, semilla=0)
    try:
        v89.entrenar(m, _texto_que_se_repite(5000), largo=32, lote=2, pasos=2, registro=lambda *a: None)
    except ValueError as e:
        assert 'repartir_fsdp' in str(e)
    else:
        raise AssertionError("debio avisar")


# ================================================================================================================
# v89: el tamano que quieras
# ================================================================================================================

def test_formula_de_parametros_exacta():
    import random
    r = random.Random(0)
    for _ in range(200):
        kw = dict(V=r.randint(2, 5000), d=r.randint(1, 200), n_capas=r.randint(2, 20), largo_cada=r.randint(1, 4),
                  ff=r.choice([None, r.randint(1, 300)]), k=r.randint(1, 6), n_tablas=r.randint(2, 10),
                  n_bits=r.randint(1, 16), afin_oculto=r.randint(1, 300), confirmar=r.randint(1, 30),
                  sufijo=r.randint(1, 6), vistos=r.randint(1, 6), nodos_racha=r.randint(2, 20))
        kw['tablas_contexto'] = r.randint(1, kw['n_tablas'] - 1)
        c = v89.ConfigASRN(**kw)
        assert v89._contar_formula(c) == c.contar_parametros(), kw


def test_leer_cantidad():
    assert v89.leer_cantidad('3k') == 3000 and v89.leer_cantidad('50M') == 50_000_000
    assert v89.leer_cantidad('1.5b') == 1_500_000_000 and v89.leer_cantidad(7e9) == 7_000_000_000
    assert v89.leer_cantidad('125_000') == 125_000
    for malo in ('abc', '-5m', '0', ''):
        try:
            v89.leer_cantidad(malo)
        except ValueError:
            pass
        else:
            raise AssertionError(malo)


def test_para_parametros_cualquier_tamano():
    for objetivo, V in [('3k', 48), ('20k', 64), ('1m', 8000), ('12m', 8000), ('50m', 32768), ('300m', 32768),
                        ('1.5b', 32768), ('7b', 50000), ('27b', 32768), ('70b', 128000)]:
        c = v89.ConfigASRN.para_parametros(objetivo, V)
        total = c.contar_parametros()
        assert abs(total / v89.leer_cantidad(objetivo) - 1) <= 0.03, (objetivo, total)
        assert c.V == V
    chico = v89.ConfigASRN.para_parametros('3k', 48)
    assert chico.n_tablas == 2                                     # la capa normal no cabe en 3 mil: reducida
    assert v89.ConfigASRN.para_parametros('50m', 32768).n_tablas == 8    # la receta normal cuando cabe
    m = v89.crear_desde_config(chico, semilla=0)
    assert torch.isfinite(m(torch.randint(0, 48, (2, 32)))).all()


def test_para_parametros_respeta_lo_fijo():
    c = v89.ConfigASRN.para_parametros('50m', 32768, d=512)
    assert c.d == 512 and abs(c.contar_parametros() / 50e6 - 1) < 0.1
    c = v89.ConfigASRN.para_parametros('50m', 32768, n_capas=12, largo_cada=2, dropout=0.1)
    assert c.n_capas == 12 and c.largo_cada == 2 and c.dropout == 0.1
    assert abs(c.contar_parametros() / 50e6 - 1) <= 0.03


def test_linea_de_comandos_parametros():
    import io
    import contextlib
    with tempfile.TemporaryDirectory() as tmp:
        datos = os.path.join(tmp, 'p.pt')
        torch.save(torch.randint(0, 60, (6000,), dtype=torch.int32), datos)
        salida = io.StringIO()
        with contextlib.redirect_stdout(salida):
            v89._principal(['--datos', datos, '--V', '60', '--parametros', '40k', '--largo', '32', '--lote', '2',
                            '--pasos', '3', '--guardar', os.path.join(tmp, 'm.pt')])
        assert 'configuracion para 40k' in salida.getvalue()
        m = v89.cargar(os.path.join(tmp, 'm.pt'), dispositivo='cpu')
        assert abs(m.num_parametros() / 40e3 - 1) <= 0.03


# ================================================================================================================
# Revision antes de publicar: casos limite, causalidad y mensajes de error claros
# ================================================================================================================

def _falla_con(error, fn, *textos):
    """fn() debe lanzar `error` y su mensaje debe contener cada uno de `textos`."""
    try:
        fn()
    except error as e:
        for t in textos:
            assert t in str(e), (t, str(e))
        return str(e)
    raise AssertionError(f"debio fallar con {error.__name__}")


def test_causal_el_futuro_no_cambia_el_pasado():
    """Ninguna posicion ve el futuro: cambiar las piezas despues de t no cambia la salida hasta t (con todo lo nuevo
    de v89 fuera de su valor neutro, casilleros llenos y largos que no son multiplo de chunk)."""
    for V, L, kw in [(7, 61, dict(chunk=8, n_bits=3)), (3, 50, dict(chunk=4, n_bits=2)), (60, 100, {})]:
        m = v89.crear_modelo(V, d=16, n_capas=4, semilla=0, **kw)
        _abrir(m.largos)
        _mover_lo_nuevo(m.largos)
        m.eval()
        g = torch.Generator().manual_seed(0)
        x = torch.randint(0, V, (2, L), generator=g)
        x[:, L // 2:] = x[:, :L - L // 2]
        with torch.no_grad():
            base = m(x)
            for t in (0, 1, L // 3, L // 2, L - 2):
                y = x.clone()
                y[:, t + 1:] = torch.randint(0, V, (2, L - t - 1), generator=g)
                assert (m(y)[:, :t + 1] - base[:, :t + 1]).abs().max() < 1e-5, (V, L, t)


def test_casos_limite_largo_1_lote_1_vocabulario_2():
    for V, B, L, kw in [(2, 1, 1, {}), (2, 3, 1, {}), (2, 1, 33, {}), (5, 2, 7, dict(n_capas=2, largo_cada=5)),
                        (10, 2, 40, dict(chunk=1, k=1, sufijo=1, vistos=1))]:
        m = v89.crear_modelo(V, d=16, semilla=0, **kw)
        _abrir(m.largos)
        _mover_lo_nuevo(m.largos)
        lg = m(torch.randint(0, V, (B, L)))
        lg.square().mean().backward()
        assert lg.shape == (B, L, V) and torch.isfinite(lg).all()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    m = v89.crear_modelo(2, d=8, n_capas=2, semilla=0)
    datos = torch.randint(0, 2, (300,))
    h = v89.entrenar(m, datos, validacion=datos[:50], largo=1, lote=1, pasos=2, evaluar_cada=1, registrar_cada=1,
                     registro=lambda *_: None)
    assert h[-1]['paso'] == 2


def test_escribir_casos_limite():
    m = v89.crear_modelo(2, d=16, n_capas=2, semilla=0)
    # no_repetir_ngramas que termina prohibiendo todas las piezas: ese paso no se aplica (antes: error de multinomial)
    s, _ = v89.escribir(m, torch.tensor([[1]]), n_nuevos=12, no_repetir_ngramas=1)
    assert s.shape == (1, 13)
    # top_p=0: solo la mas probable (antes: todo prohibido y error)
    lg = torch.tensor([[5.0, 4.0, 0.0]])
    assert {int(v89._muestrear(lg, 1.0, 0, torch.Generator().manual_seed(0), top_p=0.0)) for _ in range(20)} == {0}
    s, _ = v89.escribir(m, torch.tensor([[1, 0]]), n_nuevos=4, top_p=0.0)
    assert s.shape == (1, 6)
    # piezas fuera del vocabulario: mensaje claro (antes: IndexError 'index out of range in self')
    _falla_con(ValueError, lambda: v89.escribir(m, torch.tensor([[5]]), n_nuevos=2), 'fuera de [0, 2)')


def test_modelo_en_16_bits_avisa():
    """Convertir el modelo a bf16/fp16 (en vez de usar autocast) fallaba con un error de tipos poco claro."""
    m = v89.crear_modelo(30, d=16, n_capas=2, semilla=0)
    for tipo in (torch.bfloat16, torch.float16):
        m16 = v89.crear_modelo(30, d=16, n_capas=2, semilla=0).to(tipo)
        _falla_con(ValueError, lambda: m16(torch.randint(0, 30, (1, 20))), 'autocast')
        _falla_con(ValueError, lambda: v89.escribir(m16, torch.tensor([[1, 2]]), n_nuevos=2), 'autocast')
    _falla_con(ValueError, lambda: v89.repartir_fsdp(m, precision_pesos=torch.bfloat16), 'precision_pesos')


def test_entrenar_valores_malos_avisan():
    m = v89.crear_modelo(20, d=16, n_capas=2, semilla=0)
    datos = torch.randint(0, 20, (2000,))
    for malo in (dict(guardar_cada=0), dict(evaluar_cada=0), dict(registrar_cada=0), dict(lote=0), dict(largo=0),
                 dict(acumular=0), dict(pasos=-1), dict(lote=2.5)):
        kw = {**dict(largo=16, lote=2, pasos=1, registro=lambda *_: None), **malo}
        _falla_con(ValueError, lambda: v89.entrenar(m, datos, validacion=datos[:200], **kw), list(malo)[0])
    _falla_con(ValueError, lambda: v89.evaluar(m, datos, largo=16, lote=0), 'lote')


def test_guardar_atomico_no_deja_temporales():
    with tempfile.TemporaryDirectory() as d:
        arch = os.path.join(d, 'm.pt')
        v89._guardar_atomico({'x': torch.ones(3)}, arch)
        try:
            v89._guardar_atomico({'x': lambda: 0}, arch)        # no se puede guardar: falla a la mitad
        except Exception:
            pass
        assert not os.path.exists(arch + '.tmp')
        assert torch.equal(v89._leer(arch)['x'], torch.ones(3))    # el anterior sigue sano
        _falla_con(ValueError, lambda: v89.cargar(arch, 'cpu'), 'no es un modelo')
        _falla_con(FileNotFoundError, lambda: v89.cargar(os.path.join(d, 'no_existe.pt'), 'cpu'), 'no existe')
        _falla_con(FileNotFoundError, lambda: v89.leer_piezas(os.path.join(d, 'no_existe.bin')), 'no existe')
        _falla_con(ValueError, lambda: v89.leer_piezas(os.path.join(d, 'p.npy')), "'.npy'", '.bin')


def test_reanudar_errores_claros_y_avisos():
    datos = torch.randint(0, 20, (3000,))
    kw = dict(largo=16, lote=2, registro=lambda *_: None)
    with tempfile.TemporaryDirectory() as d:
        arch = os.path.join(d, 'm.pt')
        m = v89.crear_modelo(20, d=16, n_capas=2, semilla=0)
        v89.entrenar(m, datos, validacion=datos[:500], pasos=4, evaluar_cada=2, guardar_en=arch, precision='bf16',
                     **kw)
        # el '.mejor.pt' (solo pesos) no sirve para reanudar: decirlo (antes: KeyError 'optimizador')
        _falla_con(ValueError, lambda: v89.entrenar(v89.cargar(arch, 'cpu'), datos, pasos=6,
                                                    reanudar=os.path.join(d, 'm.mejor.pt'), **kw), 'solo los pesos')
        # pasos ya alcanzados y ajustes distintos: avisar en vez de no hacer nada en silencio
        msgs = []
        v89.entrenar(v89.cargar(arch, 'cpu'), datos, pasos=3, reanudar=arch, **{**kw, 'registro': msgs.append})
        assert any('no queda nada que entrenar' in t for t in msgs), msgs
        assert any('pasos 4 -> 3' in t for t in msgs), msgs
        # un archivo de una corrida en bf16 se puede seguir en fp16 (antes: el escalador de fp16 fallaba)
        h = v89.entrenar(v89.cargar(arch, 'cpu'), datos, pasos=6, reanudar=arch, precision='fp16',
                         registrar_cada=1, **kw)
        assert h[-1]['paso'] == 6


def test_para_parametros_imposible_explica():
    import warnings
    _falla_con(ValueError, lambda: v89.ConfigASRN.para_parametros('3k', 32000), 'vocabulario', 'V menor')
    _falla_con(ValueError, lambda: v89.ConfigASRN.para_parametros('1b', 32000, d=64, n_capas=2), 'Suelta')
    _falla_con(ValueError, lambda: v89.ConfigASRN.para_parametros('50m', 32000, capas=12), "['capas']", 'n_capas')
    _falla_con(ValueError, lambda: v89.ConfigASRN.para_parametros('50m', 32000, d=0), 'd y k')
    _falla_con(ValueError, lambda: v89.ConfigASRN.para_parametros('inf', 32000), 'positivo')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        c = v89.ConfigASRN.para_parametros('50m', 32768, d=1100)            # +16%: se puede, pero se avisa
    assert c.d == 1100 and any('+16%' in str(x.message) for x in w), [str(x.message) for x in w]


def _linea_de_comandos(args, entorno=None):
    """(codigo de salida, texto en pantalla) de correr la linea de comandos con `args` y variables de entorno."""
    import contextlib
    import io
    salida, viejo = io.StringIO(), dict(os.environ)
    os.environ.update(entorno or {})
    try:
        with contextlib.redirect_stdout(salida), contextlib.redirect_stderr(salida):
            v89._principal(args)
        codigo = 0
    except SystemExit as e:
        codigo = e.code
    finally:
        os.environ.clear()
        os.environ.update(viejo)
    return codigo, salida.getvalue()


def test_linea_de_comandos_errores_claros():
    with tempfile.TemporaryDirectory() as tmp:
        datos = os.path.join(tmp, 'p.pt')
        torch.save(torch.randint(0, 60, (6000,), dtype=torch.int32), datos)
        base = ['--datos', datos, '--V', '60', '--ancho', '16', '--capas', '2', '--largo', '32', '--lote', '2',
                '--pasos', '2', '--guardar', os.path.join(tmp, 'm.pt')]
        for extra, entorno, texto in [
                (['--guardar_cada', '0'], None, '--guardar_cada debe ser 1 o mas'),
                (['--lote', '0'], None, '--lote debe ser 1 o mas'),
                ([], {'WORLD_SIZE': '2', 'RANK': '0'}, 'falta --fsdp'),          # torchrun sin --fsdp
                (['--fsdp'], None, 'torchrun'),                                  # --fsdp sin torchrun
                (['--reanudar', os.path.join(tmp, 'no_existe.pt')], None, 'no existe'),
                (['--parametros', '3k', '--V', '32000'], None, 'vocabulario')]:
            codigo, texto_salida = _linea_de_comandos(base + extra, entorno)
            assert codigo == 2 and texto in texto_salida, (extra, codigo, texto_salida[-400:])
        npy = os.path.join(tmp, 'p.npy')
        torch.save(torch.zeros(3), npy)
        codigo, texto_salida = _linea_de_comandos(['--datos', npy, '--V', '60'])
        assert codigo == 2 and "'.npy'" in texto_salida, texto_salida[-400:]


def test_linea_de_comandos_reanudar_avisa_lo_ignorado():
    with tempfile.TemporaryDirectory() as tmp:
        datos = os.path.join(tmp, 'p.pt')
        torch.save(torch.randint(0, 60, (6000,), dtype=torch.int32), datos)
        arch = os.path.join(tmp, 'm.pt')
        comun = ['--datos', datos, '--largo', '32', '--lote', '2', '--guardar', arch]
        codigo, _ = _linea_de_comandos(comun + ['--V', '60', '--ancho', '16', '--capas', '2', '--pasos', '3'])
        assert codigo == 0
        codigo, texto = _linea_de_comandos(comun + ['--reanudar', arch, '--pasos', '5', '--V', '99', '--capas', '9'])
        assert codigo == 0 and 'se ignoran --V, --capas' in texto and 'reanudando desde el paso 3' in texto, texto
        assert v89._leer(arch)['paso'] == 5 and v89._leer(arch)['config']['V'] == 60


if __name__ == '__main__':
    import sys
    import time
    fallas, saltadas, buenas = 0, 0, 0
    solo = sys.argv[1:]
    for nombre, f in list(globals().items()):
        if nombre.startswith('test_') and callable(f) and (not solo or any(s in nombre for s in solo)):
            t = time.time()
            try:
                f()
                buenas += 1
                print(f"ok     {nombre} ({time.time() - t:.1f}s)")
            except Saltada as e:
                saltadas += 1
                print(f"salta  {nombre}: {e}")
            except Exception as e:
                fallas += 1
                print(f"FALLA  {nombre}: {type(e).__name__}: {e}")
    print(f"\n{buenas} ok, {saltadas} saltadas, {fallas} fallas")
    sys.exit(1 if fallas else 0)
