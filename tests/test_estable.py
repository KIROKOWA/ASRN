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
"""Pruebas de asrn/estable.py (la capa ASRN con protecciones) y de la API del paquete. Correr desde la carpeta del
repositorio con:
    python tests/test_estable.py [parte_del_nombre ...]

  - sin protecciones, la capa estable es v89 bit a bit; con las del grupo A, igual salvo el redondeo
  - lo que arregla cada proteccion, comprobado contra v89: la fuga causal con un nucleo de tiempo enorme, el reloj
    corrido (lejos del inicio del texto), beta enorme, la proyeccion del reloj en 16 bits
  - el kernel Triton en modo ESTABLE (con el interprete en CPU) da lo mismo que PyTorch, y tampoco tiene la fuga
  - escritura paso a paso, convertir/guardar/cargar y el recorte dinamico del gradiente"""
import copy
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_AQUI)
if _RAIZ not in sys.path:
    sys.path.insert(0, _RAIZ)

import asrn.nucleo as v89
import asrn.estable as est


class Saltada(Exception):
    """La prueba no se puede correr aqui (falta algo opcional)."""


def _mover(capas, semilla=1, gamma=None):
    """Saca los parametros nuevos (y la confirmacion, el sufijo, los vistos) de su valor neutro."""
    g = torch.Generator().manual_seed(semilla)
    with torch.no_grad():
        for c in capas:
            for n, p in c.named_parameters():
                if n in v89._NUEVOS_V87 or n in ('cq.weight', 'w_sufijo', 'w_vistos', 'g_conf.weight'):
                    p.add_(torch.randn(p.shape, generator=g) * 0.5)
            if gamma is not None:
                c.gamma.copy_(torch.linspace(gamma[0], gamma[1], c.n_tablas))


def _modelo(V=60, d=32, n_capas=4, semilla=0, **kw):
    m = v89.crear_modelo(V, d=d, n_capas=n_capas, semilla=semilla, **kw)
    _mover(m.largos)
    return m


def _texto(B, L, V, semilla=0):
    g = torch.Generator().manual_seed(semilla)
    x = torch.randint(0, V, (B, L), generator=g)
    x[:, L // 2:L // 2 + L // 4] = x[:, :L // 4]
    return x


def _ida_y_vuelta(m, x, y):
    m.zero_grad()
    lg = m(x)
    F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1)).backward()
    return lg.detach(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def _capa_y_entradas(d=64, L=256, semilla=0, **kw):
    torch.manual_seed(semilla)
    x = _texto(2, L, 20, semilla)
    E, h = torch.randn(20, d)[x], torch.randn(2, L, d)
    return x, E, h


# ----------------------------------------------------------------------------------------------------------------
# Lo mismo que v89
# ----------------------------------------------------------------------------------------------------------------

def test_sin_protecciones_igual_que_v89_bit_a_bit():
    m = _modelo()
    x, y = _texto(2, 150, 60), _texto(2, 150, 60, semilla=1)
    ref = _ida_y_vuelta(m, x, y)
    lg, gr = _ida_y_vuelta(est.convertir(copy.deepcopy(m), est.Protecciones.ninguna()), x, y)
    assert torch.equal(lg, ref[0])
    assert all(torch.equal(gr[k], ref[1][k]) for k in ref[1])


def test_grupo_A_igual_que_v89_salvo_redondeo():
    """Con las protecciones por omision (grupo A) y valores normales, lo mismo que v89 salvo el redondeo: el reloj en
    64 bits redondea distinto, nada mas. Tambien con la capa sola (sin numeros de pieza)."""
    for kw in ({}, dict(chunk=8, n_bits=3), dict(largo_cada=1)):
        m = _modelo(**kw)
        x, y = _texto(2, 130, 60), _texto(2, 130, 60, semilla=1)
        ref = _ida_y_vuelta(m, x, y)
        lg, gr = _ida_y_vuelta(est.convertir(copy.deepcopy(m)), x, y)
        assert (lg - ref[0]).abs().max() < 1e-5, kw
        for k in ref[1]:
            assert (gr[k] - ref[1][k]).abs().max() <= 1e-4 * (ref[1][k].abs().max() + 1e-6), (kw, k)
    capa, capa_e = v89.CapaASRN(32), est.CapaASRNEstable(32)
    capa_e.load_state_dict(capa.state_dict())
    E, h = torch.randn(2, 70, 32), torch.randn(2, 70, 32)
    assert (capa(E, h) - capa_e(E, h)).abs().max() < 1e-5


def test_escritura_paso_a_paso_igual_que_entero():
    m = est.convertir(_modelo())
    m.eval()
    x = _texto(1, 60, 60)
    with torch.no_grad():
        texto, _ = v89.escribir(m, x[:, :40], n_nuevos=25, temperatura=1.0, top_k=0, semilla=0,
                                memoria_de_lo_ajeno=False)
        entero = m(texto[:, :-1])
        esc = v89._Escritor(m, memoria_de_lo_ajeno=False)
        lgs = [esc.inicio(texto[:, :40])]
        for t in range(40, texto.shape[1] - 1):
            lgs.append(esc.paso(texto[:, t]))
    assert (torch.stack(lgs, 1) - entero[:, 39:]).abs().max() < 1e-5
    assert esc.cache[0]['B'].dtype == torch.float64                     # el reloj de la escritura, en 64 bits


def test_escritura_o1_igual_que_o_t():
    """La capa estable al escribir en O(1) (anillos, Hawkes por clave, mascara exacta y reloj de 64 bits) da lo mismo
    que la escritura O(t) y que el modelo con el texto entero."""
    m = est.convertir(_modelo(chunk=4, n_bits=2))
    m.eval()
    x = _texto(2, 70, 60)
    with torch.no_grad():
        entero = m(x)
        for ajeno in (False, True):
            salidas = {}
            for o1 in (True, False):
                esc = v89._Escritor(m, memoria_de_lo_ajeno=ajeno, o1=o1)
                salidas[o1] = torch.stack([esc.inicio(x[:, :30])] + [esc.paso(x[:, t]) for t in range(30, 70)], 1)
            assert (salidas[True] - salidas[False]).abs().max() < 1e-5, ajeno
            if not ajeno:
                assert (salidas[True] - entero[:, 29:]).abs().max() < 1e-5


def test_convertir_guardar_cargar():
    m = _modelo()
    x = _texto(2, 90, 60)
    with torch.no_grad():
        antes = m(x)
    est.convertir(m)
    assert all(isinstance(c, est.CapaASRNEstable) for c in m.largos)
    with tempfile.TemporaryDirectory() as tmp:
        v89.guardar(m, os.path.join(tmp, 'm.pt'))
        cargado = est.convertir(v89.cargar(os.path.join(tmp, 'm.pt'), 'cpu'))     # sin 'cpu', con GPU va a la GPU
    with torch.no_grad():
        assert (cargado(x) - antes).abs().max() < 1e-5
    est.convertir(cargado, est.Protecciones.ninguna())                  # ya estable: solo cambia las protecciones
    with torch.no_grad():
        assert torch.equal(cargado(x), antes)


# ----------------------------------------------------------------------------------------------------------------
# Lo que arregla cada proteccion
# ----------------------------------------------------------------------------------------------------------------

def _fuga(capa, L=96, corte=60, d=32):
    torch.manual_seed(0)
    E, h, x = torch.randn(1, L, d), torch.randn(1, L, d), torch.randint(0, 7, (1, L))
    E2, h2, x2 = E.clone(), h.clone(), x.clone()
    E2[:, corte:], h2[:, corte:] = torch.randn(1, L - corte, d), torch.randn(1, L - corte, d)
    x2[:, corte:] = torch.randint(0, 7, (1, L - corte))
    with torch.no_grad():
        return (capa(E, h, x)[:, :corte] - capa(E2, h2, x2)[:, :corte]).abs().max().item()


def test_api_del_paquete_usa_la_capa_estable():
    """import asrn: crear_modelo, crear_desde_config (tambien en 'meta') y cargar dan capas estables con el grupo A, con
    los mismos pesos que el nucleo para la misma semilla; protecciones= se respeta; los archivos cruzan en los dos
    sentidos."""
    import asrn
    x = torch.randint(0, 60, (2, 48))
    m = asrn.crear_modelo(60, 32, 2, semilla=3)
    n = v89.crear_modelo(60, 32, 2, semilla=3)
    assert all(isinstance(c, est.CapaASRNEstable) for c in m.largos)
    assert all(c.protecciones == est.Protecciones() for c in m.largos)
    assert not any(isinstance(c, est.CapaASRNEstable) for c in n.largos)
    for (a, p), (b, q) in zip(m.state_dict().items(), n.state_dict().items()):
        assert a == b and torch.equal(p, q), a
    with torch.no_grad():
        assert (m.eval()(x) - n.eval()(x)).abs().max() <= 1e-5
    largos = asrn.crear_modelo(60, 32, 2, semilla=3, protecciones=est.Protecciones.textos_largos(512))
    assert largos.largos[0].protecciones.tope_puerta is not None
    vacio = asrn.crear_desde_config(asrn.ConfigASRN(V=60, d=32, n_capas=2), dispositivo='meta')
    assert isinstance(vacio.largos[0], est.CapaASRNEstable) and next(vacio.parameters()).is_meta
    asrn.materializar(vacio, 'cpu', 3)
    with torch.no_grad():
        assert torch.equal(vacio.eval()(x), m(x))
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, 'm.pt')
        v89.guardar(n, f)                                   # guardado por el nucleo, cargado por el paquete
        c = asrn.cargar(f, dispositivo='cpu')
        assert isinstance(c.largos[0], est.CapaASRNEstable)
        asrn.guardar(c, f)                                  # y al reves
        assert not isinstance(v89.cargar(f, dispositivo='cpu').largos[0], est.CapaASRNEstable)
    assert asrn.__version__ == '1.0.0' and set(asrn.__all__) <= set(dir(asrn))


def test_alias_en_ingles_son_los_mismos_objetos():
    """Los alias en ingles del paquete (ASRNLayer, create_model, train, generate, ...) son exactamente los objetos en
    espanol, estan en __all__, y una capa suelta y un modelo se construyen y corren con ellos."""
    import asrn
    pares = {'ASRNLayer': 'CapaASRNEstable', 'ASRNCoreLayer': 'CapaASRN', 'ASRNModel': 'ModeloASRN',
             'ASRNConfig': 'ConfigASRN', 'LocalBlock': 'BloqueLocal', 'Protections': 'Protecciones',
             'DynamicClip': 'RecorteDinamico', 'create_model': 'crear_modelo',
             'create_from_config': 'crear_desde_config', 'convert': 'convertir', 'train': 'entrenar',
             'evaluate': 'evaluar', 'generate': 'escribir', 'save': 'guardar', 'load': 'cargar'}
    for ingles, espanol in pares.items():
        assert getattr(asrn, ingles) is getattr(asrn, espanol), ingles
        assert ingles in asrn.__all__ and espanol in asrn.__all__, ingles
    assert asrn.ASRNLayer is est.CapaASRNEstable and asrn.ASRNCoreLayer is v89.CapaASRN
    torch.manual_seed(0)
    capa = asrn.ASRNLayer(32)                                   # una capa suelta: (E, h, piezas) -> (B, L, d)
    piezas = torch.randint(0, 50, (2, 40))
    E, h = torch.randn(50, 32)[piezas], torch.randn(2, 40, 32)
    assert capa(E, h, piezas).shape == (2, 40, 32)
    modelo = asrn.create_model(V=50, d=32, n_capas=2, semilla=0)
    assert isinstance(modelo, asrn.ASRNModel) and isinstance(modelo.largos[0], asrn.ASRNLayer)
    secuencias, _ = asrn.generate(modelo, piezas[0, :8], n_nuevos=4, devolver_logits=False)
    assert secuencias.shape == (1, 12)


def test_linea_de_comandos_del_paquete_entrena_con_la_capa_estable():
    """python -m asrn (y asrn-train): la linea de comandos del nucleo con la capa estable (preparar=convertir). Se
    revisa el modelo que la linea de comandos entrena y guarda, no solo que el archivo cargue: si faltara la conversion
    (capas CapaASRN del nucleo), la prueba falla. El archivo guardado tiene esos mismos pesos y, cargado con
    asrn.cargar, da capas estables."""
    import contextlib
    import io
    import asrn
    import asrn.__main__ as principal
    with tempfile.TemporaryDirectory() as tmp:
        datos = os.path.join(tmp, 'p.pt')
        torch.save(torch.randint(0, 60, (6000,), dtype=torch.int32), datos)
        visto = []

        def preparar(modelo):
            modelo = est.convertir(modelo)
            visto.append(type(modelo.largos[0]))
            return modelo
        with contextlib.redirect_stdout(io.StringIO()):
            v89._principal(['--datos', datos, '--V', '60', '--ancho', '16', '--capas', '2', '--largo', '32',
                            '--lote', '2', '--pasos', '2', '--guardar', os.path.join(tmp, 'm.pt')], preparar=preparar)
        assert visto == [est.CapaASRNEstable]
        assert os.path.exists(os.path.join(tmp, 'm.pt'))

        entrenados, entrenar_nucleo = [], v89.entrenar

        def entrenar_y_mirar(modelo, *a, **kw):                         # el modelo que se entrena y se guarda
            entrenados.append(modelo)
            return entrenar_nucleo(modelo, *a, **kw)
        archivo = os.path.join(tmp, 'm2.pt')
        v89.entrenar = entrenar_y_mirar                                 # _principal la llama por su nombre en nucleo
        try:
            with contextlib.redirect_stdout(io.StringIO()):             # python -m asrn / asrn-train
                principal.main(['--datos', datos, '--V', '60', '--ancho', '16', '--capas', '2', '--largo', '32',
                                '--lote', '2', '--pasos', '1', '--guardar', archivo])
        finally:
            v89.entrenar = entrenar_nucleo
        assert len(entrenados) == 1
        modelo = entrenados[0]
        assert modelo.largos and all(type(c) is est.CapaASRNEstable for c in modelo.largos), \
            [type(c).__name__ for c in modelo.largos]
        assert all(c.protecciones == est.Protecciones() for c in modelo.largos)   # el grupo A, como asrn.crear_modelo
        cargado = asrn.cargar(archivo, dispositivo='cpu')
        assert all(type(c) is est.CapaASRNEstable for c in cargado.largos)
        guardado, entrenado = cargado.state_dict(), modelo.state_dict()
        assert list(guardado) == list(entrenado)
        assert all(torch.equal(guardado[n], entrenado[n].cpu()) for n in guardado)


def test_mascara_exacta_cierra_la_fuga_causal():
    """Con gamma enorme todos los puntajes validos quedan bajo -1e9: v89 manda la masa a llaves no validas (la propia
    pregunta, que copia E_{t+1}, y piezas posteriores) y el futuro cambia el pasado. La capa estable no."""
    for gamma in (1e9, 1e11):
        vieja = est.CapaASRNEstable(32, est.Protecciones.ninguna(), chunk=16, n_bits=3)
        nueva = est.CapaASRNEstable(32, chunk=16, n_bits=3)
        nueva.load_state_dict(vieja.state_dict())
        with torch.no_grad():
            vieja.gamma.fill_(gamma)
            nueva.gamma.fill_(gamma)
        assert _fuga(vieja) > 1e-3, gamma                                 # v89: fuga
        assert _fuga(nueva) == 0.0, gamma


def test_mascara_exacta_en_la_escritura():
    """Lo mismo en la escritura paso a paso (_atender): con gamma enorme, escribir sigue dando lo que da el modelo
    entero (que no ve el futuro)."""
    m = est.convertir(_modelo())
    for c in m.largos:
        with torch.no_grad():
            c.gamma.fill_(1e11)
    m.eval()
    x = _texto(1, 50, 60)
    with torch.no_grad():
        entero = m(x)
        esc = v89._Escritor(m, memoria_de_lo_ajeno=False)
        lgs = [esc.inicio(x[:, :30])] + [esc.paso(x[:, t]) for t in range(30, 50)]
    assert torch.isfinite(entero).all() and (torch.stack(lgs, 1) - entero[:, 29:]).abs().max() < 1e-4


def test_reloj_corrido_no_cambia_la_salida():
    """La capa depende solo de diferencias del reloj: sumarle una constante (estar a millones de piezas del inicio)
    no debe cambiar la salida. v89 (reloj de 32 bits) cambia 0.5% con 1.6e7 y ~10% con 1e9; la estable, nada."""
    x, E, h = _capa_y_entradas()
    for prot, tope in ((est.Protecciones.ninguna(), None), (est.Protecciones(), 1e-6)):
        torch.manual_seed(1)
        capa = est.CapaASRNEstable(64, prot, chunk=16)
        _mover([capa], gamma=(0.5, 4.0))
        with torch.no_grad():
            sig = v89.firmas_de_piezas(x)
            Qc, Kc, beta, la = capa._contexto(h)
            Bt = beta.cumsum(1)

            def salida(corrimiento):
                acc = capa._acumular(E, Qc, Kc, Bt + corrimiento, la, sig)
                acc.update(capa._repeticion(sig, beta, Bt + corrimiento, la))
                return capa._combinar(acc, h)
            base = salida(0.0)
            errores = [(salida(c) - base).abs().max().item() / base.abs().max().item() for c in (1.6e7, 1e9)]
        if tope is None:
            assert errores[0] > 1e-3 and errores[1] > 1e-2, errores       # v89: se nota
        else:
            assert max(errores) < tope, errores


def test_reloj_64_no_se_detiene():
    """Con 2**25 piezas de beta ~ 0.8, sumar en 32 bits detiene el reloj en parte de los pasos (B + beta == B); la capa
    estable acumula en 64 bits y cada paso avanza lo que dice beta."""
    torch.manual_seed(0)
    beta = F.softplus(torch.randn(1, 2 ** 25))
    B32 = torch.from_numpy(beta.numpy().cumsum(1, dtype='float32'))       # acumulador de 32 bits (como en la GPU)
    B64 = beta.double().cumsum(1)
    paso32 = (B32[:, 1:] - B32[:, :-1]).double()
    paso64 = B64[:, 1:] - B64[:, :-1]
    assert (paso32[:, -2 ** 22:] == 0).float().mean() > 0.2                # 32 bits: el reloj se detiene
    assert ((paso64 - beta[:, 1:].double()).abs() / beta[:, 1:]).max() < 1e-6


def test_beta_enorme_no_rompe_la_capa():
    """Pesos rotos del reloj (beta ~ 1e37): en v89 el reloj pasa de 3.4e38 a las pocas piezas, B = inf y el nucleo
    da inf - inf = NaN; la estable acota beta y sigue finita, con gradiente finito."""
    x, E, h = _capa_y_entradas(d=32, L=80)
    for prot, finita in ((est.Protecciones.ninguna(), False), (est.Protecciones(), True)):
        capa = est.CapaASRNEstable(32, prot, chunk=16)
        with torch.no_grad():
            capa.b_reloj[0] = 1e37
            capa.gamma.fill_(0.5)
        Er = E.clone().requires_grad_()
        y = capa(Er, h, x)
        y.square().mean().backward()
        ok = bool(torch.isfinite(y).all()) and bool(torch.isfinite(Er.grad).all())
        assert ok == finita, prot


def test_reloj_en_32_bits_con_autocast():
    """Con autocast bf16, v89 calcula w . h_norm del reloj en 16 bits; la estable en 32. beta queda mas cerca del de
    fp32 (al menos 10 veces), y B_t (que acumula ese error) tambien."""
    x, E, h = _capa_y_entradas(d=64, L=512)
    capa = est.CapaASRNEstable(64, est.Protecciones.ninguna())
    _mover([capa])
    capa_e = est.CapaASRNEstable(64)
    capa_e.load_state_dict(capa.state_dict())
    with torch.no_grad():
        verdad = capa._contexto(h)[2].double()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            b16, b32 = capa._contexto(h)[2].double(), capa_e._contexto(h)[2]
    err16 = (b16.cumsum(1) - verdad.cumsum(1)).abs().max()
    err32 = (b32.cumsum(1) - verdad.cumsum(1)).abs().max()
    assert err32 * 10 < err16, (err32, err16)


def test_proyeccion_del_reloj_sin_tf32():
    """_ProyeccionIEEE (el reloj en GPU, inmune a TF32): mismo resultado que F.linear en 32 bits y gradiente correcto."""
    torch.manual_seed(0)
    x = torch.randn(3, 7, 16, dtype=torch.double, requires_grad=True)
    w = torch.randn(2, 16, dtype=torch.double, requires_grad=True)
    b = torch.randn(2, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(est._ProyeccionIEEE.apply, (x, w, b))
    xf, wf, bf = x.detach().float(), w.detach().float(), b.detach().float()
    assert (est._ProyeccionIEEE.apply(xf, wf, bf) - F.linear(xf, wf, bf)).abs().max() < 1e-5


def test_grupo_B_tope_de_puerta_y_piso_de_kappa():
    capa = est.CapaASRNEstable(32, est.Protecciones(tope_puerta=2.0, kappa_min=0.01))
    with torch.no_grad():
        capa.g.bias.fill_(50.0)                                           # logit enorme: la puerta se satura
        capa.log_kappa.fill_(-20.0)
        capa.log_kappa_racha.fill_(-20.0)
    k, kr = capa._kappas()
    assert torch.all(k == 0.01) and torch.all(kr == 0.01)
    x, E, h = _capa_y_entradas(d=32, L=64)
    acc = capa._acumular(E, *capa._contexto(h)[:2], capa._contexto(h)[2].cumsum(1), capa._contexto(h)[3],
                         v89.firmas_de_piezas(x))
    acc.update(capa._repeticion(v89.firmas_de_piezas(x), capa._contexto(h)[2], capa._contexto(h)[2].cumsum(1),
                                capa._contexto(h)[3]))
    with torch.no_grad():
        capa.o.weight.copy_(torch.eye(32))
        acc['salida'] = torch.ones_like(acc['salida'])
        acc['peso_total'] = torch.ones_like(acc['peso_total'])
        puerta = capa._combinar(acc, torch.zeros_like(h))                 # h = 0 y aporte = 1: la salida es la puerta
    assert puerta.max() <= torch.sigmoid(torch.tensor(2.0)) + 1e-6


# ----------------------------------------------------------------------------------------------------------------
# El kernel Triton en modo ESTABLE (interprete de Triton en CPU)
# ----------------------------------------------------------------------------------------------------------------

_PRUEBA_KERNEL = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
import torch, asrn.nucleo as v89, asrn.estable as est
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


def caso(d, kw, B, L, bf16=False, corrimiento=0.0):
    torch.manual_seed(0)
    capa = est.CapaASRNEstable(d, **kw)
    with torch.no_grad():
        for n, p in capa.named_parameters():
            p.add_(torch.randn_like(p) * (0.5 if n in v89._NUEVOS_V87 else 0.05))
        capa.gamma.abs_()
        capa.b_reloj[0] += corrimiento                       # beta enorme: el reloj llega lejos enseguida
    x = torch.randint(0, 12, (B, L))
    x[:, L // 2:L // 2 + L // 4] = x[:, :L // 4]
    E0, h0 = torch.randn(12, d)[x], torch.randn(B, L, d)
    if not bf16:
        (y0, g0), (y1, g1) = gradientes(capa, x, E0, h0, False), gradientes(capa, x, E0, h0, True)
        assert (y0 - y1).abs().max() <= 1e-5 * y0.abs().max(), (d, kw, L)
        for k in g0:
            assert (g0[k] - g1[k]).abs().max() <= 1e-4 * (g0[k].abs().max() + 1e-3), (d, kw, L, k)
    else:
        _, verdad = gradientes(capa, x, E0, h0, False)
        _, ref = gradientes(capa, x, E0, h0, False, True)
        _, tri = gradientes(capa, x, E0, h0, True, True)
        for k in verdad:
            n = verdad[k].abs().max() + 1e-3
            assert (tri[k] - verdad[k]).abs().max() / n <= 1.5 * (ref[k] - verdad[k]).abs().max() / n + 1e-2, k


caso(32, dict(chunk=16, n_bits=4), 2, 90)
caso(32, dict(chunk=16, n_bits=4), 1, 5)
caso(96, dict(n_bits=14), 2, 150)
caso(48, dict(chunk=32, n_bits=6, confirmar=20, sufijo=6), 2, 200, corrimiento=2e5)
caso(64, dict(), 2, 160, bf16=True)

# la fuga: con gamma enorme, en el kernel de v89 el futuro cambia el pasado; en modo ESTABLE no
os.environ['ASRN_TRITON_CASILLEROS'] = '1'
for prot, fuga in ((est.Protecciones.ninguna(), True), (est.Protecciones(), False)):
    torch.manual_seed(0)
    capa = est.CapaASRNEstable(32, prot, chunk=16, n_bits=3)
    with torch.no_grad():
        capa.gamma.fill_(1e9)
    L, corte = 96, 60
    E, h, x = torch.randn(1, L, 32), torch.randn(1, L, 32), torch.randint(0, 7, (1, L))
    E2, h2, x2 = E.clone(), h.clone(), x.clone()
    E2[:, corte:], h2[:, corte:] = torch.randn(1, L - corte, 32), torch.randn(1, L - corte, 32)
    x2[:, corte:] = torch.randint(0, 7, (1, L - corte))
    with torch.no_grad():
        dif = (capa(E, h, x)[:, :corte] - capa(E2, h2, x2)[:, :corte]).abs().max().item()
    assert (dif > 1e-3) == fuga, (prot, dif)
print('ok')
"""


def test_kernel_estable_igual_que_pytorch_y_sin_fuga():
    """El kernel de la atencion por casilleros en modo ESTABLE (reloj en dos partes y mascara exacta) da lo mismo que
    _acumular de la capa estable en PyTorch (fp32 salvo el redondeo; bf16 no mas lejos de fp32 que PyTorch en bf16),
    tambien con el reloj lejos del inicio, y no tiene la fuga causal que si tiene el kernel de v89."""
    import subprocess
    try:
        import triton  # noqa: F401
    except ImportError:
        raise Saltada('Triton no esta instalado')
    r = subprocess.run([sys.executable, '-c', _PRUEBA_KERNEL, _RAIZ], capture_output=True, text=True,
                       env={**os.environ, 'TRITON_INTERPRET': '1'}, timeout=1800)
    assert r.returncode == 0 and r.stdout.strip().endswith('ok'), r.stderr[-3000:]


# ----------------------------------------------------------------------------------------------------------------
# Recorte dinamico
# ----------------------------------------------------------------------------------------------------------------

def _con_gradientes(m, valor_grupo, semilla=0):
    """Gradientes al azar de norma ~1 por grupo; el grupo 'reloj' multiplicado por valor_grupo."""
    g = torch.Generator().manual_seed(semilla)
    for n, p in m.named_parameters():
        p.grad = torch.randn(p.shape, generator=g) * 1e-2
    for c in m.largos:
        for n in ('w_reloj', 'b_reloj', 'gamma'):
            getattr(c, n).grad.mul_(valor_grupo)


def test_recorte_dinamico_grupos_y_umbral():
    m = _modelo()
    r = est.RecorteDinamico(m, tope=10.0, z=3.0, calentar=5)
    assert set(r.grupos) == {'reloj', 'puerta', 'resto'}
    nombres = {id(p): n for n, p in m.named_parameters()}
    assert {nombres[id(p)].rsplit('.', 1)[-1] for p in r.grupos['reloj']} == {'w_reloj', 'b_reloj', 'gamma',
                                                                              'log_kappa', 'log_kappa_racha'}
    assert any(nombres[id(p)].endswith('g_conf.weight') for p in r.grupos['puerta'])
    for s in range(20):                                                   # historia normal
        _con_gradientes(m, 1.0, s)
        r()
    _con_gradientes(m, 1e4, 99)                                           # pico solo en el reloj
    resto_antes = [p.grad.clone() for p in r.grupos['resto']]
    reloj_antes = torch.cat([p.grad.flatten() for p in r.grupos['reloj']]).norm()
    total = r()
    res = r.resumen()
    assert res['reloj'][2] < 1e-2 and res['resto'][2] == 1.0             # se recorta solo el grupo del pico
    assert all(torch.equal(a, p.grad) for a, p in zip(resto_antes, r.grupos['resto']))
    reloj_despues = torch.cat([p.grad.flatten() for p in r.grupos['reloj']]).norm()
    assert reloj_despues < reloj_antes / 100 and abs(reloj_despues - res['reloj'][1]) < 1e-3 * res['reloj'][1]
    assert total > reloj_antes * 0.99                                     # devuelve la norma de antes de recortar
    umbral_antes = res['reloj'][1]
    _con_gradientes(m, 1.0, 100)
    r()
    assert r.resumen()['reloj'][1] < 2 * umbral_antes                     # el pico no subio el umbral


def test_recorte_dinamico_con_nan_y_en_entrenar():
    m = _modelo()
    r = est.RecorteDinamico(m, calentar=2)
    for s in range(4):
        _con_gradientes(m, 1.0, s)
        r()
    media = r.media.clone()
    for p in m.parameters():
        p.grad[...] = float('nan')
    assert not torch.isfinite(r())
    assert torch.equal(r.media, media)                                    # NaN no ensucia la historia
    datos = torch.randint(0, 60, (4000,))
    m = est.convertir(_modelo())
    h = v89.entrenar(m, datos, largo=32, lote=4, pasos=6, registrar_cada=2, registro=lambda *_: None,
                     recorte=est.RecorteDinamico(m, calentar=2))
    assert len(h) == 3 and all(torch.isfinite(torch.tensor(f['perdida'])) for f in h)


if __name__ == '__main__':
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
