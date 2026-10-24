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
"""Hace las figuras del articulo (PDF vectorial, en ingles y en espanol) con los numeros de results/.

    python paper/hacer_figuras.py

Escribe paper/figuras/en/*.pdf y paper/figuras/es/*.pdf. Los datos vienen de results/banco_a100 (el banco de ASRN 1.0
en una A100) y de results/citas (el experimento de citas acotadas). El diagrama de la capa sale de
history/docs/arquitectura_capa.svg (traducido para la version en ingles); convertirlo a PDF necesita Playwright con
Chromium (si no esta, se salta y queda el PDF que ya habia). Necesita matplotlib."""
import json
import os
import statistics as st

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
BANCO = os.path.join(RAIZ, 'results', 'banco_a100')
CITAS = os.path.join(RAIZ, 'results', 'citas')

ANCHO = 6.3                                          # pulgadas: el ancho del texto en el articulo
COLORES = {'asrn': '#2a78d6', 'asrn_c': '#86b6ef', 'transformer': '#eb6834', 'transformer_c': '#f4a582',
           'mamba': '#1baf7a', 'rwkv': '#eda100', 'real': '#0b0b0b', 'sin_hawkes': '#e87ba4',
           'sin_senales': '#eda100', 'sin_puerta': '#eb6834', 'sin_reloj': '#1baf7a', 'gris': '#898781'}
MODELOS = {'ASRN': 'asrn', 'Transformer++': 'transformer', 'Mamba-2': 'mamba', 'RWKV-7': 'rwkv'}
VARIANTES = {'ASRN 1.0': 'asrn', 'sin Hawkes ni racha': 'sin_hawkes', 'puerta sin senales': 'sin_senales',
             'sin puerta': 'sin_puerta', 'sin reloj': 'sin_reloj'}
MUESTREO = 'T=0.8, top_k=40, memoria_de_lo_ajeno=True'   # el muestreo por omision

TEXTOS = {
    'en': {
        'asrn': 'ASRN 1.0', 'asrn_c': 'ASRN 1.0, compiled', 'transformer': 'Transformer++',
        'transformer_c': 'Transformer++, compiled', 'mamba': 'Mamba-2', 'rwkv': 'RWKV-7',
        'capa_asrn': 'ASRN layer', 'capa_transformer': 'Attention (FlashAttention-2)', 'capa_mamba': 'Mamba-2 block',
        'capa_rwkv': 'RWKV-7 layer',
        'largo': 'sequence length (tokens)', 'ms': 'time per forward + backward (ms)', 'gb': 'peak memory (GB)',
        'kps': 'training throughput (thousand tokens/s)', 'largo_texto': 'tokens per sequence',
        'ppl': 'perplexity (lower is better)', 'contexto': 'context {L} tokens', 'acierto': 'accuracy',
        'panel_copia': '{L} tokens ({k}×)', 'panel_mqar': '{n} pairs ({L} tokens)',
        'racha': 'streak of already-seen 4-grams ρ', 'respuesta': 'learned gate response to ρ',
        'cola': 'fraction of streaks longer than x', 'largo_racha': 'streak length x (tokens)',
        'real': 'real continuation', 'sin_hawkes': 'without Hawkes and streak', 'sin_senales': 'gate without signals',
        'sin_puerta': 'no gate', 'sin_reloj': 'no clock', 'semilla': 'seed {s}',
    },
    'es': {
        'asrn': 'ASRN 1.0', 'asrn_c': 'ASRN 1.0, compilado', 'transformer': 'Transformer++',
        'transformer_c': 'Transformer++, compilado', 'mamba': 'Mamba-2', 'rwkv': 'RWKV-7',
        'capa_asrn': 'Capa ASRN', 'capa_transformer': 'Atención (FlashAttention-2)', 'capa_mamba': 'Bloque de Mamba-2',
        'capa_rwkv': 'Capa de RWKV-7',
        'largo': 'largo del texto (piezas)', 'ms': 'tiempo de ida y vuelta (ms)', 'gb': 'memoria máxima (GB)',
        'kps': 'velocidad de entrenamiento (miles de piezas/s)', 'largo_texto': 'piezas por texto',
        'ppl': 'perplejidad (más bajo es mejor)', 'contexto': 'contexto de {L} piezas', 'acierto': 'aciertos',
        'panel_copia': '{L} piezas ({k}×)', 'panel_mqar': '{n} pares ({L} piezas)',
        'racha': 'racha de 4-gramas ya vistos ρ', 'respuesta': 'respuesta aprendida de la puerta a ρ',
        'cola': 'fracción de rachas más largas que x', 'largo_racha': 'largo de la racha x (piezas)',
        'real': 'continuación real', 'sin_hawkes': 'sin Hawkes ni racha', 'sin_senales': 'puerta sin señales',
        'sin_puerta': 'sin puerta', 'sin_reloj': 'sin reloj', 'semilla': 'semilla {s}',
    },
}

# Diagrama de la capa: las etiquetas de history/docs/arquitectura_capa.svg en ingles
DIAGRAMA_EN = {
    '8 tablas de casilleros': '8 bucket tables', 'confianza': 'confidence',
    'confirmación (contexto)': 'confirmation (context)', 'cuánto y cuán seguido se repite': 'how much and how often',
    'de 1, 2, 3 y 4': 'of 1, 2, 3 and 4', 'de la búsqueda': 'of the search', 'entre 0 y 1 · usa h': 'in (0, 1) · uses h',
    'escaneo asociativo': 'associative scan', 'estado normalizado': 'normalized state', 'grupos de piezas': 'token n-grams',
    'hash aprendido': 'learned hash', 'lo copiado': 'copied', 'mirar los 32 anteriores': 'look at the 32 previous',
    'núcleo de tiempo': 'time kernel', 'números de pieza → sufijo exacto (hasta 4)': 'token ids → exact suffix (up to 4)',
    'ordenar por casillero': 'sort by bucket', '>piezas<': '>tokens<', 'promedio': 'average', '>puerta<': '>gate<',
    'puerta · copiado': 'gate · copied', 'puntuar y elegir': 'score and select', 'racha ℓ': 'streak ℓ',
    'reloj propio': 'proper-time clock', '>reloj<': '>clock<', 'softmax por tabla': 'softmax per table',
    'trabajo O(N)': 'O(N) work', 'valores: el vector de la pieza que vino DESPUÉS': 'values: embedding of the NEXT token',
    'vistos · Hawkes': 'seen · Hawkes', 'ĥ = norma(h)': 'ĥ = LN(h)',
}


def _leer(carpeta, nombre='resultados.json'):
    with open(os.path.join(carpeta, nombre)) as f:
        return json.load(f)


def _num(v, idioma):
    """262,144 en ingles; 262 144 en espanol."""
    return f'{v:,.0f}' if idioma == 'en' else f'{v:,.0f}'.replace(',', ' ')


def _k(v):
    return f'{v / 1024:.0f}k' if v >= 1024 else f'{v:.0f}'


def _estilo():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.titlesize': 8.5,
        'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e', 'text.color': '#0b0b0b',
        'xtick.color': '#c3c2b7', 'ytick.color': '#c3c2b7', 'xtick.labelcolor': '#52514e',
        'ytick.labelcolor': '#52514e', 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': '#e6e5df', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'xtick.major.size': 0, 'ytick.major.size': 0, 'xtick.minor.size': 0, 'ytick.minor.size': 0,
        'legend.frameon': False, 'pdf.fonttype': 42, 'figure.dpi': 150,
    })


def _guardar(fig, idioma, nombre):
    import matplotlib.pyplot as plt
    carpeta = os.path.join(AQUI, 'figuras', idioma)
    os.makedirs(carpeta, exist_ok=True)
    fig.savefig(os.path.join(carpeta, nombre + '.pdf'), bbox_inches='tight', pad_inches=0.02,
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def _linea(ax, xs, ys, color, etiqueta, guiones=False):
    ax.plot(xs, ys, color=color, lw=1.6, ls=(0, (4, 2)) if guiones else '-', marker='o', ms=3.6,
            mfc='white' if guiones else color, mec=color if guiones else 'white', mew=0.9, label=etiqueta, zorder=3)


def _eje_largo(ax, largos):
    import matplotlib.ticker as tk
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_locator(tk.FixedLocator(largos))
    ax.xaxis.set_major_formatter(tk.FuncFormatter(lambda v, _: _k(v)))
    ax.xaxis.set_minor_locator(tk.NullLocator())
    ax.grid(axis='x', visible=False)


def fig_capa(B, idioma):
    """Una capa, ida y vuelta: tiempo y memoria contra el largo (la capa estable de ASRN 1.0 y los originales)."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tk
    T = TEXTOS[idioma]
    filas = B['capa_gpu']['filas']
    largos = sorted({f['largo'] for f in filas})
    series = [('ASRN (estable)', 'asrn'), ('Transformer++', 'transformer'), ('Mamba-2', 'mamba'), ('RWKV-7', 'rwkv')]
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 2.7))
    for ax, campo, escala, etiqueta in ((ejes[0], 'segundos', 1e3, T['ms']), (ejes[1], 'memoria_gb', 1, T['gb'])):
        for modelo, i in series:
            pts = sorted((f['largo'], escala * f[campo]) for f in filas if f['modelo'] == modelo)
            _linea(ax, [p[0] for p in pts], [p[1] for p in pts], COLORES[i], T['capa_' + i])
        _eje_largo(ax, [1024 * 4 ** j for j in range(5)])
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(tk.LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(tk.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_locator(tk.NullLocator())
        ax.set_xlabel(T['largo'])
        ax.set_ylabel(etiqueta)
    manijas, nombres = ejes[0].get_legend_handles_labels()
    fig.legend(manijas, nombres, loc='upper center', ncol=4, handlelength=1.8, columnspacing=1.0,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=2.0, rect=(0, 0, 1, 0.9))
    _guardar(fig, idioma, 'capa')


def fig_modelo(B, idioma):
    """Modelos de ~50 M: piezas por segundo al entrenar contra las piezas por texto (sin compilar y compilados)."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tk
    T = TEXTOS[idioma]
    filas = B['modelo_gpu']['filas']
    largos = sorted({f['largo'] for f in filas})
    fig, ax = plt.subplots(figsize=(ANCHO * 0.62, 2.5))
    for modelo, i in MODELOS.items():
        for comp in (False, True):                   # las filas compiladas se llaman 'ASRN compilado', etc.
            pts = sorted((f['largo'], f['piezas_por_s'] / 1e3) for f in filas
                         if f['modelo'] == modelo + (' compilado' if comp else ''))
            if pts:
                k = i + ('_c' if comp else '')
                _linea(ax, [p[0] for p in pts], [p[1] for p in pts], COLORES[k], T[k], guiones=comp)
    _eje_largo(ax, largos)
    ax.set_ylim(0, 280)
    ax.yaxis.set_major_locator(tk.MultipleLocator(50))
    ax.set_xlabel(T['largo_texto'])
    ax.set_ylabel(T['kps'])
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=2.2)
    fig.tight_layout()
    _guardar(fig, idioma, 'modelo')


def _elegidas(B, tarea, modelo):
    """Las 3 corridas de un modelo en una tarea, con la tasa elegida (semillas 0, 1, 2)."""
    c = B['calidad']
    lr = c['mejor_lr'][f'{tarea}/{modelo}']
    return sorted((r for r in c['corridas'] if r['tarea'] == tarea and r['modelo'] == modelo and r['lr'] == lr),
                  key=lambda r: r['semilla'])


def _puntos_por_semilla(B, idioma, tarea, paneles, nombre):
    """Barras con la media de 3 semillas y un punto por semilla (la copia es bimodal: se ve cada semilla)."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tk
    T = TEXTOS[idioma]
    fig, ejes = plt.subplots(1, len(paneles), figsize=(ANCHO, 1.95), sharey=True)
    ys = list(range(len(MODELOS)))[::-1]
    for ax, (largo, titulo) in zip(ejes, paneles):
        for y, (modelo, i) in zip(ys, MODELOS.items()):
            v = [100 * r['evaluacion'][str(largo)]['exactitud'] for r in _elegidas(B, tarea, modelo)]
            m = st.mean(v)
            ax.barh(y, m, height=0.62, color=COLORES[i], alpha=0.35, zorder=2)
            ax.scatter(v, [y] * len(v), s=11, color=COLORES[i], edgecolor='white', linewidth=0.6, zorder=3)
            ax.text(max(v) + 3, y, f'{m:.1f}%' if m < 10 else f'{m:.0f}%', va='center', fontsize=7)
        ax.set_yticks(ys, [T[i] for i in MODELOS.values()])
        ax.set_xlim(0, 130)
        ax.xaxis.set_major_locator(tk.FixedLocator([0, 50, 100]))
        ax.xaxis.set_major_formatter(tk.FuncFormatter(lambda v, _: f'{v:.0f}%'))
        ax.grid(axis='y', visible=False)
        ax.set_title(titulo, loc='left', color='#52514e')
    ejes[len(paneles) // 2].set_xlabel(T['acierto'])
    fig.tight_layout(w_pad=1.0)
    _guardar(fig, idioma, nombre)


def fig_copia(B, idioma):
    L = B['calidad']['tareas']['copia']['largo']
    T = TEXTOS[idioma]
    _puntos_por_semilla(B, idioma, 'copia', [(k * L, T['panel_copia'].format(L=_num(k * L, idioma), k=k))
                                             for k in (1, 4, 16)], 'copia')


def fig_mqar(B, idioma):
    t = B['calidad']['tareas']['mqar']
    L, pares = t['largo'], t['largo'] // t['div_pares']
    T = TEXTOS[idioma]
    _puntos_por_semilla(B, idioma, 'mqar', [(k * L, T['panel_mqar'].format(n=k * pares, L=_num(k * L, idioma)))
                                            for k in (1, 2, 4)], 'mqar')


def fig_texto(B, idioma):
    """WikiText-103: perplejidad con 1 024 y 4 096 piezas de contexto (media ± desviacion y cada semilla)."""
    import matplotlib.pyplot as plt
    T = TEXTOS[idioma]
    evs = B['calidad']['tareas']['wikitext']['evaluar']
    fig, ejes = plt.subplots(1, len(evs), figsize=(ANCHO, 1.9), sharey=True)
    ys = list(range(len(MODELOS)))[::-1]
    for ax, largo in zip(ejes, evs):
        todos = []
        for y, (modelo, i) in zip(ys, MODELOS.items()):
            v = [r['evaluacion'][str(largo)]['perplejidad'] for r in _elegidas(B, 'wikitext', modelo)]
            m, d = st.mean(v), st.stdev(v)
            todos += v
            ax.errorbar(m, y, xerr=d, fmt='o', color=COLORES[i], mec='white', mew=0.8, ms=5, elinewidth=1.4,
                        capsize=2.5, zorder=3)
            ax.scatter(v, [y - 0.22] * len(v), s=7, color=COLORES[i], alpha=0.7, linewidth=0, zorder=3)
            ax.annotate(f'{m:.1f}', (max(m + d, max(v)), y), xytext=(5, 0), textcoords='offset points', va='center',
                        fontsize=7.5)
        ax.set_yticks(ys, [T[i] for i in MODELOS.values()])
        ax.set_ylim(-0.7, len(MODELOS) - 0.4)
        ax.grid(axis='y', visible=False)
        ax.set_title(T['contexto'].format(L=_num(largo, idioma)), loc='left', color='#52514e')
        ax.set_xlabel(T['ppl'])
        ax.set_xlim(min(todos) - 4, max(todos) + 14)
    fig.tight_layout(w_pad=1.5)
    _guardar(fig, idioma, 'texto')


def fig_citas(idioma):
    """Experimento de citas acotadas: la respuesta aprendida de la puerta a la racha (capa completa, 3 semillas) y la
    cola del largo de las rachas al escribir (muestreo por omision, 3 semillas juntas)."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tk
    T = TEXTOS[idioma]
    R = _leer(CITAS)
    semillas = [_leer(CITAS, f'semilla_{s}.json') for s in R['semillas']]
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 2.4))
    ax = ejes[0]
    for s, por_capa in sorted(R['respuesta_racha']['ASRN 1.0'].items()):
        pts = por_capa[0]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORES['asrn'], lw=1.3, marker='o', ms=2.8,
                alpha=[1.0, 0.7, 0.45][int(s)], label=T['semilla'].format(s=s), zorder=3)
    ax.axhline(0, color='#c3c2b7', lw=0.8, zorder=1)
    ax.set_xscale('symlog', linthresh=1)
    ax.xaxis.set_major_locator(tk.FixedLocator([0, 1, 4, 16, 64, 256]))
    ax.xaxis.set_major_formatter(tk.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(tk.NullLocator())
    ax.set_xlabel(T['racha'])
    ax.set_ylabel(T['respuesta'])
    ax.legend(loc='lower left', handlelength=1.6)

    def cola(rachas):
        total = sum(rachas.values())
        xs = list(range(1, 41))
        return xs, [sum(n for k, n in rachas.items() if int(k) > x) / total for x in xs]

    ax = ejes[1]
    real = {}
    for k, n in semillas[0]['texto_real']['rachas'].items():
        real[k] = n
    xs, ys = cola(real)
    ax.plot(xs, ys, color=COLORES['real'], lw=1.2, ls=(0, (1, 1.5)), label=T['real'], zorder=4)
    for variante, i in VARIANTES.items():
        juntas = {}
        for sem in semillas:
            for k, n in sem['modelos'][variante]['escritura'][MUESTREO]['rachas'].items():
                juntas[k] = juntas.get(k, 0) + n
        xs, ys = cola(juntas)
        ax.plot(xs, ys, color=COLORES[i], lw=1.5 if i == 'asrn' else 1.1, ls=(0, (4, 2)) if i == 'sin_reloj' else '-',
                label=T[i], zorder=5 if i == 'asrn' else 3)
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1.2)
    ax.set_xlim(0, 40)
    ax.yaxis.set_major_formatter(tk.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xlabel(T['largo_racha'])
    ax.set_ylabel(T['cola'])
    ax.legend(loc='upper right', handlelength=1.8, fontsize=6.8)
    fig.tight_layout(w_pad=2.0)
    _guardar(fig, idioma, 'citas')


def diagrama(idioma):
    """history/docs/arquitectura_capa.svg -> PDF (traducido al ingles para 'en'). Necesita Playwright con Chromium."""
    with open(os.path.join(RAIZ, 'history', 'docs', 'arquitectura_capa.svg')) as f:
        svg = f.read()
    if idioma == 'en':
        for a, b in sorted(DIAGRAMA_EN.items(), key=lambda e: -len(e[0])):
            svg = svg.replace(a, b)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('   (sin Playwright: el diagrama no se regenera)')
        return
    destino = os.path.join(AQUI, 'figuras', idioma, 'capa_diagrama.pdf')
    html = ('<html><head><style>@page{size:1040px 520px;margin:0}html,body{margin:0}</style></head><body>'
            + svg.split('?>', 1)[-1] + '</body></html>')
    with sync_playwright() as p:
        ruta = os.environ.get('CHROMIUM')
        if not ruta and os.path.exists('/opt/pw-browsers/chromium'):
            ruta = '/opt/pw-browsers/chromium'
        b = p.chromium.launch(executable_path=ruta, args=['--disable-gpu', '--no-sandbox']) if ruta else \
            p.chromium.launch()
        pg = b.new_page()
        pg.set_content(html)
        pg.pdf(path=destino, width='1040px', height='520px', print_background=True)
        b.close()


def main():
    _estilo()
    B = _leer(BANCO)
    for idioma in TEXTOS:
        for hacer in (fig_capa, fig_modelo, fig_copia, fig_mqar, fig_texto):
            hacer(B, idioma)
        fig_citas(idioma)
        diagrama(idioma)
        print(f'figuras/{idioma}: listas')


if __name__ == '__main__':
    main()
