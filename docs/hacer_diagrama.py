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
"""Dibuja docs/asrn_architecture.svg (ingles) y docs/asrn_arquitectura.svg (espanol) desde una sola plantilla.

    python docs/hacer_diagrama.py docs
"""
import re
import sys

T = {
 'en': dict(
  aria="ASRN architecture. Left: the language model, causal local convolution blocks with an ASRN layer after every "
       "second block. Right: inside an ASRN layer. Where to copy: learned sign hashes put each position in a bucket of "
       "each of 8 tables; each token scores its 32 most recent bucket mates and copies the vector of the token that "
       "followed each match. A learned proper-time clock sets the time kernel. Whether to copy: seen flags of 1- to "
       "4-token n-grams, Hawkes intensities and a repetition streak feed a gate. Output h' = h + g W_o c. Training costs "
       "O(N log N); writing costs O(1) per new token.",
  titulo="ASRN 1.0 — architecture",
  sub="Adaptive Sparse Recurrence Network: exact retrieval of earlier occurrences with bounded work per token",
  modelo="Language model", fichas="tokens x_1 … x_t", emb="embedding E", bloque="local block",
  bloque_d="h += GELU(conv_4(LN h))", bloque_d2="h += MLP(LN h)", capa="ASRN layer", rep="× N/2",
  rep_d="an ASRN layer after every second", rep_d2="local block, not after the last", ln="LayerNorm",
  cabeza="output head = Embᵀ (tied)", pred="p(x_{t+1} | x_{≤t})",
  dentro="Inside an ASRN layer", ent_E="E  vectors", ent_h="h  state", ent_x="x  token ids",
  donde="WHERE TO COPY — sparse retrieval", hash_t="learned sign hash", hash_d="8 tables × 14 bits:",
  hash_d2="4 read the token,", hash_d3="4 the token + the previous",
  anillo_t="bucket mates", anillo_d="in each table, the token", anillo_d2="sees the C = 32 latest", anillo_d3="positions in its bucket",
  punt_t="score of each mate s", punt_d="hash similarity φ_t · φ_s", punt_d2="+ confirmation Qĥ_t · Kĥ_s",
  punt_d3="+ exact suffix (≤ 4 tokens of x)", punt_d4="+ log α_s − γ_τ (B_t − B_s) / C",
  copia_t="copy the NEXT token", copia_d="softmax per table over", copia_d2="v(E_{s+1}); mean of the", copia_d3="tables → c_t",
  salida="h′ = h + g·W_o c",
  reloj="PROPER-TIME CLOCK", ln_h="ĥ = LN(h)", beta="β_t = softplus(·) ≥ 0", alfa="α_t = 2σ(·) ∈ (0, 2)",
  bt="B_t = Σ β  (64-bit)", nucleo="→ kernel α_s e^{−κ(B_t − B_s)}",
  si="WHETHER TO COPY — regulation", ngr_t="n-grams of 1–4", ngr_d="seen flags v_n",
  hk_t="Hawkes intensity λ_n", hk_d="sum over earlier occurrences", hk_d2="of the same n-gram",
  racha_t="repetition streak ρ_t", racha_d="ρ_t = v_4 (e^{−κ β_t} ρ_{t−1} + 1)",
  puerta_t="gate g ∈ (0, 1)", puerta_d="σ( w·h + w·conf + Σ w_n v_n",
  puerta_d2="+ Σ u_n log(1 + λ_n) + f(ρ_t) )",
  f_d="learned, non-monotone",
  conf=["confidence of the search:", "tables with candidates,", "best score, peak attention"],
  costo1_t="Training: the whole text at once",
  costo1=["sort by bucket O(N log N), fixed windows of C keys;"],
  costo1b=["all recurrences in one segmented associative scan", "(O(N) work, O(log N) depth); saved memory O(N)"],
  costo2_t="Writing: O(1) per new token",
  costo2=["a ring of the C most recent positions per bucket and", "table, the Hawkes state of each n-gram's last"],
  costo2b=["occurrence, and the streak (one number)"],
  costo3_t="Numerical protections (stable layer)",
  costo3=["exact mask; 64-bit clock passed to the kernel in", "two 32-bit parts; bounds on β, α, κ and the best"],
  costo3b=["score; clock projection without TF32"],
  leyenda="Every new parameter starts neutral: a freshly initialized layer computes the simpler base layer."),
 'es': dict(
  aria="Arquitectura de ASRN. Izquierda: el modelo de lenguaje, bloques locales de convolución causal con una capa ASRN "
       "cada dos bloques. Derecha: dentro de una capa ASRN. Dónde copiar: hashes de signo aprendidos ponen cada posición "
       "en un casillero de cada una de 8 tablas; cada pieza puntúa a sus 32 compañeros de casillero más recientes y "
       "copia el vector de la pieza que siguió a cada coincidencia. Un reloj de tiempo propio aprendido fija el núcleo de "
       "tiempo. Si copiar: los vistos de grupos de 1 a 4 piezas, intensidades de Hawkes y una racha de repetición "
       "alimentan una puerta. Salida h' = h + g W_o c. Entrenar cuesta O(N log N); escribir, O(1) por pieza nueva.",
  titulo="ASRN 1.0 — arquitectura",
  sub="Adaptive Sparse Recurrence Network: recuperación exacta de apariciones anteriores con trabajo acotado por pieza",
  modelo="Modelo de lenguaje", fichas="piezas x_1 … x_t", emb="embedding E", bloque="bloque local",
  bloque_d="h += GELU(conv_4(LN h))", bloque_d2="h += MLP(LN h)", capa="capa ASRN", rep="× N/2",
  rep_d="una capa ASRN cada dos bloques", rep_d2="locales, no después del último", ln="LayerNorm",
  cabeza="salida = Embᵀ (atada)", pred="p(x_{t+1} | x_{≤t})",
  dentro="Dentro de una capa ASRN", ent_E="E  vectores", ent_h="h  estado", ent_x="x  números",
  donde="DÓNDE COPIAR — búsqueda dispersa", hash_t="hash de signo aprendido", hash_d="8 tablas × 14 bits:",
  hash_d2="4 leen la pieza,", hash_d3="4 la pieza + la anterior",
  anillo_t="compañeros de casillero", anillo_d="en cada tabla, la pieza ve", anillo_d2="las C = 32 posiciones",
  anillo_d3="más recientes del casillero",
  punt_t="puntaje de cada compañero s", punt_d="similitud de hash φ_t · φ_s", punt_d2="+ confirmación Qĥ_t · Kĥ_s",
  punt_d3="+ sufijo exacto (≤ 4 piezas de x)", punt_d4="+ log α_s − γ_τ (B_t − B_s) / C",
  copia_t="copiar la SIGUIENTE", copia_d="softmax por tabla sobre", copia_d2="v(E_{s+1}); promedio de", copia_d3="las tablas → c_t",
  salida="h′ = h + g·W_o c",
  reloj="RELOJ DE TIEMPO PROPIO", ln_h="ĥ = LN(h)", beta="β_t = softplus(·) ≥ 0", alfa="α_t = 2σ(·) ∈ (0, 2)",
  bt="B_t = Σ β  (64 bits)", nucleo="→ núcleo α_s e^{−κ(B_t − B_s)}",
  si="SI COPIAR — regulación", ngr_t="grupos de 1 a 4", ngr_d="vistos v_n",
  hk_t="intensidad de Hawkes λ_n", hk_d="suma sobre las apariciones", hk_d2="anteriores del mismo grupo",
  racha_t="racha de repetición ρ_t", racha_d="ρ_t = v_4 (e^{−κ β_t} ρ_{t−1} + 1)",
  puerta_t="puerta g ∈ (0, 1)", puerta_d="σ( w·h + w·conf + Σ w_n v_n",
  puerta_d2="+ Σ u_n log(1 + λ_n) + f(ρ_t) )",
  f_d="aprendida, no monótona",
  conf=["confianza de la búsqueda:", "tablas con candidatos,", "mejor puntaje, pico"],
  costo1_t="Entrenar: todo el texto a la vez",
  costo1=["ordenar por casillero O(N log N), ventanas fijas de C;"],
  costo1b=["todas las recurrencias en un escaneo asociativo", "segmentado (trabajo O(N), profundidad O(log N)); memoria O(N)"],
  costo2_t="Escribir: O(1) por pieza nueva",
  costo2=["un anillo con las C posiciones más recientes de cada", "casillero y tabla, el estado de Hawkes de la última"],
  costo2b=["aparición de cada grupo, y la racha (un número)"],
  costo3_t="Protecciones numéricas (capa estable)",
  costo3=["máscara exacta; reloj de 64 bits pasado al kernel en", "dos partes de 32; topes para β, α, κ y el mejor"],
  costo3b=["puntaje; proyección del reloj sin TF32"],
  leyenda="Todo parámetro nuevo empieza neutro: una capa recién creada calcula la capa base, más simple."),
}

CSS = """
  .fondo{fill:#ffffff;stroke:#d9dee3;stroke-width:1}
  .panel{fill:#f7f8fa;stroke:#d9dee3;stroke-width:1}
  .t1{font:700 22px Inter,"Segoe UI",Helvetica,Arial,sans-serif;fill:#14181c}
  .t2{font:400 13.5px Inter,"Segoe UI",Helvetica,Arial,sans-serif;fill:#57606a}
  .h{font:700 13px Inter,"Segoe UI",Helvetica,Arial,sans-serif;letter-spacing:.06em}
  .b{font:600 13.5px Inter,"Segoe UI",Helvetica,Arial,sans-serif;fill:#14181c}
  .d{font:400 12px Inter,"Segoe UI",Helvetica,Arial,sans-serif;fill:#57606a}
  .m{font:400 12px "JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#30363d}
  .n{fill:#ffffff;stroke:#adb5bd;stroke-width:1.2}
  .ra{fill:#e6f4f1;stroke:#0f7b6c;stroke-width:1.4} .ca{fill:#0f7b6c}
  .rr{fill:#eef1fb;stroke:#3b5bdb;stroke-width:1.4} .cr{fill:#3b5bdb}
  .rg{fill:#fcefe8;stroke:#c2541f;stroke-width:1.4} .cg{fill:#c2541f}
  .ro{fill:#14181c} .rot{font:700 13px "JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;fill:#ffffff}
  .l{stroke:#6e7781;stroke-width:1.4;fill:none}
  .la{stroke:#0f7b6c;stroke-width:1.8;fill:none} .lr{stroke:#3b5bdb;stroke-width:1.6;fill:none;stroke-dasharray:5 4}
  .lg{stroke:#c2541f;stroke-width:1.8;fill:none} .lk{stroke:#14181c;stroke-width:1.8;fill:none}
  .dash{stroke:#adb5bd;stroke-width:1.2;fill:none;stroke-dasharray:4 4}
"""


def fmt(txt):
    """'_x' and '_{..}' -> subscript tspans, '^{..}' -> superscript tspans (dy shifts work in every SVG renderer)."""
    txt = txt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    def sub(m, dy):
        return f'<tspan dy="{dy}" font-size="78%">{m.group(1) or m.group(2)}</tspan><tspan dy="{-dy}">​</tspan>'
    txt = re.sub(r'_\{([^}]*)\}|_([A-Za-z0-9τρα-ω])', lambda m: sub(m, 3.5), txt)
    txt = re.sub(r'\^\{([^}]*)\}()', lambda m: sub(m, -5), txt)
    return txt


def caja(x, y, w, h, clase, lineas, rx=8):
    """A rounded box with its lines centred (first line usually bold)."""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{clase}"/>']
    n = len(lineas)
    y0 = y + h / 2 - (n - 1) * 8.5 + 4.5
    for i, (txt, cl) in enumerate(lineas):
        out.append(f'<text x="{x + w / 2}" y="{y0 + i * 17:.1f}" text-anchor="middle" class="{cl}">{fmt(txt)}</text>')
    return '\n'.join(out)


def texto(x, y, txt, cl, ancla='start', color=None):
    c = f' fill="{color}"' if color else ''
    return f'<text x="{x}" y="{y}" text-anchor="{ancla}" class="{cl}"{c}>{fmt(txt)}</text>'


def flecha(d, clase='l', punta='pl', extra=''):
    return f'<path d="{d}" class="{clase}" marker-end="url(#{punta})"{extra}/>'


def dibujar(L):
    s = []
    W, H = 1320, 862
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="{L["aria"]}">')
    s.append('<title>' + L['titulo'] + '</title>')
    s.append('<defs><style>' + CSS + '</style>')
    for nombre, color in (('pl', '#6e7781'), ('pa', '#0f7b6c'), ('pr', '#3b5bdb'), ('pg', '#c2541f'), ('pk', '#14181c')):
        s.append(f'<marker id="{nombre}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                 f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')
    s.append('</defs>')
    s.append(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="14" class="fondo"/>')
    s.append(texto(28, 42, L['titulo'], 't1'))
    s.append(texto(28, 66, L['sub'], 't2'))

    # ---------------- panel A: the language model -----------------------------------------------------------
    s.append('<rect x="20" y="86" width="268" height="612" rx="12" class="panel"/>')
    s.append(texto(38, 112, L['modelo'].upper(), 'h', color='#57606a'))
    cx, w = 44, 220
    s.append(caja(cx, 130, w, 40, 'n', [(L['fichas'], 'b')]))
    s.append(caja(cx, 194, w, 40, 'n', [(L['emb'], 'b')]))
    s.append(caja(cx, 258, w, 58, 'n', [(L['bloque'], 'b'), (L['bloque_d'], 'm')]))
    s.append(caja(cx, 334, w, 58, 'n', [(L['bloque'], 'b'), (L['bloque_d2'], 'm')]))
    s.append(caja(cx, 410, w, 46, 'ra', [(L['capa'], 'b')]))
    s.append(texto(cx + w / 2, 484, L['rep'], 'b', 'middle'))
    s.append(texto(cx + w / 2, 503, L['rep_d'], 'd', 'middle'))
    s.append(texto(cx + w / 2, 519, L['rep_d2'], 'd', 'middle'))
    s.append(caja(cx, 538, w, 40, 'n', [(L['ln'], 'b')]))
    s.append(caja(cx, 598, w, 40, 'n', [(L['cabeza'], 'b')]))
    s.append(texto(cx + w / 2, 668, L['pred'], 'm', 'middle'))
    for y0, y1 in ((170, 194), (234, 258), (316, 334), (392, 410), (456, 470), (526, 538), (578, 598), (638, 652)):
        s.append(flecha(f'M{cx + w / 2},{y0} L{cx + w / 2},{y1 - 1}'))
    s.append('<path d="M38,258 C30,258 30,262 30,268 L30,450 C30,454 30,456 38,456" class="dash"/>')
    s.append('<path d="M264,433 L304,433" class="la" marker-end="url(#pa)"/>')

    # ---------------- panel B: inside an ASRN layer ---------------------------------------------------------
    s.append('<rect x="306" y="86" width="994" height="612" rx="12" class="panel" style="stroke:#0f7b6c;stroke-width:1.4"/>')
    s.append(texto(326, 112, L['dentro'].upper(), 'h', color='#0f7b6c'))
    s.append(caja(322, 186, 118, 36, 'n', [(L['ent_E'], 'm')]))
    s.append(caja(322, 371, 118, 36, 'n', [(L['ent_h'], 'm')]))
    s.append(caja(322, 496, 118, 36, 'n', [(L['ent_x'], 'm')]))

    # where to copy
    s.append('<rect x="456" y="124" width="828" height="196" rx="10" fill="none" stroke="#0f7b6c" stroke-opacity=".35"/>')
    s.append(texto(470, 144, L['donde'], 'h', color='#0f7b6c'))
    s.append(caja(470, 158, 180, 96, 'ra', [(L['hash_t'], 'b'), (L['hash_d'], 'd'), (L['hash_d2'], 'd'), (L['hash_d3'], 'd')]))
    s.append(caja(666, 158, 168, 96, 'ra', [(L['anillo_t'], 'b'), (L['anillo_d'], 'd'), (L['anillo_d2'], 'd'), (L['anillo_d3'], 'd')]))
    s.append(caja(850, 150, 240, 112, 'ra', [(L['punt_t'], 'b'), (L['punt_d'], 'd'), (L['punt_d2'], 'd'),
                                             (L['punt_d3'], 'd'), (L['punt_d4'], 'd')]))
    s.append(caja(1106, 158, 166, 96, 'ra', [(L['copia_t'], 'b'), (L['copia_d'], 'd'), (L['copia_d2'], 'd'), (L['copia_d3'], 'd')]))
    s.append(flecha('M440,204 L469,206', 'la', 'pa'))
    s.append(flecha('M650,206 L665,206', 'la', 'pa'))
    s.append(flecha('M834,206 L849,206', 'la', 'pa'))
    s.append(flecha('M1090,206 L1105,206', 'la', 'pa'))
    s.append('<rect x="1106" y="272" width="166" height="36" rx="18" class="ro"/>')
    s.append(texto(1189, 295, L['salida'], 'rot', 'middle'))
    s.append(flecha('M1189,254 L1189,271', 'la', 'pa'))

    # proper-time clock
    s.append('<rect x="456" y="334" width="828" height="98" rx="10" fill="none" stroke="#3b5bdb" stroke-opacity=".35"/>')
    s.append(texto(1270, 354, L['reloj'], 'h', 'end', '#3b5bdb'))
    s.append(caja(470, 364, 130, 50, 'rr', [(L['ln_h'], 'm')]))
    s.append(caja(620, 364, 210, 50, 'rr', [(L['beta'], 'm'), (L['alfa'], 'm')]))
    s.append(caja(850, 364, 190, 50, 'rr', [(L['bt'], 'm')]))
    s.append(texto(1112, 394, L['nucleo'], 'd', color='#3b5bdb'))
    s.append(flecha('M440,389 L469,389'))
    s.append(flecha('M600,389 L619,389'))
    s.append(flecha('M830,389 L849,389'))
    s.append(flecha('M535,364 C535,300 880,330 900,263', 'lr', 'pr'))
    s.append(flecha('M985,364 L985,263', 'lr', 'pr'))
    s.append(flecha('M880,414 C880,450 800,452 790,481', 'lr', 'pr'))

    # whether to copy
    s.append('<rect x="456" y="446" width="828" height="236" rx="10" fill="none" stroke="#c2541f" stroke-opacity=".35"/>')
    s.append(texto(470, 466, L['si'], 'h', color='#c2541f'))
    s.append(caja(470, 482, 160, 64, 'rg', [(L['ngr_t'], 'b'), (L['ngr_d'], 'd')]))
    s.append(caja(648, 482, 200, 64, 'rg', [(L['hk_t'], 'b'), (L['hk_d'], 'd'), (L['hk_d2'], 'd')]))
    s.append(caja(866, 482, 230, 64, 'rg', [(L['racha_t'], 'b'), (L['racha_d'], 'd')]))
    s.append(caja(648, 592, 448, 76, 'rg', [(L['puerta_t'], 'b'), (L['puerta_d'], 'd'), (L['puerta_d2'], 'd')]))
    s.append(flecha('M440,514 L469,514', 'lg', 'pg'))
    s.append(flecha('M630,514 L647,514', 'lg', 'pg'))
    s.append(flecha('M600,546 C600,568 940,566 960,547', 'lg', 'pg'))
    s.append(flecha('M560,546 C560,580 650,578 672,591', 'lg', 'pg'))
    s.append(flecha('M748,546 L748,591', 'lg', 'pg'))
    s.append(flecha('M1010,546 L1010,591', 'lg', 'pg'))
    s.append('<rect x="576" y="616" width="44" height="28" rx="6" class="n"/>' + texto(598, 635, 'h', 'm', 'middle'))
    s.append(flecha('M620,630 L647,630'))
    s.append('<g transform="translate(1112,592)"><rect width="160" height="76" rx="8" class="n"/>'
             '<path d="M14,50 L146,50" class="l" style="stroke-width:1"/>'
             '<path d="M16,40 C30,20 42,14 56,22 C66,30 70,44 78,58 L146,58" stroke="#c2541f" stroke-width="2" fill="none"/>'
             + texto(80, 70, L['f_d'], 'd', 'middle') + texto(10, 16, 'f(ρ)', 'm') + '</g>')
    s.append(flecha('M1112,630 L1097,630', 'lg', 'pg'))
    for k, linea in enumerate(L['conf']):
        s.append(texto(1112, 494 + 16 * k, linea, 'd', color='#0f7b6c'))
    s.append(flecha('M1084,262 C1104,300 1104,560 1084,591', 'la', 'pa', ' stroke-dasharray="6 4"'))
    s.append(flecha('M1000,668 C1000,692 1290,694 1290,640 L1290,320 C1290,300 1286,291 1273,291', 'lg', 'pg'))
    s.append(texto(1282, 470, 'g', 'b', 'end', '#c2541f'))

    # ---------------- footer -----------------------------------------------------------------------------
    anchos = 1280 / 3
    for i, (t, a, b) in enumerate(((L['costo1_t'], L['costo1'], L['costo1b']), (L['costo2_t'], L['costo2'], L['costo2b']),
                                   (L['costo3_t'], L['costo3'], L['costo3b']))):
        x = 20 + i * anchos
        s.append(f'<rect x="{x:.1f}" y="714" width="{anchos - 12:.1f}" height="100" rx="10" class="panel"/>')
        s.append(texto(round(x + 16, 1), 738, t, 'b'))
        for k, linea in enumerate(a + b):
            s.append(texto(round(x + 16, 1), 760 + 17 * k, linea, 'd'))
    s.append(texto(W / 2, 844, L['leyenda'], 'd', 'middle'))
    s.append('</svg>')
    return '\n'.join(s) + '\n'


if __name__ == '__main__':
    destino = sys.argv[1]
    for idioma, nombre in (('en', 'asrn_architecture.svg'), ('es', 'asrn_arquitectura.svg')):
        with open(f'{destino}/{nombre}', 'w', encoding='utf-8') as f:
            f.write(dibujar(T[idioma]))
    print('ok')
