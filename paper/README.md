# Paper

The LaTeX sources of the ASRN preprint, in English and Spanish, with its figures and the script that draws them.

The preprint presents the ASRN layer (sparse retrieval of exact earlier occurrences through learned hash buckets, a
learned proper-time clock, and a copy gate that reads Hawkes intensities and a repetition streak), the numerically
protected layer of release 1.0, and its evaluation on one A100 against the original Mamba-2 and RWKV-7 and a
FlashAttention-2 Transformer++: one layer from 1k to 262k tokens, training throughput of ~50M-parameter models
(uncompiled and compiled), generation, long-range copy, MQAR, WikiText-103, an ablation, and the bounded-quotation
experiment (on CPU, replicated on the A100) in which the gate learns when to stop copying. Every result comes from the
ASRN 1.0 results ([`results/`](../results/README.md): `banco_a100/`, `citas/`, `citas_a100/`; the previous A100 run in
`history/results/banco_a100_pt211/` only as a replication, and the development runs in `history/` only as labelled
development history). [`docs/TECHNICAL_REPORT.md`](../docs/TECHNICAL_REPORT.md) is its long version.

| File | What it is |
|---|---|
| `ASRN_Calvario_2026_en.tex` | The paper in English (main source) |
| `ASRN_Calvario_2026_es.tex` | The same paper in Spanish |
| `seccion_citas_en.tex`, `seccion_citas_es.tex` | The section on the bounded-quotation experiment (ASRN 1.0, 3 seeds, CPU and A100), included by the main source of each language (`\input`) |
| `referencias.bib` | The bibliography (BibTeX, shared by both languages) |
| `figuras/en/`, `figuras/es/` | The figures as vector PDF, one folder per language: `capa_diagrama.pdf` (layer diagram), `capa.pdf` (one layer: time and memory against length), `modelo.pdf` (training throughput, uncompiled and compiled), `copia.pdf` (long-range copy, each seed shown), `mqar.pdf` (MQAR), `texto.pdf` (WikiText-103 perplexity), `citas.pdf` (bounded-quotation experiment: learned gate response and streak lengths) |
| `hacer_figuras.py` | Draws the figures in both languages from the result files: `results/banco_a100/resultados.json` (A100 benchmark) and `results/citas/` (bounded-quotation experiment, CPU run); the layer diagram comes from `history/docs/arquitectura_capa.svg` |
| `zenodo.json` | Metadata draft for the Zenodo deposit (title, description, keywords, creators, license, related identifiers) |
| `LICENSE.md` | The license of the paper text and figures (CC BY 4.0) |

## Building the PDFs

```bash
python paper/hacer_figuras.py      # the figures; needs matplotlib (the diagram also needs Playwright with Chromium,
                                   # otherwise it is skipped and the existing PDF is kept)
cd paper
latexmk -pdf ASRN_Calvario_2026_en.tex
latexmk -pdf ASRN_Calvario_2026_es.tex
```

This writes `ASRN_Calvario_2026_en.pdf` and `ASRN_Calvario_2026_es.pdf` in `paper/`. The PDFs are build outputs and
are not committed. Without a local LaTeX installation, the folder can be compiled in Overleaf.

## License

The paper text and figures are licensed under [CC BY 4.0](LICENSE.md); the code of the repository is under the
[Apache License 2.0](../LICENSE).

## Español

Las fuentes en LaTeX del preprint de ASRN, en inglés y en español, con sus figuras y el programa que las dibuja.

El preprint presenta la capa ASRN (búsqueda dispersa de apariciones exactas anteriores con casilleros de hash
aprendidos, un reloj de tiempo propio aprendido y una puerta de copia que lee intensidades de Hawkes y una racha de
repetición), la capa con protecciones numéricas de la versión 1.0 y su evaluación en una A100 frente a los originales
de Mamba-2 y RWKV-7 y un Transformer++ con FlashAttention-2: una capa de 1k a 262k piezas, la velocidad de
entrenamiento de modelos de ~50 M de parámetros (sin compilar y compilados), la escritura, la copia desde lejos, MQAR,
WikiText-103, una ablación y el experimento de citas acotadas (en CPU, repetido en la A100) en el que la puerta aprende
cuándo dejar de copiar. Todos los resultados salen de los de ASRN 1.0 ([`results/`](../results/README.md):
`banco_a100/`, `citas/`, `citas_a100/`; la corrida anterior en la A100, en `history/results/banco_a100_pt211/`, solo
como repetición, y las corridas de desarrollo de `history/` solo como historia del desarrollo, marcada como tal).
[`docs/TECHNICAL_REPORT_es.md`](../docs/TECHNICAL_REPORT_es.md) es su versión larga.

| Archivo | Qué es |
|---|---|
| `ASRN_Calvario_2026_en.tex` | El artículo en inglés (la fuente principal) |
| `ASRN_Calvario_2026_es.tex` | El mismo artículo en español |
| `seccion_citas_en.tex`, `seccion_citas_es.tex` | La sección del experimento de citas acotadas (ASRN 1.0, 3 semillas, CPU y A100), que incluye la fuente principal de cada idioma (`\input`) |
| `referencias.bib` | La bibliografía (BibTeX, la misma para los dos idiomas) |
| `figuras/en/`, `figuras/es/` | Las figuras en PDF vectorial, una carpeta por idioma: `capa_diagrama.pdf` (diagrama de la capa), `capa.pdf` (una capa: tiempo y memoria según el largo), `modelo.pdf` (velocidad de entrenamiento, sin compilar y compilada), `copia.pdf` (copia desde lejos, con cada semilla), `mqar.pdf` (MQAR), `texto.pdf` (perplejidad en WikiText-103), `citas.pdf` (experimento de citas acotadas: respuesta aprendida de la puerta y largo de las rachas) |
| `hacer_figuras.py` | Dibuja las figuras en los dos idiomas a partir de los archivos de resultados: `results/banco_a100/resultados.json` (banco en la A100) y `results/citas/` (experimento de citas acotadas, corrida en CPU); el diagrama de la capa sale de `history/docs/arquitectura_capa.svg` |
| `zenodo.json` | Borrador de los metadatos del depósito en Zenodo (título, descripción, palabras clave, autores, licencia, identificadores relacionados) |
| `LICENSE.md` | La licencia del texto y las figuras del artículo (CC BY 4.0) |

### Compilar los PDF

```bash
python paper/hacer_figuras.py      # las figuras; necesita matplotlib (el diagrama necesita además Playwright con
                                   # Chromium; si no está, se salta y queda el PDF que ya había)
cd paper
latexmk -pdf ASRN_Calvario_2026_en.tex
latexmk -pdf ASRN_Calvario_2026_es.tex
```

Así quedan `ASRN_Calvario_2026_en.pdf` y `ASRN_Calvario_2026_es.pdf` en `paper/`. Los PDF son productos de la
compilación y no están en el repositorio. Sin LaTeX instalado, la carpeta se puede compilar en Overleaf.

### Licencia

El texto y las figuras del artículo tienen licencia [CC BY 4.0](LICENSE.md); el código del repositorio, la
[Licencia Apache 2.0](../LICENSE).
