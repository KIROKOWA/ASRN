# Documentation

| Document | English | Español |
|---|---|---|
| Architecture: the layer and the model in diagrams, with tensor shapes and the code that implements each part | [ARCHITECTURE.md](ARCHITECTURE.md) | [ARCHITECTURE_es.md](ARCHITECTURE_es.md) |
| Technical report: every formula, the numerical stability analysis, the benchmark with its caveats, the ablation and the limitations | [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | [TECHNICAL_REPORT_es.md](TECHNICAL_REPORT_es.md) |
| Release notes of 1.0: what changed, known issues, roadmap and how each result was verified | [RELEASE_NOTES.md](RELEASE_NOTES.md) | [RELEASE_NOTES_es.md](RELEASE_NOTES_es.md) |
| Architecture diagram (SVG): the model and the inside of an ASRN layer, with the training cost, O(1) writing and the numerical protections | [asrn_architecture.svg](asrn_architecture.svg) | [asrn_arquitectura.svg](asrn_arquitectura.svg) |

Both diagrams are drawn from one template by [`hacer_diagrama.py`](hacer_diagrama.py):
`python docs/hacer_diagrama.py docs` rewrites the two SVG files.

Start with the [README](../README.md) for installation and a quick start. The raw results behind the figures are in
[`results/`](../results/README.md), and the development history of v89 is in [`history/`](../history/README.md).

## Español

La tabla de arriba enlaza cada documento en los dos idiomas: la arquitectura (diagramas, formas de los tensores y el
código de cada parte), el informe técnico (cada fórmula, el análisis de estabilidad numérica, el banco con sus
salvedades, la ablación y las limitaciones), las notas de la versión 1.0 (cambios, problemas conocidos, hoja de ruta y
cómo se verificó cada resultado) y el diagrama de la arquitectura en SVG (el modelo y el interior de una capa ASRN, con
el costo de entrenar, la escritura O(1) y las protecciones numéricas). Los dos diagramas salen de una sola plantilla con
[`hacer_diagrama.py`](hacer_diagrama.py) (`python docs/hacer_diagrama.py docs` reescribe los dos SVG). Para instalar y
probar el paquete, empieza por el [README](../README.es.md). Los resultados crudos están en
[`results/`](../results/README.md) y la historia de desarrollo de v89 en [`history/`](../history/README.md).
