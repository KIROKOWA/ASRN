# Training a 100-million-parameter ASRN on FineWeb

Everything needed to train from scratch an ASRN 1.0 language model of **103.5 million parameters** in Google Colab,
with **FineWeb** data downloaded from Hugging Face.

| File | What it does |
|---|---|
| `entrenar_asrn_100m.ipynb` | The Colab notebook: open this one. It does everything in order and can be re-run if it is interrupted |
| `preparar_fineweb.py` | Downloads FineWeb, tokenizes it with the GPT-2 tokenizer and writes the `.bin` files (continues where it stopped if interrupted) |
| `entrenar.py` | Trains with `asrn.entrenar()` (stable layer); saves every 100 steps and resumes from there by itself |

## The model and the recipe

| | |
|---|---|
| Shape | 12 layers of width 768 (like GPT-2 small); one ASRN layer every two blocks, 6 in total |
| Parameters | **103,473,612** in total; 64.9 million without the vocabulary (the vocabulary matrix is shared by input and output) |
| Tokenizer | GPT-2 (`tiktoken`), 50,257 tokens; `<\|endoftext\|>` between documents |
| Data | FineWeb, 10-billion-token sample (`HuggingFaceFW/fineweb`, `sample/10BT`). Training: 2 billion tokens (~20 per parameter, the Chinchilla ratio). Validation: 10 million from a file that training does not use |
| Step | 16 texts of 1,024 tokens, 32 accumulated batches: 524,288 tokens per step, 3,815 steps |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1, gradient clipping 1.0 |
| Learning rate | 2e-3; 200 warm-up steps and cosine down to 2e-4 |
| Precision | bf16 (autocast); with `COMPILAR = True`, PyTorch 2.14 and `torch.compile` by parts |

Other shapes above 100 million (changing `ANCHO` and `CAPAS` in the notebook): 14 × 768 = 114 million,
13 × 832 = 123 million, 16 × 704 = 108 million. The notebook and `entrenar.py` stop if the model falls below
100 million.

## How to run it

1. Open `entrenar_asrn_100m.ipynb` in Google Colab (**File → Upload notebook**).
2. **Runtime → Change runtime type → A100 GPU**.
3. **Runtime → Run all**. It asks for permission to use Google Drive and, if the repository is private, for the
   repository ZIP (on GitHub: **Code → Download ZIP**).

Everything is kept in `My Drive/asrn_100m/`:

```
asrn_100m/
├── codigo/       copy of the repository (every session uses the same code)
├── datos/        val.bin, train_000.bin, ... and info.json (~4 GB)
└── modelo/       modelo.pt (to resume), modelo.mejor.pt, asrn_100m.pt (the final one), registro.txt, curvas.html
```

**If Colab disconnects**, open the notebook again and **Run all**: the data is not prepared again and training
continues from the last saved point (every 100 steps, about 10 minutes of work).

**Without the notebook**, from the repository root (see each script's `--help` for the options):

```bash
python examples/train_100m/preparar_fineweb.py --salida DATA_DIR [--piezas 2000000000] [--val 10000000]
python examples/train_100m/entrenar.py --datos DATA_DIR --salida MODEL_DIR [options]
```

To change the notebook, edit its builder and regenerate it:
`python benchmarks/builders/construir_entrenar_100m.py examples/train_100m/entrenar_asrn_100m.ipynb`.

## Time and cost

- Preparing the data: 20–40 minutes, only once.
- Training: about 5–7 hours on an A100 with 2 billion tokens (depends on the A100 you get and on whether it compiles).
  The log shows the speed and how much is left. With `PIEZAS = 1_000_000_000` it takes half.
- Space in Drive: ~6 GB.

## Using the model

```python
import asrn, tiktoken, torch
modelo = asrn.cargar('asrn_100m.pt', 'cuda')
enc = tiktoken.get_encoding('gpt2')
x = torch.tensor([enc.encode_ordinary('The history of the city began')], device='cuda')
salida, _ = asrn.escribir(modelo, x, n_nuevos=80, temperatura=0.8, top_k=40, devolver_logits=False)
print(enc.decode(salida[0].tolist()))
```

## Tested

The whole pipeline (prepare, train, interrupt halfway with Ctrl+C, resume, plot, evaluate, generate) was tested on CPU
with a small model, local parquet files and a fake tokenizer of the same shape (no network access to Hugging Face). It
has not been run on a real A100 yet: the memory, speed and quality of the 100-million model are still to be measured.

## Español

### Entrenar ASRN de 100 millones de parámetros con FineWeb

Todo lo necesario para entrenar desde cero un modelo de lenguaje ASRN 1.0 de **103.5 millones de parámetros** en
Google Colab, con datos de **FineWeb** bajados de Hugging Face.

| Archivo | Qué hace |
|---|---|
| `entrenar_asrn_100m.ipynb` | El cuaderno de Colab: abre este. Hace todo en orden y se puede volver a correr si se corta |
| `preparar_fineweb.py` | Baja FineWeb, lo tokeniza con el tokenizador de GPT-2 y escribe los `.bin` (sigue donde quedó si se corta) |
| `entrenar.py` | Entrena con `asrn.entrenar()` (capa estable); guarda cada 100 pasos y sigue solo desde ahí |

### El modelo y la receta

| | |
|---|---|
| Forma | 12 capas de ancho 768 (como GPT-2 small); una capa ASRN cada dos bloques, 6 en total |
| Parámetros | **103 473 612** en total; 64.9 millones sin el vocabulario (la matriz de vocabulario es la misma de entrada y de salida) |
| Tokenizador | GPT-2 (`tiktoken`), 50 257 piezas; `<\|endoftext\|>` entre documentos |
| Datos | FineWeb, muestra de 10 mil millones de piezas (`HuggingFaceFW/fineweb`, `sample/10BT`). Entrenamiento: 2 mil millones de piezas (~20 por parámetro, la proporción de Chinchilla). Validación: 10 millones de un archivo que el entrenamiento no usa |
| Paso | 16 textos de 1 024 piezas, 32 lotes acumulados: 524 288 piezas por paso, 3 815 pasos |
| Optimizador | AdamW, betas (0.9, 0.95), decaimiento 0.1, recorte de gradiente 1.0 |
| Tasa | 2e-3; 200 pasos de calentamiento y coseno hasta 2e-4 |
| Precisión | bf16 (autocast); con `COMPILAR = True`, PyTorch 2.14 y `torch.compile` por partes |

Otras formas de más de 100 millones (cambiando `ANCHO` y `CAPAS` en el cuaderno): 14 × 768 = 114 millones,
13 × 832 = 123 millones, 16 × 704 = 108 millones. El cuaderno y `entrenar.py` se detienen si el modelo queda por
debajo de 100 millones.

### Cómo correrlo

1. Abre `entrenar_asrn_100m.ipynb` en Google Colab (**Archivo → Subir cuaderno**).
2. **Entorno de ejecución → Cambiar tipo de entorno de ejecución → A100 GPU**.
3. **Entorno de ejecución → Ejecutar todas**. Pide permiso para usar Google Drive y, si el repositorio es privado,
   el ZIP del repositorio (en GitHub: **Code → Download ZIP**).

Todo queda en `Mi unidad/asrn_100m/` de tu Drive:

```
asrn_100m/
├── codigo/       copia del repositorio (todas las sesiones usan el mismo código)
├── datos/        val.bin, train_000.bin, ... e info.json (~4 GB)
└── modelo/       modelo.pt (para seguir), modelo.mejor.pt, asrn_100m.pt (el final), registro.txt, curvas.html
```

**Si Colab se desconecta**, abre el cuaderno otra vez y **Ejecutar todas**: los datos no se repiten y el
entrenamiento sigue desde el último punto guardado (cada 100 pasos, unos 10 minutos de trabajo).

**Sin el cuaderno**, desde la raíz del repositorio (las opciones, con `--help` de cada programa):

```bash
python examples/train_100m/preparar_fineweb.py --salida DIR_DATOS [--piezas 2000000000] [--val 10000000]
python examples/train_100m/entrenar.py --datos DIR_DATOS --salida DIR_MODELO [opciones]
```

Para cambiar el cuaderno, edita su programa y vuelve a generarlo:
`python benchmarks/builders/construir_entrenar_100m.py examples/train_100m/entrenar_asrn_100m.ipynb`.

### Tiempo y costo

- Preparar los datos: 20–40 minutos, una sola vez.
- Entrenar: unas 5–7 horas en una A100 con 2 mil millones de piezas (depende de la A100 que toque y de si compila).
  El registro muestra la velocidad y cuánto falta. Con `PIEZAS = 1_000_000_000` es la mitad.
- Espacio en Drive: ~6 GB.

### Usar el modelo

```python
import asrn, tiktoken, torch
modelo = asrn.cargar('asrn_100m.pt', 'cuda')
enc = tiktoken.get_encoding('gpt2')
x = torch.tensor([enc.encode_ordinary('The history of the city began')], device='cuda')
salida, _ = asrn.escribir(modelo, x, n_nuevos=80, temperatura=0.8, top_k=40, devolver_logits=False)
print(enc.decode(salida[0].tolist()))
```

### Probado

Toda la cadena (preparar, entrenar, cortar a la mitad con Ctrl+C, seguir, graficar, evaluar, escribir) se probó en
CPU con un modelo chico, parquet locales y un tokenizador falso de igual forma (sin red a Hugging Face). En una A100
de verdad todavía no se ha corrido: la memoria, la velocidad y la calidad del modelo de 100 millones están por medirse.
