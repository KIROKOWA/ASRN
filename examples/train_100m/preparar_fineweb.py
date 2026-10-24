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
"""Prepara FineWeb para entrenar ASRN: baja los archivos parquet de Hugging Face, los tokeniza con el tokenizador de
GPT-2 (tiktoken, 50 257 piezas; al final de cada documento va la pieza <|endoftext|>) y escribe las piezas en
archivos .bin (uint16, el formato que lee asrn.leer_piezas).

    python examples/train_100m/preparar_fineweb.py --salida DIR [--piezas 2000000000] [--val 10000000]
        [--fuente fineweb]

Escribe en DIR:
  val.bin                    validacion: los primeros documentos del ULTIMO archivo de la fuente (el entrenamiento
                             empieza por el primero, asi que nunca se mezclan)
  train_000.bin, ...         entrenamiento, un .bin por archivo parquet, hasta juntar --piezas
  info.json                  que hay en cada archivo; cada .bin terminado queda anotado

Se puede cortar en cualquier momento: al volver a correrlo, sigue desde el primer archivo sin terminar (lo que esta
en info.json no se repite). Cada parquet se baja a --temporal (disco local, rapido) y se borra al terminarlo; el .bin
se escribe primero en --temporal y despues se mueve a DIR (puede ser Google Drive).

--local DIR_PARQUET usa archivos parquet que ya estan en disco (con una columna 'text') en vez de bajarlos."""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

FUENTES = {
    'fineweb': ('HuggingFaceFW/fineweb', 'sample/10BT/'),             # ~10 mil millones de piezas, texto web filtrado
    'fineweb-edu': ('HuggingFaceFW/fineweb-edu', 'sample/10BT/'),     # lo mismo, solo paginas educativas
}
V_GPT2 = 50257


def _archivos(fuente, local):
    """Los parquet de la fuente, en orden."""
    if local:
        return sorted(glob.glob(os.path.join(local, '*.parquet')))
    from huggingface_hub import HfApi
    repo, prefijo = FUENTES[fuente]
    todos = HfApi().list_repo_files(repo, repo_type='dataset')
    return sorted(f for f in todos if f.startswith(prefijo) and f.endswith('.parquet'))


def _bajar(fuente, archivo, local, temporal):
    """Ruta local del parquet (lo baja de Hugging Face si hace falta)."""
    if local:
        return archivo, False
    from huggingface_hub import hf_hub_download
    repo, _ = FUENTES[fuente]
    for intento in range(5):                    # la red de Colab a veces corta una descarga larga
        try:
            return hf_hub_download(repo, archivo, repo_type='dataset', local_dir=temporal), True
        except Exception as e:
            if intento == 4:
                raise
            print(f'   la descarga fallo ({type(e).__name__}: {e}); reintento en {10 * (intento + 1)} s', flush=True)
            time.sleep(10 * (intento + 1))


def _tokenizar(parquet, destino, maximo, enc, hilos):
    """Escribe en `destino` (uint16) las piezas de los documentos de `parquet`, hasta `maximo` piezas. Devuelve
    (piezas escritas, documentos)."""
    import pyarrow.parquet as pq
    eot = enc.eot_token
    escritas = documentos = 0
    t0 = time.time()
    with open(destino, 'wb') as f:
        for lote in pq.ParquetFile(parquet).iter_batches(batch_size=4096, columns=['text']):
            textos = lote.column(0).to_pylist()
            ids = enc.encode_ordinary_batch(textos, num_threads=hilos)
            trozos = []
            for doc in ids:
                doc.append(eot)
                trozos.append(np.asarray(doc, dtype=np.uint16))
            arr = np.concatenate(trozos)
            if escritas + arr.size >= maximo:
                arr = arr[:maximo - escritas]
            arr.tofile(f)
            escritas += arr.size
            documentos += len(textos)
            if escritas >= maximo:
                break
            if documentos % (4096 * 50) == 0:
                print(f'   {escritas:,} piezas ({escritas / max(time.time() - t0, 1e-9):,.0f} piezas/s)', flush=True)
    return escritas, documentos


def _guardar_info(salida, info):
    tmp = os.path.join(salida, 'info.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(info, f, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(salida, 'info.json'))


def preparar(salida, piezas, val, fuente='fineweb', temporal='/tmp/asrn_fineweb', local=None, hilos=None):
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    assert enc.n_vocab == V_GPT2, enc.n_vocab
    hilos = hilos or os.cpu_count() or 4
    os.makedirs(salida, exist_ok=True)
    os.makedirs(temporal, exist_ok=True)
    ruta_info = os.path.join(salida, 'info.json')
    info = json.load(open(ruta_info)) if os.path.exists(ruta_info) else {}
    if info and (info.get('fuente') != fuente or info.get('piezas_pedidas') != piezas or info.get('val_pedidas') != val):
        raise SystemExit(f'{salida} ya tiene datos de otra preparacion ({info.get("fuente")}, '
                         f'{info.get("piezas_pedidas"):,} piezas). Usa otra carpeta o borra esa.')
    info = info or {'fuente': fuente, 'tokenizador': 'gpt2 (tiktoken)', 'V': V_GPT2, 'piezas_pedidas': piezas,
                    'val_pedidas': val, 'val': None, 'train': [], 'terminado': False}
    if info['terminado']:
        print(f'Los datos ya estan listos en {salida}: {sum(t["piezas"] for t in info["train"]):,} piezas de '
              f'entrenamiento y {info["val"]["piezas"]:,} de validacion.')
        return info
    archivos = _archivos(fuente, local)
    if len(archivos) < 2:
        raise SystemExit(f'hacen falta al menos 2 archivos parquet en la fuente (hay {len(archivos)})')
    print(f'{fuente}: {len(archivos)} archivos parquet. Validacion: {archivos[-1]}', flush=True)

    def hacer(origen, nombre, maximo):
        t = time.time()
        print(f'-> {nombre}: {origen} (hasta {maximo:,} piezas)', flush=True)
        parquet, bajado = _bajar(fuente, origen, local, temporal)
        tmp_bin = os.path.join(temporal, nombre)
        n, docs = _tokenizar(parquet, tmp_bin, maximo, enc, hilos)
        if bajado:
            os.remove(parquet)
        shutil.move(tmp_bin, os.path.join(salida, nombre))
        print(f'   {nombre}: {n:,} piezas de {docs:,} documentos en {(time.time() - t) / 60:.1f} min', flush=True)
        return {'archivo': nombre, 'origen': origen, 'piezas': n, 'documentos': docs}

    if info['val'] is None:
        info['val'] = hacer(archivos[-1], 'val.bin', val)
        _guardar_info(salida, info)
    total = sum(t['piezas'] for t in info['train'])
    i = len(info['train'])
    while total < piezas:
        if i >= len(archivos) - 1:              # el ultimo archivo es de la validacion
            print(f'aviso: la fuente se acabo en {total:,} piezas (pediste {piezas:,})', flush=True)
            break
        fila = hacer(archivos[i], f'train_{i:03d}.bin', piezas - total)
        info['train'].append(fila)
        total += fila['piezas']
        _guardar_info(salida, info)
        print(f'   total: {total:,} de {piezas:,} piezas ({100 * total / piezas:.0f}%)', flush=True)
        i += 1
    info['terminado'] = True
    info['piezas_train'] = total
    _guardar_info(salida, info)
    print(f'Listo: {total:,} piezas de entrenamiento y {info["val"]["piezas"]:,} de validacion en {salida}')
    return info


def leer_datos(carpeta):
    """(entrenamiento, validacion) como tensores int32 de torch, juntando los train_*.bin en el orden de info.json."""
    import torch
    info = json.load(open(os.path.join(carpeta, 'info.json')))
    if not info.get('terminado'):
        raise SystemExit(f'los datos de {carpeta} no estan terminados: corre preparar_fineweb.py otra vez')
    partes = [np.fromfile(os.path.join(carpeta, t['archivo']), dtype=np.uint16) for t in info['train']]
    train = np.empty(sum(p.size for p in partes), dtype=np.int32)
    i = 0
    for p in partes:
        train[i:i + p.size] = p
        i += p.size
    val = np.fromfile(os.path.join(carpeta, info['val']['archivo']), dtype=np.uint16).astype(np.int32)
    return torch.from_numpy(train), torch.from_numpy(val), info


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--salida', required=True, help='carpeta donde quedan los .bin (por ejemplo, en Google Drive)')
    a.add_argument('--piezas', type=int, default=2_000_000_000, help='piezas de entrenamiento (por omision 2 mil millones)')
    a.add_argument('--val', type=int, default=10_000_000, help='piezas de validacion (por omision 10 millones)')
    a.add_argument('--fuente', choices=sorted(FUENTES), default='fineweb')
    a.add_argument('--temporal', default='/tmp/asrn_fineweb', help='disco local para los parquet y los .bin a medias')
    a.add_argument('--local', help='carpeta con parquet ya bajados (en vez de Hugging Face)')
    a.add_argument('--hilos', type=int, help='hilos para tokenizar (por omision, todos los de la CPU)')
    o = a.parse_args()
    preparar(o.salida, o.piezas, o.val, o.fuente, o.temporal, o.local, o.hilos)
    sys.exit(0)
