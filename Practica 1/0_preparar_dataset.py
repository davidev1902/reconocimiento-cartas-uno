"""
Prepara dataset desde data_origen (fotos propias):
    data_origen/dosmil/   -> dataset/train, val, test
    data_origen/diesmil/  -> dataset/train, val, test

Split: 70% train | 15% val | 15% test  (aleatorio, estratificado)
"""
import os
import shutil
import random

BASE  = os.path.dirname(os.path.abspath(__file__))
SRC   = os.path.join(BASE, "data_origen")
DEST  = os.path.join(BASE, "dataset")

CLASES = ["dosmil", "diesmil"]
SPLITS = ["train", "val", "test"]
RATIO  = (0.70, 0.15, 0.15)   # train / val / test
SEED   = 42

# 1. Limpiar y recrear carpetas destino
print("Preparando carpetas...")
for split in SPLITS:
    for clase in CLASES:
        carpeta = os.path.join(DEST, split, clase)
        if os.path.exists(carpeta):
            shutil.rmtree(carpeta)
        os.makedirs(carpeta)

# 2. Dividir y copiar
random.seed(SEED)
contadores = {s: {c: 0 for c in CLASES} for s in SPLITS}

for clase in CLASES:
    src_carpeta = os.path.join(SRC, clase)
    archivos = sorted([
        f for f in os.listdir(src_carpeta)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    random.shuffle(archivos)

    n     = len(archivos)
    n_tr  = int(n * RATIO[0])
    n_val = int(n * RATIO[1])

    splits_archivos = {
        "train": archivos[:n_tr],
        "val":   archivos[n_tr:n_tr + n_val],
        "test":  archivos[n_tr + n_val:]
    }

    for split, lista in splits_archivos.items():
        for i, archivo in enumerate(lista):
            ext = os.path.splitext(archivo)[1]
            shutil.copy2(
                os.path.join(src_carpeta, archivo),
                os.path.join(DEST, split, clase, f"{clase}_{i}{ext}")
            )
            contadores[split][clase] += 1

print("\nDataset listo:")
for split in SPLITS:
    for clase in CLASES:
        print(f"  {split}/{clase}: {contadores[split][clase]} imagenes")
