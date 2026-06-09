"""
Orquestador de experimentos.
Ejecuta secuencialmente:
  1. 2_CNN_Arq_Propia.py            — arquitectura propia, sin pesos
  2. 3_CNN_TransferLearning...py    — VGG16, ResNet50, DenseNet121 x TL + FT
  3. 4_Transformer_TIMM...py        — ViT-Tiny, Swin-Tiny, DeiT-Small x TL + FT
"""
import subprocess
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__))
PY   = sys.executable

def correr(script, env_vars=None):
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    nombre = os.path.basename(script)
    print(f"\n{'='*60}")
    print(f"  Corriendo: {nombre}")
    if env_vars:
        print(f"  Params   : {env_vars}")
    print(f"{'='*60}")
    result = subprocess.run([PY, script], env=env, cwd=BASE)
    if result.returncode != 0:
        print(f"  [ERROR] {nombre} terminó con código {result.returncode}")

# ============================================================
# 1. CNN Arquitectura Propia (sin parámetros extra)
# ============================================================
correr(os.path.join(BASE, "2_CNN_Arq_Propia.py"))

# ============================================================
# 2. CNN Transfer Learning / Fine Tuning con Keras
# ============================================================
script_3 = os.path.join(BASE, "3_CNN_TransferLearning_FineTunning.py")

for modelo in ["vgg16", "resnet50", "densenet121"]:
    for congelar in [True, False]:
        correr(script_3, {
            "EXP_MODELO":   modelo,
            "EXP_PESOS":    "imagenet",
            "EXP_CONGELAR": str(congelar)
        })

# ============================================================
# 3. Transformer TIMM con PyTorch
# ============================================================
script_4 = os.path.join(BASE, "4_Transformer_TIMM_TransferLearning_FineTuning.py")

for modelo in ["vit_tiny", "swin_tiny", "deit_small"]:
    for congelar in [True, False]:
        correr(script_4, {
            "EXP_MODELO":   modelo,
            "EXP_PESOS":    "imagenet",
            "EXP_CONGELAR": str(congelar)
        })

print("\n" + "="*60)
print("  Todos los experimentos completados.")
print("  Resultados en: models/resultados_f1.txt")
print("="*60)
