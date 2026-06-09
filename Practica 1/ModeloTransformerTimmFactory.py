import timm
import torch.nn as nn


class ModeloTransformerTimmFactory:
    """
    Factory para crear modelos tipo Transformer / híbridos desde TIMM.

    Librería:
        pip install torch torchvision timm scikit-learn matplotlib pillow opencv-python

    Modelos soportados por alias:
        - vit_tiny
        - vit_small
        - vit_base
        - deit_tiny
        - deit_small
        - swin_tiny
        - swin_small
        - coat_tiny
        - coat_lite_tiny
        - convnext_tiny

    Modos:
        pesos="imagenet" -> pretrained=True
        pesos="none"     -> pretrained=False

        congelar_base=True  -> transfer learning
        congelar_base=False -> fine tuning
    """

    MODELOS = {
        "vit_tiny": "vit_tiny_patch16_224",
        "vit_small": "vit_small_patch16_224",
        "vit_base": "vit_base_patch16_224",

        "deit_tiny": "deit_tiny_patch16_224",
        "deit_small": "deit_small_patch16_224",

        "swin_tiny": "swin_tiny_patch4_window7_224",
        "swin_small": "swin_small_patch4_window7_224",

        "coat_tiny": "coat_tiny",
        "coat_lite_tiny": "coat_lite_tiny",

        # No es transformer puro, pero es moderno y se usa mucho como baseline.
        "convnext_tiny": "convnext_tiny"
    }

    @staticmethod
    def crear(
        nombreModelo,
        numeroCategorias,
        pesos="imagenet",
        congelar_base=True
    ):
        nombreModelo = nombreModelo.lower()

        if nombreModelo not in ModeloTransformerTimmFactory.MODELOS:
            raise ValueError(
                f"Modelo '{nombreModelo}' no soportado. "
                f"Usa uno de estos: {list(ModeloTransformerTimmFactory.MODELOS.keys())}"
            )

        if pesos == "imagenet":
            pretrained = True
        elif pesos == "none":
            pretrained = False
        else:
            raise ValueError("pesos debe ser 'imagenet' o 'none'.")

        nombre_timm = ModeloTransformerTimmFactory.MODELOS[nombreModelo]

        model = timm.create_model(
            nombre_timm,
            pretrained=pretrained,
            num_classes=numeroCategorias
        )

        if congelar_base:
            ModeloTransformerTimmFactory._congelar_base_y_dejar_cabeza(model)
        else:
            ModeloTransformerTimmFactory._activar_todo(model)

        return model, nombre_timm

    @staticmethod
    def _activar_todo(model):
        for param in model.parameters():
            param.requires_grad = True

    @staticmethod
    def _congelar_base_y_dejar_cabeza(model):
        """
        Congela toda la red y deja entrenable la cabeza clasificadora.

        En TIMM muchos modelos llaman la última capa:
            - head
            - classifier
            - fc

        Este método cubre la mayoría de modelos ViT, DeiT, Swin, CoaT,
        ConvNeXt, ResNet, EfficientNet, etc.
        """
        for param in model.parameters():
            param.requires_grad = False

        nombres_cabeza = ["head", "classifier", "fc"]

        for nombre, modulo in model.named_modules():
            if any(clave in nombre.lower() for clave in nombres_cabeza):
                for param in modulo.parameters():
                    param.requires_grad = True

        # Fallback: si por alguna razón no encontró cabeza entrenable,
        # activa todo para evitar entrenar con cero parámetros.
        entrenables = [p for p in model.parameters() if p.requires_grad]
        if len(entrenables) == 0:
            for param in model.parameters():
                param.requires_grad = True

    @staticmethod
    def contar_parametros(model):
        total = sum(p.numel() for p in model.parameters())
        entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
        congelados = total - entrenables

        return {
            "total": total,
            "entrenables": entrenables,
            "congelados": congelados
        }
