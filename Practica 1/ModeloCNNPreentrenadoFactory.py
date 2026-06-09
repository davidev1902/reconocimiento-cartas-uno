import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


# ============================================================
# FACTORY DE MODELOS PREENTRENADOS DE KERAS
# ============================================================
class ModeloPreentrenadoFactory:
    """
    Factory para crear modelos usando arquitecturas oficiales de Keras:

    - VGG16
    - VGG19
    - ResNet50
    - MobileNetV2
    - DenseNet121

    Puede trabajar de dos formas:

    1. Entrenamiento desde cero:
       pesos="none"

    2. Transfer learning:
       pesos="imagenet" y congelar_base=True

    3. Fine tuning:
       pesos="imagenet" y congelar_base=False
    """

    @staticmethod
    def crear(
        nombreModelo,
        formaImagen,
        numeroCategorias,
        pesos="imagenet",
        congelar_base=True,
        learning_rate=0.0001
    ):
        nombreModelo = nombreModelo.lower()

        if pesos == "none":
            weights = None
        elif pesos == "imagenet":
            weights = "imagenet"
        else:
            raise ValueError("pesos debe ser 'none' o 'imagenet'.")

        if nombreModelo == "vgg16":
            base_model = tf.keras.applications.VGG16(
                include_top=False,
                weights=weights,
                input_shape=formaImagen
            )
            preprocess_input = tf.keras.applications.vgg16.preprocess_input

        elif nombreModelo == "vgg19":
            base_model = tf.keras.applications.VGG19(
                include_top=False,
                weights=weights,
                input_shape=formaImagen
            )
            preprocess_input = tf.keras.applications.vgg19.preprocess_input

        elif nombreModelo == "resnet50":
            base_model = tf.keras.applications.ResNet50(
                include_top=False,
                weights=weights,
                input_shape=formaImagen
            )
            preprocess_input = tf.keras.applications.resnet50.preprocess_input

        elif nombreModelo == "mobilenetv2":
            base_model = tf.keras.applications.MobileNetV2(
                include_top=False,
                weights=weights,
                input_shape=formaImagen
            )
            preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        elif nombreModelo == "densenet121":
            base_model = tf.keras.applications.DenseNet121(
                include_top=False,
                weights=weights,
                input_shape=formaImagen
            )
            preprocess_input = tf.keras.applications.densenet.preprocess_input

        else:
            raise ValueError(
                f"Modelo '{nombreModelo}' no soportado. "
                "Usa: vgg16, vgg19, resnet50, mobilenetv2 o densenet121."
            )

        base_model.trainable = not congelar_base

        entradas = Input(shape=formaImagen)

        x = preprocess_input(entradas)
        x = base_model(x, training=False if congelar_base else True)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        salidas = Dense(numeroCategorias, activation="softmax")(x)

        model = Model(inputs=entradas, outputs=salidas)

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model