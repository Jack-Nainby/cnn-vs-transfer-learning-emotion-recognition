from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
    Resizing,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
    Activation,
    MaxPooling2D,
    Conv2D,
    SpatialDropout2D,
    ReLU,
    Add
)
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from src.config import INPUT_SHAPE, NUM_CLASSES

def create_Base_CNN():
    """
    Create the Baseline Custom CNN Architecture
    """
    model = Sequential([
        Input(INPUT_SHAPE),

        Conv2D(filters=2, kernel_size=10, use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2),

        Conv2D(filters=4, kernel_size=6, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2),

        Conv2D(filters=4, kernel_size=5, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),

        Conv2D(filters=4, kernel_size=4, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),

        Conv2D(filters=4, kernel_size=3, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),

        Conv2D(filters=8, kernel_size=3, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),

        Conv2D(filters=8, kernel_size=3, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),

        MaxPooling2D(pool_size=2),

        GlobalAveragePooling2D(),

        Dense(128, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax", dtype='float32')
    ])

    return model

def create_CNN3():
    """
    Create a Large Custom CNN Architecture.
    """
    model = Sequential([
        Input(INPUT_SHAPE),
        Conv2D(filters=2, kernel_size=10, use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2),
        
        Conv2D(filters=4, kernel_size=8, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2),
        
        Conv2D(filters=8, kernel_size=7, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        
        Conv2D(filters=32, kernel_size=6, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),

        # Spatial dropout applied to reduce overfitting
        SpatialDropout2D(rate=0.2),
        
        Conv2D(filters=64, kernel_size=5, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),

        # Spatial dropout applied to reduce overfitting
        SpatialDropout2D(rate=0.2),
        
        Conv2D(filters=128, kernel_size=4, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),

        # Spatial dropout applied to reduce overfitting
        SpatialDropout2D(rate=0.3),
        
        Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),

        # Spatial dropout applied to reduce overfitting
        SpatialDropout2D(rate=0.3),
        
        Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),

        # Spatial dropout applied to reduce overfitting
        SpatialDropout2D(rate=0.4),
        
        Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        
        MaxPooling2D(pool_size=2),

        # Spatial dropout applied to reduce overfitting
        SpatialDropout2D(rate=0.5),
        
        GlobalAveragePooling2D(),
        
        Dense(512, activation='relu'),

        # Output layer producing emotion class probabilities.
        Dense(NUM_CLASSES, activation='softmax', dtype='float32')
    ])
    return model

def create_CNN4():
    """
    Create a Custom CNN Architecture with Data 
    Augmentation.
    """
    model = Sequential([
        Input(INPUT_SHAPE),

        # Data augmentation performed within the model.
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),
        RandomTranslation(0.1, 0.1),
        
        Conv2D(filters=32, kernel_size=4, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(2),
        
        Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(2),
        
        Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(2),
        
        Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),

        # Spatial dropout applied to reduce overfitting
        SpatialDropout2D(rate=0.3),
        
        Conv2D(filters=512, kernel_size=3, padding='same', use_bias=False),
        BatchNormalization(),
        Activation("relu"),

        # Spatial dropout applied to reduce overfitting
        SpatialDropout2D(rate=0.4),
        
        GlobalAveragePooling2D(),
        
        Dense(256, activation='relu'),

        # Standard dropout applied before classification.
        Dropout(rate=0.5),

        # Output layer producing emotion class probabilities.
        Dense(NUM_CLASSES, activation='softmax', dtype='float32')
    ])
    return model

def create_CNN5():
    """
    Create a Custom CNN Architecture with Residual 
    Connections and Data Augmentation.
    """
    inputs = Input(shape=INPUT_SHAPE)

    # Data augmentation performed within the model.
    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.1)(x)
    x = RandomZoom(0.1)(x)
    x = RandomTranslation(0.1, 0.1)(x)

    x = Conv2D(filters=32, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(2)(x)

    # Store input for first residual connection.
    shortcut = x
    
    x = Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Match channel dimensions before residual addition.
    shortcut = Conv2D(filters=64, kernel_size=1, padding='same', use_bias=False)(shortcut)

    shortcut = BatchNormalization()(shortcut)

    # Residual connection.
    x = Add()([x, shortcut])
    x = ReLU()(x)
    x = MaxPooling2D(2)(x)

    # Store x for second residual connection.
    shortcut = x

    x = Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Match channel dimensions before residual addition.
    shortcut = Conv2D(filters=128, kernel_size=1, padding='same', use_bias=False)(shortcut)

    shortcut = BatchNormalization()(shortcut)

    # Residual connection.
    x = Add()([x, shortcut])
    x = ReLU()(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SpatialDropout2D(0.3)(x)

    x = Conv2D(filters=512, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SpatialDropout2D(0.4)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Output layer producing emotion class probabilities.
    outputs = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

    model = Model(inputs, outputs)

    return model

def create_vgg16_model(
    unfrozen_layers=0,
    extra_dense_layers=None,
    weights="imagenet"
):
    """
    Create a transfer learning model based on VGG16. 
    Supports: fully frozen feature extraction, 
    partial fine-tuning, 
    optional additional dense layers.
    """
    if extra_dense_layers is None:
        extra_dense_layers = []

    # Load pretrained VGG16 feature extractor.
    base_model = VGG16(
        weights=weights,
        include_top=False,
        input_shape=INPUT_SHAPE
    )

    # Freeze all pretrained layers.
    for layer in base_model.layers:
        layer.trainable = False

    # Optionally unfreeze final layers for fine-tuning.
    if unfrozen_layers > 0:
        for layer in base_model.layers[-unfrozen_layers:]:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)

    # Optional custom classification head.
    for units in extra_dense_layers:
        x = Dense(units, activation="relu")(x)
        x = Dropout(0.3)(x)

    # Output layer producing emotion class probabilities.
    output = Dense(NUM_CLASSES, activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=output)

def create_resnet50_model(
    unfrozen_layers=0,
    extra_dense_layers=None,
    weights="imagenet"
):
    """
    Create a transfer learning model based on ResNet50. 
    Supports: fully frozen feature extraction, 
    partial fine-tuning, 
    optional additional dense layers.
    """
    if extra_dense_layers is None:
        extra_dense_layers = []

    # Load pretrained ResNet50 feature extractor.
    base_model = ResNet50(
        weights=weights,
        include_top=False,
        input_shape=INPUT_SHAPE
    )

    # Freeze all pretrained layers.
    for layer in base_model.layers:
        layer.trainable = False

    # Optionally unfreeze final layers for fine-tuning.
    if unfrozen_layers > 0:
        for layer in base_model.layers[-unfrozen_layers:]:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)

    # Optional custom classification head.
    for units in extra_dense_layers:
        x = Dense(units, activation="relu")(x)
        x = Dropout(0.3)(x)

    # Output layer producing emotion class probabilities.
    output = Dense(NUM_CLASSES, activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=output)

def create_inceptionv3_model(
    unfrozen_layers=0,
    extra_dense_layers=None,
    weights="imagenet"
):
    """
    Create a transfer learning model based on InceptionV3. 
    Supports: fully frozen feature extraction, 
    partial fine-tuning, 
    optional additional dense layers.
    """
    if extra_dense_layers is None:
        extra_dense_layers = []

    inputs = Input(shape=INPUT_SHAPE)

    # Resize FER2013 images to match InceptionV3 requirements.
    resized = Resizing(75, 75)(inputs)

    # Load pretrained InceptionV3 feature extractor.
    base_model = InceptionV3(
        weights=weights,
        include_top=False,
        input_tensor=resized
    )

    # Freeze all pretrained layers.
    for layer in base_model.layers:
        layer.trainable = False

    # Optionally unfreeze final layers for fine-tuning.
    if unfrozen_layers > 0:
        for layer in base_model.layers[-unfrozen_layers:]:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)

    # Optional custom classification head.
    for units in extra_dense_layers:
        x = Dense(units, activation="relu")(x)
        x = Dropout(0.3)(x)

    # Output layer producing emotion class probabilities.
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs)