from src.models import (
    create_vgg16_model,
    create_resnet50_model,
    create_inceptionv3_model,
    create_Base_CNN,
    create_CNN3,
    create_CNN4,
    create_CNN5
)
import tensorflow as tf

# Transfer learning model configurations.
# Each dictionary entry defines:
# - the model creation function
# - model-specific parameters
# - starting batch size
# - loss function
# - class weighting strategy

TL_MODELS = {
    "VGG16_Frozen": {
        "model_fn": create_vgg16_model,
        "model_kwargs": {
            "unfrozen_layers": 0,
            "extra_dense_layers": [],
            "weights": "imagenet"
        },
        "batch_size": 256,
        "loss": "categorical_crossentropy",
        "class_weight": None
    },

    "VGG16_Unfreeze4": {
        "model_fn": create_vgg16_model,
        "model_kwargs": {
            "unfrozen_layers": 4,
            "extra_dense_layers": [],
            "weights": "imagenet"
        },
        "batch_size": 256,
        "loss": "categorical_crossentropy",
        "class_weight": None
    },

    "VGG16_Unfreeze4_Head": {
        "model_fn": create_vgg16_model,
        "model_kwargs": {
            "unfrozen_layers": 4,
            "extra_dense_layers": [512, 256],
            "weights": "imagenet"
        },
        "batch_size": 256,
        "loss": "categorical_crossentropy",
        "class_weight": None
    },
    
    "ResNet50_Frozen": {
        "model_fn": create_resnet50_model,
        "model_kwargs": {
            "unfrozen_layers": 0,
            "extra_dense_layers": [],
            "weights": "imagenet"
        },
        "batch_size": 256,
        "loss": "categorical_crossentropy",
        "class_weight": None
    },

    "ResNet50_Unfreeze4": {
        "model_fn": create_resnet50_model,
        "model_kwargs": {
            "unfrozen_layers": 4,
            "extra_dense_layers": [],
            "weights": "imagenet"
        },
        "batch_size": 256,
        "loss": "categorical_crossentropy",
        "class_weight": None
    },

    "ResNet50_Unfreeze4_Head": {
        "model_fn": create_resnet50_model,
        "model_kwargs": {
            "unfrozen_layers": 4,
            "extra_dense_layers": [512, 256],
            "weights": "imagenet"
        },
        "batch_size": 128,
        "loss": "categorical_crossentropy",
        "class_weight": None
    },
    
    "InceptionV3_Frozen": {
        "model_fn": create_inceptionv3_model,
        "model_kwargs": {
            "unfrozen_layers": 0,
            "extra_dense_layers": [],
            "weights": "imagenet"
        },
        "batch_size": 128,
        "loss": "categorical_crossentropy",
        "class_weight": None
    },

    "InceptionV3_Unfreeze4": {
        "model_fn": create_inceptionv3_model,
        "model_kwargs": {
            "unfrozen_layers": 4,
            "extra_dense_layers": [],
            "weights": "imagenet"
        },
        "batch_size": 64,
        "loss": "categorical_crossentropy",
        "class_weight": None
    },

    "InceptionV3_Unfreeze4_Head": {
        "model_fn": create_inceptionv3_model,
        "model_kwargs": {
            "unfrozen_layers": 4,
            "extra_dense_layers": [512, 256],
            "weights": "imagenet"
        },
        "batch_size": 64,
        "loss": "categorical_crossentropy",
        "class_weight": None
    }
}

# Custom CNN model configurations.
# Each dictionary entry defines:
# - the model creation function
# - starting batch size
# - loss function
# - class weighting strategy

CNN_MODELS = {
    "Base_CNN_Model": {
        "model_fn": create_Base_CNN,
        "model_kwargs": None,
        "batch_size": 512,
        "loss": "categorical_crossentropy",
        "class_weight": None
    },

    "CNN_Model_Expanded": {
        "model_fn": create_CNN3,
        "model_kwargs": None,
        "batch_size": 256,
        "loss": "categorical_crossentropy",
        "class_weight": None
    },

    "CNN_Model_Augmentation": {
        "model_fn": create_CNN4,
        "model_kwargs": None,
        "batch_size": 512,
        "loss": "categorical_crossentropy",
        "class_weight": None
    },

    "CNN_Model_Residual": {
        "model_fn": create_CNN5,
        "model_kwargs": None,
        "batch_size": 512,
        "loss": "categorical_crossentropy",
        "class_weight": None
    },
    
    "Base_CNN_Model_Weighted": {
        "model_fn": create_Base_CNN,
        "model_kwargs": None,
        "batch_size": 512,
        "loss": "categorical_crossentropy",
        "class_weight": {
            int(0): float(1.0266046844269623),
            int(1): float(9.406618610747051),
            int(2): float(1.0010460615781582),
            int(3): float(0.5684387684387684),
            int(4): float(0.8491274770777877),
            int(5): float(1.293372978330405),
            int(6): float(0.8260394187886635),
        }
    },

    "CNN_Model_Expanded_Weighted": {
        "model_fn": create_CNN3,
        "model_kwargs": None,
        "batch_size": 256,
        "loss": "categorical_crossentropy",
        "class_weight": {
            int(0): float(1.0266046844269623),
            int(1): float(9.406618610747051),
            int(2): float(1.0010460615781582),
            int(3): float(0.5684387684387684),
            int(4): float(0.8491274770777877),
            int(5): float(1.293372978330405),
            int(6): float(0.8260394187886635),
        }
    },

    "CNN_Model_Augmentation_Weighted": {
        "model_fn": create_CNN4,
        "model_kwargs": None,
        "batch_size": 512,
        "loss": "categorical_crossentropy",
        "class_weight": {
            int(0): float(1.0266046844269623),
            int(1): float(9.406618610747051),
            int(2): float(1.0010460615781582),
            int(3): float(0.5684387684387684),
            int(4): float(0.8491274770777877),
            int(5): float(1.293372978330405),
            int(6): float(0.8260394187886635),
        }
    },

    "CNN_Model_Residual_Weighted": {
        "model_fn": create_CNN5,
        "model_kwargs": None,
        "batch_size": 512,
        "loss": "categorical_crossentropy",
        "class_weight": {
            int(0): float(1.0266046844269623),
            int(1): float(9.406618610747051),
            int(2): float(1.0010460615781582),
            int(3): float(0.5684387684387684),
            int(4): float(0.8491274770777877),
            int(5): float(1.293372978330405),
            int(6): float(0.8260394187886635),
        }
    },

    "Base_CNN_Model_fnloss": {
        "model_fn": create_Base_CNN,
        "model_kwargs": None,
        "batch_size": 512,
        "loss": tf.keras.losses.CategoricalFocalCrossentropy(),
        "class_weight": None
    },

    "CNN_Model_Expanded_fnloss": {
        "model_fn": create_CNN3,
        "model_kwargs": None,
        "batch_size": 256,
        "loss": tf.keras.losses.CategoricalFocalCrossentropy(),
        "class_weight": None
    },

    "CNN_Model_Augmentation_fnloss": {
        "model_fn": create_CNN4,
        "model_kwargs": None,
        "batch_size": 512,
        "loss": tf.keras.losses.CategoricalFocalCrossentropy(),
        "class_weight": None
    },

    "CNN_Model_Residual_fnloss": {
        "model_fn": create_CNN5,
        "model_kwargs": None,
        "batch_size": 512,
        "loss": tf.keras.losses.CategoricalFocalCrossentropy(),
        "class_weight": None
    },

    "Base_CNN_Model_Weighted_fnloss": {
        "model_fn": create_Base_CNN,
        "model_kwargs": None,
        "batch_size": 512,
        "loss": tf.keras.losses.CategoricalFocalCrossentropy(),
        "class_weight": {
            int(0): float(1.0266046844269623),
            int(1): float(9.406618610747051),
            int(2): float(1.0010460615781582),
            int(3): float(0.5684387684387684),
            int(4): float(0.8491274770777877),
            int(5): float(1.293372978330405),
            int(6): float(0.8260394187886635),
        }
    },
    
    "CNN_Model_Expanded_Weighted_fnloss": {
        "model_fn": create_CNN3,
        "model_kwargs": None,
        "batch_size": 256,
        "loss": tf.keras.losses.CategoricalFocalCrossentropy(),
        "class_weight": {
            int(0): float(1.0266046844269623),
            int(1): float(9.406618610747051),
            int(2): float(1.0010460615781582),
            int(3): float(0.5684387684387684),
            int(4): float(0.8491274770777877),
            int(5): float(1.293372978330405),
            int(6): float(0.8260394187886635),
        }
    },

    "CNN_Model_Augmentation_Weighted_fnloss": {
        "model_fn": create_CNN4,
        "model_kwargs": None,
        "batch_size": 512,
        "loss": tf.keras.losses.CategoricalFocalCrossentropy(),
        "class_weight": {
            int(0): float(1.0266046844269623),
            int(1): float(9.406618610747051),
            int(2): float(1.0010460615781582),
            int(3): float(0.5684387684387684),
            int(4): float(0.8491274770777877),
            int(5): float(1.293372978330405),
            int(6): float(0.8260394187886635),
        }
    },

    "CNN_Model_Residual_Weighted_fnloss": {
        "model_fn": create_CNN5,
        "model_kwargs": None,
        "batch_size": 512,
        "loss": tf.keras.losses.CategoricalFocalCrossentropy(),
        "class_weight": {
            int(0): float(1.0266046844269623),
            int(1): float(9.406618610747051),
            int(2): float(1.0010460615781582),
            int(3): float(0.5684387684387684),
            int(4): float(0.8491274770777877),
            int(5): float(1.293372978330405),
            int(6): float(0.8260394187886635),
        }
    }
}