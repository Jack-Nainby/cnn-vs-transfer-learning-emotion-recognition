# Defines the target image resolution used throughout the project.
# All facial images are resized to 48x48 pixels before being passed into the models.

IMAGE_RESCALE = (48, 48)

# Defines the input tensor shape expected by the models.
# The first two dimensions represent image height and width,
# while the third dimension represents the RGB colour channels.

INPUT_SHAPE = (IMAGE_RESCALE[0], IMAGE_RESCALE[1], 3)

# Stores the names of the seven facial emotion classes present within FER2013.

EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# Calculates the number of output classes required by the final classification layer of each model.

NUM_CLASSES = len(EMOTION_LABELS)

# Collection of random seeds used to initialise training runs.
# Different seeds allow model performance to be evaluated across multiple independent runs while maintaining reproducibility.
# Current max num_runs is limited to the number of seeds.

SEEDS = [
    42, 123, 456, 789, 1024,
    2048, 4096, 8192, 1111, 2222,
    3333, 4444, 5555, 6666, 7777,
    8888, 9999, 1357, 2468, 9876
]

# Maps the numerical FER2013 emotion identifiers to their corresponding human-readable emotion labels.

EMOTION_LOOKUP = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Defines the smallest batch size allowed when recovering from GPU OOM errors during training.

MIN_BATCH_SIZE = 4