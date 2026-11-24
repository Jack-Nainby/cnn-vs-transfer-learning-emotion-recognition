import tensorflow as tf
from src.config import IMAGE_RESCALE

def load_fer_image(pixel_string, label):
    """
    Converts a FER2013 pixel string into a normalised RGB image
    and one-hot encoded emotion label suitable for CNN training.
    """
    # Split the space-separated pixel string into individual values.
    pixels = tf.strings.split(pixel_string)
    
    # Convert string values to floating-point numbers.
    pixels = tf.strings.to_number(pixels, out_type=tf.float32)

    # Reshape flat pixel array into a 48x48 grayscale image.
    pixels = tf.reshape(pixels, (48, 48, 1))

    # Normalise pixel values from [0,255] to [0,1].
    pixels = pixels / 255.0

    # Resize image using dimensions defined in config.py.
    image = tf.image.resize(pixels, IMAGE_RESCALE)

    # Convert grayscale image to RGB for compatibility with transfer learning architectures.
    image = tf.image.grayscale_to_rgb(image)

    # Convert integer emotion class into one-hot encoded vector.
    label = tf.one_hot(label, depth=7)
    
    return image, label

def create_dataset(df, batch_size, shuffle=False):
    """
    Creates an efficient TensorFlow dataset pipeline from a
    FER2013 dataframe.
    """
    # Create a TensorFlow dataset directly from the dataframe.
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            df["pixels"].values,
            df["emotion"].values
        )
    )

    # Apply image preprocessing in parallel.
    dataset = dataset.map(
        load_fer_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Randomise training sample order to reduce learning bias.
    if shuffle:
        dataset = dataset.shuffle(1000)

    # Group samples into mini-batches.
    dataset = dataset.batch(batch_size)

    # Prefetch future batches while the GPU processes the current batch to improve throughput.
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset