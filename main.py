import tensorflow as tf
from tensorflow import keras
from tensorflow.tools.docs.doc_controls import T
from model import model
import argparse

image_size = (180, 180)
batch_size = 32
parser = argparse.ArgumentParser(description='Cat vs Dog classification')

parser.add_argument('--batch_size', type=int, help='batch size', default=32)
parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--input_height',type=int, help='input height', default=180)
parser.add_argument('--input_width', type=int, help='input width', default=180)
parser.add_argument('--num_epochs', type=int, help='number of epochs', default=30)
parser.add_argument('--log_directory', type=str, help='directory to save checkpoints and summaries', default='saved_models')
parser.add_argument('--checkpoint_path', type=str,   help='path to a specific checkpoint to load', default='')

args = parser.parse_args()

def training_data(image_size, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "PetImages",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    return train_ds

def validation_data(image_size, batch_size):
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "PetImages",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    return val_ds


if __name__ == "__main__":

    image_size = (args.input_height, args.input_width)

    if args.mode == "train":
        model = model(input_shape=image_size + (3,), num_classes=2)
        model = model.create_model()
        train_ds = training_data(image_size, args.batch_size)
        val_ds = validation_data(image_size, args.batch_size)
        train_ds = train_ds.prefetch(buffer_size=args.batch_size)
        val_ds = val_ds.prefetch(buffer_size=args.batch_size)
        callbacks = [
            keras.callbacks.ModelCheckpoint(args.log_directory+"/"+"save_at_{epoch}.h5", save_best_only=True),
        ]
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds, epochs=args.num_epochs, callbacks=callbacks, validation_data=val_ds,
        )
    
    if args.mode == "test":
        val_ds = validation_data(image_size, args.batch_size)
        model = keras.models.load_model(args.checkpoint_path)
        model.evaluate(val_ds)