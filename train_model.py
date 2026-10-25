from pathlib import Path
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

tf.keras.utils.set_random_seed(42)

root = Path(__file__).resolve().parent
data_dir = root / "data" / "prepared"
results_dir = root / "results"
results_dir.mkdir(exist_ok=True)

def load_dataset(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        str(data_dir / split),
        class_names=["Cat", "Dog"],
        label_mode="binary",
        image_size=(128, 128),
        batch_size=32,
        shuffle=shuffle,
        seed=42,
    ).prefetch(tf.data.AUTOTUNE)

train_data = load_dataset("train", True)
val_data = load_dataset("validation", False)

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1.0 / 255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(results_dir / "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
    ),
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=callbacks,
)

model.save(str(results_dir / "dogs_cats_model.keras"))

with open(results_dir / "history.json", "w") as file:
    json.dump(history.history, file)

with open(results_dir / "class_names.json", "w") as file:
    json.dump(["Cat", "Dog"], file)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for axis, metric in zip(axes, ["accuracy", "loss"]):
    axis.plot(history.history[metric], label="Training")
    axis.plot(history.history[f"val_{metric}"], label="Validation")
    axis.set_title(metric.capitalize())
    axis.set_xlabel("Epoch")
    axis.set_ylabel(metric.capitalize())
    axis.legend()

fig.tight_layout()
fig.savefig(results_dir / "training_graphs.png")
plt.close(fig)

print("\nTraining complete! Files saved in the results folder.")