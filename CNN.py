# train_model.py

import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# =========================================================
# KONFIGURACJA
# =========================================================

DATA_DIR = "digits"
MODEL_PATH = "retro_digits_model.keras"

IMG_HEIGHT = 32
IMG_WIDTH = 32
BATCH_SIZE = 16

SEED = 123
#===============================
# STRUKTURA FOLDERÓW:
#
# digits/
# ├── train/
# │   ├── 0/
# │   ├── 1/
# │   └── ...
# │
# ├── val/
# │   ├── 0/
# │   ├── 1/
# │   └── ...
# │
# └── test/
#     ├── 0/
#     ├── 1/
#     └── ...
#
# =========================================================

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# =========================================================
# ŁADOWANIE DANYCH
# =========================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)

print(f"\nWykryte klasy: {class_names}")

# =========================================================
# WYDAJNOŚĆ
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# =========================================================
# AUGMENTACJA
# =========================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.05),
    ]
)

# =========================================================
# MODEL
# =========================================================

model = models.Sequential(
    [
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

        layers.Rescaling(1.0 / 255),

        data_augmentation,

        # CNN
        layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),

        layers.Dense(64, activation="relu"),

        layers.Dropout(0.5),

        layers.Dense(num_classes, activation="softmax"),
    ]
)

# =========================================================
# KOMPILACJA
# =========================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# =========================================================
# CALLBACK
# =========================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

# =========================================================
# TRENING
# =========================================================

EPOCHS = 50

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping],
)

# =========================================================
# EWALUACJA
# =========================================================
print("Walidacja:")

test_loss, test_acc = model.evaluate(test_ds)

print(f"\nTest accuracy: {test_acc * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

# =========================================================
# WYKRESY
# =========================================================

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label="Train accuracy")
plt.plot(epochs_range, val_acc, label="Validation accuracy")

plt.legend(loc="lower right")
plt.title("Accuracy")

# Loss
plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label="Train loss")
plt.plot(epochs_range, val_loss, label="Validation loss")

plt.legend(loc="upper right")
plt.title("Loss")

plt.tight_layout()
plt.show()

# =========================================================
# ZAPIS MODELU
# =========================================================

model.save(MODEL_PATH)

print(f"\nModel zapisano jako: {MODEL_PATH}")