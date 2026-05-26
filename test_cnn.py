# evaluate_model.py

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img, img_to_array

# =========================================================
# KONFIGURACJA
# =========================================================

MODEL_PATH = "retro_digits_model.keras"

TEST_DIR = "digits/test"

IMG_HEIGHT = 32
IMG_WIDTH = 32

NUM_SAMPLES = 12

# =========================================================
# ŁADOWANIE MODELU
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Nie znaleziono modelu: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("Model załadowany poprawnie.")

# =========================================================
# KLASY
# =========================================================

class_names = sorted(
    [
        d
        for d in os.listdir(TEST_DIR)
        if os.path.isdir(os.path.join(TEST_DIR, d))
    ]
)

print(f"Wykryte klasy: {class_names}")

all_images = []

for root, dirs, files in os.walk(TEST_DIR):

    for file in files:

        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            full_path = os.path.join(root, file)

            true_label = os.path.basename(root)

            all_images.append((full_path, true_label))

if len(all_images) == 0:
    raise ValueError("Brak obrazów testowych.")

# =========================================================
# PREDYKCJE
# =========================================================

print("\n=================================================")
print("PREDYKCJE NA NIEZALEŻNYM TEST SET")
print("=================================================")

print(f"{'Plik':<40} | {'Prawda':<8} | {'Predykcja':<10} | {'Pewność'}")
print("-" * 85)

correct_images = []
wrong_images = []

for img_path, true_label in all_images:

    # =====================================================
    # PREPROCESSING
    # =====================================================

    img = load_img(
        img_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="grayscale",
    )

    img_array = img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    # =====================================================
    # PREDYKCJA
    # =====================================================

    predictions = model.predict(img_array, verbose=0)

    predicted_class_idx = np.argmax(predictions[0])

    predicted_label = class_names[predicted_class_idx]

    confidence = 100 * np.max(predictions[0])

    # =====================================================
    # STATYSTYKI
    # =====================================================

    is_correct = predicted_label == true_label

    # =====================================================
    # LOG
    # =====================================================

    short_name = os.path.basename(img_path)

    print(
        f"{short_name:<40} | "
        f"{true_label:<8} | "
        f"{predicted_label:<10} | "
        f"{confidence:.2f}%"
    )

    if is_correct:
        correct_images.append((img_path, true_label, predicted_label))
    else:
        wrong_images.append((img_path, true_label, predicted_label))

# =========================================================
# PODSUMOWANIE
# =========================================================

accuracy = len(correct_images) / len(all_images) * 100

print("-" * 85)
print(f"Accuracy na całym zbiorze testowym: {accuracy:.2f}%")
print(f"Poprawne: {len(correct_images)}, Błędne: {len(wrong_images)}")


# =====================================================
# WIZUALIZACJA
# =====================================================

def show_results(image_list, title_prefix, num_to_show=12):
    if not image_list:
        print(f"Brak obrazów do wyświetlenia dla: {title_prefix}")
        return

    random.shuffle(image_list)
    subset = image_list[:num_to_show]
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"{title_prefix} (pierwsze {len(subset)})")

    for i, (img_path, true, pred) in enumerate(subset):
        plt.subplot(3, 4, i + 1)
        display_img = load_img(img_path)
        plt.imshow(display_img)
        plt.title(f"T: {true}\nP: {pred}", color="green" if true == pred else "red")
        plt.axis("off")
    plt.tight_layout()


show_results(correct_images, "Poprawne Predykcje")
show_results(wrong_images, "Błędne Predykcje")

plt.show()