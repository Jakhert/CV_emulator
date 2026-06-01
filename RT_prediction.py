import os
import sys
from typing import Sequence

import cv2
import mss
import numpy as np
import pywinctl as gw
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

TARGET_TITLE = "RetroArch"

TITLE_BAR_H = 65
BORDER = 12

IMSHOW_WIN_NAME = "Frame view"
IMSHOW_SCORE_WIN_NAME = "Score view"

# x1, y1, x2, y2
SCORE_POS = (0.68, 0.105, 0.942, 0.145)

DIGIT_WIDTH = 35
RIGHT_MARGIN = 0
NUM_DIGITS = 7
PIXEL_COMPENSATION = -2
LEFT_OFFSET = 16

MODEL_PATH = "retro_digits_model.keras"
TEST_DIR = "digits/test"
IMG_HEIGHT = 32
IMG_WIDTH = 32

CONFIDENCE_THRESHOLD = 80.0


def get_window():
    win_lst = gw.getWindowsWithTitle(
        TARGET_TITLE,
        condition=gw.Re.CONTAINS,
        flags=gw.Re.IGNORECASE
    )

    if len(win_lst) > 0:
        return win_lst[0]
    else:
        return None


def get_size_dict(win):
    return {
        "top": win.top + TITLE_BAR_H,
        "left": win.left + BORDER,
        "width": win.width - (BORDER * 2),
        "height": win.height - TITLE_BAR_H - BORDER
    }


def get_bounding_rect(
    image: cv2.typing.MatLike,
    threshold: int = 10
) -> Sequence[int]:

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rows_with_pixels = np.any(gray > threshold, axis=1)
    cols_with_pixels = np.any(gray > threshold, axis=0)

    non_empty_rows = np.where(rows_with_pixels)[0]
    non_empty_cols = np.where(cols_with_pixels)[0]

    if len(non_empty_rows) == 0 or len(non_empty_cols) == 0:
        return 0, image.shape[0], 0, image.shape[1]

    y1, y2 = non_empty_rows[0], non_empty_rows[-1] + 1
    x1, x2 = non_empty_cols[0], non_empty_cols[-1] + 1

    return y1, y2, x1, x2


def load_class_names(test_dir: str) -> list[str]:
    if os.path.isdir(test_dir):
        class_names = sorted(
            [
                d for d in os.listdir(test_dir)
                if os.path.isdir(os.path.join(test_dir, d))
            ]
        )
        if class_names:
            return class_names

    return [str(i) for i in range(10)]


def preprocess_digit(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    img_array = cv2.imread  # tylko żeby nie dać przypadkowo zły typ w edytorze
    img_array = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=-1)  # (32, 32, 1)
    return img_array


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}", file=sys.stderr)
        return

    print("Ładowanie modelu...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model załadowany.")

    class_names = load_class_names(TEST_DIR)
    print(f"Klasy: {class_names}")

    win = get_window()
    if win is None:
        print("No RetroArch window!", file=sys.stderr)
        return

    win_bbox = (0, 0, 0, 0)

    size_dict: dict[str, int]
    bounding_rect: cv2.typing.Rect

    last_score_str = ""

    def cropped_image(sct: mss.MSS) -> cv2.typing.MatLike:
        nonlocal win_bbox
        nonlocal size_dict
        nonlocal bounding_rect

        recalc_bounding_rect: bool = False

        # handle move / resize
        if win_bbox != win.bbox:
            win_bbox = win.bbox
            recalc_bounding_rect = True
            size_dict = get_size_dict(win)

        screenshot = sct.grab(size_dict)
        frame_bgra = np.array(screenshot)

        if recalc_bounding_rect:
            bounding_rect = get_bounding_rect(frame_bgra)

        y1, y2, x1, x2 = bounding_rect
        cropped_bgra = frame_bgra[y1:y2, x1:x2]

        return cv2.cvtColor(cropped_bgra, cv2.COLOR_BGRA2BGR)

    with mss.mss() as sct:
        try:
            while True:
                im = cropped_image(sct)

                cv2.imshow(IMSHOW_WIN_NAME, im)

                height, width = im.shape[:2]

                # wycinanie score area
                x1 = int(SCORE_POS[0] * width)
                y1 = int(SCORE_POS[1] * height)
                x2 = int(SCORE_POS[2] * width)
                y2 = int(SCORE_POS[3] * height)

                score_img = im[y1:y2, x1:x2]
                score_display = score_img.copy()

                # offset z lewej
                score_offset = score_img[:, LEFT_OFFSET:]
                sh, sw = score_offset.shape[:2]

                digit_crops = []
                digit_boxes = []

                # rysowanie ramek digitów
                for i in range(NUM_DIGITS):
                    shift = i * (DIGIT_WIDTH + PIXEL_COMPENSATION)

                    x_end = sw - RIGHT_MARGIN - shift
                    x_start = x_end - DIGIT_WIDTH

                    if x_start < 0:
                        break

                    digit_bgr = score_offset[0:sh, x_start:x_end]

                    if digit_bgr.size == 0:
                        continue

                    processed_digit = preprocess_digit(digit_bgr)
                    digit_crops.append(processed_digit)
                    digit_boxes.append((x_start, x_end))

                if digit_crops:
                    batch_array = np.array(digit_crops)
                    predictions = model.predict(batch_array, verbose=0)

                    current_score_chars = []

                    for i, pred in enumerate(predictions):
                        predicted_class_idx = int(np.argmax(pred))
                        predicted_label = class_names[predicted_class_idx]
                        confidence = float(np.max(pred) * 100.0)

                        x_start, x_end = digit_boxes[i]
                        color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)

                        cv2.rectangle(
                            score_display,
                            (LEFT_OFFSET + x_start, 0),
                            (LEFT_OFFSET + x_end, sh),
                            color,
                            2
                        )

                        cv2.putText(
                            score_display,
                            f"{i + 1}",
                            (LEFT_OFFSET + x_start, 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1
                        )

                        # do wyniku bierzemy tylko pewne cyfry
                        if confidence >= CONFIDENCE_THRESHOLD:
                            current_score_chars.append(predicted_label)

                    current_score_chars.reverse()
                    current_score = "".join(current_score_chars)

                    if current_score != last_score_str:
                        if current_score:
                            print(f"Estimated score: {current_score}")
                        else:
                            print("Estimated score: <brak>")
                        last_score_str = current_score

                cv2.imshow(IMSHOW_SCORE_WIN_NAME, score_display)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    break

        except KeyboardInterrupt:
            print("End by user input")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()