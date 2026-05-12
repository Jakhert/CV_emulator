import cv2
import os
import glob


def slice_digits():
    input_folder = "screenshots"
    output_folder = "digits"

    digit_width = 48
    right_margin = 16
    num_digits = 6

    pixel_compensation = -2

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = glob.glob(os.path.join(input_folder, "shot_*.png"))

    if not images:
        print(f"Files not found in {input_folder}")
        return

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None: continue

        left_offset = 16
        img = img[:, left_offset:]

        h, w = img.shape[:2]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        for i in range(num_digits):

            shift = i * (digit_width + pixel_compensation)

            x_end = w - right_margin - shift
            x_start = x_end - digit_width

            if x_start < 0:
                break

            digit = img[0:h, x_start:x_end]

            save_path = os.path.join(output_folder, f"{img_name}_pos_{num_digits-i}.png")
            cv2.imwrite(save_path, digit)

    print("Digits saved.")


if __name__ == "__main__":
    slice_digits()