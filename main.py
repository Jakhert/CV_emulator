import mss
import pygetwindow as gw
import cv2
import numpy as np
import time
import os


def capture_loop(interval=30):
    target_title = 'RetroArch'

    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')

    print(f"Loop started. Capturing every {interval}s. Ctrl + C to stop.")

    try:
        while True:
            all_windows = gw.getAllTitles()
            ra_windows = [w for w in all_windows if target_title in w]

            if not ra_windows:
                print(f"Window not found: '{target_title}'. Waiting...")
                time.sleep(5)
                continue

            win = gw.getWindowsWithTitle(ra_windows[0])[0]

            if win.isMinimized:
                print("Window minimized, skipping frame.")
                time.sleep(interval)
                continue

            region = {
                "top": win.top,
                "left": win.left,
                "width": win.width,
                "height": win.height
            }

            with mss.mss() as sct:
                screenshot = sct.grab(region)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                h, w = img.shape[:2]

                roi_procenty = {"y_start": 0.17, "y_end": 0.21, "x_start": 0.65, "x_end": 0.84}
                y1, y2 = int(h * roi_procenty["y_start"]), int(h * roi_procenty["y_end"])
                x1, x2 = int(w * roi_procenty["x_start"]), int(w * roi_procenty["x_end"])

                wynik_img = img[y1:y2, x1:x2]

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"screenshots/shot_{timestamp}.png"

                cv2.imwrite(filename, wynik_img)
                print(f"Saved: {filename} (Window size: {w}x{h})")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nLoop stopped via user input.")


if __name__ == "__main__":
    capture_loop(interval=30)