import cv2
import mss
import numpy as np
import pywinctl as gw
import sys
from typing import Sequence

TARGET_TITLE = 'RetroArch'
TITLE_BAR_H = 75 # to są parametry dla mojego monitora 4k, normalnie trochę mniej, możńa zrocić mapę tego w zależnośći od res / OS
BORDER = 14
IMSHOW_WIN_NAME = "Frame view"

def get_window():
    win_lst = gw.getWindowsWithTitle(TARGET_TITLE, condition=gw.Re.CONTAINS, flags=gw.Re.IGNORECASE)
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

def get_bounding_rect(image: cv2.typing.MatLike, threshold: int = 10) -> Sequence[int]:
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

def main() -> None:
    win = get_window()
    if win is None:
        print("No RetroArch window!", file=sys.stderr)
        return

    
    win_bbox = (0,0,0,0) # trigger recalc in first frame
    size_dict: dict[str, int]
    bounding_rect: cv2.typing.Rect

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

    with mss.MSS() as sct:
        try:
            while True:
                im = cropped_image(sct)
                cv2.imshow(IMSHOW_WIN_NAME, im)
                key = cv2.waitKey(1)

        except KeyboardInterrupt:
            print("End by user input")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()