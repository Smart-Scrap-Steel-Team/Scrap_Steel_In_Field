import sys
import os
import cv2
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CAMERA_CONFIG, RECTANGLE_DETECT_PARAMS, SAVE_CONFIG
from camera import Camera
from vision import (
    detect_yellow_rectangle,
    visualize_detection,
    transform_corners_to_world
)


def ensure_dirs():
    dirs = [
        os.path.join(SAVE_CONFIG['base_dir'], SAVE_CONFIG['sub_dirs']['original']),
        os.path.join(SAVE_CONFIG['base_dir'], SAVE_CONFIG['sub_dirs']['result'])
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def main():
    print("="*60)
    print("    Yellow Rectangle Detection (Live)")
    print("="*60)

    _, save_result_dir = ensure_dirs()

    try:
        print("\nConnecting to camera:", CAMERA_CONFIG['camera_ip'])

        cam = Camera(
            CAMERA_CONFIG['camera_ip'],
            CAMERA_CONFIG['username'],
            CAMERA_CONFIG['password']
        )

        print("\nControls:")
        print("  [SPACE] - Save detection result")
        print("  [q] - Quit")
        print("\nStarting detection...")

        params = {
            'hsv_lower': np.array(RECTANGLE_DETECT_PARAMS['hsv_lower']),
            'hsv_upper': np.array(RECTANGLE_DETECT_PARAMS['hsv_upper']),
            'min_area': RECTANGLE_DETECT_PARAMS['min_area']
        }

        while True:
            frame = cam.get_snapshot(CAMERA_CONFIG['channel'])
            if frame is None:
                frame = cam.get_snapshot_rtsp(CAMERA_CONFIG['channel'], CAMERA_CONFIG['rtsp_path'])
                if frame is None:
                    continue

            corners, mask = detect_yellow_rectangle(frame, params)

            h, w = frame.shape[:2]

            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            result_img = frame.copy()
            if corners is not None:
                cv2.drawContours(result_img, [corners.reshape(-1,1,2)], -1, (0,255,0), 3)
                for i, (x, y) in enumerate(corners):
                    cv2.circle(result_img, (int(x), int(y)), 12, (0,0,255), -1)
                    cv2.putText(result_img, str(i+1), (int(x)-15, int(y)-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 3)
            else:
                cv2.putText(result_img, "No Rectangle Found", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

            top_row = np.hstack([frame, mask_colored])
            bottom_row = np.hstack([result_img, np.zeros_like(frame)])
            full_display = np.vstack([top_row, bottom_row])

            scale = 0.6
            full_display = cv2.resize(full_display, (int(full_display.shape[1]*scale), int(full_display.shape[0]*scale)))
            window_name = "Rectangle Detection"

            # 1. 创建窗口，并设置标志为 WINDOW_NORMAL（允许调整大小）
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            # 2. 强制设置窗口的初始尺寸（例如宽 1000 像素，高 800 像素）
            cv2.resizeWindow(window_name, 1000, 800)
            cv2.imshow(window_name, full_display)

            key = cv2.waitKey(200) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord(' '):
                if corners is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(save_result_dir, "detection_" + timestamp + ".jpg")
                    cv2.imwrite(save_path, result_img)
                    print("\nSaved:", save_path)
                    print("  Corners detected:", len(corners))
                else:
                    print("\nNo rectangle detected")

    except Exception as e:
        print("\nError:", e)
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
