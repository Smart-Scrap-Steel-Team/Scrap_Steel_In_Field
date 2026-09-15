
import sys
import os
import cv2
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CAMERA_CONFIG, RECTANGLE_DETECT_PARAMS, SAVE_CONFIG
from camera import Camera
from vision import (
    RectangleDetector,
    CoordinateTransformer,
    calculate_grasp_points
)


def ensure_dirs():
    dirs = [
        os.path.join(SAVE_CONFIG['base_dir'], SAVE_CONFIG['sub_dirs']['original']),
        os.path.join(SAVE_CONFIG['base_dir'], SAVE_CONFIG['sub_dirs']['result'])
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def draw_info_panel(image, corners, world_corners, grasp_points):
    panel_height = 300
    panel = np.ones((panel_height, image.shape[1], 3), dtype=np.uint8) * 30

    y_pos = 30
    cv2.putText(panel, "=== Pixel Coordinates ===", (30, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    y_pos += 30

    if corners is not None:
        for i, (x, y) in enumerate(corners):
            text = "  Corner " + str(i+1) + ": (" + str(x) + "," + str(y) + ")"
            cv2.putText(panel, text, (50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y_pos += 25
    else:
        cv2.putText(panel, "  No rectangle detected", (50, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,100,255), 1)
        y_pos += 25

    y_pos += 10
    cv2.putText(panel, "=== World Coordinates ===", (30, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
    y_pos += 30

    if world_corners is not None:
        for i, (x, y) in enumerate(world_corners):
            text = "  Corner " + str(i+1) + ": (" + str(round(x,1)) + "," + str(round(y,1)) + ")"
            cv2.putText(panel, text, (50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y_pos += 25
    else:
        cv2.putText(panel, "  No transform matrix or corners", (50, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,100,255), 1)
        y_pos += 25

    y_pos += 10
    cv2.putText(panel, "=== Grasp Points ===", (30, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    y_pos += 30

    if grasp_points is not None:
        labels = ["25%", "50%", "75%"]
        for i, (x, y) in enumerate(grasp_points):
            text = "  " + labels[i] + ": (" + str(round(x,1)) + "," + str(round(y,1)) + ")"
            cv2.putText(panel, text, (50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y_pos += 25

    return panel


def main():
    print("="*60)
    print("  Coordinate Transform & Grasp Points (Live)")
    print("="*60)

    _, save_result_dir = ensure_dirs()

    try:
        print("\nConnecting to camera:", CAMERA_CONFIG['camera_ip'])

        cam = Camera(
            CAMERA_CONFIG['camera_ip'],
            CAMERA_CONFIG['username'],
            CAMERA_CONFIG['password']
        )

        detector = RectangleDetector(
            hsv_lower=RECTANGLE_DETECT_PARAMS['hsv_lower'],
            hsv_upper=RECTANGLE_DETECT_PARAMS['hsv_upper'],
            min_area=RECTANGLE_DETECT_PARAMS['min_area']
        )

        transformer = CoordinateTransformer("transform_matrix.json")

        if transformer.is_ready():
            print("Transform matrix loaded")
        else:
            print("Warning: No transform matrix found")

        print("\nControls:")
        print("  [SPACE] - Save result")
        print("  [q] - Quit")
        print("\nStarting...")

        while True:
            frame = cam.get_snapshot(CAMERA_CONFIG['channel'])
            if frame is None:
                frame = cam.get_snapshot_rtsp(CAMERA_CONFIG['channel'], CAMERA_CONFIG['rtsp_path'])
                if frame is None:
                    continue

            corners, mask = detector.detect(frame)

            world_corners = None
            grasp_points = None
            if corners is not None and transformer.is_ready():
                world_corners = transformer.transform_corners(corners)
                if world_corners is not None:
                    grasp_points = calculate_grasp_points(world_corners)

            result_img = frame.copy()

            if corners is not None:
                cv2.drawContours(result_img, [corners.reshape(-1,1,2)], -1, (0,255,0), 3)
                for i, (x, y) in enumerate(corners):
                    color = (0, 0, 255) if i % 2 == 0 else (255, 0, 0)
                    cv2.circle(result_img, (int(x), int(y)), 12, color, -1)
                    cv2.putText(result_img, str(i+1), (int(x)-15, int(y)-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 3)

            info_panel = draw_info_panel(result_img, corners, world_corners, grasp_points)

            scale = 0.6
            result_scaled = cv2.resize(result_img, (int(result_img.shape[1]*scale), int(result_img.shape[0]*scale)))
            info_panel_scaled = cv2.resize(info_panel, (result_scaled.shape[1], int(info_panel.shape[0]*scale)))
            full_display = np.vstack([result_scaled, info_panel_scaled])

            cv2.imshow("Coordinate Transform", full_display)

            key = cv2.waitKey(200) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord(' '):
                if corners is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(save_result_dir, "coordinate_" + timestamp + ".jpg")
                    cv2.imwrite(save_path, result_img)
                    print("\nSaved:", save_path)
                    if grasp_points:
                        print("  Grasp points:")
                        labels = ["25%", "50%", "75%"]
                        for i, (x, y) in enumerate(grasp_points):
                            print("   ", labels[i], ":", round(x,1), ",", round(y,1))
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

