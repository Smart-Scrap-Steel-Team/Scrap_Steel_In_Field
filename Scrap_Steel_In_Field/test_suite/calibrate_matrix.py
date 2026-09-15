import cv2
import numpy as np
import json
import os
import sys
from datetime import datetime  # 确保导入了 datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CAMERA_CONFIG, RECTANGLE_DETECT_PARAMS
from camera import Camera
from vision import detect_yellow_rectangle

# ... 此处省略 get_real_world_points 函数，保持不变 ...
def get_real_world_points():
    print("=" * 60)
    print("       Calibration - Real World Coordinates")
    print("=" * 60)
    print("\nPlease enter the real-world coordinates of the rectangle corners.")
    print("Coordinates should be in millimeters (mm) or centimeters (cm).")
    print("\nCorner order:")
    print("  1 = Top-Left (左上)")
    print("  2 = Top-Right (右上)")
    print("  3 = Bottom-Right (右下)")
    print("  4 = Bottom-Left (左下)")
    print("\n" + "-" * 60)

    world_points = []
    for i in range(4):
        while True:
            try:
                coord = input(f"  Corner {i + 1} (x,y): ").strip()
                x_str, y_str = coord.replace(' ', '').split(',')
                x = float(x_str)
                y = float(y_str)
                world_points.append((x, y))
                break
            except KeyboardInterrupt:
                print("\nAborting...")
                return None
            except Exception:
                print("    Invalid format! Please enter like: 100, 50")

    print("\n" + "-" * 60)
    print("You entered:")
    for i, (x, y) in enumerate(world_points):
        print(f"  Corner {i + 1}: ({x}, {y})")

    confirm = input("\nConfirm? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return None

    return np.array(world_points, dtype=np.float32)


def main():
    print("=" * 60)
    print("      Transform Matrix Calibration Tool")
    print("=" * 60)

    print("\nStep 1: Connecting to camera...")
    cam = Camera(
        CAMERA_CONFIG['camera_ip'],
        CAMERA_CONFIG['username'],
        CAMERA_CONFIG['password']
    )

    print("\nStep 2: Capture image and detect yellow rectangle")
    print("Press [SPACE] to capture when ready, [q] to quit")

    # 提前设置好窗口属性，避免画面闪烁
    window_name = "Calibration Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)  # 强制将窗口大小固定为 800x600

    detected_corners = None
    frame = None

    while True:
        frame = cam.get_snapshot(CAMERA_CONFIG['channel'])
        if frame is None:
            frame = cam.get_snapshot_rtsp(CAMERA_CONFIG['channel'], CAMERA_CONFIG['rtsp_path'])
            if frame is None:
                continue

        params = {
            'hsv_lower': np.array(RECTANGLE_DETECT_PARAMS['hsv_lower']),
            'hsv_upper': np.array(RECTANGLE_DETECT_PARAMS['hsv_upper']),
            'min_area': RECTANGLE_DETECT_PARAMS['min_area']
        }

        corners, mask = detect_yellow_rectangle(frame, params)

        display_frame = frame.copy()

        if corners is not None:
            cv2.drawContours(display_frame, [corners.reshape(-1, 1, 2)], -1, (0, 255, 0), 3)

            # 【修改重点】根据实际检测顺序调整标签269.7,
            # 检测顺序：左上 -> 左下 -> 右下 -> 右上
            corner_names = ["1:Top-Left","2:Top-Right", "3:Bot-Right","4:Bot-Left"]

            for i, (x, y) in enumerate(corners):
                cv2.circle(display_frame, (int(x), int(y)), 12, (0, 0, 255), -1)
                label = corner_names[i]
                cv2.putText(display_frame, label, (int(x) - 40, int(y) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.putText(display_frame, "DETECTED - Press SPACE!", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            cv2.putText(display_frame, "NO RECTANGLE FOUND", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # 之前的 scale 缩放可以去掉了，因为窗口已经强制固定大小，图片会自动适应
        # scale = 0.6
        # display_frame = cv2.resize(display_frame, ...)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(200) & 0xFF
        if key == ord('q'):
            print("\nAborted.")
            cv2.destroyAllWindows()
            return
        elif key == ord(' '):
            if corners is not None:
                detected_corners = corners
                print("\nRectangle detected and captured!")

                # ========== 保存有效图片和像素坐标 ==========
                save_dir = "calibration_data"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_save_path = os.path.join(save_dir, f"capture_{timestamp}.jpg")
                txt_save_path = os.path.join(save_dir, f"capture_{timestamp}.txt")

                cv2.imwrite(img_save_path, frame)
                print(f"✅ 图片已保存至: {img_save_path}")

                with open(txt_save_path, 'w', encoding='utf-8') as f:
                    f.write("Detected Pixel Corners (1:Top-Left, 2:Top-Right, 3:Bottom-Right, 4:Bottom-Left):\n")
                    for i, (x, y) in enumerate(detected_corners):
                        f.write(f"Corner {i + 1}: ({x:.1f}, {y:.1f})\n")
                print(f"✅ 像素坐标已保存至: {txt_save_path}")
                # ==============================================

                break
            else:
                print("\nNo rectangle detected! Try again.")

    cv2.destroyAllWindows()

    # ... 后续标定与矩阵计算代码保持不变 ...
    if detected_corners is None:
        print("\nNo corners detected, cannot calibrate.")
        return

    print("\nDetected pixel corners:")
    for i, (x, y) in enumerate(detected_corners):
        print(f"  Corner {i + 1}: ({x:.1f}, {y:.1f})")

    world_points = get_real_world_points()
    if world_points is None:
        return

    print("\n" + "=" * 60)
    print("Step 3: Calculating transform matrix...")
    print("=" * 60)

    pixel_points = np.array(detected_corners, dtype=np.float32)
    transform_matrix, _ = cv2.findHomography(pixel_points, world_points)

    print("\nTransform Matrix:")
    print(transform_matrix)

    matrix_list = transform_matrix.tolist()
    output_path = os.path.join(os.path.dirname(__file__), "transform_matrix.json")
    with open(output_path, 'w') as f:
        json.dump(matrix_list, f, indent=2)

    print("\n" + "=" * 60)
    print("SUCCESS! Transform matrix saved to:")
    print(f"  {output_path}")
    print("=" * 60)

    print("\nTesting the transform:")
    for i, (px, py) in enumerate(detected_corners):
        px_h = np.array([px, py, 1.0])
        trans = np.dot(transform_matrix, px_h)
        wx, wy = trans[:2] / trans[2]
        print(
            f"  Pixel ({px:.1f}, {py:.1f}) -> World ({wx:.1f}, {wy:.1f}) (Expected: {world_points[i][0]:.1f}, {world_points[i][1]:.1f})")

    print("\nCalibration complete!")


if __name__ == "__main__":
    main()