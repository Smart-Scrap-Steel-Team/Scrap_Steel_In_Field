
import sys
import os
import cv2
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    CAMERA_CONFIG,
    RECTANGLE_DETECT_PARAMS,
    ARM_CONFIG,
    SAVE_CONFIG
)
from camera import Camera
from vision import (
    RectangleDetector,
    CoordinateTransformer,
    calculate_grasp_points
)
from arm_control import ArmController


def ensure_dirs():
    dirs = [
        os.path.join(SAVE_CONFIG['base_dir'], SAVE_CONFIG['sub_dirs']['original']),
        os.path.join(SAVE_CONFIG['base_dir'], SAVE_CONFIG['sub_dirs']['result'])
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def save_result_images(frame, detection_img, corners, world_corners, grasp_points):
    save_original, save_result = ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    original_path = os.path.join(save_original, "full_orig_" + timestamp + ".jpg")
    result_path = os.path.join(save_result, "full_result_" + timestamp + ".jpg")

    cv2.imwrite(original_path, frame)

    y_offset = 50
    for i, (x, y) in enumerate(grasp_points):
        label = "Gr" + str(i+1) + ": " + str(round(x,1)) + "," + str(round(y,1))
        cv2.putText(detection_img, label, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_offset += 30

    cv2.imwrite(result_path, detection_img)
    return original_path, result_path


def main():
    print("="*60)
    print("      Full Workflow Test")
    print("="*60)

    try:
        print("\n[Step 1/5] Initializing modules...")

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

        arm = ArmController(
            host=ARM_CONFIG['server_ip'],
            port=ARM_CONFIG['server_port']
        )

        print("Modules initialized")

        print("\n[Step 2/5] Waiting for arm to connect...")
        print("Configure arm to connect to: 192.168.3.193:8770")

        arm.start_server()

        if not arm.is_connected():
            print("Arm connection timeout!")
            return

        print("Arm connected")

        while True:
            print("\n" + "-"*60)
            print("Ready, waiting for 'yes_grab' from arm")
            print("Press Ctrl+C to exit")
            print("-"*60)

            while True:
                msg = arm.get_message(timeout=5)
                if msg == "yes_grab":
                    print("\nReceived yes_grab! Starting...")
                    break
                elif msg:
                    print("Received message:", msg)

            print("\n[Step 3/5] Capturing image...")
            frame = cam.get_snapshot(CAMERA_CONFIG['channel'])
            if frame is None:
                frame = cam.get_snapshot_rtsp(CAMERA_CONFIG['channel'], CAMERA_CONFIG['rtsp_path'])
                if frame is None:
                    print("Capture failed!")
                    continue
            print("Capture done")

            print("\n[Step 4/5] Detecting and calculating...")
            corners, mask = detector.detect(frame)

            if corners is None:
                print("No rectangle detected!")
                continue

            print("Detected", len(corners), "corners")

            world_corners = None
            grasp_points = None

            if transformer.is_ready():
                world_corners = transformer.transform_corners(corners)
                if world_corners:
                    grasp_points = calculate_grasp_points(world_corners)
            else:
                print("Warning: No transform matrix, using dummy points")
                grasp_points = [(100, 200), (300, 400), (500, 600)]

            if not grasp_points:
                print("Grasp points calculation failed!")
                continue

            print("Grasp points:")
            labels = ["25%", "50%", "75%"]
            for i, (x, y) in enumerate(grasp_points):
                print("  ", labels[i], ":", round(x,1), ",", round(y,1))

            detection_img = frame.copy()
            cv2.drawContours(detection_img, [corners.reshape(-1,1,2)], -1, (0,255,0), 3)
            for i, (x, y) in enumerate(corners):
                cv2.circle(detection_img, (int(x), int(y)), 10, (0,0,255), -1)

            original_path, result_path = save_result_images(frame, detection_img, corners, world_corners, grasp_points)
            print("Results saved")
            print("  Original:", original_path)
            print("  Result:", result_path)

            print("\n[Step 5/5] Sending grasp points to arm...")

            confirm_chars = ['A', 'B', 'C']

            success = True
            for i, (x, y) in enumerate(grasp_points):
                print("  Point", i+1, ":", round(x,1), ",", round(y,1), "...")
                arm.send_grasp_point(x, y, 0)

                wait_start = time.time()
                confirmed = False
                while time.time() - wait_start < 30:
                    msg = arm.get_message(timeout=0.5)
                    if msg == confirm_chars[i]:
                        print("    Confirmed:", confirm_chars[i])
                        confirmed = True
                        break
                    elif msg:
                        print("    Other message:", msg)

                if not confirmed:
                    print("    Timeout waiting for", confirm_chars[i])
                    success = False
                    break

            if success:
                print("\n" + "="*60)
                print("  Workflow completed successfully!")
                print("="*60)

            print("\nWaiting for next cycle...")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print("\nError:", e)
        import traceback
        traceback.print_exc()
    finally:
        try:
            arm.stop()
        except:
            pass


if __name__ == "__main__":
    main()

