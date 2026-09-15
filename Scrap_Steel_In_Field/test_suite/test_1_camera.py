
import sys
import os
import cv2
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CAMERA_CONFIG, SAVE_CONFIG
from camera import Camera


def ensure_dir():
    save_dir = os.path.join(SAVE_CONFIG['base_dir'], SAVE_CONFIG['sub_dirs']['original'])
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def main():
    print("="*60)
    print("       Camera Test (Live Preview + Capture)")
    print("="*60)

    save_dir = ensure_dir()

    try:
        print("\nConnecting to camera:", CAMERA_CONFIG['camera_ip'])
        print("Channel:", CAMERA_CONFIG['channel'])

        cam = Camera(
            CAMERA_CONFIG['camera_ip'],
            CAMERA_CONFIG['username'],
            CAMERA_CONFIG['password']
        )

        print("\nControls:")
        print("  [SPACE] - Save capture")
        print("  [q] - Quit")
        print("\nTrying to connect...")

        test_frame = cam.get_snapshot(CAMERA_CONFIG['channel'])
        if test_frame is None:
            test_frame = cam.get_snapshot_rtsp(CAMERA_CONFIG['channel'], CAMERA_CONFIG['rtsp_path'])
            if test_frame is None:
                print("\nCannot connect to camera!")
                return

        print("\nCamera connected!")
        print("Showing live video...")

        window_name = "Camera - SPACE to save, q to quit"

        while True:
            try:
                frame = cam.get_snapshot(CAMERA_CONFIG['channel'])
                if frame is None:
                    frame = cam.get_snapshot_rtsp(CAMERA_CONFIG['channel'], CAMERA_CONFIG['rtsp_path'])
                    if frame is None:
                        print("Frame capture failed, retrying...")
                        continue

                h, w = frame.shape[:2]
                scale = 1.0
                if w > 800:
                    scale = 800 / w
                display_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

                info_text = "Camera: " + CAMERA_CONFIG['camera_ip'] + " | Ch:" + str(CAMERA_CONFIG['channel'])
                cv2.putText(display_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(200) & 0xFF

                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord(' '):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(save_dir, "capture_" + timestamp + ".jpg")
                    cv2.imwrite(save_path, frame)
                    print("\nSaved:", save_path)

            except Exception as e:
                print("\nError:", e)
                continue

    except Exception as e:
        print("\nError:", e)
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

