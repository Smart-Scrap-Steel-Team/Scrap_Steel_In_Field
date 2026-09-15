import sys
import os
import cv2
import numpy as np
import json
import socket
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    CAMERA_CONFIG,
    RECTANGLE_DETECT_PARAMS,
    ARM_CONFIG,
    SAVE_CONFIG
)
from camera import Camera
from vision import (
    detect_yellow_rectangle,
    transform_corners_to_world,
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


def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def start_arm_server(grasp_points):
    print("\n" + "=" * 60)
    print("      Arm Server - Waiting for Connection")
    print("=" * 60)
    print("Port: " + str(ARM_CONFIG['server_port']))

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind(('0.0.0.0', ARM_CONFIG['server_port']))
        server_socket.listen(1)
        print("\nWaiting for arm to connect...")
        print("You can also run 'python test_4_arm.py' in another terminal as client simulator!")

        client_sock, addr = server_socket.accept()
        print("\nConnected from: " + str(addr))

        print("\nSending: START")
        client_sock.sendall("START".encode())

        while True:
            try:
                client_sock.settimeout(1.0)
                data = client_sock.recv(1024)

                if not data:
                    print("\nArm disconnected!")
                    break

                msg = data.decode('utf-8', errors='ignore').strip()
                print("\nReceived: [" + msg + "]")

                if msg == "yes_grab":
                    print("\nSending grasp points:")
                    for i, (x, y) in enumerate(grasp_points):
                        send_msg = "{:.1f},{:.1f},0;".format(x, y)
                        time.sleep(0.5)
                        try:
                            client_sock.sendall(send_msg.encode())
                            print("  Sent point " + str(i + 1) + ": " + send_msg)
                        except Exception as e:
                            print("  Send error: " + str(e))
                            break

                    print("\nAll points sent!")
                    print("\nTo test again, have arm send 'yes_grab'")

            except socket.timeout:
                continue

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print("\nError: " + str(e))
        import traceback
        traceback.print_exc()
    finally:
        if 'client_sock' in locals():
            client_sock.close()
        server_socket.close()


def main():
    print("=" * 70)
    print("            FULL REAL SCENARIO TEST")
    print("=" * 70)

    ensure_dirs()

    local_ip = get_local_ip()
    print("\nYour local IP: " + local_ip)
    print("Make sure arm is configured to connect to: " + local_ip + ":" + str(ARM_CONFIG['server_port']))

    print("\n" + "-" * 70)
    print("Step 1: Connecting to camera...")

    try:
        cam = Camera(
            CAMERA_CONFIG['camera_ip'],
            CAMERA_CONFIG['username'],
            CAMERA_CONFIG['password']
        )
        print("Camera connected OK!")
    except Exception as e:
        print("Failed to connect to camera: " + str(e))
        return

    print("\n" + "-" * 70)
    print("Step 2: Checking transform matrix...")

    matrix_path = os.path.join(os.path.dirname(__file__), "transform_matrix.json")
    if not os.path.exists(matrix_path):
        print("ERROR: transform_matrix.json not found!")
        print("Please run calibrate_matrix.py first.")
        return

    try:
        with open(matrix_path, 'r') as f:
            matrix_list = json.load(f)
        transform_matrix = np.array(matrix_list)
        print("Transform matrix loaded OK!")
        print(transform_matrix)
    except Exception as e:
        print("ERROR loading matrix: " + str(e))
        return

    print("\n" + "-" * 70)
    print("Step 3: Live detection - press SPACE to capture & process")
    print("        press q to quit")
    print("\n" + "-" * 70)

    captured_frame = None
    captured_corners = None

    try:
        while True:
            frame = cam.get_snapshot(CAMERA_CONFIG['channel'])
            if frame is None:
                frame = cam.get_snapshot_rtsp(CAMERA_CONFIG['channel'], CAMERA_CONFIG['rtsp_path'])
                if frame is None:
                    print("Waiting for camera...")
                    time.sleep(0.5)
                    continue

            params = {
                'hsv_lower': np.array(RECTANGLE_DETECT_PARAMS['hsv_lower']),
                'hsv_upper': np.array(RECTANGLE_DETECT_PARAMS['hsv_upper']),
                'min_area': RECTANGLE_DETECT_PARAMS['min_area']
            }

            corners, mask = detect_yellow_rectangle(frame, params)

            display_frame = frame.copy()

            status_text = "DETECTED - Press SPACE!"
            status_color = (0, 255, 0)

            if corners is not None:
                cv2.drawContours(display_frame, [corners.reshape(-1, 1, 2)], -1, (0, 255, 0), 3)
                for i, (x, y) in enumerate(corners):
                    cv2.circle(display_frame, (int(x), int(y)), 12, (0, 0, 255), -1)
                    cv2.putText(display_frame, str(i + 1), (int(x) - 15, int(y) - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            else:
                status_text = "NO RECTANGLE FOUND"
                status_color = (0, 0, 255)

            cv2.putText(display_frame, status_text, (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)

            window_name = "Real Test - Detecting..."
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 必须先创建窗口

            # --- 关键修改：强制设置像素大小 ---
            # 这里的 (400, 300) 可以根据你想要的大小随意修改
            cv2.resizeWindow(window_name, 400, 300)
            # -------------------------------------

            # 为了防止图片在小窗口里变形，我们简单缩放一下图片
            # 如果不缩放，图片会自动适应窗口但可能被拉伸
            display_resized = cv2.resize(display_frame, (400, 300), interpolation=cv2.INTER_AREA)

            cv2.imshow(window_name, display_resized)

            key = cv2.waitKey(200) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                if corners is not None:
                    captured_frame = frame.copy()
                    captured_corners = corners
                    print("\nRectangle captured!")
                    cv2.destroyAllWindows()
                    break
                else:
                    print("\nNo rectangle detected! Try again.")

        if captured_corners is None:
            return

        print("\n" + "-" * 70)
        print("Step 4: Processing captured frame...")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        original_dir, result_dir = ensure_dirs()
        orig_path = os.path.join(original_dir, "real_test_" + timestamp + ".jpg")
        cv2.imwrite(orig_path, captured_frame)
        print("Original image saved: " + orig_path)

        print("\nDetected pixel corners:")
        for i, (x, y) in enumerate(captured_corners):
            print("  " + str(i + 1) + ". (" + str(round(x, 1)) + "," + str(round(y, 1)) + ")")

        print("\nTransforming to world coordinates...")
        world_corners = transform_corners_to_world(captured_corners, matrix_path)

        if world_corners is None:
            print("ERROR: Transform failed!")
            return

        print("\nWorld coordinates:")
        for i, (x, y) in enumerate(world_corners):
            print("  " + str(i + 1) + ". (" + str(round(x, 1)) + "," + str(round(y, 1)) + ")")

        print("\nCalculating grasp points...")
        grasp_points = calculate_grasp_points(world_corners)

        if grasp_points is None:
            print("ERROR: Failed to calculate grasp points!")
            return

        print("\nGrasp points (25%, 50%, 75%):")
        labels = ["25%", "50%", "75%"]
        for i, (x, y) in enumerate(grasp_points):
            print("  " + labels[i] + ": (" + str(round(x, 1)) + "," + str(round(y, 1)) + ")")

        result_frame = captured_frame.copy()
        cv2.drawContours(result_frame, [captured_corners.reshape(-1, 1, 2)], -1, (0, 255, 0), 3)
        for i, (x, y) in enumerate(captured_corners):
            cv2.circle(result_frame, (int(x), int(y)), 12, (0, 0, 255), -1)
            cv2.putText(result_frame, str(i + 1), (int(x) - 15, int(y) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        result_path = os.path.join(result_dir, "real_test_" + timestamp + ".jpg")
        cv2.imwrite(result_path, result_frame)
        print("\nResult image saved: " + result_path)

        print("\n" + "-" * 70)
        print("Step 5: Ready to send to arm!")
        print("\nMake sure:")
        print("  1. Arm is ready to connect")
        print("  2. Arm is configured to connect to: " + local_ip + ":" + str(ARM_CONFIG['server_port']))
        print("\nGrasp points to send:")
        for i, (x, y) in enumerate(grasp_points):
            print("  Point " + str(i + 1) + ": " + str(round(x, 1)) + "," + str(round(y, 1)))

        confirm = input("\nStart arm communication? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return

        start_arm_server(grasp_points)

    except KeyboardInterrupt:
        print("\nInterrupted!")
    except Exception as e:
        print("\nERROR: " + str(e))
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
