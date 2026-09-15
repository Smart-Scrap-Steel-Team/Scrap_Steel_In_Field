import sys
import os
import cv2
import numpy as np
import json
import socket
import threading
import time

# 获取当前脚本所在的目录 (假设脚本在 test_suite 文件夹内)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 将当前目录添加到系统路径，以便导入 config 和 vision
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from config import ARM_CONFIG
from vision import (
    detect_yellow_rectangle,
    _sort_corners,
    transform_corners_to_world,
    calculate_grasp_points
)


def load_image_from_file():
    print("=" * 60)
    print("    Offline Arm Test - Load Image")
    print("=" * 60)

    # ==========================================
    # 修改点 1: 指定具体的图片路径
    # ==========================================
    img_filename = "capture_20260519_162048.jpg"
    img_path = os.path.join(CURRENT_DIR, "calibration_data", img_filename)

    print(f"\nAttempting to load specific image: {img_path}")

    if not os.path.exists(img_path):
        print(f"\nError: Image not found at {img_path}!")
        print("Please check if the file exists.")
        return None

    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image (file might be corrupted or not an image).")
        return None

    print("Image loaded successfully.")
    return img


def process_image_for_arm(img):
    print("\n" + "=" * 60)
    print("  Processing Image for Arm")
    print("=" * 60)

    from config import RECTANGLE_DETECT_PARAMS

    params = {
        'hsv_lower': np.array(RECTANGLE_DETECT_PARAMS['hsv_lower']),
        'hsv_upper': np.array(RECTANGLE_DETECT_PARAMS['hsv_upper']),
        'min_area': RECTANGLE_DETECT_PARAMS['min_area']
    }

    corners, mask = detect_yellow_rectangle(img, params)

    if corners is None:
        print("\nError: No yellow rectangle detected in image!")
        print("You can manually enter pixel coordinates.")

        choice = input("\nManually enter corners? (y/n): ").strip().lower()
        if choice != 'y':
            return None

        corners = []
        print("\nEnter 4 corners (pixel x,y):")
        for i in range(4):
            while True:
                coord = input(f"  Corner {i + 1}: ").strip()
                try:
                    x_str, y_str = coord.replace(' ', '').split(',')
                    x = float(x_str)
                    y = float(y_str)
                    corners.append((x, y))
                    break
                except:
                    print("  Invalid! Use format: x,y")

        corners = np.array(corners)
        corners = _sort_corners(corners)

    print("\nDetected/Entered pixel corners:")
    for i, (x, y) in enumerate(corners):
        print(f"  {i + 1}. ({x:.1f}, {y:.1f})")

    # ==========================================
    # 修改点 2: 指定具体的 Matrix 路径
    # ==========================================
    matrix_path = os.path.join(CURRENT_DIR, "transform_matrix.json")

    if not os.path.exists(matrix_path):
        print(f"\nError: Transform matrix not found at {matrix_path}!")
        return None

    world_corners = transform_corners_to_world(corners, matrix_path)

    if world_corners is None:
        print("\nError: Transform failed!")
        choice = input("\nUse pixel coordinates directly? (y/n): ").strip().lower()
        if choice != 'y':
            return None
        world_corners = [(float(x), float(y)) for x, y in corners]

    print("\nWorld coordinates:")
    for i, (x, y) in enumerate(world_corners):
        print(f"  {i + 1}. ({x:.1f}, {y:.1f})")

    grasp_points = calculate_grasp_points(world_corners)

    if grasp_points is None:
        print("\nError: Failed to calculate grasp points!")
        return None

    print("\nCalculated grasp points:")
    labels = ["25%", "50%", "75%"]
    for i, (x, y) in enumerate(grasp_points):
        print(f"  {labels[i]}: ({x:.1f}, {y:.1f})")

    return grasp_points


def start_arm_server(grasp_points):
    print("\n" + "=" * 60)
    print("      Arm Server - Waiting for Connection")
    print("=" * 60)
    print(f"Port: {ARM_CONFIG['server_port']}")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind(('0.0.0.0', ARM_CONFIG['server_port']))
        server_socket.listen(1)
        print("\nWaiting for arm to connect...")
        print("\nYou can also run 'python test_4_arm.py' in another terminal as client simulator!")

        client_sock, addr = server_socket.accept()
        print(f"\nConnected from: {addr}")

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
                print(f"\nReceived: [{msg}]")

                if msg == "yes_grab":
                    print("\nSending grasp points:")
                    for i, (x, y) in enumerate(grasp_points):
                        send_msg = f"{x:.1f},{y:.1f},0;"
                        time.sleep(0.5)
                        try:
                            client_sock.sendall(send_msg.encode())
                            print(f"  Sent point {i + 1}: {send_msg}")
                        except Exception as e:
                            print(f"  Send error: {e}")
                            break

                    print("\nAll points sent!")
                    print("\nTo test again, have arm send 'yes_grab'")
                else:
                    print(f"  Got ack: {msg}")

            except socket.timeout:
                continue

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'client_sock' in locals():
            client_sock.close()
        server_socket.close()


def main():
    img = load_image_from_file()

    if img is None:
        return

    grasp_points = process_image_for_arm(img)

    if grasp_points is None:
        return

    print("\n" + "=" * 60)
    print("Start arm server to send these points?")
    print("=" * 60)

    choice = input("Start server? (y/n, default=y): ").strip().lower()
    if choice == 'n':
        print("\nDone.")
        return

    start_arm_server(grasp_points)


if __name__ == "__main__":
    main()