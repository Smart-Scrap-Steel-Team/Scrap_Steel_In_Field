
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ARM_CONFIG
from arm_control import ArmController, ArmSimulator


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


def test_arm_server():
    print("="*60)
    print("    Arm Server Mode (Wait for real arm)")
    print("="*60)

    local_ip = "192.168.3.193"
    print("\nConfigure arm to connect to:", local_ip + ":" + str(ARM_CONFIG['server_port']))

    arm = ArmController(
        host=ARM_CONFIG['server_ip'],
        port=ARM_CONFIG['server_port']
    )

    try:
        print("\nStarting server...")
        arm.start_server()

        if not arm.is_connected():
            print("\nNo arm connected")
            return

        print("\n" + "="*60)
        print("  Arm connected!")
        print("="*60)

        while True:
            print("\n--- Menu ---")
            print("1. Send test points")
            print("2. Manual single point")
            print("3. Manual 3 points")
            print("4. Check received messages")
            print("q. Quit")

            choice = input("\nChoose: ").strip().lower()

            if choice == 'q':
                break
            elif choice == '1':
                print("\nSending test points...")
                test_points = [(100, 200), (300, 400), (500, 600)]
                for i, (x, y) in enumerate(test_points):
                    print("  Point", i+1, ":", x, ",", y)
                    arm.send_grasp_point(x, y, 0)
                    time.sleep(0.3)
            elif choice == '2':
                xy = input("\nEnter x,y: ").strip()
                try:
                    x_str, y_str = xy.split(',')
                    x = float(x_str.strip())
                    y = float(y_str.strip())
                    print("Sending:", x, ",", y)
                    arm.send_grasp_point(x, y, 0)
                except:
                    print("Invalid format!")
            elif choice == '3':
                points = []
                for i in range(3):
                    try:
                        xy = input("Point " + str(i+1) + " (x,y): ").strip()
                        x_str, y_str = xy.split(',')
                        x = float(x_str.strip())
                        y = float(y_str.strip())
                        points.append((x, y))
                    except:
                        print("Invalid format!")
                        break
                if len(points) == 3:
                    for i, (x, y) in enumerate(points):
                        print("Point", i+1, ":", x, ",", y)
                        arm.send_grasp_point(x, y, 0)
                        time.sleep(0.3)
            elif choice == '4':
                print("\nRecent messages:")
                count = 0
                while count < 10:
                    msg = arm.get_message(timeout=1)
                    if msg:
                        print("  ", count+1, ".", msg)
                    else:
                        break
                    count += 1
                if count == 0:
                    print("  (No messages)")

    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print("\nError:", e)
    finally:
        arm.stop()


def test_arm_simulator():
    print("="*60)
    print("    Arm Simulator Mode")
    print("="*60)
    print("\nFirst run server mode in another window!")

    server_ip = input("\nServer IP (default 127.0.0.1): ").strip() or "127.0.0.1"

    sim = ArmSimulator(server_ip, ARM_CONFIG['server_port'])

    try:
        print("\nConnecting to", server_ip + ":" + str(ARM_CONFIG['server_port']), "...")
        if not sim.connect():
            print("Connection failed!")
            return

        print("\nSimulator connected!")

        print("\n--- Auto Simulation ---")

        print("\n1. Waiting for START...")
        start_time = time.time()
        while time.time() - start_time < 10:
            msg = sim.receive()
            if msg == "START":
                print("Received START")
                break
            time.sleep(0.1)

        print("\n2. Sending yes_grab...")
        sim.send("yes_grab")

        print("\n3. Waiting for grasp points...")
        confirm_chars = ['A', 'B', 'C']

        for i in range(3):
            print("  Waiting point", i+1, "...")
            wait_start = time.time()
            while time.time() - wait_start < 30:
                msg = sim.receive()
                if msg and msg.endswith(';'):
                    print("   Received:", msg)
                    time.sleep(0.5)
                    print("   Sending confirm:", confirm_chars[i])
                    sim.send(confirm_chars[i])
                    break
                time.sleep(0.1)

        print("\n" + "="*60)
        print("  Simulation done!")
        print("="*60)

        print("\n--- Manual Mode ---")
        print("Type message to send, q to quit\n")

        while True:
            user_input = input("> ").strip()

            if user_input.lower() == 'q':
                break

            if user_input:
                sim.send(user_input)

            while True:
                msg = sim.receive()
                if msg:
                    print("Received:", msg)
                else:
                    break

    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print("\nError:", e)
    finally:
        sim.disconnect()


def main():
    print("\n" + "="*60)
    print("      Arm Test")
    print("="*60)
    print("\nChoose mode:")
    print("  1. Server (wait for real arm)")
    print("  2. Simulator (simulate arm)")
    print("  3. Offline mode (use saved image + transform_matrix)")

    choice = input("\nChoose (1-3): ").strip()

    if choice == '1':
        test_arm_server()
    elif choice == '2':
        test_arm_simulator()
    elif choice == '3':
        print("\n" + "-"*60)
        print("Running offline mode...")
        print("-"*60)
        import subprocess
        import sys
        script_path = os.path.join(os.path.dirname(__file__), "test_4_arm_offline.py")
        subprocess.run([sys.executable, script_path])
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()

