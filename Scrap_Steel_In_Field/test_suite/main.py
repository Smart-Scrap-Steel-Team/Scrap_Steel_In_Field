
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    print("\n" + "="*60)
    print("       废钢装箱系统 - 测试套件")
    print("="*60)


def print_menu():
    print("\n请选择测试项目:\n")
    print("  [1] 摄像头测试 - 实时预览+拍照")
    print("  [2] 矩形检测测试 - 黄色矩形检测可视化")
    print("  [3] 坐标转换测试 - 像素坐标→世界坐标+抓取点")
    print("  [4] 机械臂测试 - 机械臂连接和通讯")
    print("  [5] 完整工作流 - 拍照-&gt;检测-&gt;抓取全流程")
    print("\n  [q] 退出")


def run_test(test_num):
    script_name = f"test_{test_num}"
    if test_num == 1:
        script_name += "_camera.py"
    elif test_num == 2:
        script_name += "_detection.py"
    elif test_num == 3:
        script_name += "_coordinate.py"
    elif test_num == 4:
        script_name += "_arm.py"
    elif test_num == 5:
        script_name += "_full.py"
    else:
        print("无效选择!")
        return

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)

    if not os.path.exists(script_path):
        print(f"脚本不存在: {script_path}")
        return

    print(f"\n启动测试脚本: {script_name}\n")
    print("-"*60 + "\n")

    # 运行脚本
    import subprocess
    subprocess.run([sys.executable, script_path])

    print("\n" + "-"*60)
    print("\n测试结束，按 Enter 返回菜单...")
    input()


def main():
    while True:
        print_banner()
        print_menu()

        choice = input("\n请选择 (1-5/q): ").strip().lower()

        if choice == 'q':
            print("\n再见!\n")
            break
        elif choice in ['1', '2', '3', '4', '5']:
            run_test(int(choice))
        else:
            print("\n无效选择，请重试!")


if __name__ == "__main__":
    main()

