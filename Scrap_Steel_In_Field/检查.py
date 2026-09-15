# -*- coding: utf-8 -*-
"""
check_data.py
用于检查三号选手传来的矩阵数据是否完整、无错误
支持 CSV 和 JSON 两种格式
"""

import os
import csv
import json
import sys

# ============ 配置区：根据你实际的数据文件修改 ============
DATA_FILE = "test_suite/transform_matrix.json"   # 如果是 json，改成 data/matrix.json
# ========================================================


def check_csv(path):
    """检查 CSV 数据"""
    if not os.path.exists(path):
        print(f"[错误] 文件不存在：{path}")
        return False

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 0:
        print("[错误] CSV 文件为空")
        return False

    header = rows[0]
    data_rows = rows[1:]
    total = len(data_rows)

    # 统计缺失值
    empty_count = 0
    for row in data_rows:
        for cell in row:
            if cell is None or str(cell).strip() == "":
                empty_count += 1

    # 统计列数是否一致
    col_mismatch = sum(1 for row in data_rows if len(row) != len(header))

    print("=" * 40)
    print(f"文件路径   ：{path}")
    print(f"表头字段   ：{header}")
    print(f"数据行数   ：{total}")
    print(f"缺失值数量 ：{empty_count}")
    print(f"列数不一致 ：{col_mismatch}")
    print("-" * 40)

    if empty_count == 0 and col_mismatch == 0:
        print("格式校验：通过 ✅")
        print("数据完整无错误")
        return True
    else:
        print("格式校验：不通过 ❌")
        print("请检查缺失值或列数不一致的问题")
        return False


def check_json(path):
    """检查 JSON 数据"""
    if not os.path.exists(path):
        print(f"[错误] 文件不存在：{path}")
        return False

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[错误] JSON 解析失败：{e}")
        return False

    if isinstance(data, list):
        total = len(data)
        empty_count = sum(
            1 for item in data
            if isinstance(item, dict) and any(v in ("", None) for v in item.values())
        )
    elif isinstance(data, dict):
        total = len(data)
        empty_count = sum(1 for v in data.values() if v in ("", None))
    else:
        print("[错误] JSON 顶层结构不是 list 或 dict")
        return False

    print("=" * 40)
    print(f"文件路径   ：{path}")
    print(f"数据条数   ：{total}")
    print(f"缺失值数量 ：{empty_count}")
    print("-" * 40)

    if empty_count == 0:
        print("格式校验：通过 ✅")
        print("数据完整无错误")
        return True
    else:
        print("格式校验：不通过 ❌")
        return False


def main():
    path = DATA_FILE
    print(f"开始检查数据文件：{path}\n")

    if path.lower().endswith(".csv"):
        ok = check_csv(path)
    elif path.lower().endswith(".json"):
        ok = check_json(path)
    else:
        print("[错误] 仅支持 .csv 或 .json 文件")
        sys.exit(1)

    print("=" * 40)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()