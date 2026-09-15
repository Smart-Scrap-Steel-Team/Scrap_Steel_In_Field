import os
import subprocess

# 配置你的模型文件名
models = [
    {"pt": "scrap_steel.pt", "type": "yolov8", "imgsz": 640},
    {"pt": "dangerous.pt", "type": "yolov5", "imgsz": 640}
]

def check_onnx_inference(onnx_path):
    try:
        import onnxruntime as ort
        import numpy as np
        # 假设输入名为 'images'，shape 为 [1, 3, 640, 640]
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        dummy = np.random.randn(*[d if isinstance(d, int) else 1 for d in input_shape]).astype(np.float32)
        session.run(None, {input_name: dummy})
        print(f"ONNX inference check passed for {onnx_path}")
    except Exception as e:
        print(f"[Warning] ONNX inference failed for {onnx_path}: {e}")

for m in models:
    pt = os.path.join("models", m["pt"])
    onnx = pt.replace(".pt", ".onnx")
    engine = pt.replace(".pt", ".engine")
    imgsz = m["imgsz"]

    # 1. 导出ONNX
    if m["type"] == "yolov8":
        # yolov8官方导出，使用静态shape
        cmd = f"yolo export model={pt} format=onnx imgsz={imgsz} dynamic=False"
    elif m["type"] == "yolov5":
        # yolov5官方导出，使用静态shape，指定缓存路径
        yolov5_export = r"C:\\Users\\xindao\\.cache\\torch\\hub\\ultralytics_yolov5_master\\export.py"
        cmd = f"python {yolov5_export} --weights {pt} --include onnx --img {imgsz}"
    else:
        raise ValueError("Unknown model type")
    print(f"Exporting {pt} to ONNX...")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Exported {onnx}")
    except subprocess.CalledProcessError as e:
        print(f"[Error] ONNX export failed for {pt}: {e}")
        continue

    # 可选：ONNX推理检查
    try:
        check_onnx_inference(onnx)
    except ImportError:
        print("onnxruntime not installed, skip ONNX inference check.")

    # 2. ONNX → TensorRT engine (FP16)
    print(f"Converting {onnx} to TensorRT engine...")
    trt_cmd = f"trtexec --onnx={onnx} --saveEngine={engine} --fp16"
    try:
        subprocess.run(trt_cmd, shell=True, check=True)
        print(f"Converted {engine}")
    except subprocess.CalledProcessError as e:
        print(f"[Error] TensorRT engine export failed for {onnx}: {e}")
        continue

print("All done.")