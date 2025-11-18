import os
from typing import List, Tuple, Dict

OUTPUT_DIR = "output"  # 之后你可以改成更合适的位置

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_images_batch(image_paths: List[str]) -> Tuple[List[Dict], List[str]]:
    """
    批处理多张图片，返回 (results, errors)
    results: List[dict] —— 每个 dict 对应一张图片的 GazeResult JSON
    errors: List[str]   —— 错误信息字符串
    """
    ensure_output_dir()

    results: List[Dict] = []
    errors: List[str] = []

    for path in image_paths:
        try:
            res = process_single_image(path)
            if res is None:
                errors.append(f"ProcessingFailed: {path}")
            else:
                results.append(res)
        except FileNotFoundError:
            errors.append(f"FileMissing: {path}")
        except Exception as e:
            errors.append(f"Exception: {path}: {e}")

    return results, errors

def process_single_image(image_path: str) -> Dict:
    """
    处理单张图片：
    - 这里目前是占位实现：不做真正图像处理，只返回一个 fake 结果
    - 目的：先打通 Flutter <-> Python 的 HTTP 通道和 JSON 结构

    后续会在这里加入：
    - MediaPipe 人脸/眼睛/虹膜检测
    - 眼位几何标准化（双眼连线水平 + 鼻梁居中）
    - 2:1 裁剪
    - dx/dy 计算与九宫格方向分类
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    # TODO: 这里后面替换为真正的图像读取与处理逻辑
    cropped_path = image_path  # 目前先用原图路径占位

    # 先全部当作 "center" 方向，置信度 0.8，方便你在 B 区看到效果
    fake_direction = "center"
    fake_confidence = 0.8
    fake_rotation = 0.0

    result = {
        "original_path": image_path,
        "cropped_path": cropped_path,
        "direction": fake_direction,
        "confidence": float(fake_confidence),
        "rotation_angle": float(fake_rotation),
    }
    return result