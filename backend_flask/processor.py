# backend_flask/processor.py

import math
import uuid
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import mediapipe as mp

# -------------- 路径与全局对象 -----------------

BASE_DIR = Path(__file__).resolve().parent
CROP_DIR = BASE_DIR / "static" / "crops"
CROP_DIR.mkdir(parents=True, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 常用 landmark 索引（MediaPipe FaceMesh）
LEFT_EYE_IDX = [33, 133, 159, 145]
RIGHT_EYE_IDX = [362, 263, 386, 374]
LEFT_IRIS_IDX = [468, 469, 470, 471]
RIGHT_IRIS_IDX = [473, 474, 475, 476]


# -------------- 辅助函数 -----------------


def _landmark_xy(landmarks, idx, width, height):
    lm = landmarks[idx]
    return lm.x * width, lm.y * height


def _eye_center(landmarks, indices, width, height):
    xs, ys = [], []
    for i in indices:
        x, y = _landmark_xy(landmarks, i, width, height)
        xs.append(x)
        ys.append(y)
    return float(np.mean(xs)), float(np.mean(ys))


def _iris_center(landmarks, indices, width, height):
    xs, ys = [], []
    for i in indices:
        x, y = _landmark_xy(landmarks, i, width, height)
        xs.append(x)
        ys.append(y)
    return float(np.mean(xs)), float(np.mean(ys))


def classify_direction(dx: float, dy: float) -> str:
    """
    根据 dx, dy 判定 9 宫格方向。
    dx < 0: 向左看；dx > 0: 向右看
    dy < 0: 向上看；dy > 0: 向下看

    这里把“中心区域”收紧一点，尤其是竖直方向，
    让轻微的上/下视更容易被分类为 up / down。
    """
    # 中心区域阈值（可根据临床反馈再调）
    horiz_center_th = 0.25   # 左右中心：|dx| < 0.25
    vert_center_th = 0.18    # 上下中心：|dy| < 0.18（比左右略紧）

    # 水平判定：left / center / right
    if dx < -horiz_center_th:
        horiz = "left"
    elif dx > horiz_center_th:
        horiz = "right"
    else:
        horiz = "center"

    # 垂直判定：up / center / down
    if dy < -vert_center_th:
        vert = "up"
    elif dy > vert_center_th:
        vert = "down"
    else:
        vert = "center"

    # 组合成 9 宫格方向
    if horiz == "left" and vert == "up":
        return "upLeft"
    if horiz == "center" and vert == "up":
        return "up"
    if horiz == "right" and vert == "up":
        return "upRight"
    if horiz == "left" and vert == "center":
        return "left"
    if horiz == "center" and vert == "center":
        return "center"
    if horiz == "right" and vert == "center":
        return "right"
    if horiz == "left" and vert == "down":
        return "downLeft"
    if horiz == "center" and vert == "down":
        return "down"
    if horiz == "right" and vert == "down":
        return "downRight"

    return "center"


def compute_confidence(dx: float, dy: float, direction: str) -> float:
    """
    根据“偏离中心”的距离估算一个 0.5 ~ 1.0 的置信度。
    dx, dy 已经在 [-1, 1]。
    """
    dist = math.sqrt(dx * dx + dy * dy)  # 0 ~ sqrt(2)
    conf = 0.5 + 0.5 * min(dist / 1.2, 1.0)
    return float(conf)


def _crop_eye_region(image_bgr: np.ndarray, landmarks) -> np.ndarray:
    """
    根据双眼相关的 landmark，裁剪出包含双眼的区域（BGR）。
    """
    h, w, _ = image_bgr.shape
    idx_all = LEFT_EYE_IDX + RIGHT_EYE_IDX + LEFT_IRIS_IDX + RIGHT_IRIS_IDX

    xs = []
    ys = []
    for i in idx_all:
        x, y = _landmark_xy(landmarks, i, w, h)
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        # 回退：直接裁剪整张脸的大致中间区域
        x_min, y_min = int(w * 0.2), int(h * 0.2)
        x_max, y_max = int(w * 0.8), int(h * 0.8)
    else:
        x_min = max(int(min(xs) - 0.05 * w), 0)
        x_max = min(int(max(xs) + 0.05 * w), w - 1)
        y_min = max(int(min(ys) - 0.05 * h), 0)
        y_max = min(int(max(ys) + 0.05 * h), h - 1)

    if x_max <= x_min or y_max <= y_min:
        return image_bgr

    crop = image_bgr[y_min:y_max, x_min:x_max]
    return crop


def _compute_gaze_vector(landmarks, width: int, height: int):
    """
    通过左右眼的中心和虹膜中心，估算视线偏移向量 (dx, dy)，并做归一化。
    """
    # 左眼
    le_center_x, le_center_y = _eye_center(landmarks, LEFT_EYE_IDX, width, height)
    le_iris_x, le_iris_y = _iris_center(landmarks, LEFT_IRIS_IDX, width, height)

    # 右眼
    re_center_x, re_center_y = _eye_center(landmarks, RIGHT_EYE_IDX, width, height)
    re_iris_x, re_iris_y = _iris_center(landmarks, RIGHT_IRIS_IDX, width, height)

    # 眼睛宽高，用于归一化
    le_w = abs(
        _landmark_xy(landmarks, LEFT_EYE_IDX[0], width, height)[0]
        - _landmark_xy(landmarks, LEFT_EYE_IDX[1], width, height)[0]
    )
    le_h = abs(
        _landmark_xy(landmarks, LEFT_EYE_IDX[2], width, height)[1]
        - _landmark_xy(landmarks, LEFT_EYE_IDX[3], width, height)[1]
    )
    re_w = abs(
        _landmark_xy(landmarks, RIGHT_EYE_IDX[0], width, height)[0]
        - _landmark_xy(landmarks, RIGHT_EYE_IDX[1], width, height)[0]
    )
    re_h = abs(
        _landmark_xy(landmarks, RIGHT_EYE_IDX[2], width, height)[1]
        - _landmark_xy(landmarks, RIGHT_EYE_IDX[3], width, height)[1]
    )

    # 防止除零
    le_w = le_w if le_w > 1e-3 else 1.0
    le_h = le_h if le_h > 1e-3 else 1.0
    re_w = re_w if re_w > 1e-3 else 1.0
    re_h = re_h if re_h > 1e-3 else 1.0

    # 对每只眼睛分别计算偏移（iris 相对 eye center）
    le_dx = (le_iris_x - le_center_x) / le_w
    le_dy = (le_iris_y - le_center_y) / le_h
    re_dx = (re_iris_x - re_center_x) / re_w
    re_dy = (re_iris_y - re_center_y) / re_h

    # 双眼取平均
    dx = float((le_dx + re_dx) / 2.0)
    dy = float((le_dy + re_dy) / 2.0)

    # 限制在 [-1, 1] 区间，便于后续阈值判断
    dx = max(min(dx, 1.0), -1.0)
    dy = max(min(dy, 1.0), -1.0)

    return dx, dy


# 映射到 9 宫格 slot 名称，方便前端直接使用
# （和前端 B 区格子 key 完全一致）
VALID_SLOTS = {
    "upLeft",
    "up",
    "upRight",
    "left",
    "center",
    "right",
    "downLeft",
    "down",
    "downRight",
}


def process_images(images: List[np.ndarray]) -> List[Dict[str, Any]]:
    """
    核心对外接口：
    - 参数 images: List[np.ndarray]，每个元素是 BGR 格式的图像（app.py 已经完成解码）
    - 返回: List[Dict]，每个 dict 形如：
        {
          "slot": "upLeft" | ... | "downRight",
          "crop_url": "/static/crops/xxxx.jpg" 或 None,
          "confidence": float,
          "dx": float,
          "dy": float,
          "error": str 或 None
        }
    app.py 会把这个 list 包装成 JSON {"results": [...]} 返回给前端。
    """
    results: List[Dict[str, Any]] = []

    for idx, img_bgr in enumerate(images):
        try:
            if img_bgr is None or img_bgr.size == 0:
                raise ValueError("EmptyImage")

            h, w, _ = img_bgr.shape
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            mp_result = mp_face_mesh.process(img_rgb)
            if not mp_result.multi_face_landmarks:
                raise ValueError("NoFaceDetected")

            landmarks = mp_result.multi_face_landmarks[0].landmark

            # 1) 计算 gaze 向量
            dx, dy = _compute_gaze_vector(landmarks, w, h)

            # 2) 方向 + 置信度
            direction = classify_direction(dx, dy)
            confidence = compute_confidence(dx, dy, direction)

            if direction not in VALID_SLOTS:
                direction = "center"

            # 3) 裁剪双眼区域
            crop_bgr = _crop_eye_region(img_bgr, landmarks)
            crop_bgr = cv2.resize(crop_bgr, (512, 512))

            # 4) 保存裁剪图到 static/crops 下
            filename = f"crop_{idx}_{uuid.uuid4().hex}.jpg"
            save_path = CROP_DIR / filename
            cv2.imwrite(str(save_path), crop_bgr)

            crop_url = f"/static/crops/{filename}"

            results.append(
                {
                    "slot": direction,
                    "crop_url": crop_url,
                    "confidence": confidence,
                    "dx": dx,
                    "dy": dy,
                    "error": None,
                }
            )
        except Exception as e:
            # 出错时也返回一条记录，方便前端显示“处理失败”的信息
            results.append(
                {
                    "slot": "center",
                    "crop_url": None,
                    "confidence": 0.0,
                    "dx": 0.0,
                    "dy": 0.0,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    return results