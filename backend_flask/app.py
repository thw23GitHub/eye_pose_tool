import io
import base64
import logging
from typing import List, Tuple, Dict, Optional

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from PIL import Image, ImageOps
import numpy as np
import mediapipe as mp

# -------------------------
# Flask 基础设置
# -------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")


# -------------------------
# MediaPipe FaceMesh 初始化
# -------------------------
mp_face_mesh = mp.solutions.face_mesh

# static_image_mode=True：单张图检测
# refine_landmarks=True：包含虹膜关键点（iris）
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# 9 个标准方向 + 对应 B 区 slot 名称
DIRECTION_TO_SLOT = {
    "up_left": "up_left",
    "up": "up",
    "up_right": "up_right",
    "left": "left",
    "center": "center",
    "right": "right",
    "down_left": "down_left",
    "down": "down",
    "down_right": "down_right",
}

SLOT_ORDER = [
    "up_left", "up", "up_right",
    "left", "center", "right",
    "down_left", "down", "down_right",
]


# -------------------------
# 工具函数：从 FaceMesh 提取关键点
# -------------------------
def _landmarks_to_np_array(landmarks, image_width: int, image_height: int) -> np.ndarray:
    """
    将 mediapipe 的 normalized landmark 转成 (N, 2) 像素坐标数组。
    """
    pts = []
    for lm in landmarks:
        x = lm.x * image_width
        y = lm.y * image_height
        pts.append((x, y))
    return np.array(pts, dtype=np.float32)


def _get_eye_iris_centers(landmarks_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    粗略计算左右眼、左右虹膜的中心点。

    FaceMesh 索引说明（基于 mediapipe 官方图）：
    - 左眼虹膜：468~471
    - 右眼虹膜：473~476
    - 左眼轮廓（大致）：[33, 133, 159, 145]
    - 右眼轮廓（大致）：[362, 263, 386, 374]
    """
    iris_left_idx = [468, 469, 470, 471]
    iris_right_idx = [473, 474, 475, 476]
    eye_left_idx = [33, 133, 159, 145]
    eye_right_idx = [362, 263, 386, 374]

    iris_left = landmarks_np[iris_left_idx, :]
    iris_right = landmarks_np[iris_right_idx, :]
    eye_left = landmarks_np[eye_left_idx, :]
    eye_right = landmarks_np[eye_right_idx, :]

    iris_left_center = iris_left.mean(axis=0)
    iris_right_center = iris_right.mean(axis=0)
    eye_left_center = eye_left.mean(axis=0)
    eye_right_center = eye_right.mean(axis=0)

    return (iris_left_center, eye_left_center), (iris_right_center, eye_right_center)


def _estimate_gaze_direction(
        iris_left: np.ndarray,
        eye_left_center: np.ndarray,
        iris_right: np.ndarray,
        eye_right_center: np.ndarray,
        eye_left_box: Tuple[float, float],
        eye_right_box: Tuple[float, float],
        tx: float = 0.25,
        ty: float = 0.25,
) -> str:
    """
    按文档 2.4 的公式做视线方向判定：
      dx = (pupil_x - eye_center_x) / eye_width
      dy = (pupil_y - eye_center_y) / eye_height

    这里左右眼各算一遍，然后取平均 dx, dy 再做 9 宫格分类。
    """
    # 左眼
    eye_left_w, eye_left_h = eye_left_box
    dx_left = (iris_left[0] - eye_left_center[0]) / max(eye_left_w, 1e-5)
    dy_left = (iris_left[1] - eye_left_center[1]) / max(eye_left_h, 1e-5)

    # 右眼
    eye_right_w, eye_right_h = eye_right_box
    dx_right = (iris_right[0] - eye_right_center[0]) / max(eye_right_w, 1e-5)
    dy_right = (iris_right[1] - eye_right_center[1]) / max(eye_right_h, 1e-5)

    dx = (dx_left + dx_right) / 2.0
    dy = (dy_left + dy_right) / 2.0

    # 水平方向
    if dx <= -tx:
        horiz = "left"
    elif dx >= tx:
        horiz = "right"
    else:
        horiz = "center"

    # 垂直方向（注意图像坐标 y 向下增加）
    if dy <= -ty:
        vert = "up"
    elif dy >= ty:
        vert = "down"
    else:
        vert = "center"

    direction = f"{vert}_{horiz}"
    # 特殊处理“center_center”
    if direction == "center_center":
        direction = "center"

    # 映射到 9 宫格标准名称
    if direction == "up_left":
        return "up_left"
    if direction == "up_right":
        return "up_right"
    if direction == "down_left":
        return "down_left"
    if direction == "down_right":
        return "down_right"
    if direction == "up_center":
        return "up"
    if direction == "center_left":
        return "left"
    if direction == "center_right":
        return "right"
    if direction == "down_center":
        return "down"
    if direction == "center":
        return "center"

    # 兜底：当做 center
    return "center"


def _get_eye_boxes(landmarks_np: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    粗略计算左右眼的宽高（像素）。
    用于归一化 dx、dy。
    """
    left_idx = [33, 133, 159, 145]
    right_idx = [362, 263, 386, 374]

    left_pts = landmarks_np[left_idx, :]
    right_pts = landmarks_np[right_idx, :]

    left_x_min, left_y_min = left_pts.min(axis=0)
    left_x_max, left_y_max = left_pts.max(axis=0)
    right_x_min, right_y_min = right_pts.min(axis=0)
    right_x_max, right_y_max = right_pts.max(axis=0)

    left_w = left_x_max - left_x_min
    left_h = left_y_max - left_y_min
    right_w = right_x_max - right_x_min
    right_h = right_y_max - right_y_min

    return (float(left_w), float(left_h)), (float(right_w), float(right_h))


# -------------------------
# 工具函数：根据双眼裁剪 2:1 眼区
# -------------------------
def crop_eye_region_2to1(
        image: Image.Image,
        iris_left_center: np.ndarray,
        iris_right_center: np.ndarray,
) -> Image.Image:
    """
    按文档要求的“2:1 裁剪”：以双眼中点为中心，生成宽:高 = 2:1 的矩形。
    """
    w, h = image.size

    cx = float((iris_left_center[0] + iris_right_center[0]) / 2.0)
    cy = float((iris_left_center[1] + iris_right_center[1]) / 2.0)

    eye_dist = float(np.linalg.norm(iris_right_center - iris_left_center))
    if eye_dist <= 1e-3:
        eye_dist = min(w, h) / 5.0

    box_width = eye_dist * 2.5
    box_height = box_width / 2.0  # 2:1

    top = cy - box_height * 0.55
    bottom = top + box_height
    left = cx - box_width / 2.0
    right = left + box_width

    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)

    if right - left <= 0 or bottom - top <= 0:
        return image.copy()

    return image.crop((left, top, right, bottom))


# -------------------------
# 主处理函数：单张图 -> 方向 + 裁剪图
# -------------------------
def process_single_image(pil_img: Image.Image) -> Dict:
    """
    对单张图片执行：
    - EXIF 方向矫正
    - FaceMesh 检测关键点
    - 根据虹膜 & 眼眶计算眼区裁剪 2:1
    - 根据 pupil/eye_center + Tx/Ty 计算 9 宫格方向
    """
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    w, h = pil_img.size

    img_np = np.array(pil_img)
    results = face_mesh.process(img_np)

    if not results.multi_face_landmarks:
        return {
            "direction": "center",
            "cropped_image": pil_img
        }

    landmarks = results.multi_face_landmarks[0].landmark
    landmarks_np = _landmarks_to_np_array(landmarks, w, h)

    (iris_left_center, eye_left_center), (iris_right_center, eye_right_center) = \
        _get_eye_iris_centers(landmarks_np)

    eye_left_box, eye_right_box = _get_eye_boxes(landmarks_np)

    direction = _estimate_gaze_direction(
        iris_left_center,
        eye_left_center,
        iris_right_center,
        eye_right_center,
        eye_left_box,
        eye_right_box,
        tx=0.25,
        ty=0.25,
    )

    cropped = crop_eye_region_2to1(pil_img, iris_left_center, iris_right_center)

    return {
        "direction": direction,
        "cropped_image": cropped
    }


# -------------------------
# /process_images 路由
# -------------------------
@app.route("/process_images", methods=["POST"])
def process_images():
    content_type = request.content_type
    logger.info("Content-Type from client: %s", content_type)

    if "images" not in request.files:
        return jsonify({"error": "No 'images' field in form-data."}), 400

    files = request.files.getlist("images")
    logger.info("request.files keys: %s", list(request.files.keys()))
    logger.info("Parsed %d images from request", len(files))

    files = files[:9]

    # 1. 逐张处理：方向 + 裁剪图
    processed_items: List[Dict] = []
    for idx, file in enumerate(files):
        filename = secure_filename(file.filename) or f"image_{idx}.jpg"
        try:
            pil_img = Image.open(io.BytesIO(file.read()))
        except Exception:
            logger.exception("读取图片失败：%s", filename)
            continue

        result = process_single_image(pil_img)
        direction = result["direction"]
        cropped = result["cropped_image"]

        logger.info("图像 %d 分类结果: %s", idx, direction)

        processed_items.append({
            "original_index": idx,
            "filename": filename,
            "direction": direction,
            "cropped_image": cropped
        })

    if not processed_items:
        return jsonify({"error": "No valid images processed."}), 400

    # 2. 先按 direction 填充 slot（每种方向取第一张）
    slot_to_item: Dict[str, Optional[Dict]] = {s: None for s in SLOT_ORDER}

    for item in processed_items:
        dir_name = item["direction"]
        slot = DIRECTION_TO_SLOT.get(dir_name, "center")
        if slot_to_item[slot] is None:
            slot_to_item[slot] = item

    # 3. 用剩余的图片补齐所有空的 slot，保证尽量 9 个格子都被占用，
    #    并且每张图片最多只用一次。
    used_indices = {
        v["original_index"] for v in slot_to_item.values() if v is not None
    }
    remaining_items = [
        item for item in processed_items
        if item["original_index"] not in used_indices
    ]
    remaining_iter = iter(remaining_items)

    for slot in SLOT_ORDER:
        if slot_to_item[slot] is None:
            try:
                slot_to_item[slot] = next(remaining_iter)
            except StopIteration:
                # 图片数量 < 9，或有几张检测失败，就允许少于 9 格
                break

    # 4. 构造返回给前端的结果
    grid_order: List[int] = []
    results_for_frontend: List[Dict] = []

    for slot in SLOT_ORDER:
        item = slot_to_item.get(slot)
        if item is None:
            grid_order.append(-1)
            continue

        original_index = item["original_index"]
        grid_order.append(original_index)

        buf = io.BytesIO()
        item["cropped_image"].save(buf, format="JPEG", quality=95)
        img_bytes = buf.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        results_for_frontend.append({
            "slot": slot,
            "image_base64": img_b64
        })

    logger.info("九宫格排序结果(位置->原始索引): %s", grid_order)
    logger.info(
        "Return %d results, %d errors",
        len(results_for_frontend),
        len(files) - len(processed_items),
        )

    return jsonify({"results": results_for_frontend})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)