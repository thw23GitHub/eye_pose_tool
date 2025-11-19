import base64
import io
import math
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageOps

# -------------------------
# MediaPipe FaceMesh 初始化
# -------------------------
mp_face_mesh = mp.solutions.face_mesh
# 注意：在 serious_python 环境中，模型初始化应在应用启动时一次性完成
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# 常用 landmark 索引（MediaPipe FaceMesh）
LEFT_EYE_IDX = [33, 133, 159, 145]
RIGHT_EYE_IDX = [362, 263, 386, 374]
LEFT_IRIS_IDX = [468, 469, 470, 471]
RIGHT_IRIS_IDX = [473, 474, 475, 476]

# -------------------------
# 工具函数：关键点提取与几何计算
# -------------------------

def _landmark_xy(landmarks: List, idx: int, width: int, height: int) -> Tuple[float, float]:
    """将归一化坐标转换为像素坐标"""
    lm = landmarks[idx]
    return lm.x * width, lm.y * height

def _center_from_indices(landmarks: List, indices: List[int], width: int, height: int) -> Tuple[float, float]:
    """计算一组关键点的中心像素坐标"""
    xs, ys = [], []
    for i in indices:
        x, y = _landmark_xy(landmarks, i, width, height)
        xs.append(x)
        ys.append(y)
    return float(np.mean(xs)), float(np.mean(ys))

def _get_eye_boxes(landmarks: List, width: int, height: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    计算左右眼的外接框宽高 (w, h)，用于归一化 dx, dy。
    """
    LEFT_EYE_FULL_IDX = [33, 133, 159, 145, 144, 163, 7, 173, 161]
    RIGHT_EYE_FULL_IDX = [362, 263, 386, 374, 373, 390, 285, 467, 388]

    left_pts = np.array([_landmark_xy(landmarks, i, width, height) for i in LEFT_EYE_FULL_IDX])
    right_pts = np.array([_landmark_xy(landmarks, i, width, height) for i in RIGHT_EYE_FULL_IDX])

    left_w = left_pts[:, 0].max() - left_pts[:, 0].min()
    left_h = left_pts[:, 1].max() - left_pts[:, 1].min()
    right_w = right_pts[:, 0].max() - right_pts[:, 0].min()
    right_h = right_pts[:, 1].max() - right_pts[:, 1].min()

    return (max(left_w, 1.0), max(left_h, 1.0)), (max(right_w, 1.0), max(right_h, 1.0))

def _get_rotation_angle(le_center: Tuple[float, float], re_center: Tuple[float, float]) -> float:
    """ 计算双眼连线与水平线的夹角 (角度制) """
    angle_rad = math.atan2(re_center[1] - le_center[1], re_center[0] - le_center[0])
    return -angle_rad * 180 / math.pi

def _compute_gaze_vector(landmarks: List, width: int, height: int) -> Tuple[float, float]:
    """ 估算视线偏移向量 (dx, dy)，归一化后在 [-1, 1] 之间。 """
    # 1. 计算中心
    le_center = _center_from_indices(landmarks, LEFT_EYE_IDX, width, height)
    le_iris = _center_from_indices(landmarks, LEFT_IRIS_IDX, width, height)
    re_center = _center_from_indices(landmarks, RIGHT_EYE_IDX, width, height)
    re_iris = _center_from_indices(landmarks, RIGHT_IRIS_IDX, width, height)

    # 2. 获取眼框宽高
    (le_w, le_h), (re_w, re_h) = _get_eye_boxes(landmarks, width, height)

    # 3. 归一化偏移量
    le_dx = (le_iris[0] - le_center[0]) / le_w
    le_dy = (le_iris[1] - le_center[1]) / le_h
    re_dx = (re_iris[0] - re_center[0]) / re_w
    re_dy = (re_iris[1] - re_center[1]) / re_h

    # --- 调试输出：关键原始数据 ---
    print(f"DEBUG_LE: Iris ({le_iris[0]:.2f}, {le_iris[1]:.2f}), Center ({le_center[0]:.2f}, {le_center[1]:.2f})")
    print(f"DEBUG_RE: Iris ({re_iris[0]:.2f}, {re_iris[1]:.2f}), Center ({re_center[0]:.2f}, {re_center[1]:.2f})")
    print(f"DEBUG_LW: {le_w:.2f}, LH: {le_h:.2f} | RW: {re_w:.2f}, RH: {re_h:.2f}")
    print(f"DEBUG_LE_dx: {le_dx:.4f}, LE_dy: {le_dy:.4f}")
    print(f"DEBUG_RE_dx: {re_dx:.4f}, RE_dy: {re_dy:.4f}")
    # ------------------------------

    # 4. 双眼取平均
    dx = float((le_dx + re_dx) / 2.0)
    dy = float((le_dy + re_dy) / 2.0)

    # 限制在 [-1, 1] 区间
    dx = max(min(dx, 1.0), -1.0)
    dy = max(min(dy, 1.0), -1.0)

    return dx, dy

def classify_direction(dx: float, dy: float) -> str:
    horiz_center_th = 0.25
    vert_center_th = 0.18

    # --- 调试输出：最终分类依据 ---
    print(f"DEBUG_FINAL: dx={dx:.4f}, dy={dy:.4f} (Tx={horiz_center_th}, Ty={vert_center_th})")
    # -----------------------------

    if dx < -horiz_center_th:
        horiz = "left"
    elif dx > horiz_center_th:
        horiz = "right"
    else:
        horiz = "center"

    if dy < -vert_center_th:
        vert = "up"
    elif dy > vert_center_th:
        vert = "down"
    else:
        vert = "center"

    direction_map = {
        ("up", "left"): "upLeft",
        ("up", "center"): "up",
        ("up", "right"): "upRight",
        ("center", "left"): "left",
        ("center", "center"): "center",
        ("center", "right"): "right",
        ("down", "left"): "downLeft",
        ("down", "center"): "down",
        ("down", "right"): "downRight",
    }

    final_direction = direction_map.get((vert, horiz), "center")
    print(f"DEBUG_FINAL_DIR: {final_direction}")
    return final_direction


def _crop_eye_region_2to1(
        img_bgr: np.ndarray,
        landmarks: List,
        rotation_angle: float
) -> np.ndarray:
    """
    执行旋正 (Roll Correction) 和严格 2:1 (宽:高) 裁剪。
    """
    h, w, _ = img_bgr.shape

    # 1. 旋正
    (center_x, center_y) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
    rotated_img = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # 2. 确定裁剪中心和尺寸
    le_center = _center_from_indices(landmarks, LEFT_EYE_IDX, w, h)
    re_center = _center_from_indices(landmarks, RIGHT_EYE_IDX, w, h)

    eye_dist = math.sqrt((le_center[0] - re_center[0])**2 + (le_center[1] - re_center[1])**2)

    # 目标裁剪宽度：保证覆盖脸颊 (例如眼距的 4.5 倍)
    target_crop_width = int(eye_dist * 4.5)
    target_crop_width = min(target_crop_width, int(w * 0.9))

    # 严格 2:1 宽高比 (宽:高)
    target_crop_height = target_crop_width // 2

    # 裁剪中心 Y: 眼睛在裁剪高度的 40% 处 (0.4 * H)
    cy_target = int(le_center[1] - (target_crop_height * 0.40))

    # 裁剪区域 (基于旋转后的图片坐标系)
    x1 = int((le_center[0] + re_center[0]) / 2 - target_crop_width // 2)
    y1 = int(cy_target)
    x2 = x1 + target_crop_width
    y2 = y1 + target_crop_height

    # 边界检查
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    crop = rotated_img[y1:y2, x1:x2]

    return crop

# -------------------------
# 主处理函数 (包含调试输出)
# -------------------------

def process_single_image(image_b64: str) -> Dict:
    result = {
        "direction": "center",
        "image_b64": None,
        "confidence": 0.0,
        "dx": 0.0,
        "dy": 0.0,
        "rotation_angle": 0.0,
        "error": None,
    }

    try:
        img_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(img_bytes))

        img_np_rgb = np.array(ImageOps.exif_transpose(pil_img).convert("RGB"))
        img_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
        h, w, _ = img_bgr.shape

        print(f"\n--- DEBUG: New Image WxH: {w}x{h} ---")

        mp_result = face_mesh.process(img_np_rgb)
        if not mp_result.multi_face_landmarks:
            result['error'] = "NoFaceDetected"
            return result

        landmarks = mp_result.multi_face_landmarks[0].landmark

        le_center = _center_from_indices(landmarks, LEFT_EYE_IDX, w, h)
        re_center = _center_from_indices(landmarks, RIGHT_EYE_IDX, w, h)
        rotation_angle = _get_rotation_angle(le_center, re_center)
        result['rotation_angle'] = rotation_angle

        # 裁剪操作（确保 2:1 修正生效）
        cropped_bgr = _crop_eye_region_2to1(img_bgr, landmarks, rotation_angle)

        # 计算 Gaze 向量和方向 (包含调试输出)
        dx, dy = _compute_gaze_vector(landmarks, w, h)
        direction = classify_direction(dx, dy)

        result['direction'] = direction
        result['confidence'] = 0.5 + 0.5 * math.sqrt(dx * dx + dy * dy)
        result['dx'] = float(dx)
        result['dy'] = float(dy)

        # 编码裁剪图为 Base64 (最高质量 Q100)
        success, encoded_image = cv2.imencode(".jpg", cropped_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if not success:
            result['error'] = "CropEncodeFailed"
            return result

        result['image_b64'] = base64.b64encode(encoded_image).decode('utf-8')

    except Exception as e:
        result['error'] = f"ProcessingException: {type(e).__name__}: {str(e)}"

    return result

# --- (process_batch 和 make_mosaic 函数保持不变) ---
def process_batch(image_b64_list: List[str]) -> Dict[str, Any]:
    """ 对外接口 (适配 serious_python)：接收 Base64 列表，返回结果列表。 """
    results: List[Dict] = []
    errors: List[str] = []

    for b64 in image_b64_list:
        res = process_single_image(b64)
        if res['error']:
            errors.append(res['error'])
            results.append(res)
        else:
            results.append(res)

    return {"results": results, "errors": errors}

def make_mosaic(b64_list_9: List[str], gaps: int = 0) -> Dict[str, Any]:
    """ 接收 9 张 Base64 图片 (按 B 区顺序)，拼接成一张完整的 Q100 JPEG 图片。 """
    tiles: List[np.ndarray] = []
    try:
        for b64 in b64_list_9:
            if not b64:
                tile_w, tile_h = 400, 200
                if tiles:
                    tile_h, tile_w = tiles[0].shape[:2]

                blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                blank[:] = (200, 200, 200)
                tiles.append(blank)
                continue

            img_bytes = base64.b64decode(b64)
            img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            tiles.append(img_np)

        if not tiles:
            return {"error": "No valid images decoded for mosaic", "mosaic_b64": None}

        tile_h, tile_w = tiles[0].shape[:2]

        total_w = tile_w * 3 + gaps * 2
        total_h = tile_h * 3 + gaps * 2
        mosaic = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        mosaic[:] = (0, 0, 0)

        idx = 0
        for row in range(3):
            for col in range(3):
                if idx >= len(tiles): break

                dstX0 = col * (tile_w + gaps)
                dstY0 = row * (tile_h + gaps)

                endX = dstX0 + tile_w
                endY = dstY0 + tile_h

                current_tile = cv2.resize(tiles[idx], (tile_w, tile_h))

                mosaic[dstY0:endY, dstX0:endX] = current_tile
                idx += 1

        success, encoded_image = cv2.imencode(".jpg", mosaic, [cv2.IMWRITE_JPEG_QUALITY, 100])

        if not success:
            return {"error": "MosaicEncodeFailed", "mosaic_b64": None}

        mosaic_b64 = base64.b64encode(encoded_image).decode('utf-8')

        return {"error": None, "mosaic_b64": mosaic_b64}

    except Exception as e:
        return {"error": f"MosaicCreationException: {str(e)}", "mosaic_b64": None}