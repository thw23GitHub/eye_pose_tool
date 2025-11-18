import base64
import logging
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# ----------------- 基础配置 -----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger("app")

app = Flask(__name__)
CORS(app)

# MediaPipe Face Mesh，全局实例
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,  # 使用虹膜关键点
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 九宫格标签顺序（固定映射）
GRID_LABELS: List[str] = [
    "up_left", "up_center", "up_right",
    "mid_left", "mid_center", "mid_right",
    "down_left", "down_center", "down_right",
]


# ----------------- 工具函数 -----------------


def file_to_bgr_image(file_storage) -> np.ndarray:
    """将上传的文件对象转换为 OpenCV BGR 图像。"""
    data = file_storage.read()
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码为图像")
    return img


def image_to_base64_png(img_bgr: np.ndarray) -> str:
    """将 BGR 图像编码为 PNG 的 base64 字符串。"""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("PNG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ----------------- 视线方向分类（新版） -----------------


def classify_gaze_single(img_bgr: np.ndarray, face_mesh) -> Optional[str]:
    """
    使用 MediaPipe FaceMesh 粗略判断注视方向，返回 GRID_LABELS 中的一个或 None。
    分别对左右眼做归一化，然后取平均值来判断左右 / 上下。
    """
    try:
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        # 关键点索引（MediaPipe FaceMesh + iris）
        left_eye_idx = [33, 133, 160, 159, 158, 157, 173, 246]
        right_eye_idx = [362, 263, 387, 386, 385, 384, 398, 466]
        left_iris_idx = [474, 475, 476, 477]
        right_iris_idx = [469, 470, 471, 472]

        def landmark_xy(idx_list):
            xs, ys = [], []
            for idx in idx_list:
                lm = face_landmarks.landmark[idx]
                xs.append(lm.x * w)
                ys.append(lm.y * h)
            return np.array(xs), np.array(ys)

        # 左右眼轮廓
        l_ex, l_ey = landmark_xy(left_eye_idx)
        r_ex, r_ey = landmark_xy(right_eye_idx)
        # 左右眼虹膜点
        l_ix, l_iy = landmark_xy(left_iris_idx)
        r_ix, r_iy = landmark_xy(right_iris_idx)

        eps = 1e-6

        def norm_pos(iris_xs, iris_ys, eye_xs, eye_ys):
            eye_min_x, eye_max_x = eye_xs.min(), eye_xs.max()
            eye_min_y, eye_max_y = eye_ys.min(), eye_ys.max()
            cx, cy = float(np.mean(iris_xs)), float(np.mean(iris_ys))
            x_n = (cx - eye_min_x) / (eye_max_x - eye_min_x + eps)
            y_n = (cy - eye_min_y) / (eye_max_y - eye_min_y + eps)
            return x_n, y_n

        # 分别计算左右眼的归一化位置，然后取平均
        lx_n, ly_n = norm_pos(l_ix, l_iy, l_ex, l_ey)
        rx_n, ry_n = norm_pos(r_ix, r_iy, r_ex, r_ey)

        x_norm = (lx_n + rx_n) / 2.0
        y_norm = (ly_n + ry_n) / 2.0

        # ------- 左 / 右 判定 -------
        if x_norm < 0.40:
            horiz = "left"
        elif x_norm > 0.60:
            horiz = "right"
        else:
            horiz = "center"

        # ------- 上 / 下 判定 -------
        if y_norm < 0.45:
            vert = "up"
        elif y_norm > 0.55:
            vert = "down"
        else:
            vert = "mid"

        label_map = {
            ("up", "left"): "up_left",
            ("up", "center"): "up_center",
            ("up", "right"): "up_right",
            ("mid", "left"): "mid_left",
            ("mid", "center"): "mid_center",
            ("mid", "right"): "mid_right",
            ("down", "left"): "down_left",
            ("down", "center"): "down_center",
            ("down", "right"): "down_right",
        }

        label = label_map.get((vert, horiz))
        return label
    except Exception as e:
        logger.warning("classify_gaze_single 异常: %s", e)
        return None


# ----------------- Flask 路由 -----------------


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/process_images", methods=["POST"])
def process_images():
    """
    接收 9 张图片：
      - Content-Type: multipart/form-data
      - 字段名: images（可以是多个）

    返回（兼容前端）：
      {
        "status": "ok",
        "results": [               # ✅ 前端主要使用这个
          {
            "original_index": 0,
            "label": "mid_center",
            "image_base64": "...."
          },
          ...
        ],
        "sorted_indices": [...],   # 可选调试字段
        "sorted_labels":  [...],
        "errors": [...]
      }
    """
    try:
        logger.info("Content-Type from client: %s", request.content_type)

        if "images" not in request.files:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "未找到字段 'images'",
                        "results": [],
                        "errors": [{"error": "no_images_field"}],
                    }
                ),
                400,
            )

        files = request.files.getlist("images")
        logger.info("request.files keys: %s", list(request.files.keys()))
        logger.info("Parsed %d images from request", len(files))

        # 原始结果列表（未排序）
        raw_results = []
        errors = []

        for idx, fs in enumerate(files):
            try:
                img_bgr = file_to_bgr_image(fs)
                label = classify_gaze_single(img_bgr, face_mesh)
                logger.info("图像 %d 分类结果: %s", idx, label)

                img_b64 = image_to_base64_png(img_bgr)
                raw_results.append(
                    {
                        "original_index": idx,
                        "label": label,
                        "image_base64": img_b64,
                    }
                )
            except Exception as e:
                logger.exception("处理第 %d 张图片异常", idx)
                errors.append({"index": idx, "error": str(e)})

        # 根据 label 做九宫格排序
        label_to_pos = {label: i for i, label in enumerate(GRID_LABELS)}

        def sort_key(i: int) -> int:
            label = raw_results[i]["label"]
            return label_to_pos.get(label, 999)

        # raw_results 的下标列表
        sorted_result_indices = sorted(range(len(raw_results)), key=sort_key)

        # 按九宫格顺序重排结果，供前端使用
        sorted_results = [raw_results[i] for i in sorted_result_indices]

        sorted_indices = [r["original_index"] for r in sorted_results]
        sorted_labels = [r["label"] for r in sorted_results]

        logger.info("九宫格排序结果(位置->原始索引): %s", sorted_indices)
        logger.info("Return %d results, %d errors", len(sorted_results), len(errors))

        return jsonify(
            {
                "status": "ok",
                # ✅ 兼容前端：results[].image_base64
                "results": sorted_results,
                # 额外调试字段
                "sorted_indices": sorted_indices,
                "sorted_labels": sorted_labels,
                "errors": errors,
            }
        )

    except Exception as e:
        logger.exception("顶层 /process_images 异常")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"服务器异常: {e}",
                    "results": [],
                    "errors": [{"error": str(e)}],
                }
            ),
            500,
        )


# ----------------- 主入口 -----------------

if __name__ == "__main__":
    # 绑定 0.0.0.0 方便手机访问
    app.run(host="0.0.0.0", port=8000, debug=True)