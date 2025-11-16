# app.py  —— 无 mediapipe 版本，使用 Flask + OpenCV 裁剪眼部九宫格
#
# 接口约定：
#   POST /process
#     form-data: files=[多张图片]  (或者 images=[多张图片] 也兼容)
#   返回值：单张 JPEG 九宫格图片（二进制流）
#
#   GET /health  -> {"status": "ok"}

from flask import Flask, request, send_file, jsonify
from PIL import Image
import numpy as np
import cv2
import io
import math

app = Flask(__name__)

# 使用 OpenCV 自带的人脸检测模型（不需要单独下载）
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ========== 工具函数 ==========

def detect_face_bbox(rgb_img):
    """
    输入：RGB ndarray (H, W, 3)
    输出：人脸 bounding box (x, y, w, h)
    若检测不到，返回整个图像中心区域作为备用。
    """
    h, w, _ = rgb_img.shape
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        # 兜底：取中间区域
        x = int(0.1 * w)
        y = int(0.15 * h)
        fw = int(0.8 * w)
        fh = int(0.7 * h)
        return x, y, fw, fh

    # 若检测到多张脸，取最大的一个
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(fw), int(fh)


def crop_eye_strip(pil_img):
    """
    根据人脸 bbox，从“眉毛到鼻梁”裁剪一条横向眼带。
    尽量让“眼睛位于竖直方向的中间”。
    """
    rgb = np.array(pil_img.convert("RGB"))
    h, w, _ = rgb.shape

    x, y, fw, fh = detect_face_bbox(rgb)

    # 根据人脸框推算眼带区域（这些比例是经验值，可以以后再微调）
    top = y + int(0.20 * fh)       # 眉毛稍下
    bottom = y + int(0.60 * fh)    # 到鼻梁附近
    left = x - int(0.10 * fw)      # 左右略放宽一点
    right = x + int(1.10 * fw)

    # 边界裁剪
    top = max(0, top)
    bottom = min(h, bottom)
    left = max(0, left)
    right = min(w, right)

    if bottom <= top or right <= left:
        # 极端情况兜底：取图像中间一条横带
        top = int(0.25 * h)
        bottom = int(0.65 * h)
        left = 0
        right = w

    crop = rgb[top:bottom, left:right, :]
    return Image.fromarray(crop)


def build_mosaic(strips, tile_width=512, tile_height=256, gap=4):
    """
    将最多 9 张横向眼带拼成 3x3 九宫格。
    输入：PIL.Image list
    输出：PIL.Image (九宫格)
    """
    if not strips:
        raise ValueError("strips is empty")

    # 最多 9 张
    strips = strips[:9]
    n = len(strips)
    rows = cols = 3

    # 把每个 strip 统一缩放为 tile_width x tile_height
    resized = []
    for im_strip in strips:
        r = im_strip.resize((tile_width, tile_height), Image.LANCZOS)
        resized.append(r)

    # 画布大小
    W = cols * tile_width + (cols - 1) * gap
    H = rows * tile_height + (rows - 1) * gap
    canvas = Image.new("RGB", (W, H), (0, 0, 0))

    for idx, img in enumerate(resized):
        r = idx // cols
        c = idx % cols
        x = c * (tile_width + gap)
        y = r * (tile_height + gap)
        canvas.paste(img, (x, y))

    return canvas


# ========== Flask 路由 ==========

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/process", methods=["POST"])
def process():
    """
    接收 1~9 张完整面部照片，裁剪眼带并生成九宫格 JPG 返回。
    前端可以用字段名 files 或 images 之一上传。
    """
    files = request.files.getlist("files")
    if not files:
        files = request.files.getlist("images")

    if not files:
        return jsonify({"error": "no files uploaded; use form-data with key 'files'"}), 400

    strips = []
    for f in files[:9]:
        try:
            pil_img = Image.open(f.stream).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"cannot open image: {e}"}), 400

        strip = crop_eye_strip(pil_img)
        strips.append(strip)

    mosaic = build_mosaic(strips)

    buf = io.BytesIO()
    mosaic.save(buf, format="JPEG", quality=95)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="image/jpeg",
        as_attachment=False,
        download_name="mosaic.jpg",
    )


if __name__ == "__main__":
    # 你之前前端是访问 http://127.0.0.1:27865/process
    # 所以这里保持 host=0.0.0.0, port=27865
    app.run(host="0.0.0.0", port=27865, debug=True)