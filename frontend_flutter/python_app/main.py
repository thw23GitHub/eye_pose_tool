from flask import Flask, request, jsonify
from processor import process_images_batch

app = Flask(__name__)

@app.route("/process_images", methods=["POST"])
def process_images():
    """
    接收 Flutter 传来的 JSON:
    {
      "images": ["/path/to/img1.jpg", "/path/to/img2.jpg", ...]
    }
    返回:
    {
      "results": [
        {
          "original_path": "...",
          "cropped_path": "...",
          "direction": "upLeft",
          "confidence": 0.92,
          "rotation_angle": -3.1
        },
        ...
      ],
      "errors": ["NoFaceDetected: /xx/yy.jpg", ...]
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    images = data.get("images", [])

    results, errors = process_images_batch(images)

    return jsonify({
        "results": results,
        "errors": errors,
    })


if __name__ == "__main__":
    # 开发期在 Mac 上本地调试用
    app.run(host="0.0.0.0", port=8000, debug=True)