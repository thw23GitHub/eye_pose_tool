// lib/mosaic_builder.dart
import 'dart:typed_data';
import 'package:image/image.dart' as im;

/// 生成 3×3 拼图（含 2px 白色间隔线）。
Uint8List buildMosaic(List<Uint8List> tiles,
    {int tileW = 800, int tileH = 400, int gap = 25}) {
  const cols = 3, rows = 3;
  final totalW = cols * tileW + (cols + 1) * gap;
  final totalH = rows * tileH + (rows + 1) * gap;

  // ✅ 直接用背景色创建白底画布 —— 兼容 image 4.5.x
  final canvas = im.Image(
    width: totalW,
    height: totalH,
    backgroundColor: im.ColorRgb8(255, 255, 255),
  );

  for (var k = 0; k < 9; k++) {
    final r = k ~/ 3, c = k % 3;
    final x0 = gap + c * (tileW + gap);
    final y0 = gap + r * (tileH + gap);

    if (k >= tiles.length) continue;
    final img = im.decodeImage(tiles[k]);
    if (img == null) continue;

    // 新 API：需要使用命名参数 size
    final fitted = im.copyResizeCropSquare(img, size: tileH < tileW ? tileH : tileW);
    final resized = im.copyResize(fitted, width: tileW, height: tileH);

    // 新 API：合成函数
    im.compositeImage(canvas, resized, dstX: x0, dstY: y0);
  }

  return Uint8List.fromList(im.encodeJpg(canvas, quality: 95));
}