// lib/pipeline.dart
import 'dart:typed_data';
import 'package:image/image.dart' as im;

/// 生成九宫格（本地离线版）
/// 输入：1~9 张原始照片字节，输出：九宫格 JPG 字节
Future<Uint8List> processBatchToMosaic(List<Uint8List> images) async {
  if (images.isEmpty) {
    throw Exception("没有输入图片");
  }

  final tiles = <im.Image>[];

  // 处理最多 9 张
  for (final bytes in images.take(9)) {
    final src = im.decodeImage(bytes);
    if (src == null) continue;

    final crop = _cropFaceRegion(src);
    tiles.add(crop);
  }

  if (tiles.isEmpty) {
    throw Exception("所有图片解码失败");
  }

  // 不足 9 张补空白
  while (tiles.length < 9) {
    final w = tiles[0].width;
    final h = tiles[0].height;

    final blank = im.Image(width: w, height: h);
    final gray = im.ColorRgb8(240, 240, 240);

    // 手动填充灰色背景
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        blank.setPixel(x, y, gray);
      }
    }

    tiles.add(blank);
  }

  final mosaic = _makeMosaic(tiles);

  return Uint8List.fromList(im.encodeJpg(mosaic, quality: 95));
}

/// 简单裁剪区域：
/// 略偏上（眉毛到鼻梁附近）并左右放宽一点，
/// 保证眼睛大致在竖直方向中心且不被左右切掉。
im.Image _cropFaceRegion(im.Image src) {
  final w = src.width;
  final h = src.height;

  // 中心略偏上
  final cx = (w / 2).toInt();
  final cy = (h * 0.45).toInt();

  final cropW = (w * 0.60).toInt(); // 左右适当放宽
  final cropH = (h * 0.40).toInt(); // 上下只截一部分

  int x0 = cx - cropW ~/ 2;
  int y0 = cy - cropH ~/ 2;

  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;

  int cw = cropW;
  int ch = cropH;

  if (x0 + cw > w) cw = w - x0;
  if (y0 + ch > h) ch = h - y0;

  if (cw <= 0) cw = w;
  if (ch <= 0) ch = h;

  return im.copyCrop(src, x: x0, y: y0, width: cw, height: ch);
}

/// 合成 3×3 九宫格
im.Image _makeMosaic(List<im.Image> tiles) {
  final tileW = tiles[0].width;
  final tileH = tiles[0].height;

  const gap = 4; // 格子之间的间距（像素）

  final totalW = tileW * 3 + gap * 2;
  final totalH = tileH * 3 + gap * 2;

  final mosaic = im.Image(width: totalW, height: totalH);
  final black = im.ColorRgb8(0, 0, 0);

  // 填充背景为黑色
  for (int y = 0; y < totalH; y++) {
    for (int x = 0; x < totalW; x++) {
      mosaic.setPixel(x, y, black);
    }
  }

  int idx = 0;
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      final dstX0 = col * (tileW + gap);
      final dstY0 = row * (tileH + gap);
      final srcImg = tiles[idx];

      // 把每一张小图逐像素贴到 mosaic 上
      for (int y = 0; y < tileH; y++) {
        for (int x = 0; x < tileW; x++) {
          final mx = dstX0 + x;
          final my = dstY0 + y;
          if (mx < 0 || mx >= totalW || my < 0 || my >= totalH) continue;

          final c = srcImg.getPixel(x, y);
          mosaic.setPixel(mx, my, c);
        }
      }

      idx++;
    }
  }

  return mosaic;
}

//test, hello ShiChuan    --time 2021-08-05
// 测试用，hello ShiChuan    --time 2021-08-05
// 再测试，hello ShiChuan    --time 2021-08-05
// 代码修改测试，hello ShiChuan    --time 2021-08-05
