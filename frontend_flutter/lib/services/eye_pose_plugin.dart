import 'dart:typed_data';
import 'package:flutter/services.dart';

class EyePosePlugin {
  static const MethodChannel _channel = MethodChannel('eye_pose_plugin');

  /// 调用原生 MediaPipe 处理单张图片：
  /// 传入：JPG/PNG 的原始字节
  /// 返回：List<Map<String,double>>，每个元素是 {x, y}
  static Future<List<Map<String, double>>> processImage(
      Uint8List bytes) async {
    final result = await _channel.invokeMethod('processImage', bytes);
    if (result == null) return <Map<String, double>>[];

    final list = (result as List).cast<dynamic>();
    return list
        .map<Map<String, double>>((e) {
          final m = (e as Map);
          final x = (m['x'] as num).toDouble();
          final y = (m['y'] as num).toDouble();
          return {'x': x, 'y': y};
        })
        .toList();
  }
}