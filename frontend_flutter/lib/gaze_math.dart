// lib/gaze_math.dart
import 'dart:math' as math;

/// 输入 MediaPipe 的归一化点 [{x,y}, ...]，返回 (hx, vy)
Map<String, double> gazeOffset(List<Map<String, double>> lms) {
  double x(int i) => lms[i]['x']!;
  double y(int i) => lms[i]['y']!;
  // 索引：与 Python 版本一致
  const L_OUT = 33, L_IN = 133, L_UP = 159, L_DN = 145, L_IRIS = 468;
  const R_IN  = 362, R_OUT= 263, R_UP = 386, R_DN = 374, R_IRIS = 473;

  Map<String, double> eyeHV(int outI, int inI, int upI, int dnI, int irisI) {
    final cx = (x(outI) + x(inI)) / 2.0;
    final cy = (y(upI) + y(dnI)) / 2.0;
    final hxDen = ((x(inI) - x(outI)) / 2.0).abs();
    final vyDen = ((y(dnI) - y(upI)) / 2.0).abs();
    final hx = (x(irisI) - cx) / (hxDen > 1e-6 ? hxDen : 1e-6);
    final vy = (y(irisI) - cy) / (vyDen > 1e-6 ? vyDen : 1e-6);
    return {'hx': hx, 'vy': vy};
  }

  final L = eyeHV(L_OUT, L_IN, L_UP, L_DN, L_IRIS);
  final R = eyeHV(R_OUT, R_IN, R_UP, R_DN, R_IRIS);

  return {'hx': (L['hx']! + R['hx']!) / 2.0, 'vy': (L['vy']! + R['vy']!) / 2.0};
}

/// 稳健归一化到 [-1,1]：median + IQR，clamp=1.5（与 Python 一致）
List<double> robustScale(List<double> v, {double clamp = 1.5}) {
  if (v.isEmpty) return [];
  final s = List<double>.from(v)..sort();
  double median() => s.length.isOdd
      ? s[s.length ~/ 2]
      : (s[s.length ~/ 2 - 1] + s[s.length ~/ 2]) / 2.0;
  final med = median();

  double percentile(double p) {
    if (s.isEmpty) return 0;
    final idx = p * (s.length - 1);
    final i = idx.floor();
    final f = idx - i;
    if (i + 1 < s.length) return s[i] * (1 - f) + s[i + 1] * f;
    return s[i];
    }
  final q1 = percentile(0.25), q3 = percentile(0.75);
  final iqr = q3 - q1;
  final madAlt = v.map((e) => (e - med).abs()).reduce((a, b) => a + b) / v.length;
  final scale = iqr > 1e-6 ? (iqr / 1.349) : (madAlt * 1.253 + 1e-6);

  final out = <double>[];
  for (final t in v) {
    var z = (t - med) / (scale > 1e-6 ? scale : 1e-6);
    if (z < -clamp) z = -clamp;
    if (z > clamp) z = clamp;
    out.add(z / clamp); // [-1,1]
  }
  return out;
}

/// 构造 9×9 代价矩阵：每张图的 (HX,VY) 与九宫格目标的 L2 距离平方
List<List<double>> buildCostMatrix(List<double> HX, List<double> VY) {
  const targets = <List<double>>[
    [-1,-1],[0,-1],[1,-1],
    [-1, 0],[0, 0],[1, 0],
    [-1, 1],[0, 1],[1, 1],
  ];
  final n = HX.length;
  final cost = List.generate(n, (_) => List<double>.filled(n, 0));
  for (var i = 0; i < n; i++) {
    for (var j = 0; j < n; j++) {
      final dx = HX[i] - targets[j][0];
      final dy = VY[i] - targets[j][1];
      cost[i][j] = dx*dx + dy*dy;
    }
  }
  return cost;
}