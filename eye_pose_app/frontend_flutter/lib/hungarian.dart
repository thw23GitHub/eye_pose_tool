// lib/hungarian.dart
import 'dart:math' as math;

/// 经典匈牙利算法（方阵）；返回每一行分配到的列索引。
List<int> hungarian(List<List<double>> cost) {
  final n = cost.length;
  final a = List.generate(n, (i) => List<double>.from(cost[i]));
  // 行列最小值归约
  for (var i = 0; i < n; i++) {
    final m = a[i].reduce(math.min);
    for (var j = 0; j < n; j++) a[i][j] -= m;
  }
  for (var j = 0; j < n; j++) {
    var m = a[0][j];
    for (var i = 1; i < n; i++) m = math.min(m, a[i][j]);
    for (var i = 0; i < n; i++) a[i][j] -= m;
  }

  final starred = List.generate(n, (_) => List<bool>.filled(n, false));
  final primed  = List.generate(n, (_) => List<bool>.filled(n, false));
  final rowCov  = List<bool>.filled(n, false);
  final colCov  = List<bool>.filled(n, false);

  // 初始打星
  for (var i = 0; i < n; i++) {
    for (var j = 0; j < n; j++) {
      if (a[i][j] == 0 && !rowCov[i] && !colCov[j]) {
        starred[i][j] = true;
        rowCov[i] = true;
        colCov[j] = true;
      }
    }
  }
  for (var i = 0; i < n; i++) rowCov[i] = false;
  for (var j = 0; j < n; j++) colCov[j] = false;

  void coverCols() {
    for (var j = 0; j < n; j++) {
      colCov[j] = false;
      for (var i = 0; i < n; i++) {
        if (starred[i][j]) { colCov[j] = true; break; }
      }
    }
  }

  int? starInRow(int r) {
    for (var j = 0; j < n; j++) if (starred[r][j]) return j;
    return null;
  }
  int? starInCol(int c) {
    for (var i = 0; i < n; i++) if (starred[i][c]) return i;
    return null;
  }
  int? primeInRow(int r) {
    for (var j = 0; j < n; j++) if (primed[r][j]) return j;
    return null;
  }

  List<int>? findZero() {
    for (var i = 0; i < n; i++) {
      if (rowCov[i]) continue;
      for (var j = 0; j < n; j++) {
        if (!colCov[j] && a[i][j] == 0) return [i, j];
      }
    }
    return null;
  }

  void augment(List<List<int>> path) {
    for (final rc in path) {
      final r = rc[0], c = rc[1];
      starred[r][c] = !starred[r][c];
    }
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) primed[i][j] = false;
      rowCov[i] = false;
    }
    for (var j = 0; j < n; j++) colCov[j] = false;
  }

  coverCols();
  while (colCov.where((v) => v).length < n) {
    var z = findZero();
    while (z == null) {
      // 调整矩阵
      double m = double.infinity;
      for (var i = 0; i < n; i++) {
        if (rowCov[i]) continue;
        for (var j = 0; j < n; j++) {
          if (!colCov[j]) m = math.min(m, a[i][j]);
        }
      }
      for (var i = 0; i < n; i++) {
        if (!rowCov[i]) for (var j = 0; j < n; j++) a[i][j] -= m;
      }
      for (var j = 0; j < n; j++) {
        if (colCov[j]) for (var i = 0; i < n; i++) a[i][j] += m;
      }
      z = findZero();
    }
    final r = z[0], c = z[1];
    primed[r][c] = true;
    final s = starInRow(r);
    if (s != null) {
      rowCov[r] = true;
      colCov[s] = false;
    } else {
      final path = <List<int>>[];
      path.add([r, c]);
      var cstar = c;
      var rstar = starInCol(cstar);
      while (rstar != null) {
        path.add([rstar, cstar]);
        final cprime = primeInRow(rstar)!;
        path.add([rstar, cprime]);
        cstar = cprime;
        rstar = starInCol(cstar);
      }
      augment(path);
      coverCols();
    }
  }

  final assign = List<int>.filled(n, -1);
  for (var i = 0; i < n; i++) {
    for (var j = 0; j < n; j++) if (starred[i][j]) { assign[i] = j; break; }
  }
  return assign;
}