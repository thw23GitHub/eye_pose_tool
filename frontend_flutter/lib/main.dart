import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as im;
import 'package:image_picker/image_picker.dart';

import 'pipeline.dart'; // 你现有的自动识别 + 排序 + 拼图函数

void main() {
  runApp(const EyePoseApp());
}

/// 根应用
class EyePoseApp extends StatelessWidget {
  const EyePoseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '眼位九宫格工具（离线）',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: const EyePoseMosaicPage(),
    );
  }
}

/// 主页面
class EyePoseMosaicPage extends StatefulWidget {
  const EyePoseMosaicPage({super.key});

  @override
  State<EyePoseMosaicPage> createState() => _EyePoseMosaicPageState();
}

class _EyePoseMosaicPageState extends State<EyePoseMosaicPage> {
  final ImagePicker _picker = ImagePicker();

  /// 和原生交互的通道（用于保存到系统相册）
  static const MethodChannel _nativeChannel =
      MethodChannel('eye_pose_native');

  /// 相册里选出来的原始完整脸照片（最多 9 张）
  final List<XFile> _pickedFaces = [];

  /// 预览九宫格用的小图（按选择顺序排列）
  List<Uint8List> _previewTiles = [];

  /// 自动识别 + 排序 + 拼图之后的九宫格被切成的 9 张小图
  List<Uint8List> _tiles = [];

  /// 当前九宫格显示顺序（0–8），用来支持拖拽换位
  List<int> _order = List<int>.generate(9, (i) => i);

  /// 是否在处理中（防止重复点击）
  bool _busy = false;

  /// 当前正在被拖拽的格子下标（用于高亮）
  int? _draggingIndex;

  void _showSnack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg)),
    );
  }

  // ==================== 选择照片 & 预览 ====================

  /// 选择 1–9 张原始照片，并生成九宫格预览小图
  Future<void> _pickImages() async {
    try {
      debugPrint('[_pickImages] start pickMultiImage');
      final files = await _picker.pickMultiImage();
      if (files.isEmpty) return;

      final selected = files.take(9).toList(); // 最多 9 张

      // 生成预览图（3x3 小方块）
      final previewTiles = await _generatePreviewTiles(selected);

      setState(() {
        _pickedFaces
          ..clear()
          ..addAll(selected);
        _previewTiles = previewTiles;
        _tiles = [];
        _order = List<int>.generate(9, (i) => i);
      });

      debugPrint('[_pickImages] picked count = ${_pickedFaces.length}');
    } catch (e) {
      _showSnack('选择照片失败：$e');
    }
  }

  /// 将原始照片缩放为方形小图，用于上方的预览九宫格
  Future<List<Uint8List>> _generatePreviewTiles(List<XFile> files) async {
    final result = <Uint8List>[];

    for (final f in files) {
      final bytes = await f.readAsBytes();
      final img = im.decodeImage(bytes);
      if (img == null) continue;

      // 按 image 4.5.4 的 API，只能传一个位置参数 + 命名参数 size
      final square = im.copyResizeCropSquare(img, size: 600);

      final outBytes = Uint8List.fromList(
        im.encodeJpg(square, quality: 90),
      );
      result.add(outBytes);
    }

    return result;
  }

  // ==================== 调用后端生成九宫格 ====================

  /// 调用后端生成九宫格，并在 Dart 侧切成 9 块小图，支持拖拽排序
  Future<void> _processFaces() async {
    if (_pickedFaces.isEmpty) {
      _showSnack('请先从相册选择 1–9 张照片');
      return;
    }

    setState(() => _busy = true);

    try {
      // 1. 读取原始图片字节
      final faceBytes = <Uint8List>[];
      for (final f in _pickedFaces) {
        faceBytes.add(await f.readAsBytes());
      }

      // 2. 调用你现有的自动识别 + 排序 + 拼图函数（在 pipeline.dart 里）
      final mosaicBytes = await processBatchToMosaic(faceBytes);

      // 3. 把后端生成的九宫格切成 3×3 共 9 块小图
      final tiles = _splitMosaicToTiles(mosaicBytes);

      setState(() {
        _tiles = tiles;
        _order = List<int>.generate(_tiles.length, (i) => i);
      });

      _showSnack('处理完成，可以长按下面九宫格中的小图拖拽调整顺序');
    } catch (e) {
      _showSnack('处理失败：$e');
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  /// 从整张九宫格图像中切出 9 张小图
  List<Uint8List> _splitMosaicToTiles(Uint8List mosaicBytes) {
    final img = im.decodeImage(mosaicBytes);
    if (img == null) {
      throw Exception('无法解码自动生成的九宫格图片');
    }

    final tileW = img.width ~/ 3;
    final tileH = img.height ~/ 3;

    final tiles = <Uint8List>[];

    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        final x = col * tileW;
        final y = row * tileH;
        final tile = im.copyCrop(
          img,
          x: x,
          y: y,
          width: tileW,
          height: tileH,
        );
        final bytes = Uint8List.fromList(
          im.encodeJpg(tile, quality: 95),
        );
        tiles.add(bytes);
      }
    }

    return tiles;
  }

  /// 交换两个格子的顺序（拖拽时调用）
  void _swapOrder(int from, int to) {
    setState(() {
      final tmp = _order[from];
      _order[from] = _order[to];
      _order[to] = tmp;
    });
  }

  // ==================== 按当前顺序重新拼成九宫格并保存 ====================

  /// 将 9 张 tile 按当前顺序合成为一张大图（3x3），返回 PNG 字节
  Future<Uint8List> _composeMosaicFromTiles(
      List<Uint8List> orderedTiles) async {
    final images = <ui.Image>[];

    for (final bytes in orderedTiles) {
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      images.add(frame.image);
    }

    if (images.isEmpty) {
      throw Exception('没有可用的 tile 图像');
    }

    final tileW = images[0].width.toDouble();
    final tileH = images[0].height.toDouble();
    final width = (tileW * 3).toInt();
    final height = (tileH * 3).toInt();

    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);
    final paint = Paint();

    for (int i = 0; i < images.length; i++) {
      final img = images[i];
      final row = i ~/ 3;
      final col = i % 3;

      final src = Rect.fromLTWH(
        0,
        0,
        img.width.toDouble(),
        img.height.toDouble(),
      );

      final dst = Rect.fromLTWH(
        col * tileW,
        row * tileH,
        tileW,
        tileH,
      );

      canvas.drawImageRect(img, src, dst, paint);
    }

    final picture = recorder.endRecording();
    final ui.Image merged = await picture.toImage(width, height);
    final byteData =
        await merged.toByteData(format: ui.ImageByteFormat.png);

    if (byteData == null) {
      throw Exception('合成九宫格失败（无法导出 PNG）');
    }

    return byteData.buffer.asUint8List();
  }

  Future<void> _saveMosaic() async {
    if (_tiles.isEmpty) {
      _showSnack('请先点击“处理照片”生成九宫格');
      return;
    }

    setState(() => _busy = true);

    try {
      // 1. 根据当前拖拽顺序重新排列 tile
      final orderedTiles = _order.map((i) => _tiles[i]).toList();

      // 2. 在 Flutter 里重新合成一张九宫格（先 PNG）
      final mergedPngBytes = await _composeMosaicFromTiles(orderedTiles);

      // 3. 为了和原生 saveToGallery 兼容，转成 JPEG 再给原生
      final decoded = im.decodeImage(mergedPngBytes);
      if (decoded == null) {
        throw Exception('无法解码合成后的九宫格 PNG');
      }
      final mergedJpgBytes =
          Uint8List.fromList(im.encodeJpg(decoded, quality: 95));

      final fileName =
          'eye_pose_mosaic_${DateTime.now().millisecondsSinceEpoch}.jpg';

      // 4. 使用已经验证过“能在相册看到”的原生方法保存到 /Pictures
      final ok = await _nativeChannel.invokeMethod<bool>(
        'saveToGallery',
        {
          'bytes': mergedJpgBytes,
          'filename': fileName,
        },
      );

      if (ok == true) {
        _showSnack('九宫格已保存到系统相册（/Pictures 等目录）');
      } else {
        _showSnack('保存失败：原生方法返回 false');
      }
    } on PlatformException catch (e) {
      _showSnack('保存失败（平台异常）：${e.message}');
    } catch (e) {
      _showSnack('保存失败：$e');
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  // ==================== UI ====================

  @override
  Widget build(BuildContext context) {
    final canProcess = _pickedFaces.isNotEmpty && !_busy;
    final canSave = _tiles.isNotEmpty && !_busy;

    return Scaffold(
      appBar: AppBar(
        title: const Text('眼位九宫格工具（离线）'),
      ),
      body: SafeArea(
        child: Column(
          children: [
            // 顶部提示
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      '已选择：${_pickedFaces.length} 张（最多 9 张）',
                      style: const TextStyle(fontSize: 14),
                    ),
                  ),
                ],
              ),
            ),

            // 中间整体可以滚动：预览九宫格 + 处理后九宫格
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 4),
                    Text(
                      '上：原始照片预览（按选择顺序）',
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                    const SizedBox(height: 8),
                    _buildPreviewGrid(),

                    const SizedBox(height: 12),
                    Text(
                      '下：自动识别后的眼位九宫格（可拖拽微调顺序）',
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                    const SizedBox(height: 8),
                    _buildDraggableGrid(),
                  ],
                ),
              ),
            ),

            // 底部按钮条（随整体滚动）
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 4, 16, 12),
              child: Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _busy ? null : _pickImages,
                      icon: const Icon(Icons.photo_library_outlined),
                      label: const Text('选择照片'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: FilledButton.icon(
                      onPressed: canProcess ? _processFaces : null,
                      icon: const Icon(Icons.play_arrow_rounded),
                      label: const Text('处理照片'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: FilledButton.icon(
                      onPressed: canSave ? _saveMosaic : null,
                      icon: const Icon(Icons.save_alt_rounded),
                      label: const Text('保存九宫格'),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// 上方预览九宫格
  Widget _buildPreviewGrid() {
    if (_previewTiles.isEmpty) {
      return const SizedBox(
        height: 150,
        child: Center(
          child: Text(
            '请选择 1–9 张照片，上方将以九宫格方式预览原始照片。',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.black45),
          ),
        ),
      );
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        final width = constraints.maxWidth;
        final tileSize = width / 3;

        return SizedBox(
          width: width,
          height: tileSize * 3,
          child: GridView.builder(
            physics: const NeverScrollableScrollPhysics(),
            padding: EdgeInsets.zero,
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 3,
              mainAxisSpacing: 4,
              crossAxisSpacing: 4,
            ),
            itemCount: _previewTiles.length,
            itemBuilder: (context, index) {
              final bytes = _previewTiles[index];
              return ClipRRect(
                borderRadius: BorderRadius.circular(6),
                child: Image.memory(
                  bytes,
                  fit: BoxFit.cover,
                ),
              );
            },
          ),
        );
      },
    );
  }

  /// 下方可拖拽的九宫格
  Widget _buildDraggableGrid() {
    if (_tiles.isEmpty) {
      return const SizedBox(
        height: 220,
        child: Center(
          child: Text(
            '请先选择照片并点击“处理照片”，\n'
            '系统会自动识别和排序，然后可以在这里拖拽微调。',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.black45),
          ),
        ),
      );
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        final width = constraints.maxWidth;
        final tileSize = width / 3;

        return SizedBox(
          width: width,
          height: tileSize * 3,
          child: GridView.builder(
            physics: const NeverScrollableScrollPhysics(),
            padding: EdgeInsets.zero,
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 3,
              mainAxisSpacing: 6,
              crossAxisSpacing: 6,
            ),
            itemCount: _tiles.length,
            itemBuilder: (context, index) {
              final tileIndex = _order[index];
              final bytes = _tiles[tileIndex];

              return LongPressDraggable<int>(
                data: index,
                dragAnchorStrategy: pointerDragAnchorStrategy,
                feedback: SizedBox(
                  width: tileSize,
                  height: tileSize,
                  child: _buildTile(bytes, dragging: true),
                ),
                onDragStarted: () {
                  setState(() {
                    _draggingIndex = index;
                  });
                },
                onDragEnd: (_) {
                  setState(() {
                    _draggingIndex = null;
                  });
                },
                child: DragTarget<int>(
                  onWillAccept: (from) => from != null && from != index,
                  onAccept: (from) {
                    _swapOrder(from, index);
                  },
                  builder: (context, candidates, rejects) {
                    final isTarget = candidates.isNotEmpty;
                    return AnimatedOpacity(
                      duration: const Duration(milliseconds: 120),
                      opacity: _draggingIndex == index ? 0.7 : 1.0,
                      child: _buildTile(
                        bytes,
                        highlight: isTarget,
                        indexLabel: index + 1,
                      ),
                    );
                  },
                ),
              );
            },
          ),
        );
      },
    );
  }

  Widget _buildTile(
    Uint8List bytes, {
    bool dragging = false,
    bool highlight = false,
    int? indexLabel,
  }) {
    return DecoratedBox(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: highlight ? Colors.teal : Colors.black12,
          width: highlight ? 3 : 1,
        ),
        color: Colors.black12,
      ),
      child: Stack(
        fit: StackFit.expand,
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(7),
            child: Image.memory(
              bytes,
              fit: BoxFit.cover,
            ),
          ),
          if (indexLabel != null && !dragging)
            Positioned(
              right: 4,
              top: 4,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  '$indexLabel',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 10,
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}