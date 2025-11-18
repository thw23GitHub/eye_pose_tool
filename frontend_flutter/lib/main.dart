import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const EyePoseApp());
}

/// ★ 这里改成你自己的后端地址。
/// 你现在是 10.215.237.32:8000，对吧。
const String backendBaseUrl = 'http://10.215.237.32:8000';

class EyePoseApp extends StatelessWidget {
  const EyePoseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '眼位九宫格工具',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const EyePoseHomePage(),
    );
  }
}

class EyePoseHomePage extends StatefulWidget {
  const EyePoseHomePage({super.key});

  @override
  State<EyePoseHomePage> createState() => _EyePoseHomePageState();
}

class _EyePoseHomePageState extends State<EyePoseHomePage> {
  final ImagePicker _picker = ImagePicker();

  // A 区原始照片
  final List<Uint8List> _originalImageBytes = [];
  final List<XFile> _originalFiles = [];

  // B 区九宫格（固定 9 格）
  final List<Uint8List?> _processedGrid =
  List<Uint8List?>.filled(9, null, growable: false);

  // B 区截图用
  final GlobalKey _processedGridKey = GlobalKey();

  String? _statusMessage;
  bool _isProcessing = false;

  // slot -> index
  static const Map<String, int> _slotIndexMap = {
    'up_left': 0,
    'up': 1,
    'up_right': 2,
    'left': 3,
    'center': 4,
    'right': 5,
    'down_left': 6,
    'down': 7,
    'down_right': 8,
  };

  @override
  Widget build(BuildContext context) {
    const horizontalPadding = 16.0;
    const gridSpacing = 4.0;

    return Scaffold(
      backgroundColor: Colors.grey[100],
      body: SafeArea(
        child: Scrollbar(
          thumbVisibility: true,
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(
              horizontal: horizontalPadding,
              vertical: 12,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // 标题
                Center(
                  child: Text(
                    '眼位九宫格工具',
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                const SizedBox(height: 8),

                // 状态信息
                if (_statusMessage != null) ...[
                  Text(
                    _statusMessage!,
                    style: const TextStyle(
                      color: Colors.red,
                      fontSize: 14,
                    ),
                  ),
                  const SizedBox(height: 8),
                ],

                // A 区
                const SizedBox(height: 4),
                const Text(
                  'A 区：原始照片',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                AspectRatio(
                  aspectRatio: 1, // 正方形区域，保证 3×3 不被裁剪
                  child: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.grey.shade400),
                    ),
                    child: _originalImageBytes.isEmpty
                        ? const Center(
                      child: Text(
                        '尚未选择照片',
                        style: TextStyle(color: Colors.grey),
                      ),
                    )
                        : GridView.builder(
                      physics: const NeverScrollableScrollPhysics(),
                      shrinkWrap: true,
                      gridDelegate:
                      const SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 3,
                        crossAxisSpacing: gridSpacing,
                        mainAxisSpacing: gridSpacing,
                      ),
                      itemCount: _originalImageBytes.length,
                      itemBuilder: (context, index) {
                        return ClipRRect(
                          borderRadius: BorderRadius.circular(4),
                          child: Image.memory(
                            _originalImageBytes[index],
                            fit: BoxFit.cover,
                          ),
                        );
                      },
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // B 区
                const Text(
                  'B 区：处理后九宫格',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                RepaintBoundary(
                  key: _processedGridKey,
                  child: AspectRatio(
                    aspectRatio: 1,
                    child: Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: Colors.grey.shade400),
                      ),
                      child: _processedGrid.every((b) => b == null)
                          ? const Center(
                        child: Text(
                          '暂未生成九宫格',
                          style: TextStyle(color: Colors.grey),
                        ),
                      )
                          : GridView.builder(
                        physics: const NeverScrollableScrollPhysics(),
                        shrinkWrap: true,
                        gridDelegate:
                        const SliverGridDelegateWithFixedCrossAxisCount(
                          crossAxisCount: 3,
                          crossAxisSpacing: gridSpacing,
                          mainAxisSpacing: gridSpacing,
                        ),
                        itemCount: 9,
                        itemBuilder: (context, index) {
                          final bytes = _processedGrid[index];
                          if (bytes == null) {
                            return Container(
                              decoration: BoxDecoration(
                                color: Colors.grey[200],
                                borderRadius: BorderRadius.circular(4),
                              ),
                            );
                          }
                          return ClipRRect(
                            borderRadius: BorderRadius.circular(4),
                            child: Image.memory(
                              bytes,
                              fit: BoxFit.cover,
                            ),
                          );
                        },
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // C 区按钮
                const Text(
                  'C 区：操作按钮',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),

                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Expanded(
                      child: _buildActionButton(
                        label: '选择照片',
                        onTap: _isProcessing ? null : _pickImages,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: _buildActionButton(
                        label: _isProcessing ? '处理中…' : '处理照片',
                        onTap: _isProcessing ? null : _processImages,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: _buildActionButton(
                        label: '保存照片',
                        onTap:
                        _isProcessing ? null : _saveProcessedGridAsImage,
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 16),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // ===== 按钮 =====
  Widget _buildActionButton({
    required String label,
    required VoidCallback? onTap,
  }) {
    final enabled = onTap != null;
    return ElevatedButton(
      onPressed: onTap,
      style: ElevatedButton.styleFrom(
        padding: const EdgeInsets.symmetric(vertical: 12),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(24),
        ),
        backgroundColor: enabled ? Colors.blue : Colors.grey,
      ),
      child: Text(
        label,
        style: const TextStyle(fontSize: 16, color: Colors.white),
      ),
    );
  }

  // ===== A 区：选择照片 =====
  Future<void> _pickImages() async {
    setState(() {
      _statusMessage = null;
    });

    try {
      final List<XFile> picked = await _picker.pickMultiImage(
        maxWidth: 2000,
        maxHeight: 2000,
        imageQuality: 95,
      );

      if (picked.isEmpty) return;

      if (picked.length > 9) {
        setState(() {
          _statusMessage = '已选择 ${picked.length} 张照片，将仅使用前 9 张。';
        });
      }

      final limited = picked.take(9).toList();
      final bytesList = <Uint8List>[];

      for (final f in limited) {
        bytesList.add(await f.readAsBytes());
      }

      setState(() {
        _originalFiles
          ..clear()
          ..addAll(limited);
        _originalImageBytes
          ..clear()
          ..addAll(bytesList);
        _processedGrid.fillRange(0, _processedGrid.length, null);
      });
    } catch (e) {
      setState(() {
        _statusMessage = '选择照片时出错：$e';
      });
    }
  }

  // ===== 调用后端处理 =====
  Future<void> _processImages() async {
    if (_originalFiles.isEmpty) {
      setState(() {
        _statusMessage = '请先在 A 区选择 1–9 张原始照片。';
      });
      return;
    }

    setState(() {
      _isProcessing = true;
      _statusMessage = null;
      _processedGrid.fillRange(0, _processedGrid.length, null);
    });

    try {
      final uri = Uri.parse('$backendBaseUrl/process_images');
      final request = http.MultipartRequest('POST', uri);

      for (final file in _originalFiles) {
        request.files.add(
          await http.MultipartFile.fromPath(
            'images',
            file.path,
            contentType: MediaType('image', 'jpeg'),
          ),
        );
      }

      final streamedResponse =
      await request.send().timeout(const Duration(seconds: 30));
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode != 200) {
        setState(() {
          _statusMessage =
          '处理照片失败：后端返回状态码 ${response.statusCode}。\n'
              '请检查 Flask 日志和 /process_images 接口实现。';
        });
        return;
      }

      final dynamic decoded = jsonDecode(response.body);

      if (decoded is! Map || decoded['results'] == null) {
        setState(() {
          _statusMessage =
          '前端未能从后端 JSON 中解析出任何图像数据。\n'
              '请检查 /process_images 是否返回：results[].image_base64 字段。';
        });
        return;
      }

      final results = decoded['results'];
      if (results is! List || results.isEmpty) {
        setState(() {
          _statusMessage =
          '后端已返回 JSON，但 results 为空。\n'
              '请确认 /process_images 逻辑是否正确生成九宫格图像。';
        });
        return;
      }

      final List<Uint8List?> newGrid =
      List<Uint8List?>.filled(9, null, growable: false);

      for (final item in results) {
        if (item is! Map) continue;

        final String? base64Str = item['image_base64'] as String?;
        if (base64Str == null || base64Str.isEmpty) continue;

        final String? slot = item['slot'] as String?;
        int index;
        if (slot != null && _slotIndexMap.containsKey(slot)) {
          index = _slotIndexMap[slot]!;
        } else {
          index = results.indexOf(item);
          if (index < 0 || index >= 9) continue;
        }

        try {
          final bytes = base64Decode(base64Str);
          newGrid[index] = bytes;
        } catch (_) {
          // 单个格子解码失败，忽略
        }
      }

      if (newGrid.every((b) => b == null)) {
        setState(() {
          _statusMessage =
          '已成功从后端收到 JSON，但没有任何可用的 image_base64 图像。\n'
              '请检查后端是否正确填充 results[].image_base64。';
        });
        return;
      }

      setState(() {
        for (var i = 0; i < 9; i++) {
          _processedGrid[i] = newGrid[i];
        }
        _statusMessage = null;
      });
    } on SocketException catch (e) {
      setState(() {
        _statusMessage =
        '处理照片失败：无法连接到后端服务器。\n'
            '请确认：\n'
            '1）手机与服务器（Flask）在同一网络；\n'
            '2）backendBaseUrl = $backendBaseUrl 是否正确；\n'
            '3）Flask 是否已启动并监听对应端口。\n'
            '系统错误信息：$e';
      });
    } on TimeoutException {
      setState(() {
        _statusMessage =
        '处理照片失败：连接后端超时。\n'
            '请检查网络状况以及 Flask 是否在 /process_images 及时返回。';
      });
    } catch (e) {
      setState(() {
        _statusMessage = '处理照片失败：$e';
      });
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  // ===== 保存 B 区九宫格 =====
  Future<void> _saveProcessedGridAsImage() async {
    if (_processedGrid.every((b) => b == null)) {
      setState(() {
        _statusMessage = '当前没有可保存的九宫格，请先点击“处理照片”。';
      });
      return;
    }

    if (!await _ensureStoragePermission()) {
      setState(() {
        _statusMessage = '没有存储权限，无法保存图片。请在系统设置中授予存储权限。';
      });
      return;
    }

    try {
      final boundary = _processedGridKey.currentContext?.findRenderObject()
      as RenderRepaintBoundary?;
      if (boundary == null) {
        setState(() {
          _statusMessage = '保存失败：无法获取九宫格渲染对象。';
        });
        return;
      }

      final ui.Image image =
      await boundary.toImage(pixelRatio: 3.0); // 提高导出清晰度
      final byteData =
      await image.toByteData(format: ui.ImageByteFormat.png);
      if (byteData == null) {
        setState(() {
          _statusMessage = '保存失败：无法生成 PNG 数据。';
        });
        return;
      }

      final pngBytes = byteData.buffer.asUint8List();

      final directory = await getApplicationDocumentsDirectory();
      final ts = DateTime.now()
          .toIso8601String()
          .replaceAll(':', '-')
          .replaceAll('.', '-');
      final filePath = '${directory.path}/eye_pose_grid_$ts.png';
      final file = File(filePath);
      await file.writeAsBytes(pngBytes);

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('已保存九宫格到：$filePath'),
            duration: const Duration(seconds: 3),
          ),
        );
      }
    } catch (e) {
      setState(() {
        _statusMessage = '保存照片失败：$e';
      });
    }
  }

  Future<bool> _ensureStoragePermission() async {
    if (!Platform.isAndroid && !Platform.isIOS) return true;

    if (Platform.isAndroid) {
      final status = await Permission.storage.request();
      return status.isGranted;
    }
    return true;
  }
}