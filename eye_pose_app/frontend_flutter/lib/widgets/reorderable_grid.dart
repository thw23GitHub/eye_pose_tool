import 'dart:typed_data';
import 'package:flutter/material.dart';

/// 3×3 可拖拽九宫格：
/// - tiles: 长度<=9 的图片列表
/// - onSwap: 当 from 与 to 位置交换时回调
class ReorderableTileGrid extends StatelessWidget {
  final List<Uint8List> tiles;
  final void Function(int from, int to) onSwap;

  const ReorderableTileGrid({
    Key? key,
    required this.tiles,
    required this.onSwap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    if (tiles.isEmpty) {
      return const Center(
        child: Text(
          '请先选择并处理图片，生成九宫格',
          style: TextStyle(color: Colors.black54),
        ),
      );
    }

    return AspectRatio(
      aspectRatio: 3 / 2, // 3 列，每格近似 2:1
      child: GridView.builder(
        physics: const NeverScrollableScrollPhysics(),
        padding: const EdgeInsets.all(4),
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 3,
          crossAxisSpacing: 4,
          mainAxisSpacing: 4,
        ),
        itemCount: 9,
        itemBuilder: (context, index) {
          final hasImage = index < tiles.length;
          final imgBytes = hasImage ? tiles[index] : null;

          Widget tileWidget;
          if (imgBytes == null) {
            tileWidget = Container(
              decoration: BoxDecoration(
                color: Colors.grey.shade200,
                borderRadius: BorderRadius.circular(6),
              ),
            );
          } else {
            tileWidget = DecoratedBox(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(6),
                border: Border.all(color: Colors.black12),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(6),
                child: Image.memory(
                  imgBytes,
                  fit: BoxFit.cover,
                ),
              ),
            );
          }

          if (!hasImage) {
            // 没有图片的格子，不参与拖拽，只作为空白占位
            return tileWidget;
          }

          return DragTarget<int>(
            onWillAccept: (from) => from != null && from != index,
            onAccept: (from) {
              onSwap(from, index);
            },
            builder: (context, candidate, rejected) {
              final isHighlighted = candidate.isNotEmpty;
              return LongPressDraggable<int>(
                data: index,
                feedback: Material(
                  elevation: 4,
                  borderRadius: BorderRadius.circular(6),
                  child: SizedBox(
                    width: 80,
                    height: 40,
                    child: tileWidget,
                  ),
                ),
                childWhenDragging: Opacity(
                  opacity: 0.3,
                  child: tileWidget,
                ),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 150),
                  decoration: BoxDecoration(
                    boxShadow: isHighlighted
                        ? [
                            BoxShadow(
                              color: Colors.teal.withOpacity(0.4),
                              blurRadius: 8,
                            )
                          ]
                        : null,
                  ),
                  child: tileWidget,
                ),
              );
            },
          );
        },
      ),
    );
  }
}