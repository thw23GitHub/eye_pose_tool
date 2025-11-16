package com.example.frontend_flutter   // ← 如果你改过包名，这里改成你的实际包名

import android.content.ContentValues
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {

  private val CHANNEL = "eye_pose_native"

  override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
    super.configureFlutterEngine(flutterEngine)

    MethodChannel(
      flutterEngine.dartExecutor.binaryMessenger,
      CHANNEL
    ).setMethodCallHandler { call, result ->
      when (call.method) {
        "saveToGallery" -> {
          val bytes = call.argument<ByteArray>("bytes")
          val filename = call.argument<String>("filename")

          if (bytes == null || filename.isNullOrEmpty()) {
            result.error(
              "INVALID_ARGS",
              "bytes 或 filename 为空",
              null
            )
            return@setMethodCallHandler
          }

          saveImageToGallery(bytes, filename, result)
        }

        else -> result.notImplemented()
      }
    }
  }

  /**
   * 把 JPEG 图片字节保存到系统相册（Pictures/EyePose）
   */
  private fun saveImageToGallery(
    bytes: ByteArray,
    filename: String,
    result: MethodChannel.Result
  ) {
    try {
      val resolver = applicationContext.contentResolver

      // 目标集合：外部存储上的图片集合
      val imageCollection = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
      } else {
        MediaStore.Images.Media.EXTERNAL_CONTENT_URI
      }

      val values = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, filename)
        put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
          // 保存到 Pictures/EyePose 目录下
          put(
            MediaStore.Images.Media.RELATIVE_PATH,
            "Pictures/EyePose"
          )
          // 写入中
          put(MediaStore.Images.Media.IS_PENDING, 1)
        }
      }

      val uri = resolver.insert(imageCollection, values)
      if (uri == null) {
        Log.e("EyePoseNative", "Failed to insert MediaStore record")
        result.success(false)
        return
      }

      try {
        resolver.openOutputStream(uri)?.use { out ->
          out.write(bytes)
          out.flush()
        }

        // 标记写入完成
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
          val update = ContentValues().apply {
            put(MediaStore.Images.Media.IS_PENDING, 0)
          }
          resolver.update(uri, update, null, null)
        }

        Log.i("EyePoseNative", "Image saved: $uri")
        result.success(true)
      } catch (e: Exception) {
        Log.e("EyePoseNative", "Error writing image data", e)
        result.error("SAVE_FAILED", e.message, null)
      }
    } catch (e: Exception) {
      Log.e("EyePoseNative", "saveImageToGallery exception", e)
      result.error("SAVE_FAILED", e.message, null)
    }
  }
}