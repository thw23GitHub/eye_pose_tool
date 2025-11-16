package com.example.frontend_flutter

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity: FlutterActivity() {
  override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
    super.configureFlutterEngine(flutterEngine)
    // 手动注册我们放在 app 内部的插件
    flutterEngine.plugins.add(EyePosePlugin())
  }
}