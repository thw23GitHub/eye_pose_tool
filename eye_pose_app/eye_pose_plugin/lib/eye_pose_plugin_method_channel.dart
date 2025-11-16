import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'eye_pose_plugin_platform_interface.dart';

/// An implementation of [EyePosePluginPlatform] that uses method channels.
class MethodChannelEyePosePlugin extends EyePosePluginPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('eye_pose_plugin');

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>('getPlatformVersion');
    return version;
  }
}
