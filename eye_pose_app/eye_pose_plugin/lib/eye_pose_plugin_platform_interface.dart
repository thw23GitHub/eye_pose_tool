import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'eye_pose_plugin_method_channel.dart';

abstract class EyePosePluginPlatform extends PlatformInterface {
  /// Constructs a EyePosePluginPlatform.
  EyePosePluginPlatform() : super(token: _token);

  static final Object _token = Object();

  static EyePosePluginPlatform _instance = MethodChannelEyePosePlugin();

  /// The default instance of [EyePosePluginPlatform] to use.
  ///
  /// Defaults to [MethodChannelEyePosePlugin].
  static EyePosePluginPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [EyePosePluginPlatform] when
  /// they register themselves.
  static set instance(EyePosePluginPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
