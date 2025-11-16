import 'package:flutter_test/flutter_test.dart';
import 'package:eye_pose_plugin/eye_pose_plugin.dart';
import 'package:eye_pose_plugin/eye_pose_plugin_platform_interface.dart';
import 'package:eye_pose_plugin/eye_pose_plugin_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockEyePosePluginPlatform
    with MockPlatformInterfaceMixin
    implements EyePosePluginPlatform {

  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  final EyePosePluginPlatform initialPlatform = EyePosePluginPlatform.instance;

  test('$MethodChannelEyePosePlugin is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelEyePosePlugin>());
  });

  test('getPlatformVersion', () async {
    EyePosePlugin eyePosePlugin = EyePosePlugin();
    MockEyePosePluginPlatform fakePlatform = MockEyePosePluginPlatform();
    EyePosePluginPlatform.instance = fakePlatform;

    expect(await eyePosePlugin.getPlatformVersion(), '42');
  });
}
