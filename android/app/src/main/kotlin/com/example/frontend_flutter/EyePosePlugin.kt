package com.example.frontend_flutter

import android.graphics.BitmapFactory
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result

// ==== MediaPipe Solutions API ====
import com.google.mediapipe.solutions.facemesh.FaceMesh
import com.google.mediapipe.solutions.facemesh.FaceMeshResult
import com.google.mediapipe.solutions.facemesh.FaceMeshOptions
import com.google.mediapipe.formats.proto.LandmarkProto

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.json.JSONArray
import org.json.JSONObject
import android.util.Log

class EyePosePlugin : FlutterPlugin, MethodCallHandler {

    private lateinit var channel: MethodChannel
    private var faceMesh: FaceMesh? = null

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(binding.binaryMessenger, "eye_pose_plugin")
        channel.setMethodCallHandler(this)

        val options = FaceMeshOptions.builder()
            .setStaticImageMode(true)
            .setRefineLandmarks(true)
            .setMaxNumFaces(1)
            .build()

        faceMesh = FaceMesh(binding.applicationContext, options)
        Log.d("EyePosePlugin", "attached & initialized")
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "processImage" -> {
                val bytes = call.argument<ByteArray>("imageBytes")
                val mesh = faceMesh
                if (bytes == null || mesh == null) {
                    result.error("NO_IMAGE_OR_NOT_READY", "bytes or mesh null", null); return
                }
                try {
                    val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)

                    val latch = CountDownLatch(1)
                    var captured: FaceMeshResult? = null
                    var errorMsg: String? = null

                    mesh.setResultListener { r: FaceMeshResult ->
                        captured = r; latch.countDown()
                    }
                    mesh.setErrorListener { message: String, e: RuntimeException ->
                        errorMsg = if (message.isNotEmpty()) "$message: ${e.message}" else (e.message ?: "unknown")
                        latch.countDown()
                    }

                    mesh.send(bmp)
                    latch.await(5, TimeUnit.SECONDS)

                    errorMsg?.let { result.error("PROCESS_ERROR", it, null); return }

                    val res = captured
                    if (res == null || res.multiFaceLandmarks().isEmpty()) {
                        result.success(JSONObject().put("landmarks", JSONArray()).toString()); return
                    }

                    val lmList: LandmarkProto.NormalizedLandmarkList = res.multiFaceLandmarks()[0]
                    val out = JSONArray()
                    for (p in lmList.landmarkList) {
                        out.put(JSONObject().put("x", p.x).put("y", p.y))
                    }
                    result.success(JSONObject().put("landmarks", out).toString())
                } catch (e: Exception) {
                    result.error("PROCESS_ERROR", e.message, null)
                }
            }
            else -> result.notImplemented()
        }
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        faceMesh = null
        Log.d("EyePosePlugin", "detached")
    }
}