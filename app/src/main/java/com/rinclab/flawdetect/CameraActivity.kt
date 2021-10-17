package com.rinclab.flawdetect

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.media.Image
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.rinclab.flawdetect.databinding.ActivityCameraBinding
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.util.concurrent.Executors
import kotlin.random.Random
import org.opencv.core.Core
import org.opencv.core.CvException


class CameraActivity : AppCompatActivity() {
    private lateinit var activityCameraBinding: ActivityCameraBinding
    private val permissions = listOf(Manifest.permission.CAMERA)
    private val permissionsRequestCode = Random.nextInt(0, 10000)
    private val executor = Executors.newSingleThreadExecutor()

    private lateinit var imgMat: Mat
    private lateinit var baseMat: Mat

    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK
    private var flag: Boolean = false

    private var passFlag = false
    private var canDetect = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        OpenCVLoader.initDebug()

        activityCameraBinding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(activityCameraBinding.root)

        activityCameraBinding.controlButton.isEnabled = false

        activityCameraBinding.controlButton.setOnClickListener {
            it.isEnabled = false
            if (!canDetect) {
                activityCameraBinding.controlButton.text = "Stop Detect"
                canDetect = true
            } else {
                activityCameraBinding.controlButton.text = "Start Detect"
                canDetect = false
            }
            it.isEnabled = true
        }

        activityCameraBinding.cameraCaptureButton.setOnClickListener {
            it.isEnabled = false

            if (!flag) {
                activityCameraBinding.cameraCaptureButton.text = "Capture Again"
            }

            flag = false

            baseMat = imgMat
            activityCameraBinding.imageBase.setImageBitmap(baseMat.toBitmap())
            activityCameraBinding.imageBase.visibility = View.VISIBLE
            flag = true
            it.isEnabled = true
            activityCameraBinding.controlButton.isEnabled = true
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() = activityCameraBinding.viewFinder.post {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
                .build()

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(executor, ImageAnalysis.Analyzer { imageProxy ->
                if (passFlag) {
                    imageProxy.close()
                    return@Analyzer
                }

                imageProxy.use { proxy ->
                    val mediaImage = proxy.image
                    mediaImage?.let {
                        imgMat = it.yuvToRgba()
                        if (flag && canDetect) {
                            passFlag = true
                            val test = imgMat
                            val (baseReg, imgMask) = register(baseMat, imgMat)
                            val targetMask = Mat()
                            Core.multiply(imgMat, imgMask, targetMask)
                            val result = detect(baseReg, targetMask)
                            val boxes = listOf(Box(intArrayOf(100, 100, 100, 100, 10000)))//detect(baseReg, targetMask)
                            prediction(result, mediaImage)
                            //Core.rotate(baseReg, baseReg, Core.ROTATE_90_CLOCKWISE)
                            activityCameraBinding.imageBase.setImageBitmap(baseReg.toBitmap())

                            if (result.isNotEmpty()) {
                                activityCameraBinding.result.post{
                                    activityCameraBinding.result.text = result.size.toString() +", " + result[0].x.toString() + ", " + result[0].y.toString() + "," + result[0].w.toString()
                                }
                            }

                            passFlag = false
                        }
                    }
                }
            })

            val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this as LifecycleOwner, cameraSelector, preview, imageAnalysis)

            preview.setSurfaceProvider(activityCameraBinding.viewFinder.surfaceProvider)

        }, ContextCompat.getMainExecutor(this))
    }

    private fun prediction(results: List<Box>, image: Image) {
        val graphicOverlay = activityCameraBinding.graphicOverlay

        graphicOverlay.clear()
        results.forEach {
            val faceGraphic = BoxGraphic(graphicOverlay, it, image.cropRect)
            graphicOverlay.add(faceGraphic)
        }
        graphicOverlay.postInvalidate()
    }


    private fun hasPermission(context: Context) = permissions.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onResume() {
        super.onResume()

        if (!hasPermission(this)) {
            ActivityCompat.requestPermissions(
                this, permissions.toTypedArray(), permissionsRequestCode
            )
        } else {
            bindCameraUseCases()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == permissionsRequestCode && hasPermission(this)) {
            bindCameraUseCases()
        } else {
            finish()
        }
    }
}

private fun Mat.toBitmap(): Bitmap {
    val bitmap = Bitmap.createBitmap(cols(), rows(), Bitmap.Config.ARGB_8888)
    try {
        Utils.matToBitmap(this, bitmap)
    } catch (e: CvException) {
        Log.e("Cv", e.message+"")
    }

    return bitmap
}

fun Image.yuvToRgba(): Mat {
    val nv21: ByteArray

    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize: Int = yBuffer.remaining()
    val uSize: Int = uBuffer.remaining()
    val vSize: Int = vBuffer.remaining()

    nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuv = Mat(height + height / 2, width, CvType.CV_8UC1)
    yuv.put(0, 0, nv21)
    val rgb = Mat()
    Imgproc.cvtColor(yuv, rgb, Imgproc.COLOR_YUV2RGB_NV21, 3)
    //Core.rotate(rgb, rgb, Core.ROTATE_90_CLOCKWISE)
    return rgb
}