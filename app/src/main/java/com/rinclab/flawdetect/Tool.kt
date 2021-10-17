package com.rinclab.flawdetect

import android.content.res.TypedArray
import android.graphics.Rect
import android.util.Log
import androidx.core.graphics.times
import androidx.core.graphics.toRegion
import com.google.common.collect.ImmutableSet
import com.google.common.collect.Sets
import org.apache.commons.math3.util.CombinatoricsUtils
import org.opencv.calib3d.Calib3d.RANSAC
import org.opencv.calib3d.Calib3d.findHomography
import org.opencv.core.*
import org.opencv.core.Core.*
import org.opencv.core.CvType.*
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.COLOR_RGB2GRAY
import org.opencv.imgproc.Imgproc.warpPerspective
import kotlin.math.max
import kotlin.math.min

const val MAX_FEATURES = 5000

class Box (rectInfo: IntArray) {
    val x = rectInfo[0]
    val y = rectInfo[1]
    val w = rectInfo[2]
    val h = rectInfo[3]
    val area = w * h
    var rect = Rect(x, y, x+w, y+h)
}

fun Box.toRect(): Rect {
    return Rect(x, y, x+w, y+h)
}

fun process(base: Mat, target: Mat): Mat {
    val (baseReg, imgMask) = register(base, target)
    val targetMask = Mat()
    multiply(target, imgMask, targetMask)
    detect(baseReg, targetMask)

    return baseReg
}

fun register(base: Mat, target: Mat): Pair<Mat, Mat> {
    val baseGray = Mat()
    Imgproc.cvtColor(base, baseGray, Imgproc.COLOR_RGB2GRAY)

    val targetGray = Mat()
    Imgproc.cvtColor(target, targetGray, Imgproc.COLOR_RGB2GRAY)

    val orb = ORB.create(MAX_FEATURES)

    val kp1 = MatOfKeyPoint()
    val des1 = Mat()
    orb.detect(baseGray, kp1)
    orb.compute(baseGray, kp1, des1)

    val kp2 = MatOfKeyPoint()
    val des2 = Mat()
    orb.detect(targetGray, kp2)
    orb.compute(targetGray, kp2, des2)

    val matcher = BFMatcher.create(NORM_HAMMING)
    val matches = MatOfDMatch()
    // some matches are not good enough
    // TODO
    try {
        matcher.match(des1, des2, matches)
    } catch (e: CvException) {
        Log.e("Cv", des1.size().toString() + ", " + des2.size().toString())
        Log.e("Cv", e.message+"")
    }


    val matchesList = matches.toList()

    val kp1List = kp1.toList()
    val kp2List = kp2.toList()

    val srcPts = MatOfPoint2f()
    val dstPts = MatOfPoint2f()
    srcPts.fromList(matchesList.map { kp1List[it.queryIdx].pt })
    dstPts.fromList(matchesList.map { kp2List[it.trainIdx].pt })
    val matM = findHomography(srcPts, dstPts, RANSAC, 0.5)

    val imgOnes = Mat(target.size(), CV_8UC3, Scalar.all(1.0))

    val imgMask = Mat()
    warpPerspective(imgOnes, imgMask, matM, target.size())

    val baseReg = Mat()
    warpPerspective(base, baseReg, matM, baseGray.size())
//    baseReg.convertTo(baseReg, CV_8UC3)

    return Pair(baseReg, imgMask)
}

fun Mat.smooth(): Mat{
    val mat = Mat()
    this.copyTo(mat)
    Imgproc.medianBlur(mat, mat, 7)
    normalize(mat, mat, 0.0, 128.0, NORM_MINMAX)
    Imgproc.cvtColor(mat, mat, COLOR_RGB2GRAY)
    Imgproc.medianBlur(mat, mat, 5)
//    mat.convertTo(mat, CV_32S)
    return mat
}

fun detect(baseReg: Mat, targetMask: Mat): List<Box> {
    val img1 = baseReg.smooth()
    val img2 = targetMask.smooth()

    val diffImage = Mat()
    absdiff(img1, img2, diffImage)

    val bw = Mat()
    val bwThreshold = 5.0
    Imgproc.threshold(diffImage, bw, bwThreshold, 255.0, Imgproc.THRESH_BINARY_INV)

////    val mask = Mat()
////    compare(diffImage, Scalar(10.0), mask, CMP_GT)
////    diffImage.setTo(Scalar(255.0), mask)
//
//    val changeMap = Mat()
//    bw.convertTo(changeMap, CV_8U)
    val kernel = Mat.ones(3,3, CV_8U)
    val cleanChangeMap = Mat()
    Imgproc.erode(bw, cleanChangeMap, kernel)
    Imgproc.dilate(cleanChangeMap, cleanChangeMap, kernel)

    return getBox(bw)
}

fun getBox(img: Mat): List<Box> {
    val imgLabel = Mat()
    val props = Mat()
    val centroid = Mat()
    val thresholdMin = img.rows() * img.cols() * 0.001
    val thresholdMax = img.rows() * img.cols() * 0.5
    Imgproc.connectedComponentsWithStats(img, imgLabel, props, centroid)

    val resArea = mutableListOf<Box>()
    for (i in 0 until props.rows()) {
        val rectInfo = IntArray(5)
        props.row(i).get(0,0, rectInfo)
        val box = Box(rectInfo)
        if (box.area > thresholdMin && box.area < thresholdMax) {
            resArea.add(box)
        }
    }

    resArea.sortByDescending { it.area }

    val areaList = resArea.take(min(resArea.size, 4)).toMutableList()
    return merge(areaList)
}

fun merge(areaList: MutableList<Box>): List<Box> {
    if (areaList.size > 1) {
        while (true) {
            var found = 0
            val combinations =  CombinatoricsUtils.combinationsIterator(areaList.size, 2)

            for ((i, j) in combinations) {
                val a = areaList[i]
                val b = areaList[j]
                if (Rect.intersects(a.rect, b.rect)) {
                    a.rect.union(b.rect)
                    areaList.remove(b)
                    found = 1
                    break
                }
            }

            if (found == 0 || areaList.size == 1) break
        }
    }

    return areaList
}

fun union(a: Box, b: Box): Box {
    val x = max(a.x, b.x)
    val y = max(a.y, b.y)
    val w = min(a.x + a.w, b.x + b.w) - x
    val h = min(a.y + a.h, b.y + b.h) - y

    return Box(intArrayOf(x, y, w, h, w*h))
}

fun intersection(a: Box, b: Box): Boolean {
    val x = max(a.x, b.x)
    val y = max(a.y, b.y)
    val w = min(a.x + a.w, b.x + b.w) - x
    val h = min(a.y + a.h, b.y + b.h) - y

    return !(w < 0 || h < 0)
}