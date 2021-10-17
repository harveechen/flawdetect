package com.rinclab.flawdetect

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect

class BoxGraphic(
    overlay: GraphicOverlay,
    private val box: Box,
    private val imageRect: Rect
) : GraphicOverlay.Graphic(overlay) {

    private val boxPositionPaint: Paint
    private val idPaint: Paint
    private val boxPaint: Paint

    init {
        val selectedColor = Color.WHITE

        boxPositionPaint = Paint()
        boxPositionPaint.color = selectedColor

        idPaint = Paint()
        idPaint.color = selectedColor

        boxPaint = Paint()
        boxPaint.color = selectedColor
        boxPaint.style = Paint.Style.STROKE
        boxPaint.strokeWidth = BOX_STROKE_WIDTH
    }

    override fun draw(canvas: Canvas?) {
//        canvas?.rotate(90F)
        val rect = calculateRect(
            imageRect.height().toFloat(),
            imageRect.width().toFloat(),
            box.toRect()
        )
        canvas?.drawRect(rect, boxPaint)
//        canvas?.restore()
    }

    companion object {
        private const val BOX_STROKE_WIDTH = 5.0f
    }

}