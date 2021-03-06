package com.almuqsitalif08.trashclassificationversion2

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.drawable.toBitmap
import com.almuqsitalif08.trashclassificationversion2.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer


class MainActivity : AppCompatActivity() {

    lateinit var bitmap: Bitmap
    lateinit var imageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView2)
        val selectButton = findViewById<Button>(R.id.button)
        val predictButon = findViewById<Button>(R.id.button2)
        val showPredict = findViewById<TextView>(R.id.textView)
        val camera = findViewById<Button>(R.id.camerabtn)

        bitmap = imageView.drawable.toBitmap()

        selectButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"
            startActivityForResult(intent, 100)
        }

        predictButon.setOnClickListener {
            val labelFile = application.assets.open("mylabels.txt").bufferedReader().use { it.readText() }
            val label = labelFile.split("\n")
            val model = Model.newInstance(this)
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .build()
            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            val tBuffer = imageProcessor.process(tensorImage)
            val byteBuffer = tBuffer.buffer
            inputFeature0.loadBuffer(byteBuffer)
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
            showPredict.text = showOutput(outputFeature0.floatArray, label)
            model.close()
        }

        camera.setOnClickListener {
            checkandGetpermissions()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if(data != null){
            if (requestCode == 100){
                val uri = data.data
                imageView.setImageURI(uri)
                bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver,  uri)
            } else if (requestCode == 101 && resultCode == Activity.RESULT_OK){
                bitmap = data.extras?.get("data") as Bitmap
                bitmap = bitmap.copy(Bitmap.Config.ARGB_8888,true)
                imageView.setImageBitmap(bitmap)
            }
        }
    }

    private fun showOutput(arr: FloatArray, listString: List<String>): String {
        var index = 0
        var maxValue = 0.0f

        for (i in 0..arr.lastIndex) {
            if (arr[i] > maxValue) {
                index = i
                maxValue = arr[i]
            }
        }

        return listString[index].capitalize() + " " + (maxValue * 100).toInt().toString() + "%"
    }

    private fun checkandGetpermissions(){
        if(if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED
            } else {
                false
            }
        ){
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 200)
            }
        }
        else{
            Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show()
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(intent, 101)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(requestCode == 200){
            if(grantResults[0] == PackageManager.PERMISSION_GRANTED)
            {
                Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show()
                val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(intent, 101)
            }
            else{
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show()
            }
        }
    }
}