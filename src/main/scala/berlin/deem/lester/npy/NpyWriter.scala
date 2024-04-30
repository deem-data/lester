package berlin.deem.lester.npy

import breeze.linalg.{DenseMatrix, _}
import java.nio.{ ByteOrder, ByteBuffer }
import java.io._
import org.apache.commons.io.FileUtils

object NpyWriter {

  def writeMatrix(path: String, mat: DenseMatrix[Double]): Unit = {
    val file = new File(path)
    val array = mkDataArray(mat.data, Array(mat.rows, mat.cols), !mat.isTranspose)
    FileUtils.writeByteArrayToFile(file, array)
  }

  def writeVector(path: String, vec: DenseVector[Double]): Unit = {
    val file = new File(path)
    val array = mkDataArray(vec.data, Array(vec.length), true)
    FileUtils.writeByteArrayToFile(file, array)
  }

  /*
  def write[A: DataHandler](path: String, mat: DenseMatrix[A]): Unit = {
    write(new File(path), mat)
  }


  def write[A: DataHandler](file: File, mat: DenseMatrix[A]): Unit = {
    val array = mkDataArray(mat.data, Array(mat.rows, mat.cols), !mat.isTranspose)
    FileUtils.writeByteArrayToFile(file, array)
  }

  def write[A: DataHandler](path: String, vec: DenseVector[A]): Unit = {
    write(new File(path), vec)
  }

  def write[A: DataHandler](file: File, vec: DenseVector[A]): Unit = {
    val array = mkDataArray(vec.data, Array(vec.length), true)
    FileUtils.writeByteArrayToFile(file, array)
  }*/

  def mkDataArray(
     data: Array[Double],
     shape: Array[Int],
     order: Boolean
   ): Array[Byte] = {
    //val handler = implicitly[DataHandler[A]]
    val handler = new DoubleHandler
    if (handler.descr.isEmpty) throw new Exception("can't serialize type")
    val descr = ">" + handler.descr.get
    val header = new NpyHeader(descr, order, shape)
    val headerStr = header.toString
    val remaining = (header.toString.length + 11) % 16
    val padLen = if (remaining > 0) 16 - remaining else 0
    val headerLen = headerStr.length + padLen + 1
    val dataBytes = handler.toByteArray(data)
    val size = headerStr.length + 11 + padLen + dataBytes.length
    val array = new Array[Byte](size)
    val bb = ByteBuffer.wrap(array)
    bb.put(Array(0x93.toByte, 'N'.toByte, 'U'.toByte, 'M'.toByte, 'P'.toByte,
      'Y'.toByte, 1.toByte, 0.toByte))
    bb.order(ByteOrder.LITTLE_ENDIAN)
    bb.putShort(headerLen.toShort)
    bb.order(ByteOrder.BIG_ENDIAN)
    bb.put(headerStr.getBytes)
    bb.put(Array.fill(padLen)(' '.toByte))
    bb.put('\n'.toByte)
    bb.put(dataBytes)
    array
  }

}
