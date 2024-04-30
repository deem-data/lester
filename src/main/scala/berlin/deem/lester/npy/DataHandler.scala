package berlin.deem.lester.npy

import java.nio.{ByteBuffer, ByteOrder}
import scala.reflect._

abstract class DataHandler[A: ClassTag] {

  def descr: Option[String] = None

  def convert[B](elem: B): A

  def toByteArray(data: Array[A]): Array[Byte]

  def mkArray(header: NpyHeader, data: Array[_]): Array[A] = {
    header.dtype match {
      case "i1" | "b" => data.asInstanceOf[Array[Byte]].map(convert)
      case "i2" => data.asInstanceOf[Array[Short]].map(convert)
      case "i4" | "i" | "u1" | "u2" => data.asInstanceOf[Array[Int]].map(convert)
      case "i8" | "u4" => data.asInstanceOf[Array[Long]].map(convert)
      case "f4" | "f" | "f2" => data.asInstanceOf[Array[Float]].map(convert)
      case "f8" => data.asInstanceOf[Array[Double]].map(convert)
      case dtype => throw new Exception(s"unsupported type '$dtype'")
    }
  }

}

class BooleanHandler extends DataHandler[Boolean] {

  override def descr: Option[String] = Some("b")

  def convert[B](elem: B): Boolean = elem != 0

  def toByteArray(data: Array[Boolean]): Array[Byte] = {
    data.map(if (_) 1.toByte else 0.toByte)
  }

}

class ByteHandler extends DataHandler[Byte] {

  override def descr: Option[String] = Some("i1")

  def convert[B](elem: B): Byte = elem match {
    case n: Byte => n
    case n: Short => n.toByte
    case n: Int => n.toByte
    case n: Long => n.toByte
    case n: Float => n.toByte
    case n: Double => n.toByte
  }

  def toByteArray(data: Array[Byte]): Array[Byte] = data

}

class ShortHandler extends DataHandler[Short] {

  override def descr: Option[String] = Some("i2")

  def convert[B](elem: B): Short = elem match {
    case n: Byte => n.toShort
    case n: Short => n
    case n: Int => n.toShort
    case n: Long => n.toShort
    case n: Float => n.toShort
    case n: Double => n.toShort
  }

  def toByteArray(data: Array[Short]): Array[Byte] = {
    val array = new Array[Byte](data.length * 2)
    val bb = ByteBuffer.wrap(array)
    bb.order(ByteOrder.BIG_ENDIAN)
    val sb = bb.asShortBuffer()
    sb.put(data)
    array
  }

}

class IntHandler extends DataHandler[Int] {

  override def descr: Option[String] = Some("i4")

  def convert[B](elem: B): Int = elem match {
    case n: Byte => n.toInt
    case n: Short => n.toInt
    case n: Int => n
    case n: Long => n.toInt
    case n: Float => n.toInt
    case n: Double => n.toInt
  }

  def toByteArray(data: Array[Int]): Array[Byte] = {
    val array = new Array[Byte](data.length * 4)
    val bb = ByteBuffer.wrap(array)
    bb.order(ByteOrder.BIG_ENDIAN)
    val sb = bb.asIntBuffer()
    sb.put(data)
    array
  }

}

class LongHandler extends DataHandler[Long] {

  override def descr: Option[String] = Some("i8")

  def convert[B](elem: B): Long = elem match {
    case n: Byte => n.toLong
    case n: Short => n.toLong
    case n: Int => n.toLong
    case n: Long => n
    case n: Float => n.toLong
    case n: Double => n.toLong
  }

  def toByteArray(data: Array[Long]): Array[Byte] = {
    val array = new Array[Byte](data.length * 8)
    val bb = ByteBuffer.wrap(array)
    bb.order(ByteOrder.BIG_ENDIAN)
    val sb = bb.asLongBuffer()
    sb.put(data)
    array
  }

}

class FloatHandler extends DataHandler[Float] {

  override def descr: Option[String] = Some("f4")

  def convert[B](elem: B): Float = elem match {
    case n: Byte => n.toFloat
    case n: Short => n.toFloat
    case n: Int => n.toFloat
    case n: Long => n.toFloat
    case n: Float => n
    case n: Double => n.toFloat
  }

  def toByteArray(data: Array[Float]): Array[Byte] = {
    val array = new Array[Byte](data.length * 4)
    val bb = ByteBuffer.wrap(array)
    bb.order(ByteOrder.BIG_ENDIAN)
    val sb = bb.asFloatBuffer()
    sb.put(data)
    array
  }

}

class DoubleHandler extends DataHandler[Double] {

  override def descr: Option[String] = Some("f8")

  def convert[B](elem: B): Double = elem match {
    case n: Byte => n.toDouble
    case n: Short => n.toDouble
    case n: Int => n.toDouble
    case n: Long => n.toDouble
    case n: Float => n.toDouble
    case n: Double => n
  }

  def toByteArray(data: Array[Double]): Array[Byte] = {
    val array = new Array[Byte](data.length * 8)
    val bb = ByteBuffer.wrap(array)
    bb.order(ByteOrder.BIG_ENDIAN)
    val sb = bb.asDoubleBuffer()
    sb.put(data)
    array
  }

}

