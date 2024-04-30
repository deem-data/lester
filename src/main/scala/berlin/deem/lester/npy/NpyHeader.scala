package berlin.deem.lester.npy

import java.nio.ByteOrder

class NpyHeader(
  val descr: String,
  val fortranOrder: Boolean,
  val shape: Array[Int]
) {

  val typeParser = """^[<=>]?(\w\d*)$""".r

  val dtype: String = descr match {
    case typeParser(dataType) => dataType
  }

  val numDims: Int = shape.length

  val numElems: Int = if (shape.isEmpty) 1 else shape.product

  val byteOrder: ByteOrder = descr.head match {
    case '>' => ByteOrder.BIG_ENDIAN
    case '<' => ByteOrder.LITTLE_ENDIAN
    case _ => ByteOrder.nativeOrder()
  }

  override def toString: String = {
    val fortran = if (fortranOrder) "True" else "False"
    val shapeStr = if (numDims == 1) shape(0) + "," else shape.mkString(", ")
    s"{'descr': '$descr', 'fortran_order': $fortran, 'shape': ($shapeStr), }"
  }

}

object NpyHeader {
  def apply(headerString: String): NpyHeader = {
    val pattern = """'descr': '([^']+)', 'fortran_order': (True|False), 'shape': \(([^)]+?),?\)""".r.unanchored
    headerString match {
      case pattern(descr, fortranOrderString, shapeString) =>
        val fortranOrder = fortranOrderString == "True"
        val shape = shapeString.trim.split(", ").map(_.toInt)
        new NpyHeader(descr, fortranOrder, shape)
      case _ => throw new Exception("wrong header")
    }
  }
}
