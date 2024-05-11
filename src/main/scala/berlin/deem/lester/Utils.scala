package berlin.deem.lester

import scala.reflect.runtime.universe._
import scala.reflect.runtime.{currentMirror => cm}
import berlin.deem.lester.npy.NpyWriter
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.ml.linalg.{BreezeConverter, SparseVector}
import org.apache.spark.sql.DataFrame

import java.io.{File, PrintWriter}
import java.nio.file.{Files, Paths, StandardCopyOption}

object Utils {

  private def deleteRecursively(file: File): Unit = {
    if (file.isDirectory) {
      file.listFiles.foreach(deleteRecursively)
    }
    if (file.exists && !file.delete) {
      throw new Exception(s"Unable to delete ${file.getAbsolutePath}")
    }
  }

  def makeCallable(cls: Class[_], methodName: String): MethodMirror = {
    val classSymbol = cm.classSymbol(cls)
    val ctorSymbol = classSymbol.primaryConstructor.asMethod
    val ctor = cm.reflectClass(classSymbol).reflectConstructor(ctorSymbol)
    val methodSymbol = classSymbol.toType.decl(TermName(methodName)).asMethod

    val instance = cls.cast(ctor())
    cm.reflect(instance).reflectMethod(methodSymbol)
  }

  def sourcePathsToJson(dataMap: Map[String, String]): String = {
    dataMap.map { case (key, value) =>
      s""""$key":"$value""""
    }.mkString("{", ", ", "}")
  }

  def matrixColumnProvenanceToJson(dataMap: Map[String, (Int, Int)]): String = {
    val jsonEntries = dataMap.map { case (column, (start, stop)) =>
      val jsonSlice = s"""[${start}, ${stop}]"""
      s""""${column}": $jsonSlice"""
    }
    jsonEntries.mkString("{", ", ", "}")
  }

  def columnProvenanceToJson(dataMap: Map[String, Array[String]]): String = {
    val jsonEntries = dataMap.map { case (key, values) =>
      val jsonValues = values.map(value => s""""$value"""").mkString("[", ", ", "]")
      s""""$key": $jsonValues"""
    }
    jsonEntries.mkString("{", ", ", "}")
  }

  def asSingleParquetFile(df: DataFrame, artifactDirectoryPath: String, fileName: String): Unit = {

    df.coalesce(1)
      .write.parquet(s"${artifactDirectoryPath}/__${fileName}")

    val directory = new File(s"${artifactDirectoryPath}/__${fileName}")
    val files = directory.listFiles().filter { file =>
      file.getName.startsWith("part") && file.getName.endsWith("snappy.parquet")
    }

    files.headOption.foreach { file =>
      Files.move(
        Paths.get(file.getAbsolutePath),
        Paths.get(s"${artifactDirectoryPath}/${fileName}"),
        StandardCopyOption.REPLACE_EXISTING
      )
    }

    deleteRecursively(directory)
  }

  def vectorAsNpyFile(vectorDf: DataFrame, path: String): Unit = {
    val elems = vectorDf
      .collect()
      .map { _.get(0).asInstanceOf[Double] }

    NpyWriter.writeVector(path, DenseVector(elems: _*))
  }

  def matrixAsNpyFile(vectorsDf: DataFrame, path: String): Unit = {
    val rowVectors = vectorsDf
      .collect()
      .map { row =>
        val sparkVector = row.get(0).asInstanceOf[SparseVector]
        val breezeVector = BreezeConverter.asBreeze(sparkVector)
        breezeVector.toDenseVector
      }

    NpyWriter.writeMatrix(path, DenseMatrix(rowVectors: _*))
  }

  def jsonToFile(json: String, path: String): Unit = {
    val writer = new PrintWriter(new File(path))
    try {
      writer.write(json)
    } finally {
      writer.close()
    }
  }

}
