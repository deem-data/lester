import berlin.deem.lester.pipeline.MatrixColumnProvenance
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{HashingTF, OneHotEncoder, OneHotEncoderModel, StandardScaler, StringIndexer, StringIndexerModel, Tokenizer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class Graph {

  private val in: mutable.HashMap[String, ArrayBuffer[String]] = mutable.HashMap.empty
  private val out: mutable.HashMap[String, ArrayBuffer[String]] = mutable.HashMap.empty
  private val weights: mutable.HashMap[(String, String), Int] = mutable.HashMap.empty

  def addNode(node: String): Unit = {
    if (!in.contains(node)) {
      in(node) = ArrayBuffer.empty
    }
    if (!out.contains(node)) {
      out(node) = ArrayBuffer.empty
    }
  }

  def nodes(): Iterator[String] = {
    in.keysIterator
  }

  def addEdge(source: String, target: String): Unit = {
    out(source).append(target)
    in(target).append(source)
  }

  def addWeight(source: String, target: String, weight: Int): Unit = {
    weights((source, target)) = weight
  }

  def weight(source: String, target: String): Option[Int] = {
    weights.get(source, target)
  }

  def inDegree(node: String): Int = {
    in(node).length
  }

  def outDegree(node: String): Int = {
    out(node).length
  }

  def successors(node: String): Iterator[String] = {
    out(node).iterator
  }

  def predecessors(node: String): Iterator[String] = {
    in(node).iterator
  }

}


object playing extends App {

  val spark = SparkSession.builder()
    .master("local[2]")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()

  var df = spark.read.option("header", "true").csv("data/books.csv")
  df = df.withColumn("work_text_reviews_count", col("work_text_reviews_count").cast("integer"))
  df = df.withColumn("ratings_count", col("ratings_count").cast("integer"))
  df = df.na.drop()

  val stages = ArrayBuffer[PipelineStage]()

  val categoricalColumns = Array("original_publication_year", "authors")
  for (column <- categoricalColumns) {
    val indexer = new StringIndexer()
      .setInputCol(column)
      .setOutputCol(column + "Index")
      .setHandleInvalid("keep")

    stages += indexer
  }

  val categoricalIndexes = categoricalColumns.map(column => column + "Index")
  val categoricalFeatures = categoricalColumns.map(column => column + "Vec")

  val encoder = new OneHotEncoder()
    .setInputCols(categoricalIndexes)
    .setOutputCols(categoricalFeatures)

  stages += encoder

  val numericalColumns = Array("work_text_reviews_count", "ratings_count")

  stages += new VectorAssembler()
    .setInputCols(numericalColumns)
    .setOutputCol("numericalFeaturesRaw")

  stages += new StandardScaler()
    .setInputCol("numericalFeaturesRaw")
    .setOutputCol("numericalFeatures")
    .setWithMean(true)

  stages += new Tokenizer()
    .setInputCol("title")
    .setOutputCol("words")

  stages += new HashingTF()
    .setInputCol("words")
    .setOutputCol("textFeatures")
    .setNumFeatures(100)

  stages += new VectorAssembler()
    .setInputCols(categoricalFeatures ++ Array("numericalFeatures", "textFeatures"))
    .setOutputCol("features")

  val featureTransform = new Pipeline()
    .setStages(stages.toArray)





  val fittedFeatureTransform = featureTransform.fit(df)

  /*
  val G = new Graph()

  for (transformer <- fittedFeatureTransform.stages) {

    var inputs: Array[String] = Array.empty
    var outputs: Array[String] = Array.empty
    var isMerge: Boolean = false

    for (param <- transformer.params) {
      if (param.name == "inputCol" && transformer.isSet(param)) {
        inputs = Array(transformer.getOrDefault(param).asInstanceOf[String])
      }
      if (param.name == "inputCols" && transformer.isSet(param)) {
        inputs = transformer.getOrDefault(param).asInstanceOf[Array[String]]
      }
      if (param.name == "outputCol" && transformer.isSet(param)) {
        outputs = Array(transformer.getOrDefault(param).asInstanceOf[String])
      }
      if (param.name == "outputCols" && transformer.isSet(param)) {
        outputs = transformer.getOrDefault(param).asInstanceOf[Array[String]]
      }
    }

    if (transformer.isInstanceOf[VectorAssembler]) {
      isMerge = true
    }

    val outputSizes: Option[Array[Int]] =
      transformer match {
        case t: HashingTF =>
          val numFeaturesParam = t.params.find(_.name == "numFeatures").get
          val numFeatures: Int = transformer.getOrDefault(numFeaturesParam).asInstanceOf[Int]
          Some(Array(numFeatures))

        case t: OneHotEncoderModel =>
          // TODO there is a private method in the class which has the logic for all settings
          val sizes = t.categorySizes
            .map(_ - 1)
            .toArray
          Some(sizes)

       case t: StringIndexerModel =>
          val sizes = t.labelsArray.map(_.length).toArray
          Some(sizes)

        case _ => None
      }

    println(transformer.uid)
    println("\t" + inputs.mkString(", "))
    println("\t" + outputs.mkString(", "))

    for (column <- inputs) {
      G.addNode(column)
    }

    for (column <- outputs) {
      G.addNode(column)
    }

    if (!isMerge) {
      outputSizes match {
        case Some(sizes) =>
          for (((inputColumn, outputColumn), size) <- inputs.zip(outputs).zip(sizes)) {
            G.addEdge(inputColumn, outputColumn)
            G.addWeight(inputColumn, outputColumn, size)
          }
        case _ =>
          for ((inputColumn, outputColumn) <- inputs.zip(outputs)) {
            G.addEdge(inputColumn, outputColumn)
          }
      }
    } else {
      val outputColumn = outputs.head
      for (inputColumn <- inputs) {
        G.addEdge(inputColumn, outputColumn)
      }
    }
  }

  def walkPath(
    node: String,
    G: Graph,
    path: ArrayBuffer[(String, Int)],
    lastSize: Int
  ): (ArrayBuffer[(String, Int)], Int) = {

    var newLastSize = lastSize

    if (G.outDegree(node) == 0) {
      (path, newLastSize)
    } else {
      val successor = G.successors(node).toArray.head
      G.weight(node, successor) match {
        case Some(weight) =>
          newLastSize = weight
        case _ =>
      }

      var order = 0
      for ((successorPredecessor, index) <- G.predecessors(successor).zipWithIndex) {
        if (successorPredecessor == node) {
          order = index
        }
      }

      path.append((successor, order))

      walkPath(successor, G, path, newLastSize)
    }

  }

  val sources = G.nodes().filter(G.inDegree(_) == 0)

  import scala.math.Ordering
  implicit val arrayOrdering: Ordering[Array[Int]] = Ordering.by(_.toIterable)

  val mappings = sources
    .map({ source =>
      val (path, size) = walkPath(source, G, ArrayBuffer[(String, Int)](), 1)
      val orders = path.map({ case (_, order) => order }).toArray
      println(s"${source}: ${path} ${orders.mkString(",")}")
      (source, size, orders)
    })
    .toArray
    .sortBy({ case (_, _, orders) => orders.reverse })
    .map({ case (source, size, _) => source -> size })

  var matrixColumnProvenance: mutable.HashMap[String, Array[Int]] = mutable.HashMap.empty
  var currentIndex = 0
  for((source, size) <- mappings) {
    matrixColumnProvenance += source -> Array(currentIndex, currentIndex + size)
    currentIndex += size
  }*/
  val matrixColumnProvenance = MatrixColumnProvenance.computeMatrixColumnProvenance(fittedFeatureTransform)

  matrixColumnProvenance foreach { case (source, slice) => println(s"${source}: ${slice}")}

  spark.stop()
  System.exit(0)
}
