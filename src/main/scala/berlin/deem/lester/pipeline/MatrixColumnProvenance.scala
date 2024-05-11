package berlin.deem.lester.pipeline

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{HashingTF, OneHotEncoderModel, StringIndexerModel, VectorAssembler}

import scala.annotation.tailrec
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object MatrixColumnProvenance {

  @tailrec
  private def walkPath(
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

  def computeMatrixColumnProvenance(fittedFeatureTransform: PipelineModel): Map[String, (Int, Int)] = {

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

    val sources = G.nodes().filter(G.inDegree(_) == 0)

    import scala.math.Ordering
    implicit val arrayOrdering: Ordering[Array[Int]] = Ordering.by(_.toIterable)

    val mappings = sources
      .map({ source =>
        val (path, size) = walkPath(source, G, ArrayBuffer[(String, Int)](), 1)
        val orders = path.map({ case (_, order) => order }).toArray
        // println(s"${source}: ${path} ${orders.mkString(",")}")
        (source, size, orders)
      })
      .toArray
      .sortBy({ case (_, _, orders) => orders.reverse })
      .map({ case (source, size, _) => source -> size })

    val matrixColumnProvenance: mutable.HashMap[String, (Int, Int)] = mutable.HashMap.empty
    var currentIndex = 0
    for((source, size) <- mappings) {
      matrixColumnProvenance += source -> (currentIndex, currentIndex + size)
      currentIndex += size
    }

    matrixColumnProvenance.toMap
  }

}
