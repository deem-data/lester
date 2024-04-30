package berlin.deem.lester.example.highlyratedbooks

import berlin.deem.lester.context.TrainModel
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.LogisticRegression

import scala.collection.mutable.ArrayBuffer

class LearnWithSparkML {

  @TrainModel
  def logreg(): Pipeline = {
    val nextStages = new ArrayBuffer[PipelineStage]

    nextStages += new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)

    val model = new Pipeline()
      .setStages(nextStages.toArray)

    model
  }
}
