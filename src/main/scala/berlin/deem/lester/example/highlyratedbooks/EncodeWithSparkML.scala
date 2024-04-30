package berlin.deem.lester.example.highlyratedbooks

import berlin.deem.lester.context.{EncodeFeatures, Split}

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{HashingTF, OneHotEncoder, StandardScaler, StringIndexer, Tokenizer, VectorAssembler}
import org.apache.spark.sql.DataFrame

class EncodeWithSparkML {

  @Split
  def trainTestSplit(data: DataFrame, randomSeed: Long): (DataFrame, DataFrame) = {
    val splits = data.randomSplit(Array(0.8, 0.2), randomSeed)
    val train = splits(0)
    val test = splits(1)

    (train, test)
  }

  @EncodeFeatures
  def encodeBooks(): Pipeline = {

    val stages = ArrayBuffer[PipelineStage]()

    val categoricalColumns = Array("tag_id", "original_publication_year")
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

    val numericalColumns = Array("work_text_reviews_count")

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

    featureTransform
  }
}
