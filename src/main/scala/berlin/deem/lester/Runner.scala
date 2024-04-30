package berlin.deem.lester

import berlin.deem.lester.context.{AnnotationScanner, EncodeFeatures, Prepare, Split, TrainModel}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}

import berlin.deem.lester.dataframe.TrackedDataframe
import org.apache.spark.sql.functions.col
import java.io.File
import java.util.UUID

import Utils._

object Runner {

  private def loadInputs(
    prepareClass: Class[_],
    prepareMethodName: String,
    spark: SparkSession,
    nameToPath: Map[String, String]
  ): Array[TrackedDataframe] = {

    val prepareAnnotation = prepareClass.getDeclaredMethods
      .find(_.getName == prepareMethodName)
      .get
      .getAnnotationsByType(classOf[Prepare])
      .head

    val inputs = prepareAnnotation.sources()
      .map { source =>
        val df = spark.read.option("header", "true").csv(nameToPath(source.name()))

        if (source.trackProvenanceBy().isEmpty) {
          TrackedDataframe(df)
        } else {
          TrackedDataframe.withProvenanceColumn(df, source.name())
        }
      }

    inputs
  }



  def run(pipelineName: String, pipelinePackage: String, nameToPath: Map[String, String], randomSeed: Long) = {

    val artifactDirectoryPath = s".lester/${pipelineName}/${UUID.randomUUID()}"
    val artifactDirectory = new File(artifactDirectoryPath)

    if (!artifactDirectory.exists()) {
      artifactDirectory.mkdirs()
    }

    jsonToFile(sourcePathsToJson(nameToPath), s"${artifactDirectoryPath}/source_paths.json")

    val (prepareClassName, prepareMethodName) = AnnotationScanner.scanFor(pipelinePackage, classOf[Prepare])
    val (splitClassName, splitMethodName) = AnnotationScanner.scanFor(pipelinePackage, classOf[Split])
    val (encodeFeaturesClassName, encodeFeaturesMethodName) =
      AnnotationScanner.scanFor(pipelinePackage, classOf[EncodeFeatures])
    val (trainModelClassName, trainModelMethodName) = AnnotationScanner.scanFor(pipelinePackage, classOf[TrainModel])

    val spark = SparkSession.builder()
      .master("local[2]")
      .getOrCreate()

    val prepareClass = Class.forName(prepareClassName)
    val splitClass = Class.forName(splitClassName)
    val encodeFeaturesClass = Class.forName(encodeFeaturesClassName)
    val trainModelClass = Class.forName(trainModelClassName)

    val inputs = loadInputs(prepareClass, prepareMethodName, spark, nameToPath)

    val trackedPreparedData = makeCallable(prepareClass, prepareMethodName)
      .apply(inputs: _*)
      .asInstanceOf[TrackedDataframe]

    jsonToFile(columnProvenanceToJson(trackedPreparedData.columnProvenance),
      s"${artifactDirectoryPath}/column_provenance.json")

    val preparedData = trackedPreparedData.df
    preparedData.cache()

    val (train, test) = makeCallable(splitClass, splitMethodName)
      .apply(preparedData, randomSeed)
      .asInstanceOf[(DataFrame, DataFrame)]

    train.cache()
    test.cache()

    val trainProvenance = train
      .select(trackedPreparedData.provenanceColumns.map(col): _*)
      .coalesce(1)

    val testProvenance = test
      .select(trackedPreparedData.provenanceColumns.map(col): _*)
      .coalesce(1)

    asSingleParquetFile(trainProvenance, artifactDirectoryPath, "row_provenance_X_train.parquet")
    asSingleParquetFile(testProvenance, artifactDirectoryPath, "row_provenance_X_test.parquet")

    val featureTransform = makeCallable(encodeFeaturesClass, encodeFeaturesMethodName)
      .apply()
      .asInstanceOf[Pipeline]

    val model = makeCallable(trainModelClass, trainModelMethodName)
      .apply()
      .asInstanceOf[Pipeline]

    val fittedFeatureTransform = featureTransform.fit(train)
    val X_y_train = fittedFeatureTransform.transform(train)
    val X_y_test = fittedFeatureTransform.transform(test)

    X_y_train.cache()
    X_y_test.cache()

    val X_train = X_y_train.select("features")
    matrixAsNpyFile(X_train, s"${artifactDirectoryPath}/X_train.npy")
    val y_train = X_y_train.select("label")
    vectorAsNpyFile(y_train, s"${artifactDirectoryPath}/y_train.npy")

    val X_test = X_y_test.select("features")
    matrixAsNpyFile(X_test, s"${artifactDirectoryPath}/X_test.npy")
    val y_test = X_y_test.select("label")
    vectorAsNpyFile(y_test, s"${artifactDirectoryPath}/y_test.npy")

    println("Training model")
    val fittedModel = model.fit(X_y_train)

    val predictions = fittedModel.transform(X_y_test)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy = $accuracy")

    spark.stop()
    System.exit(0)
  }
}