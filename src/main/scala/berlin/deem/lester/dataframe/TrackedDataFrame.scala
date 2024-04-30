package berlin.deem.lester.dataframe

import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions.monotonically_increasing_id

object TrackedDataframe {
  def apply(df: DataFrame): TrackedDataframe = {
    new TrackedDataframe(df, Map.empty, Array.empty)
  }

  def withProvenanceColumn(df: DataFrame, sourceName: String): TrackedDataframe = {

    val columnProvenance = df.columns
      .map { column =>
        column -> Array(s"${sourceName}.${column}")
      }
      .toMap

    val provenanceColumnName = s"__lester_provenance_${sourceName}"
    val dfWithProvenance = df.withColumn(provenanceColumnName, monotonically_increasing_id())

    new TrackedDataframe(dfWithProvenance, columnProvenance, Array(provenanceColumnName))
  }
}

class TrackedDataframe(
    val df: DataFrame,
    val columnProvenance: Map[String, Array[String]],
    val provenanceColumns: Array[String]
) {


  val na = new DropNaIntermediate(this)

  def filter(expression: String): TrackedDataframe = {
    val resultDf = this.df.filter(expression)
    new TrackedDataframe(resultDf, this.columnProvenance, this.provenanceColumns)
  }

  def select(col: String, cols: String*): TrackedDataframe = {
    val otherCols: Array[String] = cols.toArray
    var projectionColumns = Array(col) ++ otherCols
    for (provenanceColumn <- this.provenanceColumns) {
      if (!projectionColumns.contains(provenanceColumn)) {
        projectionColumns = projectionColumns ++ Array(provenanceColumn)
      }
    }

    val resultDf = df.select(projectionColumns(0), projectionColumns.slice(1, projectionColumns.length): _*)

    new TrackedDataframe(resultDf, this.columnProvenance, this.provenanceColumns)
  }

  // TODO make this package-private
  def dropna(): TrackedDataframe = {
    val resultDf = this.df.na.drop()
    new TrackedDataframe(resultDf, this.columnProvenance, this.provenanceColumns)
  }

  def join(other: TrackedDataframe, on: Seq[String]): TrackedDataframe = {
    val resultDf = this.df.join(other.df, on)
    // TODO this needs duplicate checks and special care for self-joins
    val resultProvenanceColumns = this.provenanceColumns ++ other.provenanceColumns
    val resultColumnProvenance = this.columnProvenance ++ other.columnProvenance
    new TrackedDataframe(resultDf, resultColumnProvenance, resultProvenanceColumns)
  }

  def withColumn(colName: String, col: Column): TrackedDataframe = {
    val resultDf = this.df.withColumn(colName, col)

    val nonProvenanceColumns = this.df.columns.filterNot(_.startsWith("__lester"))

    // TODO we would need to parse the expression to understand the provenance in detail
    val resultColumnProvenance = this.columnProvenance ++ Map(colName -> nonProvenanceColumns)

    new TrackedDataframe(resultDf, resultColumnProvenance, this.provenanceColumns)
  }

}

class DropNaIntermediate(val trackedDf: TrackedDataframe) {
  def drop(): TrackedDataframe = {
    this.trackedDf.dropna()
  }
}
