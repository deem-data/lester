package berlin.deem.lester.example.highlyratedbooks

import berlin.deem.lester.context.{Datasource, Prepare}
//import org.apache.spark.sql.DataFrame
import berlin.deem.lester.dataframe.{TrackedDataframe => DataFrame}
import org.apache.spark.sql.functions.{col, when}

class PrepareWithSpark {

  @Prepare(sources = Array(
    new Datasource(name = "books", trackProvenanceBy = Array("goodreads_book_id")),
    new Datasource(name = "categories", trackProvenanceBy= Array("tag_id")),
    new Datasource(name = "bookTags")))
  def labelBooks(books: DataFrame, categories: DataFrame, bookTags: DataFrame): DataFrame = {

    val englishBooks = books
      .na.drop()
      .filter("language_code == 'eng'")

    val popularCategories = categories.filter("popularity >= 10")
    val categoriesWithBooks = popularCategories.join(bookTags, Seq("tag_id"))

    var labeledBooks = englishBooks.join(categoriesWithBooks, Seq("goodreads_book_id"))
    labeledBooks = labeledBooks.withColumn("label", when(col("average_rating") > 4.2, 1.0).otherwise(0.0))
    labeledBooks = labeledBooks
      .withColumn("work_text_reviews_count", col("work_text_reviews_count").cast("integer"))

    labeledBooks
  }
}
