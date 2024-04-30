import berlin.deem.lester.Runner

import scala.collection.immutable.HashMap

object example extends App {

  val pipelinePackage = "berlin.deem.lester.example.highlyratedbooks"

  val nameToPath = HashMap(
    "books" -> "data/books.csv",
    "categories" -> "data/categories.csv",
    "bookTags" -> "data/book_tags.csv",
  )
  val randomSeed = 42

  Runner.run("example_sparkml_scala", pipelinePackage, nameToPath, randomSeed)
}
