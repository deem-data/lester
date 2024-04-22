from lester.context import datasource, prepare, DataframeDialect
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import when, col


@prepare(
    dialect=DataframeDialect.PYSPARK,
    sources=[
        datasource('books', track_provenance_by=['goodreads_book_id']),
        datasource('categories', track_provenance_by=['tag_id']),
        datasource('book_tags')])
def label_books(books, categories, book_tags):

    english_books = books \
        .dropna() \
        .filter("language_code == 'eng'")

    popular_categories = categories.filter("popularity >= 10")
    categories_with_books = popular_categories.join(book_tags, on='tag_id')

    labeled_books = english_books.join(categories_with_books, on='goodreads_book_id')
    labeled_books = labeled_books.withColumn('label', when(col('average_rating') > 4.2, 1.0).otherwise(0.0))
    labeled_books = labeled_books \
        .withColumn('work_text_reviews_count', col('work_text_reviews_count').cast(IntegerType()))

    return labeled_books
